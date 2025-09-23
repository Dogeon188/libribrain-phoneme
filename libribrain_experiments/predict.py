from pathlib import Path
import click
import time
import numpy as np
from torch.utils.data import DataLoader
from pnpl.datasets import LibriBrainCompetitionHoldout
from lightning.pytorch.accelerators import find_usable_cuda_devices
import yaml
from tqdm import tqdm
import torch
from pnpl.datasets import LibriBrainPhoneme

from libribrain_experiments.hpo import get_run, load_search_space, runs_configs_from_search_space
from libribrain_experiments.models.scripted_modules import scripted_modules

from .models.configurable_modules.classification_module import ClassificationModule


def generate_submission(name: str = "submission", model: ClassificationModule = None, config: dict = None):
    """
    Generates a submission file for the LibriBrain competition holdout dataset.
    """
    print("Generating submission file...")

    # First, instantiate the Competition Holdout dataset
    train_dataset = LibriBrainPhoneme(
        data_path="./data/",
        tmin=0.0,
        tmax=0.5,
        standardize=True,
        partition="train",
    )
    submission_dataset = LibriBrainCompetitionHoldout(
        channel_means=train_dataset.channel_means,
        channel_stds=train_dataset.channel_stds,
        **config["data"]["dataset"]
    )

    dataset_loader = DataLoader(
        submission_dataset, **config["data"]["dataloader"])
    segments_to_predict = len(submission_dataset)

    print("Number of segments to predict: ", segments_to_predict)

    model.to(find_usable_cuda_devices()[0])

    start_time = time.time()
    submission_preds = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataset_loader, total=len(dataset_loader)):
            batch = batch.to(model.device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)
            submission_preds.extend(probs.cpu().numpy())

    print("Generated submission predictions in ",
          time.time() - start_time, " seconds")

    submission_dataset.generate_submission_in_csv(
        submission_preds,
        f"preds_{name}.csv"
    )


def main(model_config_path: Path, search_space: Path, run_index: int, submission_config_path: Path, checkpoint_path: Path):
    try:
        with open(model_config_path, 'r') as f:
            model_config_data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Config file not found. Please provide a valid path")

    try:
        with open(submission_config_path, 'r') as f:
            submission_config_data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Config file not found. Please provide a valid path")

    search_space_data = load_search_space(search_space)
    run_configs = runs_configs_from_search_space(search_space_data)
    if (run_index is None):
        run_index = np.random.randint(0, len(run_configs))
    model_config_data = get_run(model_config_data, run_configs, run_index)

    if type(model_config_data["model"]) is dict:
        model_class = scripted_modules[model_config_data["model"]["name"]]
        model = model_class.load_from_checkpoint(checkpoint_path)
    elif type(model_config_data["model"]) is list:
        model = ClassificationModule.load_from_checkpoint(checkpoint_path)

    generate_submission(name=checkpoint_path.stem,
                        model=model, config=submission_config_data)


@click.command()
@click.option('--config', 'config_path', type=click.Path(exists=True), required=True, help='Path to the config YAML file.')
@click.option('--checkpoint', 'checkpoint_path', type=click.Path(exists=True), required=True, help='Path to the model checkpoint file.')
def cli(config_path, checkpoint_path):
    main(config_path, checkpoint_path)


if __name__ == "__main__":
    cli()
