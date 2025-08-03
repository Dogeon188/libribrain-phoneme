from pathlib import Path
import click
import re
import time
from torch.utils.data import DataLoader
from pnpl.datasets import LibriBrainCompetitionHoldout
from lightning.pytorch.accelerators import find_usable_cuda_devices
import yaml
from tqdm import tqdm
import torch

from .models.configurable_modules.classification_module import ClassificationModule


def generate_submission(name: str = "submission", model: ClassificationModule = None, config: dict = None):
    """
    Generates a submission file for the LibriBrain competition holdout dataset.
    """
    # First, instantiate the Competition Holdout dataset
    print("Generating submission file...")
    submission_dataset = LibriBrainCompetitionHoldout(**config["data"]["dataset"])

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
    
    # predictions = []

    # print(f"Generating predictions for {len(dataset)} segments...")
    # print(f"Using batch size: {batch_size}, device: {device}")

    # with torch.no_grad():
    #     for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing batches")):
    #         # batch_data shape: (batch_size, 306, 125)
    #         batch_data = batch_data.to(device)

    #         # Forward pass
    #         logits = model(batch_data)  # Shape: (batch_size, 39)
    #         probs = torch.softmax(logits, dim=1)  # Convert to probabilities

    #         # Move back to CPU and store individual predictions
    #         probs_cpu = probs.cpu()
    #         for i in range(probs_cpu.shape[0]):
    #             predictions.append(probs_cpu[i])  # Shape: (39,)

    print("Generated submission predictions in ",
          time.time() - start_time, " seconds")

    submission_dataset.generate_submission_in_csv(
        submission_preds,
        f"preds_{name}.csv"
    )


def main(config_path: Path, checkpoint_path: Path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Config file not found. Please provide a valid path")

    model = ClassificationModule.load_from_checkpoint(checkpoint_path)
    generate_submission(name=checkpoint_path.stem, model=model, config=config)


@click.command()
@click.option('--config', 'config_path', type=click.Path(exists=True), required=True, help='Path to the config YAML file.')
@click.option('--checkpoint', 'checkpoint_path', type=click.Path(exists=True), required=True, help='Path to the model checkpoint file.')
def cli(config_path, checkpoint_path):
    main(config_path, checkpoint_path)


if __name__ == "__main__":
    cli()
