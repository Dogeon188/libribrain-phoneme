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

from libribrain_experiments.grouped_dataset import MyGroupedDatasetV3

from .models.configurable_modules.classification_module import ClassificationModule


def calculate_grouped_train_stats():
    CACHE_PATH = Path(
        f"./data_preprocessed/groupedv3/train_grouped_100_stats.pt")
    if CACHE_PATH.exists():
        stats = torch.load(CACHE_PATH, weights_only=False)
        return stats['mean'], stats['std']

    train_dataset = LibriBrainPhoneme(
        data_path="./data/",
        tmin=0.0,
        tmax=0.5,
        standardize=True,
        partition="train",
    )
    grouped_train_dataset = MyGroupedDatasetV3(
        train_dataset,
        grouped_samples=100,
        drop_remaining=False,
        average_grouped_samples=True,
        state_cache_path=Path(
            f"./data_preprocessed/groupedv3/train_grouped_100.pt"),
        balance=True,
        shuffle=True,
    )

    all_means = []
    all_stds = []

    train_loader = DataLoader(
        grouped_train_dataset, batch_size=256, shuffle=False, num_workers=4)

    for sample, *_ in tqdm(train_loader):
        arr = sample.numpy()
        mean = arr.mean(axis=2)  # mean over time
        std = arr.std(axis=2)    # std over time
        all_means.extend(mean)
        all_stds.extend(std)

    train_channel_means = np.mean(np.stack(all_means), axis=0)
    train_channel_stds = np.mean(np.stack(all_stds), axis=0)

    torch.save({
        'mean': train_channel_means,
        'std': train_channel_stds
    }, CACHE_PATH)

    return train_channel_means, train_channel_stds


def calculate_holdout_stats(holdout_dataset):
    all_means = []
    all_stds = []

    holdout_loader = DataLoader(
        holdout_dataset, batch_size=256, shuffle=False, num_workers=4)

    for sample in tqdm(holdout_loader):
        arr = sample.numpy()
        mean = arr.mean(axis=2)  # mean over time
        std = arr.std(axis=2)    # std over time
        all_means.extend(mean)
        all_stds.extend(std)

    holdout_channel_means = np.mean(np.stack(all_means), axis=0)
    holdout_channel_stds = np.mean(np.stack(all_stds), axis=0)

    return holdout_channel_means, holdout_channel_stds


def generate_submission(name: str = "submission", model: ClassificationModule = None, config: dict = None):
    """
    Generates a submission file for the LibriBrain competition holdout dataset.
    """
    print("Generating submission file...")

    # First, instantiate the Competition Holdout dataset
    submission_dataset = LibriBrainCompetitionHoldout(
        **config["data"]["dataset"]
    )

    # Calculate channel mean & std from grouped training dataset
    train_channel_means, train_channel_stds = calculate_grouped_train_stats()
    holdout_channel_means, holdout_channel_stds = calculate_holdout_stats(
        submission_dataset)
    
    print("Train channel means: ", train_channel_means)
    print("Train channel stds: ", train_channel_stds)
    print("Holdout channel means: ", holdout_channel_means)
    print("Holdout channel stds: ", holdout_channel_stds)

    dataset_loader = DataLoader(
        submission_dataset, **config["data"]["dataloader"])
    segments_to_predict = len(submission_dataset)

    print("Number of segments to predict: ", segments_to_predict)

    model.to(find_usable_cuda_devices()[0])

    start_time = time.time()
    submission_preds = []
    model.eval()
    with torch.no_grad():
        train_channel_means = torch.tensor(
            train_channel_means, dtype=torch.float32).to(model.device)
        train_channel_stds = torch.tensor(
            train_channel_stds, dtype=torch.float32).to(model.device)
        holdout_channel_means = torch.tensor(
            holdout_channel_means, dtype=torch.float32).to(model.device)
        holdout_channel_stds = torch.tensor(
            holdout_channel_stds, dtype=torch.float32).to(model.device)

        for batch in tqdm(dataset_loader, total=len(dataset_loader)):
            batch = batch.to(model.device)
            # assume data is gaussian
            # standardize than destandardize
            batch = (batch - holdout_channel_means[None, :, None]) / \
                holdout_channel_stds[None, :, None]
            batch = batch * train_channel_stds[None, :, None] + \
                train_channel_means[None, :, None]
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)
            submission_preds.extend(probs.cpu().numpy())

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
