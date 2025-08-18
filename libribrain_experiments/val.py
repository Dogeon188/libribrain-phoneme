from pathlib import Path

import lightning
from tqdm import tqdm

from libribrain_experiments.models.configurable_modules.classification_module import ClassificationModule
import yaml
import torch
from torchmetrics.functional.classification import f1_score
from lightning.pytorch.accelerators import find_usable_cuda_devices
from pnpl.datasets import LibriBrainPhoneme
from libribrain_experiments.grouped_dataset import MyGroupedDatasetV3


def main(config: Path, checkpoint_path: Path):
    try:
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Config file not found. Please provide a valid path")

    lightning.seed_everything(config_data["general"]["seed"])

    train_dataset = LibriBrainPhoneme(
        data_path="./data/",
        tmin=0.0,
        tmax=0.5,
        standardize=True,
        partition="train",
    )

    raw_val_dataset = LibriBrainPhoneme(
        data_path="./data/",
        tmin=0.0,
        tmax=0.5,
        standardize=True,
        partition="validation",
        channel_means=train_dataset.channel_means,
        channel_stds=train_dataset.channel_stds,
    )
    val_dataset = MyGroupedDatasetV3(
        raw_val_dataset,
        grouped_samples=100,
        drop_remaining=False,
        average_grouped_samples=True,
        state_cache_path=Path(
            "./data_preprocessed/groupedv3/val_grouped_100.pt"),
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, **config_data["data"]["dataloader"])

    model = ClassificationModule.load_from_checkpoint(checkpoint_path)
    model = model.to(find_usable_cuda_devices()[0])

    all_targets = []
    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, leave=False):
            x, y = batch[0], batch[1]
            x = x.to(model.device)
            y = y.to(model.device)
            outputs = model(x)
            preds = outputs.argmax(dim=-1)

            all_targets.extend(y)
            all_preds.extend(preds)

    all_targets = torch.stack(all_targets)
    all_preds = torch.stack(all_preds)

    accuracy = (all_targets == all_preds).float().mean().item()
    f1_macro = f1_score(
        all_preds, all_targets, num_classes=39, average='macro', task='multiclass'
    ).item()

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation F1 Macro: {f1_macro:.4f}")


def cli(config, checkpoint_path):
    """
    Command line interface for running hyperparameter optimization.
    """
    main(config, checkpoint_path)


if __name__ == "__main__":
    cli()
