#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pnpl.datasets.libribrain2025 import constants_utils
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml

from libribrain_experiments.utils import check_labels, get_dataset_partition_from_config


def configure_constants() -> None:
    constants_utils.set_remote_constants_url((Path.cwd() / "constants.json").as_uri())
    constants_utils.refresh_constants()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def allocate_worker_budget(total_workers: int) -> tuple[int, int]:
    train_workers = (total_workers + 1) // 2
    val_workers = total_workers // 2
    return train_workers, val_workers


def load_raw_train_and_val_datasets(data_config: dict):
    datasets_config = data_config["datasets"]
    if "train" not in datasets_config or "val" not in datasets_config:
        raise ValueError("Config must define both train and val datasets")

    train_dataset = get_dataset_partition_from_config(datasets_config["train"])
    train_channel_means = train_dataset.datasets[0].channel_means
    train_channel_stds = train_dataset.datasets[0].channel_stds
    labels = train_dataset.datasets[0].labels_sorted

    val_dataset = get_dataset_partition_from_config(
        datasets_config["val"],
        train_channel_means,
        train_channel_stds,
    )
    check_labels([labels, val_dataset.datasets[0].labels_sorted])
    return train_dataset, val_dataset, labels


def count_labels_in_dataset(
    dataset,
    num_classes: int,
    batch_size: int,
    num_workers: int,
    split_name: str = "dataset",
) -> torch.Tensor:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    counts = torch.zeros(num_classes, dtype=torch.int64)
    for batch in tqdm(loader, desc=f"Counting {split_name} labels", leave=False):
        labels = batch[1]
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels, dtype=torch.long)
        counts += torch.bincount(labels.to(dtype=torch.long).cpu(), minlength=num_classes)
    return counts


def build_distribution_table(
    labels: list[str],
    train_counts: torch.Tensor,
    val_counts: torch.Tensor,
) -> pd.DataFrame:
    train_total = int(train_counts.sum().item())
    val_total = int(val_counts.sum().item())

    summary = pd.DataFrame(
        {
            "phoneme": labels,
            "train_count": train_counts.tolist(),
            "train_pct": (train_counts.float() / train_total).tolist(),
            "val_count": val_counts.tolist(),
            "val_pct": (val_counts.float() / val_total).tolist(),
        }
    )
    return summary.sort_values(
        by=["train_count", "phoneme"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def render_distribution_plot(summary: pd.DataFrame, output_path: Path) -> None:
    labels = summary["phoneme"].tolist()
    train_pct = summary["train_pct"].tolist()
    val_pct = summary["val_pct"].tolist()
    train_counts = summary["train_count"].tolist()
    val_counts = summary["val_count"].tolist()

    positions = list(range(len(labels)))
    width = 0.42

    fig, ax = plt.subplots(figsize=(max(16, len(labels) * 0.45), 7))
    train_bars = ax.bar(
        [pos - width / 2 for pos in positions],
        train_pct,
        width=width,
        label="Train",
        color="#1f77b4",
    )
    val_bars = ax.bar(
        [pos + width / 2 for pos in positions],
        val_pct,
        width=width,
        label="Validation",
        color="#ff7f0e",
    )

    ax.set_title("Raw Phoneme Class Distribution by Split")
    ax.set_xlabel("Phoneme")
    ax.set_ylabel("Fraction of split")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    for bar, count in zip(train_bars, train_counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(count),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )
    for bar, count in zip(val_bars, val_counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(count),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@click.command(
    help="Plot raw phoneme class distributions for the train/validation split defined by a config."
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the experiment config file.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("distribution_plots"),
    show_default=True,
    help="Directory where the PNG and CSV summary will be written.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Batch size to use while counting labels. Defaults to the config dataloader batch size.",
)
@click.option(
    "--num-workers",
    type=int,
    default=None,
    help="Number of workers to use while counting labels. Defaults to the config dataloader num_workers.",
)
def main(
    config_path: Path,
    output_dir: Path,
    batch_size: int | None,
    num_workers: int | None,
) -> None:
    configure_constants()
    config = load_config(config_path)

    train_dataset, val_dataset, labels = load_raw_train_and_val_datasets(config["data"])

    dataloader_config = config["data"].get("dataloader", {})
    batch_size = batch_size or dataloader_config.get("batch_size", 256)
    num_workers = num_workers if num_workers is not None else dataloader_config.get("num_workers", 0)
    train_workers, val_workers = allocate_worker_budget(num_workers)

    with ThreadPoolExecutor(max_workers=2) as executor:
        train_future = executor.submit(
            count_labels_in_dataset,
            train_dataset,
            len(labels),
            batch_size,
            train_workers,
            "train",
        )
        val_future = executor.submit(
            count_labels_in_dataset,
            val_dataset,
            len(labels),
            batch_size,
            val_workers,
            "validation",
        )
        train_counts = train_future.result()
        val_counts = val_future.result()

    summary = build_distribution_table(labels, train_counts, val_counts)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "phoneme_class_distribution.csv", index=False)
    render_distribution_plot(summary, output_dir / "phoneme_class_distribution.png")


if __name__ == "__main__":
    main()
