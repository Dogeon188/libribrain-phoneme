import click
import json
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from pnpl.datasets import LibriBrainPhoneme
import re


def plot_class_specific_scores(count1, count2, name1, name2, metric_name, labels, sort=True):

    num_classes = len(labels)

    # If sorting is requested, reorder the bars based on the criteria.
    if sort:
        order = torch.argsort(count1).flip(dims=[0])
    else:
        order = torch.arange(len(count1))

    # Reorder the arrays along the class dimension (axis=1) and update the summary statistics
    count1 = count1[order]
    count2 = count2[order]
    labels = [labels[i] for i in order]
    # Positions of the groups on the x-axis
    x = np.arange(num_classes)

    # Width of each bar
    width = 0.35

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(25, 12))

    # Plot class counts bars 1
    bars1 = ax.bar(x + width/2, count1, width,
                   label=name1, capsize=5, color='salmon', edgecolor='black')

    # Plot class counts bars 2
    bars2 = ax.bar(x - width/2, count2, width,
                   label=name2, capsize=5, color='skyblue', edgecolor='black')

    # Add labels and title
    ax.set_xlabel('Phonemes', fontsize=16)
    ax.set_ylabel(metric_name, fontsize=16)
    ax.set_title(metric_name + " for each Phoneme", fontsize=20)

    # Set x-axis tick labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=16)

    # Add legend
    ax.legend(fontsize=14)

    # Add grid for better readability
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

    # Adjust layout to prevent clipping of tick-labels
    plt.tight_layout()
    
    return fig, ax


def main(pred1_path: Path, pred2_path: Path):
    pred1 = pd.read_csv(pred1_path)
    pred2 = pd.read_csv(pred2_path)
    
    # instantiate the validation dataset to get the ordered labels
    val_dataset = LibriBrainPhoneme(
        data_path="data",
        partition="validation",
        tmin=0.0,
        tmax=0.5
    )

    # for each row, pick the column (phoneme) with the highest score
    # drop segment_idx, get "phoneme_X", extract X, convert to zero‐based int
    top1_1 = (
        pred1
        .drop(columns=["segment_idx"])
        .idxmax(axis=1)
        .str.extract(r"phoneme_(\d+)")
        .astype(int)[0]
        - 1
    )
    top1_2 = (
        pred2
        .drop(columns=["segment_idx"])
        .idxmax(axis=1)
        .str.extract(r"phoneme_(\d+)")
        .astype(int)[0]
        - 1
    )

    # count occurrences of each class index, ensure all classes appear
    n_classes = len(val_dataset.labels_sorted)
    class_counts1 = top1_1.value_counts().reindex(range(n_classes), fill_value=0)
    class_counts2 = top1_2.value_counts().reindex(range(n_classes), fill_value=0)

    fig, ax = plot_class_specific_scores(
        torch.Tensor(class_counts1.values),
        torch.Tensor(class_counts2.values),
        pred1_path.stem,
        pred2_path.stem,
        "Class Counts",
        val_dataset.labels_sorted,
    )

    fig.show()
    fig.savefig(pred1_path.parent / f"class_counts-{pred1_path.stem}-{pred2_path.stem}.png")

@click.command()
@click.argument('pred1_path', type=click.Path(exists=True, path_type=Path))
@click.argument('pred2_path', type=click.Path(exists=True, path_type=Path))
def cli(pred1_path: Path, pred2_path: Path):
    """
    CLI command to visualize class-specific scores from two prediction files.
    
    Args:
        pred1_path: Path to the first prediction CSV file.
        pred2_path: Path to the second prediction CSV file.
    """
    main(pred1_path, pred2_path)


if __name__ == "__main__":
    cli()
