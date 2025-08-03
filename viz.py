import click
import json
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch
from pnpl.datasets import LibriBrainPhoneme
import re


def plot_class_specific_scores(scores, random_scores, metric_name, labels, sort=True):

    num_classes = len(labels)

    # If sorting is requested, reorder the bars based on the criteria.
    if sort:
        order = torch.argsort(scores).flip(dims=[0])
    else:
        order = torch.arange(len(scores))

    # Reorder the arrays along the class dimension (axis=1) and update the summary statistics
    scores = scores[order]
    random_scores = random_scores[order]
    labels = [labels[i] for i in order]
    # Positions of the groups on the x-axis
    x = np.arange(num_classes)

    # Width of each bar
    width = 0.35

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(25, 12))

    # Plot Random scores bars
    bars1 = ax.bar(x - width/2, random_scores, width,
                   label='Random', capsize=5, color='skyblue', edgecolor='black')

    # Plot Actual score bars
    bars2 = ax.bar(x + width/2, scores, width,
                   label='Model', capsize=5, color='salmon', edgecolor='black')

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


def main(result_path: Path):
    with open(result_path, 'r') as f:
        results = json.load(f)

    f1_keys = [k for k in results.keys() if re.match(r'^val_class_\d+_f1$', k)]
    f1_naive_keys = [k for k in results.keys() if re.match(r'^val_class_\d+_naive_f1$', k)]
    f1_random_keys = [k for k in results.keys() if re.match(r'^val_class_\d+_random_f1$', k)]

    f1_by_class = torch.tensor([results[k] for k in f1_keys])
    f1_naive_by_class = torch.tensor([results[k] for k in f1_naive_keys])
    f1_random_by_class = torch.tensor([results[k] for k in f1_random_keys])

    val_dataset = LibriBrainPhoneme(
        data_path=f"data/",
        partition="validation",
        tmin=0.0,
        tmax=0.5
    )

    fig, ax = plot_class_specific_scores(
        f1_by_class,
        f1_random_by_class,
        "F1",
        val_dataset.labels_sorted
    )

    fig.savefig(result_path.parent / "f1_by_class.png")


@click.command()
@click.argument('result_path', type=click.Path(exists=True))
def cli(result_path):
    main(Path(result_path))


if __name__ == "__main__":
    cli()
