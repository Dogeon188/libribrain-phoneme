#!/usr/bin/env python3

import argparse
import json
from copy import deepcopy
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import pdist
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libribrain_experiments.models.configurable_modules.classification_module import (
    ClassificationModule,
)
from libribrain_experiments.models.scripted_modules import scripted_modules
from libribrain_experiments.utils import get_datasets_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate layer saliency clustermaps for the neural2speech conformer model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phoneme/conformer/custom-2025-09-09-config.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "out/final-results/"
            "best-val_f1_macro-conformer-custom-2025-09-09-hpo-0-epoch=26-val_f1_macro=0.6409.ckpt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("saliency_maps/conformer-0.6409-layer-clustermap"),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def load_config_without_grouping(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_config = deepcopy(config["data"])
    general_config = data_config.setdefault("general", {})
    general_config.pop("dynamic_averaged_samples", None)
    general_config.pop("averaged_samples", None)
    general_config.pop("grouped_samples", None)
    return config, data_config


def collate(batch):
    xs = torch.stack([item[0] for item in batch], dim=0)
    ys = torch.as_tensor([int(item[1]) for item in batch], dtype=torch.long)
    return xs, ys


def camel_to_snake(name: str) -> str:
    chars = []
    for idx, char in enumerate(name):
        if char.isupper() and idx > 0 and not name[idx - 1].isupper():
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


GENERIC_CONTAINER_NAMES = {"module", "model", "modules_list", "sequential"}


def iter_named_param_modules(module: nn.Module, path: tuple[str, ...] = ()) -> list[tuple[str, nn.Module]]:
    specs = []
    has_direct_params = any(param.requires_grad for param in module.parameters(recurse=False))
    if has_direct_params and path:
        specs.append((".".join(path), module))

    type_counts = {}
    for child_name, child in module.named_children():
        type_name = camel_to_snake(type(child).__name__)
        type_index = type_counts.get(type_name, 0)
        type_counts[type_name] = type_index + 1
        if child_name in GENERIC_CONTAINER_NAMES:
            child_path = path
        else:
            segment = child_name if not child_name.isdigit() else f"{type_name}{type_index}"
            child_path = path + (segment,)
        specs.extend(iter_named_param_modules(child, child_path))
    return specs


def infer_row_specs(module: nn.Module):
    specs = iter_named_param_modules(module)
    if specs:
        return specs
    raise TypeError(f"Could not infer saliency row specs for module type {type(module).__name__}")


def filter_observed_row_specs(
    module: nn.Module, row_specs: list[tuple[str, nn.Module]], sample_inputs: torch.Tensor
) -> list[tuple[str, nn.Module]]:
    observed = set()
    handles = []

    def register(name, target_module):
        def hook(_mod, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(tensor):
                observed.add(name)

        handles.append(target_module.register_forward_hook(hook))

    for name, target_module in row_specs:
        register(name, target_module)

    try:
        with torch.no_grad():
            module(sample_inputs)
    finally:
        for handle in handles:
            handle.remove()

    return [(name, target_module) for name, target_module in row_specs if name in observed]


def get_model_class(model_config):
    if isinstance(model_config, list):
        return ClassificationModule
    if isinstance(model_config, dict):
        model_name = model_config.get("name")
        if model_name in scripted_modules:
            return scripted_modules[model_name]
    raise ValueError(f"Unsupported model config for saliency generation: {model_config!r}")


def load_model_from_checkpoint(config: dict, checkpoint_path: Path) -> nn.Module:
    model_class = get_model_class(config["model"])
    return model_class.load_from_checkpoint(
        str(checkpoint_path),
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )


def get_phoneme_lookup(dataset):
    current = dataset
    while current is not None:
        if hasattr(current, "idx_to_phoneme"):
            return current.idx_to_phoneme
        if hasattr(current, "id_to_phoneme"):
            return current.id_to_phoneme
        if hasattr(current, "original_dataset"):
            current = current.original_dataset
            continue
        if hasattr(current, "datasets") and current.datasets:
            current = current.datasets[0]
            continue
        break
    raise AttributeError(f"Could not find phoneme lookup on dataset type {type(dataset).__name__}")


def batch_first_scores(grad: torch.Tensor, batch_size: int) -> torch.Tensor:
    if grad.ndim == 1:
        return grad.unsqueeze(0)
    if grad.shape[0] == batch_size:
        return grad.abs().reshape(batch_size, -1).mean(dim=1)
    if grad.ndim > 1 and grad.shape[1] == batch_size:
        moved = torch.movedim(grad, 1, 0)
        return moved.abs().reshape(batch_size, -1).mean(dim=1)
    raise ValueError(f"Unexpected gradient shape {tuple(grad.shape)} for batch size {batch_size}")


def get_requested_activations(
    row_names: list[str], activations: dict[str, torch.Tensor]
) -> tuple[list[str], list[torch.Tensor]]:
    observed_row_names = [name for name in row_names if name in activations]
    if not observed_row_names:
        raise RuntimeError("No row activations were captured for the current batch")
    return observed_row_names, [activations[name] for name in observed_row_names]


def format_plot_row_label(row_name: str) -> str:
    label = row_name.removeprefix("conformer_speech0.")
    label = label.replace("encoder.conformer_layers.", "")
    return label


def get_cluster_map_figsize() -> tuple[int, int]:
    return (28, 12)
    # can change to e.g. (28, 30) for larger models with more layers


def render_cluster_map(df: pd.DataFrame, title: str, output_path: Path) -> None:
    values = df.to_numpy(dtype=float)
    if values.shape[0] > 1:
        row_linkage = linkage(pdist(values, metric="euclidean"), method="average")
        row_order = leaves_list(row_linkage)
    else:
        row_linkage = None
        row_order = np.arange(values.shape[0])

    if values.shape[1] > 1:
        col_linkage = linkage(pdist(values.T, metric="euclidean"), method="average")
        col_order = leaves_list(col_linkage)
    else:
        col_linkage = None
        col_order = np.arange(values.shape[1])

    ordered = values[np.ix_(row_order, col_order)]
    row_labels = [format_plot_row_label(df.index[i]) for i in row_order]
    col_labels = [df.columns[i] for i in col_order]

    fig = plt.figure(figsize=get_cluster_map_figsize())
    gs = gridspec.GridSpec(
        2,
        3,
        width_ratios=[1.5, 1.5, 20],
        height_ratios=[2, 20],
        wspace=0.1,
        hspace=0.1,
    )

    ax_cbar = fig.add_subplot(gs[0, 0])
    ax_col = fig.add_subplot(gs[0, 2])
    ax_row = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[1, 2])

    if col_linkage is not None:
        dendrogram(col_linkage, ax=ax_col, orientation="top", no_labels=True, color_threshold=None)
    ax_col.set_axis_off()

    if row_linkage is not None:
        dendrogram(row_linkage, ax=ax_row, orientation="left", no_labels=True, color_threshold=None)
    ax_row.set_axis_off()

    image = ax_heat.imshow(ordered, aspect="auto", interpolation="nearest", cmap="turbo", vmin=0.0, vmax=1.0)
    ax_heat.set_xticks(np.arange(len(col_labels)))
    ax_heat.set_xticklabels(col_labels, rotation=90, fontsize=12)
    ax_heat.set_yticks(np.arange(len(row_labels)))
    ax_heat.set_yticklabels(row_labels, fontsize=12)
    ax_heat.yaxis.tick_right()
    ax_heat.tick_params(axis="y", labelright=True, labelleft=False)
    ax_heat.set_title(title, fontsize=18, pad=14)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(image, cax=ax_cbar)
    cbar.ax.tick_params(labelsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_split(
    module: ClassificationModule,
    dataset,
    split_name: str,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
) -> None:
    module.eval()
    row_specs = infer_row_specs(module)
    sample_inputs, _ = collate([dataset[0]])
    sample_inputs = sample_inputs.to(module.device)
    row_specs = filter_observed_row_specs(module, row_specs, sample_inputs)
    if not row_specs:
        raise RuntimeError(f"No observable row specs found for {type(module).__name__}")
    row_names = [name for name, _ in row_specs]
    activations = {}
    handles = []

    def register(name, target_module):
        def hook(_mod, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(tensor):
                activations[name] = tensor

        handles.append(target_module.register_forward_hook(hook))

    for name, target_module in row_specs:
        register(name, target_module)

    phoneme_lookup = get_phoneme_lookup(dataset)
    accum = {row_name: {} for row_name in row_names}
    label_counts = {}

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )

    device = module.device
    for inputs, targets in tqdm(loader, desc=f"{split_name} saliency", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        activations.clear()
        module.zero_grad(set_to_none=True)

        logits = module(inputs)
        target_logits = logits.gather(1, targets.unsqueeze(1)).sum()
        batch_row_names, batch_activations = get_requested_activations(row_names, activations)
        grads = torch.autograd.grad(target_logits, batch_activations, allow_unused=False)

        batch_size_now = targets.shape[0]
        target_labels = [phoneme_lookup[int(idx)] for idx in targets.detach().cpu().tolist()]
        for row_name, grad in zip(batch_row_names, grads):
            scores = batch_first_scores(grad.detach(), batch_size_now).cpu().numpy()
            for label, score in zip(target_labels, scores.tolist()):
                accum[row_name].setdefault(label, []).append(float(score))

        for label in target_labels:
            label_counts[label] = label_counts.get(label, 0) + 1

    for handle in handles:
        handle.remove()

    phoneme_labels = sorted(label_counts)
    raw_matrix = np.zeros((len(row_names), len(phoneme_labels)), dtype=np.float64)
    for row_idx, row_name in enumerate(row_names):
        for col_idx, label in enumerate(phoneme_labels):
            values = accum[row_name].get(label, [])
            raw_matrix[row_idx, col_idx] = float(np.mean(values)) if values else 0.0

    raw_df = pd.DataFrame(raw_matrix, index=row_names, columns=phoneme_labels)
    row_min = raw_df.min(axis=1)
    row_max = raw_df.max(axis=1)
    denom = (row_max - row_min).replace(0.0, 1.0)
    normalized_df = raw_df.sub(row_min, axis=0).div(denom, axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(output_dir / f"{split_name}_layer_saliency_raw.csv")
    normalized_df.to_csv(output_dir / f"{split_name}_layer_saliency_rowminmax.csv")
    render_cluster_map(
        normalized_df,
        title=f"{split_name.title()} Layer Saliency",
        output_path=output_dir / f"{split_name}_layer_saliency_clustermap.png",
    )

    summary = {
        "split": split_name,
        "config_path": str(args.config.resolve()),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_samples": int(sum(label_counts.values())),
        "num_phonemes": len(phoneme_labels),
        "row_names": row_names,
        "phoneme_order": phoneme_labels,
        "label_counts": label_counts,
        "grouping_disabled_for_saliency": True,
    }
    with (output_dir / f"{split_name}_layer_saliency_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    global args
    args = parse_args()
    config, data_config = load_config_without_grouping(args.config)

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 16 if torch.cuda.is_available() else 8

    _, val_dataset, test_dataset, _ = get_datasets_from_config(data_config)
    module = load_model_from_checkpoint(config, args.checkpoint)
    if torch.cuda.is_available():
        module = module.cuda()

    analyze_split(module, val_dataset, "validation", args.output_dir, batch_size, args.num_workers)
    analyze_split(module, test_dataset, "test", args.output_dir, batch_size, args.num_workers)


if __name__ == "__main__":
    main()
