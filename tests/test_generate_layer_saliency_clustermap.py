from pathlib import Path

import click
import pandas as pd
import torch
from torch.utils.data import ConcatDataset
from torch import nn

import plot_phoneme_class_distribution as class_dist
from generate_layer_saliency_clustermap import (
    format_plot_row_label,
    get_model_class,
    get_requested_activations,
    infer_row_specs,
    get_cluster_map_figsize,
)
from plot_phoneme_class_distribution import (
    allocate_worker_budget,
    build_distribution_table,
    count_labels_in_dataset,
)
from libribrain_experiments.models.configurable_modules.classification_module import (
    ClassificationModule,
)
from libribrain_experiments.models.scripted_modules.stft import STFTClassificationModule


def test_get_model_class_uses_configurable_module_for_list_configs():
    assert get_model_class([{"relu": None}]) is ClassificationModule


def test_get_model_class_uses_scripted_module_for_named_configs():
    assert get_model_class({"name": "stft"}) is STFTClassificationModule


def test_infer_row_specs_uses_top_level_modules_for_generic_classification_module():
    module = ClassificationModule(
        model_config=[
            {"conv1d": {"in_channels": 4, "out_channels": 8, "kernel_size": 3, "padding": "same"}},
            {"relu": None},
            {"flatten": None},
            {"linear": {"in_features": 8 * 5, "out_features": 3}},
        ],
        n_classes=3,
        optimizer_config={"name": "adam", "config": {"lr": 1e-3}},
        loss_config={"name": "cross_entropy"},
    )

    row_names = [name for name, _ in infer_row_specs(module)]

    assert row_names == ["conv1d0", "linear0"]


def test_infer_row_specs_supports_stft_module():
    row_names = [name for name, _ in infer_row_specs(STFTClassificationModule())]

    assert row_names == [
        "res0.conv2d0",
        "res0.conv2d1",
        "downsample0",
        "res1.conv2d0",
        "res1.conv2d1",
        "res2.conv2d0",
        "res2.conv2d1",
        "res3.conv2d0",
        "res3.conv2d1",
        "downsample3",
        "classifier.linear0",
        "classifier.linear1",
    ]


class ToyAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
        self.proj = nn.Linear(8, 4)

    def forward(self, x):
        x, _ = self.attn(x, x, x, need_weights=False)
        return self.proj(x)


def test_filter_observed_row_specs_skips_modules_without_forward_hook_outputs():
    from generate_layer_saliency_clustermap import filter_observed_row_specs

    module = ToyAttentionBlock()
    row_specs = infer_row_specs(module)
    inputs = torch.randn(2, 5, 8)

    filtered_names = [name for name, _ in filter_observed_row_specs(module, row_specs, inputs)]

    assert filtered_names == ["attn", "proj"]


def test_get_requested_activations_skips_missing_rows():
    activations = {
        "attn": torch.randn(2, 5, 8),
        "proj": torch.randn(2, 5, 4),
    }

    observed_names, observed_activations = get_requested_activations(
        ["attn", "attn.out_proj", "proj"], activations
    )

    assert observed_names == ["attn", "proj"]
    assert observed_activations == [activations["attn"], activations["proj"]]


def test_format_plot_row_label_strips_conformer_prefixes_for_plotting():
    row_name = (
        "conformer_speech0.encoder.conformer_layers."
        "conformer_layer3.conv_module.conv1d2"
    )

    assert format_plot_row_label(row_name) == "conformer_layer3.conv_module.conv1d2"


def test_get_cluster_map_figsize_is_tall_enough_for_megconformer():
    assert get_cluster_map_figsize() == (28, 30)


class ToyLabelDataset(torch.utils.data.Dataset):
    labels_sorted = ["aa", "bb", "cc"]

    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.randn(2, 3), torch.tensor(self.labels[index], dtype=torch.long)


def test_count_labels_in_dataset_counts_each_class():
    dataset = ToyLabelDataset([0, 2, 1, 2, 2, 0])

    counts = count_labels_in_dataset(dataset, num_classes=3, batch_size=2, num_workers=0)

    assert counts.tolist() == [2, 1, 3]


def test_count_labels_in_dataset_reports_progress_bar():
    dataset = ToyLabelDataset([0, 2, 1, 2, 2, 0])
    seen = {}

    def fake_tqdm(iterable, **kwargs):
        seen["kwargs"] = kwargs
        return iterable

    original_tqdm = class_dist.tqdm
    class_dist.tqdm = fake_tqdm
    try:
        count_labels_in_dataset(
            dataset,
            num_classes=3,
            batch_size=2,
            num_workers=0,
            split_name="train",
        )
    finally:
        class_dist.tqdm = original_tqdm

    assert seen["kwargs"]["desc"] == "Counting train labels"
    assert seen["kwargs"]["leave"] is False


def test_build_distribution_table_contains_counts_and_percentages():
    labels = ["aa", "bb", "cc"]
    train_counts = torch.tensor([2, 1, 3])
    val_counts = torch.tensor([1, 1, 2])

    table = build_distribution_table(labels, train_counts, val_counts)

    expected = pd.DataFrame(
        {
            "phoneme": ["cc", "aa", "bb"],
            "train_count": [3, 2, 1],
            "train_pct": [3 / 6, 2 / 6, 1 / 6],
            "val_count": [2, 1, 1],
            "val_pct": [2 / 4, 1 / 4, 1 / 4],
        }
    )

    pd.testing.assert_frame_equal(table, expected)


def test_build_distribution_table_sorts_by_descending_train_frequency():
    labels = ["aa", "bb", "cc"]
    train_counts = torch.tensor([2, 5, 3])
    val_counts = torch.tensor([9, 1, 4])

    table = build_distribution_table(labels, train_counts, val_counts)

    assert table["phoneme"].tolist() == ["bb", "cc", "aa"]
    assert table["train_count"].tolist() == [5, 3, 2]
    assert table["val_count"].tolist() == [1, 4, 9]


def test_plot_phoneme_class_distribution_uses_click_command():
    assert isinstance(class_dist.main, click.core.Command)


def test_allocate_worker_budget_splits_total_budget():
    assert allocate_worker_budget(0) == (0, 0)
    assert allocate_worker_budget(1) == (1, 0)
    assert allocate_worker_budget(2) == (1, 1)
    assert allocate_worker_budget(5) == (3, 2)


def test_load_raw_train_and_val_datasets_uses_unwrapped_partitions():
    original_get_partition = class_dist.get_dataset_partition_from_config

    class ToyPartitionDataset(torch.utils.data.Dataset):
        def __init__(self, labels_sorted, channel_means=None, channel_stds=None):
            self.labels_sorted = labels_sorted
            self.channel_means = torch.tensor([1.0]) if channel_means is None else channel_means
            self.channel_stds = torch.tensor([2.0]) if channel_stds is None else channel_stds

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return torch.zeros(1), torch.tensor(0)

    seen = []

    def fake_get_partition(partition_config, channel_means=None, channel_stds=None):
        seen.append((partition_config, channel_means, channel_stds))
        return ConcatDataset([ToyPartitionDataset(["aa", "bb"], channel_means, channel_stds)])

    class_dist.get_dataset_partition_from_config = fake_get_partition
    try:
        train_dataset, val_dataset, labels = class_dist.load_raw_train_and_val_datasets(
            {
                "datasets": {
                    "train": [{"libribrain_phoneme": {"partition": "train"}}],
                    "val": [{"libribrain_phoneme": {"partition": "validation"}}],
                }
            }
        )
    finally:
        class_dist.get_dataset_partition_from_config = original_get_partition

    assert isinstance(train_dataset, ConcatDataset)
    assert isinstance(val_dataset, ConcatDataset)
    assert labels == ["aa", "bb"]
    assert seen[0][1] is None
    assert seen[0][2] is None
    assert torch.equal(seen[1][1], torch.tensor([1.0]))
    assert torch.equal(seen[1][2], torch.tensor([2.0]))
