from pathlib import Path

import torch
from torch import nn

from generate_layer_saliency_clustermap import (
    get_model_class,
    get_requested_activations,
    infer_row_specs,
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
