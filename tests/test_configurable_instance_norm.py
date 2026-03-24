from pathlib import Path

import torch.nn as nn
import yaml

from libribrain_experiments.models.configurable_modules.utils import modules_from_config


def test_modules_from_config_builds_instance_norm1d_without_running_stats():
    modules = modules_from_config(
        [
            {
                "instance_norm1d": {
                    "num_features": 306,
                    "affine": False,
                    "track_running_stats": False,
                }
            }
        ]
    )

    assert len(modules) == 1
    layer = modules[0]
    assert isinstance(layer, nn.InstanceNorm1d)
    assert layer.num_features == 306
    assert layer.affine is False
    assert layer.track_running_stats is False


def test_baseline_xl_instance_norm_config_starts_with_input_instance_norm():
    config_path = Path("configs/phoneme/baseline-xl-instance-norm/base-config.yaml")

    with config_path.open() as f:
        config = yaml.safe_load(f)

    first_layer = config["model"][0]
    assert list(first_layer) == ["instance_norm1d"]
    assert first_layer["instance_norm1d"] == {
        "num_features": 306,
        "affine": False,
        "track_running_stats": False,
    }
