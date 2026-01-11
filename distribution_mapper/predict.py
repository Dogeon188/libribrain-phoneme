import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml
from tqdm import tqdm

from distribution_mapper import DistributionMapper
from pnpl.datasets import LibriBrainCompetitionHoldout, LibriBrainPhoneme


class DebugShowDim(nn.Module):
    def __init__(self, name: str = "DebugShowDim"):
        super().__init__()
        self.name = name

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        print(f"{self.name} input shape: {input_.shape}")
        exit(0)
        return input_

    def extra_repr(self) -> str:
        return f"name={self.name}"


class ResnetBlock(nn.Module):
    def __init__(self, model_config: list[tuple[str, dict]], downsample: list[tuple[str, dict]] | None = None):
        super().__init__()
        self.model_config = model_config
        self.module_list = nn.ModuleList()
        self.module_list.extend(modules_from_config(model_config))
        if downsample is not None:
            self.downsample = nn.ModuleList(modules_from_config(downsample))
        else:
            self.downsample = None

    def forward(self, x):
        x_residual = x
        for module in self.module_list:
            x = module(x)
        if self.downsample is not None:
            for module in self.downsample:
                x_residual = module(x_residual)
        if x_residual.shape != x.shape:
            raise ValueError(f"Shape mismatch in ResnetBlock: "
                             f"{x_residual.shape} != {x.shape}")
        return x + x_residual


class Unsqueeze(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return input_.unsqueeze(self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class Squeeze(nn.Module):
    def __init__(self, dim: int | list[int]):
        super().__init__()
        self.dim = dim

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return input_.squeeze(self.dim)

    def extra_repr(self) -> str:
        return f"dims={self.dim}"
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # A tensor consists of all the possible positions (index) e.g 0, 1, 2, ... max length of input
        # Shape (pos) --> [max len, 1]
        pos = torch.arange(0, max_len).unsqueeze(1)
        self.encoding = torch.zeros((max_len, d_model))

        # sin for even item of position's dimension
        sin_den = 10000 ** (torch.arange(0, d_model, 2)/d_model)
        cos_den = 10000 ** (torch.arange(1, d_model, 2)/d_model)  # cos for odd

        self.encoding[:, 0::2] = torch.sin(pos / sin_den)
        self.encoding[:, 1::2] = torch.cos(pos / cos_den)

        self.encoding = self.encoding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)

        # We want pos_encoding be saved and restored in the `state_dict`, but not trained by the optimizer
        # hence registering it!
        # Source & credits: https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723/2
        self.register_buffer('pos_encoding', self.encoding)

    def forward(self, x):
        # shape (token_embedding) --> [sentence len, batch size, d_model]

        # Concatenating embeddings with positional encodings
        # Note: As we made positional encoding with the size max length of sentence in our dataset
        #       hence here we are picking till the sentence length in a batch
        #       Another thing to notice is in the Transformer's paper they used FIXED positional encoding,
        #       there are methods where we can also learn them
        return self.dropout(x + self.pos_encoding[:, :x.size(1), :]).to(x.device)

    
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import yaml
from tqdm import tqdm

from pnpl.datasets import LibriBrainCompetitionHoldout  


import torch
from torch.nn import Conv1d, ELU
from torch.nn import Softsign, GRU, Linear, ReLU, Sigmoid, GELU, BatchNorm1d
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import LinearLR, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from libribrain_experiments.models.util_layers import Permute
from torch.nn import LayerNorm
import copy

from libribrain_experiments.models.average_groups import AverageGroups
from libribrain_experiments.models.meg2vec import Meg2VecModel
from libribrain_experiments.models.meg2vec.model import Meg2VecForPreTraining

def modules_from_config(modules: list[tuple[str, dict]]):
    modules_list = []
    for module in modules:
        module_type = list(module)[0]
        config = module[module_type]
        match module_type:
            case "linear":
                module = Linear(**config)
            # convolutional layers
            case "conv1d":
                module = Conv1d(**config)
            case "conv2d":
                module = nn.Conv2d(**config)
            case "depthwise_conv1d":
                module = Conv1d(
                    **config,
                    groups=config["in_channels"])
            case "depthwise_conv2d":
                module = nn.Conv2d(
                    **config,
                    groups=config["in_channels"])
            # transformer sequence classification
            case "positional_encoding":
                module = PositionalEncoding(**config)
            case "transformer":
                module = nn.Transformer(**config)
            # source: https://github.com/maqboolkhan/Transformer_classifier_pytorch?tab=readme-ov-file
            case "transformer_encoder":
                if "encoder_layer" not in config:
                    raise ValueError(
                        "transformer_encoder requires 'encoder_layer' in config")

                encoder_config = copy.deepcopy(config)

                encoder_layer = encoder_config.pop("encoder_layer", None)
                if encoder_layer is None:
                    raise ValueError(
                        "transformer_encoder requires 'encoder_layer' in config")
                if not isinstance(encoder_layer, list):
                    encoder_layer = [encoder_layer]
                if len(encoder_layer) != 1:
                    raise ValueError(
                        "transformer_encoder requires exactly one encoder_layer in config")
                encoder_layer = modules_from_config(encoder_layer)

                norm_layer = encoder_config.pop("norm", None)
                if norm_layer is not None:
                    if not isinstance(norm_layer, list):
                        norm_layer = [norm_layer]
                    if len(norm_layer) != 1:
                        raise ValueError(
                            "transformer_encoder requires exactly one norm layer in config")
                    norm_layer = modules_from_config(norm_layer)[0]
                    encoder_config["norm"] = norm_layer

                module = nn.TransformerEncoder(
                    encoder_layer=encoder_layer[0],
                    **encoder_config
                )
            case "transformer_encoder_layer":
                module = nn.TransformerEncoderLayer(**config)
            # activation functions
            case "softsign":
                module = Softsign()
            case "relu":
                module = ReLU()
            case "elu":
                module = ELU()
            case "sigmoid":
                module = Sigmoid()
            case "gelu":
                module = GELU()
            case "silu":
                module = nn.SiLU()
            case "softmax":
                module = nn.Softmax(dim=-1)
            # dropout and normalization layers
            case "dropout":
                module = nn.Dropout(**config)
            case "dropout1d":
                module = nn.Dropout1d(**config)
            case "batch_norm1d":
                module = BatchNorm1d(**config)
            case "batch_norm2d":
                module = nn.BatchNorm2d(**config)
            case "layer_norm":
                module = LayerNorm(**config)
            case "average_groups":
                module = AverageGroups(**config)
            case "adaptive_avg_pool1d":
                module = nn.AdaptiveAvgPool1d(**config)
            case "average_pool2d":
                module = nn.AvgPool2d(**config)
            case "max_pool1d":
                module = nn.MaxPool1d(**config)
            # recurrent layers
            case "gru":
                module = GRU(**config)
            case "lstm":
                module = nn.LSTM(**config)
            # reshaping layers
            case "flatten":
                module = nn.Flatten()
            case "unsqueeze":
                module = Unsqueeze(**config)
            case "squeeze":
                module = Squeeze(**config)
            case "permute":
                module = Permute(**config)
            # custom layers
            case "resnet_block":
                module = ResnetBlock(**config)
            case "meg2vec":
                module = Meg2VecModel(**config)
            case "meg2wav":
                # Extract pretrained weights path if provided
                config_copy = copy.deepcopy(config)
                pretrained_weights = config_copy.pop("pretrained_weights", None)
                use_full_pretraining_model = config_copy.pop("use_full_pretraining_model", False)
                
                # Create the appropriate model
                if use_full_pretraining_model:
                    # Use full pre-training model (for continuing pre-training or research)
                    module = Meg2VecForPreTraining(**config_copy)
                else:
                    # Use core model (for downstream tasks and feature extraction)
                    module = Meg2VecModel(**config_copy)
                
                # Load pretrained weights if provided
                if pretrained_weights is not None:
                    try:
                        # Load checkpoint
                        checkpoint = torch.load(pretrained_weights, map_location='cpu')
                        
                        # Extract model state dict if it's a training checkpoint
                        if 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                        else:
                            state_dict = checkpoint
                        
                        if use_full_pretraining_model:
                            # Load full model weights
                            module.load_state_dict(state_dict, strict=False)
                        else:
                            # Extract only encoder weights for core model
                            encoder_state_dict = {}
                            for key, value in state_dict.items():
                                # Keep only encoder-related weights (exclude quantizer, projection head, etc.)
                                if key.startswith('meg2vec.'):
                                    # Remove 'meg2vec.' prefix to match Meg2VecModel structure
                                    new_key = key[8:]  # Remove 'meg2vec.'
                                    encoder_state_dict[new_key] = value
                                elif not any(exclude in key for exclude in [
                                    'quantizer', 'project_q', 'project_hid', 
                                    'dropout_features', 'dropout_input'
                                ]):
                                    # Keep weights that don't belong to pre-training specific components
                                    encoder_state_dict[key] = value
                            
                            module.load_state_dict(encoder_state_dict, strict=False)
                        
                        model_type = "full pre-training" if use_full_pretraining_model else "core encoder"
                        print(f"Loaded {model_type} model with pretrained weights from: {pretrained_weights}")
                        
                    except Exception as e:
                        print(f"Warning: Failed to load pretrained weights from {pretrained_weights}: {e}")
                        print("Continuing with randomly initialized weights...")
            case "_debug_show_dim":
                module = DebugShowDim(**config)
            case _:
                raise ValueError(f"Unsupported module_type: {module_type}")
        modules_list.append(module)
    return modules_list

def build_model_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    model_layers = modules_from_config(config["model"])
    return nn.Sequential(*model_layers)



class SpeechWithTeacherDataset(Dataset):
    def __init__(self, base_dataset, teacher_preds: dict | None = None):
        self.base = base_dataset
        self.teacher_preds = teacher_preds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]
        if self.teacher_preds is not None:
            teacher_prob = self.teacher_preds[idx]
            return data.float(), teacher_prob
        else:
            return data.float()


def compare_mapped_vs_raw(yaml_path, ckpt_path, mapper_ckpt, device="cuda"):

    baseline_model = build_model_from_yaml(yaml_path).to(device)

    state = torch.load(ckpt_path, map_location=device)
    if "state_dict" in state:   # pytorch-lightning 
        baseline_model.load_state_dict(state["state_dict"], strict=False)
    else:                      
        baseline_model.load_state_dict(state, strict=False)
    baseline_model.eval()

    print("Baseline model loaded")
    print("the first layer:", baseline_model[0])

    # holdout_ds = LibriBrainCompetitionHoldout(
    #     data_path="../data",
    #     standardize=True,
    #     tmin=0.0,
    #     tmax=0.5,
    #     task="phoneme"
    # )
    holdout_ds = LibriBrainPhoneme(
        data_path="../data",
        standardize=True,
        tmin=0.0,
        tmax=0.5,
        partition="test",  
    )

    holdout_loader = DataLoader(holdout_ds, batch_size=8, shuffle=False)  

    mapper = DistributionMapper().to(device)
    mapper.load_state_dict(torch.load(mapper_ckpt, map_location=device))
    mapper.eval()
    print("Mapper loaded")

    batch = next(iter(holdout_loader))
    if isinstance(batch, (list, tuple)):
        batch = batch[0]
    batch = batch.to(device)

    with torch.no_grad():
        raw_logits = baseline_model(batch)
        mapped_logits = baseline_model(mapper(batch))

    print("Raw input shape:", batch.shape)
    print("Mapped input shape:", mapper(batch).shape)
    print("Raw logits shape:", raw_logits.shape)
    print("Mapped logits shape:", mapped_logits.shape)

    raw_probs, mapped_probs = [], []

    with torch.no_grad():
        for X in tqdm(holdout_loader, desc="Predicting raw"):
            if isinstance(X, (list, tuple)):
                X = X[0]
            X = X.to(device)
            logits = baseline_model(X)
            probs = torch.softmax(logits, dim=-1)
            raw_probs.append(probs.cpu())
    raw_probs = torch.cat(raw_probs, dim=0).numpy()

    with torch.no_grad():
        for X in tqdm(holdout_loader, desc="Predicting mapped"):
            if isinstance(X, (list, tuple)):
                X = X[0]
            X = X.to(device)
            X_mapped = mapper(X)
            logits = baseline_model(X_mapped)
            probs = torch.softmax(logits, dim=-1)
            mapped_probs.append(probs.cpu())
    mapped_probs = torch.cat(mapped_probs, dim=0).numpy()

    return raw_probs, mapped_probs



if __name__ == "__main__":
    yaml_path = "../configs/phoneme/megt/base-config.yaml"
    ckpt_path = "../out/phoneme-megt/best-val_bal_acc-megt-hpo-2-epoch=24-val_f1_macro=0.5725.ckpt"
    mapper_ckpt = "out/mapper_model.pth"
    device = "cuda"

    raw_probs, mapped_probs = compare_mapped_vs_raw(yaml_path, ckpt_path, mapper_ckpt, device)

    # raw_pred_classes = np.argmax(raw_probs, axis=1)
    # mapped_pred_classes = np.argmax(mapped_probs, axis=1)

    # num_classes = raw_probs.shape[1]
    # raw_counts = np.bincount(raw_pred_classes, minlength=num_classes)
    # mapped_counts = np.bincount(mapped_pred_classes, minlength=num_classes)

    # print("Raw holdout predicted class counts:", raw_counts)
    # print("Mapped holdout predicted class counts:", mapped_counts)

    # pd.DataFrame({"segment_idx": np.arange(len(raw_pred_classes)),
    #               "pred_class": raw_pred_classes}).to_csv("raw_pred_classes.csv", index=False)
    
    # pd.DataFrame({"segment_idx": np.arange(len(mapped_pred_classes)),
    #               "pred_class": mapped_pred_classes}).to_csv("mapped_pred_classes.csv", index=False)

    # print("Saved as CSV")

    # x = np.arange(1, num_classes + 1)
    # width = 0.35

    # plt.figure(figsize=(12,6))
    # plt.bar(x - width/2, raw_counts, width, label="Raw Holdout")
    # plt.bar(x + width/2, mapped_counts, width, label="Mapped Holdout")
    # plt.xlabel("Phoneme Class (1~39)")
    # plt.ylabel("Count of Predicted Segments")
    # plt.title("Predicted Class Counts: Raw vs Mapped Holdout")
    # plt.xticks(x)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("pred_class_counts.png", dpi=300)
    # plt.show()

    # print("Saved as pred_class_counts.png")


    num_classes = raw_probs.shape[1]
    raw_df = pd.DataFrame(
        raw_probs,
        columns=[f"phoneme_{i+1}" for i in range(num_classes)]
    )
    raw_df.insert(0, "segment_idx", np.arange(len(raw_probs)))
    raw_df.to_csv("raw_pred_probs.csv", index=False)

    mapped_df = pd.DataFrame(
        mapped_probs,
        columns=[f"phoneme_{i+1}" for i in range(num_classes)]
    )
    mapped_df.insert(0, "segment_idx", np.arange(len(mapped_probs)))
    mapped_df.to_csv("mapped_pred_probs.csv", index=False)

    print("saved as raw_pred_probs.csv and mapped_pred_probs.csv")


    ### evaluation ###
    from sklearn.metrics import f1_score
    import numpy as np
    from pnpl.datasets import LibriBrainPhoneme

    test_dataset = LibriBrainPhoneme(
        data_path="../data",
        standardize=True,
        tmin=0.0,
        tmax=0.5,
        partition="test",
    )

    y_true = []
    for i in range(len(test_dataset)):
        _, label = test_dataset[i]  
        y_true.append(label.item())
    y_true = np.array(y_true)

    y_pred_mapped = np.argmax(mapped_probs, axis=1)

    f1_mapped = f1_score(y_true, y_pred_mapped, average="macro")

    print(f"Mapped input F1 macro: {f1_mapped:.4f}")


