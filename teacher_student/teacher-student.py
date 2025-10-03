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


def load_teacher_predictions(csv_path):
    df = pd.read_csv(csv_path)
    preds = {}
    for _, row in df.iterrows():
        seg_id = int(row["segment_idx"])
        probs = row.drop("segment_idx").to_numpy(dtype="float32")
        preds[seg_id] = torch.tensor(probs)
    return preds


class SpeechWithTeacherDataset(Dataset):
    def __init__(self, base_dataset, teacher_preds: dict | None = None):
        self.base = base_dataset
        self.teacher_preds = teacher_preds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx]  # base dataset return tensor shape (channels, time)
        if self.teacher_preds is not None:
            teacher_prob = self.teacher_preds[idx]  # (num_classes,)
            return data.float(), teacher_prob
        else:
            return data.float()


class ResNetBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)

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



def build_teacher_student(yaml_path, teacher_ckpt_path, device="cuda", random_student=False):
    teacher = build_model_from_yaml(yaml_path).to(device)
    student = build_model_from_yaml(yaml_path).to(device)

    ckpt = torch.load(teacher_ckpt_path, map_location=device)
    if "state_dict" in ckpt:  # PyTorch Lightning checkpoint
        state_dict = ckpt["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[len("model."):]] = v
            else:
                new_state_dict[k] = v
        teacher.load_state_dict(new_state_dict, strict=False)
        if not random_student:
            student.load_state_dict(new_state_dict, strict=False)
    else:
        teacher.load_state_dict(ckpt, strict=False)
        if not random_student:
            student.load_state_dict(ckpt, strict=False)

    teacher.eval()
    student.train()
    return teacher, student



def train_student(student, dataloader, optimizer, device, alpha=0.5):
    student.train()
    total_loss = 0.0
    loop = tqdm(dataloader, desc="Training", leave=False)
    for feats, teacher_prob in loop:
        feats, teacher_prob = feats.to(device), teacher_prob.to(device)
        logits = student(feats)
        log_probs = F.log_softmax(logits, dim=-1)
        soft_labels = teacher_prob
        hard_labels = teacher_prob.argmax(dim=-1)
        loss_kl = F.kl_div(log_probs, soft_labels, reduction="batchmean")
        loss_ce = F.cross_entropy(logits, hard_labels)
        loss = alpha * loss_kl + (1 - alpha) * loss_ce
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)



def predict_and_save(student, dataloader, output_csv, device):
    student.eval()
    predictions = []
    with torch.no_grad():
        for feats in tqdm(dataloader, desc="Predicting", leave=False):
            feats = feats.to(device)
            logits = student(feats)
            probs = torch.softmax(logits, dim=-1)
            predictions.extend(probs.cpu())


    import csv
    header = ["segment_idx"] + [f"phoneme_{i+1}" for i in range(predictions[0].shape[0])]
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, tensor in enumerate(predictions):
            writer.writerow([idx] + tensor.tolist())
    print(f"Saved predictions to {output_csv}")


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



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yaml_path = "./configs/phoneme/megt/base-config.yaml"
    teacher_ckpt_path = "./out/phoneme-megt/best-val_bal_acc-megt-hpo-2-epoch=24-val_f1_macro=0.5725.ckpt"
    teacher_preds_csv = "./preds_best-val_bal_acc-megt-hpo-2-epoch=24-val_f1_macro=0.5725.csv"

    # Teacher / Student
    teacher, student = build_teacher_student(yaml_path, teacher_ckpt_path, device, random_student=True)

    # Teacher predictions
    teacher_preds = load_teacher_predictions(teacher_preds_csv)

    # Target Dataset
    holdout_ds = LibriBrainCompetitionHoldout(
        data_path ='./data',
        standardize=True,
        tmin=0.0,
        tmax=0.5,
        task='phoneme'
    )
    dataset = SpeechWithTeacherDataset(holdout_ds, teacher_preds)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    max_epoch = 500

    # Training / Adaptation
    for epoch in range(max_epoch):
        loss = train_student(student, dataloader, optimizer, device, alpha=0.5)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    # Save student
    torch.save(student.state_dict(), "./out/megt_student_adapted.pt")

    # Prediction on target domain
    predict_loader = DataLoader(SpeechWithTeacherDataset(holdout_ds), batch_size=64, shuffle=False)
    predict_and_save(student, predict_loader, "./out/megt_student_predictions.csv", device)
