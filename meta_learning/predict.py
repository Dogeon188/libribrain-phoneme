import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml
from tqdm import tqdm

from meta_learning import DistributionMapper
from pnpl.datasets import LibriBrainCompetitionHoldout


class ResNetBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)

def modules_from_config(config_list):
    layers = []
    for cfg in config_list:
        if "conv1d" in cfg:
            args = cfg["conv1d"]
            layers.append(nn.Conv1d(**args))
        elif "linear" in cfg:
            args = cfg["linear"]
            layers.append(nn.Linear(**args))
        elif "relu" in cfg:
            layers.append(nn.ReLU(inplace=True))
        elif "elu" in cfg:
            layers.append(nn.ELU(inplace=True))
        elif "dropout" in cfg:
            args = cfg["dropout"]
            layers.append(nn.Dropout(**args))
        elif "flatten" in cfg:
            layers.append(nn.Flatten())
        elif "resnet_block" in cfg:
            block_layers = modules_from_config(cfg["resnet_block"]["model_config"])
            layers.append(ResNetBlock(nn.Sequential(*block_layers)))
        else:
            raise ValueError(f"Unsupported layer config: {cfg}")
    return layers

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

    holdout_ds = LibriBrainCompetitionHoldout(
        data_path="../data",
        standardize=True,
        tmin=0.0,
        tmax=0.5,
        task="phoneme"
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
    yaml_path = "../configs/phoneme/baseline-xl/base-config.yaml"
    ckpt_path = "../out/phoneme-baseline-xl/best-val_bal_acc-baseline-xl-hpo-2-epoch=05-val_f1_macro=0.6588.ckpt"
    mapper_ckpt = "out/mapper_model.pth"
    device = "cuda"

    raw_probs, mapped_probs = compare_mapped_vs_raw(yaml_path, ckpt_path, mapper_ckpt, device)

    raw_pred_classes = np.argmax(raw_probs, axis=1)
    mapped_pred_classes = np.argmax(mapped_probs, axis=1)

    num_classes = raw_probs.shape[1]
    raw_counts = np.bincount(raw_pred_classes, minlength=num_classes)
    mapped_counts = np.bincount(mapped_pred_classes, minlength=num_classes)

    print("Raw holdout predicted class counts:", raw_counts)
    print("Mapped holdout predicted class counts:", mapped_counts)

    pd.DataFrame({"segment_idx": np.arange(len(raw_pred_classes)),
                  "pred_class": raw_pred_classes}).to_csv("raw_pred_classes.csv", index=False)
    
    pd.DataFrame({"segment_idx": np.arange(len(mapped_pred_classes)),
                  "pred_class": mapped_pred_classes}).to_csv("mapped_pred_classes.csv", index=False)

    print("Saved as CSV")

    x = np.arange(1, num_classes + 1)
    width = 0.35

    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, raw_counts, width, label="Raw Holdout")
    plt.bar(x + width/2, mapped_counts, width, label="Mapped Holdout")
    plt.xlabel("Phoneme Class (1~39)")
    plt.ylabel("Count of Predicted Segments")
    plt.title("Predicted Class Counts: Raw vs Mapped Holdout")
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pred_class_counts.png", dpi=300)
    plt.show()

    print("Saved as pred_class_counts.png")


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

