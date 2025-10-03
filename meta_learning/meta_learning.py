import sys
from pathlib import Path
import itertools

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np

from libribrain_experiments.grouped_dataset import MyGroupedDatasetV3
from pnpl.datasets import LibriBrainCompetitionHoldout, LibriBrainPhoneme


# distribution mapping network
class DistributionMapper(nn.Module):
    def __init__(self, channels=306, time=125, hidden_dim=512):
        super().__init__()
        input_dim = channels * time
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        b, c, t = x.shape
        x = x.view(b, -1)
        out = self.net(x)
        return out.view(b, c, t)


# loss function
def compute_mmd(x, y, sigma=10.0):
    """x, y shape: (batch, features)"""
    xx = torch.cdist(x, x) ** 2
    yy = torch.cdist(y, y) ** 2
    xy = torch.cdist(x, y) ** 2

    k_xx = torch.exp(-xx / (2 * sigma**2))
    k_yy = torch.exp(-yy / (2 * sigma**2))
    k_xy = torch.exp(-xy / (2 * sigma**2))

    mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return mmd


# training
def train_mapper_dataloader(
    source_dataset,
    target_dataset,
    batch_size=128,
    epochs=100,
    device="cuda",
    loss_type="mmd"
):
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = DistributionMapper().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # target loader iterator
    target_iter = itertools.cycle(target_loader)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_src in tqdm(source_loader, desc=f"Epoch {epoch+1}"):
            if isinstance(batch_src, (list, tuple)):
                batch_src = batch_src[0]
            batch_tgt = next(target_iter)
            if isinstance(batch_tgt, (list, tuple)):
                batch_tgt = batch_tgt[0]

            batch_src = batch_src.to(device)
            batch_tgt = batch_tgt.to(device)

            mapped = model(batch_src)

            # flatten
            mapped_flat = mapped.view(mapped.size(0), -1)
            tgt_flat = batch_tgt.view(batch_tgt.size(0), -1)

            # loss function
            if loss_type == "mmd":
                loss = compute_mmd(mapped_flat, tgt_flat)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_src.size(0)

        avg_loss = total_loss / len(source_dataset)
        print(f"Epoch {epoch+1}/{epochs}, {loss_type.upper()} Loss={avg_loss:.6f}")

    torch.save(model.state_dict(), "mapper_model.pth")
    print("Mapping model saved as mapper_model.pth")
    return model


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import torch
from torch.utils.data import DataLoader

# PCA visualization
def visualize_distribution_batch(source_dataset, target_dataset, mapper, batch_size=128, device="cuda"):
    mapper.eval()
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=False)

    # first batch
    batch_src = next(iter(source_loader))
    if isinstance(batch_src, (list, tuple)):
        batch_src = batch_src[0]
    batch_src = batch_src.to(device)

    with torch.no_grad():
        mapped = mapper(batch_src)

    # flatten
    src_flat = batch_src.view(batch_src.size(0), -1).cpu().numpy()
    mapped_flat = mapped.view(mapped.size(0), -1).cpu().numpy()

    tgt_list = []
    for i in range(batch_size):
        tgt = target_dataset[i]
        if isinstance(tgt, (list, tuple)):
            tgt = tgt[0]
        tgt_list.append(tgt.view(-1).numpy())
    tgt_flat = np.stack(tgt_list, axis=0)

    # PCA 2D
    all_data = np.concatenate([src_flat, mapped_flat, tgt_flat], axis=0)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_data)

    n = batch_size
    src_reduced = reduced[:n]
    mapped_reduced = reduced[n:2*n]
    tgt_reduced = reduced[2*n:]

    plt.figure(figsize=(8, 8))
    plt.scatter(src_reduced[:, 0], src_reduced[:, 1], 
                label="Source", color="red", s=50, alpha=0.7, edgecolors="k")
    plt.scatter(mapped_reduced[:, 0], mapped_reduced[:, 1], 
                label="Mapped Source", color="blue", s=50, alpha=0.7, edgecolors="k")
    plt.scatter(tgt_reduced[:, 0], tgt_reduced[:, 1], 
                label="Target", color="green", s=50, alpha=0.7, edgecolors="k")

    plt.legend(fontsize=12)
    plt.title("PCA Projection (One Batch): Source vs Mapped Source vs Target", fontsize=14)
    plt.tight_layout()
    plt.savefig("batch_distribution.png", dpi=300)
    plt.show()
    print("Saved as batch_distribution.png")




if __name__ == "__main__":
    ds = LibriBrainPhoneme(
        data_path='../data',
        preprocessing_str="bads+headpos+sss+notch+bp+ds",
        label_type="phoneme",
        standardize=True,
        tmin=0.0,
        tmax=0.5,
        preload_files=True
    )

    target_dataset = MyGroupedDatasetV3(
        original_dataset=ds,
        grouped_samples=100,
        drop_remaining=False,
        state_cache_path=Path("/home/pomalo/libribrain_phoneme/libribrain_phoneme/data_preprocessed/groupedv3/train_grouped_100.pt"),
        average_grouped_samples=True,
        balance=False,
        augment=False,
    )

    source_dataset = LibriBrainCompetitionHoldout(
        data_path='../data',
        standardize=True,
        tmin=0.0,
        tmax=0.5,
        task='phoneme'
    )

    mapper = train_mapper_dataloader(
        source_dataset,
        target_dataset,
        batch_size=128,
        epochs=50,
        device="cuda",
        loss_type="mmd"
    )

    visualize_distribution_batch(
        source_dataset,
        target_dataset,
        mapper,
        batch_size=128,
        device="cuda"
    )
