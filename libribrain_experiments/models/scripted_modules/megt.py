import torch
from torch import nn, optim

from .base_module import BaseClassificationModule, N_CLASSES, N_CHANNELS


class RotaryPositionalEncoding(nn.Module):
    """
    Source: https://hackmd.io/@_E6GATP9Sz2h9WVqOqMDbg/SJgIu7xRp
    """

    def __init__(self, dim: int, max_len: int = 512, base: int = 10000) -> None:
        super().__init__()
        self.theta = 1 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.theta = self.theta.repeat_interleave(2)
        self.position_ids = torch.arange(0, max_len)

    def forward(self, x: torch.Tensor):
        position_matrix = torch.outer(self.position_ids, self.theta).to(x.device)
        cos = torch.cos(position_matrix)
        sin = torch.sin(position_matrix)
        _x = torch.empty_like(x)
        _x[..., 0::2] = -x[..., 1::2]
        _x[..., 1::2] = x[..., 0::2]
        _x = _x * sin
        x = x * cos
        out = x + _x
        return out


class MegFormerClassificationModule(BaseClassificationModule):
    def __init__(
            self,
            d_model: int = 256,
            nhead: int = 8,
            transformer_activation: str = "gelu",
            transformer_num_layers: int = 4,
            lr: float = 0.0001,
    ):
        super().__init__()

        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

        # Model Architecture

        # data.shape = (, 306, num_samples=125)

        # Signal Filtering
        self.layer_filter = nn.Sequential(
            nn.Conv1d(N_CHANNELS, d_model, 7, 1, "same")
        )
        self.layer_res1 = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(d_model, d_model, 3, 1, "same"),
            nn.ELU(),
            nn.Conv1d(d_model, d_model, 1, 1, "same"),
        )
        self.elu1 = nn.ELU()

        # data.shape = (, embedding_dim, 125)
        # reshape for transformer in forward
        # data.shape = (, 125, embedding_dim)

        # Encoder w/ Transformer
        self.pos_enc = RotaryPositionalEncoding(
            dim=d_model,
            max_len=125,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            activation=transformer_activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_num_layers,
        )

        # data.shape = (, 125, N_CLASSES)
        self.linear = nn.Sequential(
            nn.Linear(d_model, N_CLASSES),
            # Pool across time dimension, reduce to 1 scalar per class
            nn.AvgPool2d((125, 1), (125, 1)),
            nn.Flatten(),
        )

        # data.shape = (, N_CLASSES)

    def forward(self, x):
        # data.shape = (, 306, num_samples=125)
        x = self.layer_filter(x)
        x = self.elu1(x + self.layer_res1(x))  # Residual
        # data.shape = (, embedding_dim, 125)
        x = x.permute(0, 2, 1)  # reshape for transformer
        # data.shape = (, 125, embedding_dim)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        # data.shape = (, 125, embedding_dim)
        x = self.linear(x)
        # data.shape = (, N_CLASSES)
        return x

    def configure_optimizers(self):
        # return optimizer_from_config(self.parameters(), self.optimizer_config)
        return optim.Adam(self.parameters(), lr=0.0001)
