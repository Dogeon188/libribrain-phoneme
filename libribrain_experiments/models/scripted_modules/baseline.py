from torch import nn, optim

from .base_module import BaseClassificationModule, N_CLASSES, N_CHANNELS


class BaselineClassificationModule(BaseClassificationModule):
    def __init__(self, embedding_dim: int = 256, lr: float = 0.0001):
        super().__init__()

        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

        # Model Architecture

        # data.shape = (, N_CHANNELS, num_samples=125)

        # Signal Filtering
        self.layer_filter = nn.Sequential(
            nn.Conv1d(N_CHANNELS, embedding_dim, 7, 1, "same")
        )

        # data.shape = (, embedding_dim, 125)

        # Encoder w/ Residuals
        self.layer_res1 = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(embedding_dim, embedding_dim, 3, 1, "same"),
            nn.ELU(),
            nn.Conv1d(embedding_dim, embedding_dim, 1, 1, "same"),
        )
        self.layer_conv1 = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(embedding_dim, embedding_dim, 3, 1, "same")
        )
        self.layer_res2 = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(embedding_dim, embedding_dim, 3, 1, "same"),
            nn.ELU(),
            nn.Conv1d(embedding_dim, embedding_dim, 1, 1, "same"),
        )

        # data.shape = (, embedding_dim, 125)

        # Dense Layers
        self.layer_dense = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(embedding_dim, embedding_dim, 50, 25),
            # data.shape = (, embedding_dim, 4)
            nn.ELU(),
            nn.Conv1d(embedding_dim, embedding_dim, 7, 1, "same"),
            # data.shape = (, embedding_dim, 4)
            nn.ELU(),
            nn.Conv1d(embedding_dim, embedding_dim * 4, 4, 1, 0),
            # data.shape = (, embedding_dim * 4, 1)
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Flatten(),
            # data.shape = (, embedding_dim * 4)
            nn.Linear(embedding_dim * 4, N_CLASSES)
        )

        # data.shape = (, N_CLASSES)

    def forward(self, x):
        # data.shape = (, 306, num_samples=125)
        x = self.layer_filter(x)
        # data.shape = (, embedding_dim, 125)
        x = self.layer_conv1(x + self.layer_res1(x))
        x = self.layer_dense(x + self.layer_res2(x))
        # data.shape = (, N_CLASSES)
        return x

    def configure_optimizers(self):
        # return optimizer_from_config(self.parameters(), self.optimizer_config)
        return optim.Adam(self.parameters(), lr=0.0001)
