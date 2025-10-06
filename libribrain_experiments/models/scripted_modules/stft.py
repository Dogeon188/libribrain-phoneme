import torch
from torch import nn, optim, stft

from .base_module import BaseClassificationModule, N_CLASSES, N_CHANNELS

SEQUENCE_LENGTH = 125  # 500 ms at 250 Hz

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


class STFTClassificationModule(BaseClassificationModule):
    def __init__(self,
                 upscale_factor: int = 4, lr: float = 0.0001,
                 sr=250, n_fft=25, hop_length=5):
        super().__init__()

        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_bins = n_fft // 2 + 1
        self.time_bins = 1 + (SEQUENCE_LENGTH - n_fft) // hop_length
        self.window = torch.hann_window(n_fft)

        # Model Architecture

        # data.shape = (, N_CHANNELS, num_samples=125)

        # stft -> (, 306, freq_bins, time_bins)

        self.res0 = nn.Sequential(
            # Per-channel conv
            nn.Conv2d(
                N_CHANNELS, N_CHANNELS * upscale_factor,
                (5, 5), stride=1, padding="same",
                # groups=N_CHANNELS
            ),
            # data.shape = (, N_CHANNELS * upscale_factor, freq_bins, time_bins)
            nn.ReLU(),
            # nn.BatchNorm2d(N_CHANNELS * upscale_factor),
            nn.Conv2d(
                N_CHANNELS * upscale_factor, N_CHANNELS // 3,
                (5, 5), stride=1, padding="same",
                # groups=N_CHANNELS // 3
            ),
            # data.shape = (, N_CHANNELS // 3, freq_bins, time_bins)
            # nn.BatchNorm2d(N_CHANNELS // 3),
            nn.ReLU(),
        )
        self.downsample0 = nn.Conv2d(
            N_CHANNELS, N_CHANNELS // 3,
            (1, 1), stride=1, padding="same"
        )
        # data.shape = (, N_CHANNELS // 3, freq_bins, time_bins)
        self.res1 = nn.Sequential(
            nn.Conv2d(
                N_CHANNELS // 3, (N_CHANNELS // 3) * upscale_factor,
                (3, 3), stride=1, padding="same",
                # groups=N_CHANNELS // 3
            ),
            # data.shape = (, (N_CHANNELS // 2) * upscale_factor, freq_bins, time_bins)
            nn.ReLU(),
            # nn.BatchNorm2d((N_CHANNELS // 3) * upscale_factor),
            nn.Conv2d(
                (N_CHANNELS // 3) * upscale_factor, N_CHANNELS // 3,
                (3, 3), stride=1, padding="same",
                # groups=N_CHANNELS // 3
            ),
            # data.shape = (, N_CHANNELS // 3, freq_bins, time_bins)
            # nn.BatchNorm2d(N_CHANNELS // 3),
            nn.ReLU(),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(
                N_CHANNELS // 3, (N_CHANNELS // 3) * upscale_factor,
                (3, 3), stride=1, padding="same",
                # groups=N_CHANNELS // 3
            ),
            # data.shape = (, (N_CHANNELS // 3) * upscale_factor, freq_bins, time_bins)
            nn.ReLU(),
            # nn.BatchNorm2d((N_CHANNELS // 3) * upscale_factor),
            nn.Conv2d(
                (N_CHANNELS // 3) * upscale_factor, N_CHANNELS // 3,
                (3, 3), stride=1, padding="same",
                # groups=N_CHANNELS // 3
            ),
            # data.shape = (, N_CHANNELS // 3, freq_bins, time_bins)
            # nn.BatchNorm2d(N_CHANNELS // 3),
            nn.ReLU(),
        )
        # data.shape = (, N_CHANNELS // 3, freq_bins, time_bins)
        self.res3 = nn.Sequential(
            nn.Conv2d(
                N_CHANNELS // 3, (N_CHANNELS // 3) * upscale_factor,
                (5, 5), stride=1, padding="same",
                # groups=N_CHANNELS // 3
            ),
            # data.shape = (, (N_CHANNELS // 3) * upscale_factor, freq_bins, time_bins)
            nn.ReLU(),
            # nn.BatchNorm2d((N_CHANNELS // 3) * upscale_factor),
            nn.Conv2d(
                (N_CHANNELS // 3) * upscale_factor, N_CHANNELS // 9,
                (5, 5), stride=1, padding="same",
                # groups=N_CHANNELS // 9
            ),
            # data.shape = (, N_CHANNELS // 9, freq_bins, time_bins)
            # nn.BatchNorm2d(N_CHANNELS // 9),
            nn.ReLU(),
        )
        self.downsample3 = nn.Conv2d(
            N_CHANNELS // 3, N_CHANNELS // 9,
            (3, 3), stride=1, padding="same"
        )
        # data.shape = (, N_CHANNELS // 9, freq_bins, time_bins)
        self.classifier = nn.Sequential(
            # nn.BatchNorm2d(N_CHANNELS // 9),
            nn.Flatten(),
            nn.Linear(
                (N_CHANNELS // 9) * self.freq_bins * self.time_bins,
                256
            ),
            nn.ReLU(),
            nn.Linear(256, N_CLASSES)
        )
        # data.shape = (, N_CLASSES)

    def forward(self, x):
        # data.shape = (N, 306, num_samples=125)
        x = x.view(-1, SEQUENCE_LENGTH)
        # x.shape = (N * 306, num_samples=125)
        Zxx = stft(x, n_fft=self.n_fft,
                   hop_length=self.hop_length, return_complex=True,
                   window=self.window,
                   center=False)
        # Zxx.shape = (N * 306, freq_bins=14, time_bins=14)
        Zxx = Zxx.view(-1, N_CHANNELS, self.freq_bins, self.time_bins)
        # Zxx.shape = (, 306, freq_bins=14, time_bins=14)
        x = torch.abs(Zxx)
        x = self.res0(x) + self.downsample0(x)
        x = self.res1(x) + x
        x = self.res2(x) + x
        x = self.res3(x) + self.downsample3(x)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        # return optimizer_from_config(self.parameters(), self.optimizer_config)
        return optim.Adam(self.parameters(), lr=self.lr)

    def to(self, *args, **kwargs):
        self.window = self.window.to(*args, **kwargs)
        return super().to(*args, **kwargs)
