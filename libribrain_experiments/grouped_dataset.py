import numpy as np
import torch
import random
from tqdm import tqdm
from collections import defaultdict
import os
from torch.utils.data import DataLoader


class MyGroupedDatasetV3(torch.utils.data.Dataset):
    @staticmethod
    def _group_by_label(original_dataset, num_workers: int = -1) -> dict[int, np.ndarray]:
        label_to_indices = defaultdict(list)

        # Use DataLoader for parallel processing
        loader = DataLoader(
            original_dataset,
            batch_size=1,
            # sampler=SequentialSampler(original_dataset),
            num_workers=(os.cpu_count() if num_workers < 0 else num_workers),
            pin_memory=True,
            prefetch_factor=256,
            in_order=True  # important! ensure idx below is correct
        )

        for idx, (_, label) in enumerate(
            tqdm(loader,
                 desc="Grouping samples by label",
                 total=len(original_dataset), leave=False)):
            lbl = int(label.item())
            label_to_indices[lbl].append(idx)

        for label, indices in label_to_indices.items():
            label_to_indices[label] = np.array(indices, dtype=np.int64)
            # label_to_indices[label].sort()

        return dict(label_to_indices)

    def __init__(self, original_dataset,
                 *,
                 grouped_samples: int = 10, drop_remaining: bool = False,
                 average_grouped_samples: bool = True,
                 state_cache_path: str = None,
                 balance: bool = False, augment: bool = False,
                 repeat: int = 1, shuffle: bool = False, num_workers: int = -1
                 ):
        """
        Groups n samples from the original dataset by label, with optional data augmentation for MEG signals.

        Parameters:
        - original_dataset: The original dataset to group
        - grouped_samples: Number of samples per group
        - drop_remaining: Drop last incomplete group
        - average_grouped_samples: Average grouped samples instead of concatenating
        - state_cache_path: Path to save/load cache. If state_cache_path is provided, it will load the state from the cache if it exists
        - balance: Oversample to balance classes
        - augment: Whether to apply data augmentation to each sample
        - repeat: Number of times to repeat the dataset. If repeat is greater than 1, the dataset will be repeated that many times
        - shuffle: Shuffle dataset before grouping
        - num_workers: Number of workers to use for grouping by label. If -1, will use all available CPU cores.
        """
        if (not drop_remaining and not average_grouped_samples):
            raise ValueError(
                "drop_remaining and average_grouped_samples cannot both be False. "
                "Otherwise the dimension of the output will be inconsistent."
            )
        self.original_dataset = original_dataset
        self.grouped_samples = grouped_samples
        self.average_grouped_samples = average_grouped_samples
        self.augment = augment

        # Load/save label-indices dict

        if state_cache_path:
            if os.path.exists(state_cache_path):
                with torch.serialization.safe_globals([np.ndarray, np._core.multiarray._reconstruct, np.dtype, np.dtypes.Int64DType]):
                    state = torch.load(state_cache_path)
                    self.labels_to_indices = state['labels_to_indices']
            else:
                self.labels_to_indices = MyGroupedDatasetV3._group_by_label(
                    original_dataset, num_workers=num_workers)
                state = {
                    'labels_to_indices': self.labels_to_indices,
                }
                os.makedirs(os.path.dirname(state_cache_path), exist_ok=True)
                torch.save(state, state_cache_path)

        # Rebalance & repeat & shuffle indices

        if balance:
            max_class_size = max(len(indices)
                                 for indices in self.labels_to_indices.values())
            for label, indices in self.labels_to_indices.items():
                if len(indices) < max_class_size:
                    oversample_count = max_class_size - len(indices)
                    oversampled_indices = random.choices(
                        indices, k=oversample_count)
                    self.labels_to_indices[label] = np.concatenate(
                        [indices, oversampled_indices])
                else:
                    self.labels_to_indices[label] = indices

        if repeat > 1:
            self.labels_to_indices = {
                label: np.tile(indices, repeat) for label, indices in self.labels_to_indices.items()
            }

        if shuffle:
            for _, indices in self.labels_to_indices.items():
                indices[:] = indices[torch.randperm(len(indices))]

        # Group samples

        if drop_remaining:
            # drop last incomplete labels from each label
            self.labels_to_indices = {
                label: indices[:len(indices) //
                               grouped_samples * grouped_samples]
                for label, indices in self.labels_to_indices.items()
            }
        else:
            # append -48763 to indicatet empty
            self.labels_to_indices = {
                label: np.pad(indices, (0, grouped_samples - len(indices) %
                              grouped_samples), constant_values=-48763)
                for label, indices in self.labels_to_indices.items()
            }

        self.labels_to_indices = {
            label: indices.reshape(-1, grouped_samples)
            for label, indices in self.labels_to_indices.items()
        }

        self.group_indices = np.ndarray(shape=(0, 2), dtype=np.int64)
        for label, indices in self.labels_to_indices.items():
            pairs = np.column_stack((
                np.arange(indices.shape[0]),
                np.full(indices.shape[0], label, dtype=np.int64)))
            self.group_indices = np.concatenate(
                (self.group_indices, pairs), axis=0)

        if shuffle:
            np.random.shuffle(self.group_indices)

    def __len__(self):
        return len(self.group_indices)

    def __getitem__(self, idx):
        group_idx, label = self.group_indices[idx]
        indices = self.labels_to_indices[label][group_idx]

        # Get samples
        samples = [self.original_dataset[i] for i in indices if i != -48763]

        # if not samples:
        #     raise ValueError(
        #         f"Group {idx} with label {label} has no valid samples.")

        # Stack data
        samples_data = [sample[0] for sample in samples]
        if self.average_grouped_samples:
            # Why mul by sqrt? See examples/StdThenAvg.ipynb
            # nope.
            data = torch.stack(samples_data).mean(dim=0)  # * np.sqrt(self.grouped_samples)
        else:
            data = torch.concat(samples_data, dim=0)

        if self.augment:
            data = MyGroupedDatasetV3.apply_augmentations(data)

        return data, torch.tensor(label)

    # data augmentation
    @staticmethod
    def apply_augmentations(data):
        aug_fns = [
            MyGroupedDatasetV3.add_gaussian_noise,
            MyGroupedDatasetV3.time_shift,
            MyGroupedDatasetV3.time_mask,
            MyGroupedDatasetV3.amplitude_scaling,
            MyGroupedDatasetV3.channel_dropout,
            lambda x: MyGroupedDatasetV3.frequency_band_perturb(
                x, bands=[(8, 12), (13, 30)], scale_range=(0.8, 1.2))
        ]

        # choose a random subset of augmentations to apply
        k = np.random.random_sample((len(aug_fns),)) < 0.3
        data = data.clone()  # avoid modifying original data
        for i, aug_fn in enumerate(aug_fns):
            if k[i]:
                data = aug_fn(data)
        return data

    # data augmentation
    # @staticmethod
    # def apply_augmentation_by_id(data, aug_id: int):
    #     aug_fns = [
    #         MyGroupedDatasetV3.add_gaussian_noise,
    #         MyGroupedDatasetV3.time_shift,
    #         MyGroupedDatasetV3.time_mask,
    #         MyGroupedDatasetV3.amplitude_scaling,
    #         MyGroupedDatasetV3.channel_dropout,
    #         lambda x: MyGroupedDatasetV3.frequency_band_perturb(
    #             x, bands=[(8, 12), (13, 30)], scale_range=(0.8, 1.2))
    #     ]

    #     if not (0 <= aug_id < len(aug_fns)):
    #         raise ValueError(f"aug_id must be between 0 and {len(aug_fns)-1}")

    #     data = data.clone()  # avoid modifying original
    #     data = aug_fns[aug_id](data)
    #     return data


    # data augmentation
    # @staticmethod
    # def apply_augmentations(data):
    #     aug_fns = [
    #         MyGroupedDatasetV3.time_mask,
    #         MyGroupedDatasetV3.channel_dropout
    #     ]

    #     # choose a random subset of augmentations to apply
    #     k = np.random.random_sample((len(aug_fns),)) < 0.5
    #     data = data.clone()  # avoid modifying original data
    #     for i, aug_fn in enumerate(aug_fns):
    #         if k[i]:
    #             data = aug_fn(data)
    #     return data


    @staticmethod
    def add_gaussian_noise(data, noise_std=0.01):
        return data + torch.randn_like(data) * noise_std

    @staticmethod
    def time_shift(data, shift_max=10):
        shift = random.randint(-shift_max, shift_max)
        return torch.roll(data, shifts=shift, dims=-1)

    @staticmethod
    def time_mask(data, mask_size=20):
        t = data.size(-1)
        start = random.randint(0, max(0, t - mask_size))
        data[:, start:start+mask_size] = 0
        return data

    @staticmethod
    def amplitude_scaling(data, scale_range=(0.9, 1.1)):
        scale = random.uniform(*scale_range)
        return data * scale

    @staticmethod
    def channel_dropout(data, drop_prob=0.1):
        mask = (torch.rand(data.size(0)) > drop_prob).float().unsqueeze(1)
        return data * mask

    @staticmethod
    def frequency_band_perturb(data, bands=[(8, 12)], scale_range=(0.8, 1.2), fs=100):
        """
        Frequency band perturbation: adjust power in certain frequency ranges.
        fs: sampling rate
        """
        # FFT
        freq_data = torch.fft.rfft(data, dim=-1)
        freqs = torch.fft.rfftfreq(data.size(-1), 1/fs)

        for (low, high) in bands:
            scale = random.uniform(*scale_range)
            band_mask = (freqs >= low) & (freqs <= high)
            freq_data[:, band_mask] *= scale

        # Inverse FFT
        data = torch.fft.irfft(freq_data, n=data.size(-1), dim=-1)
        return data


class MyGroupedDatasetV2(torch.utils.data.Dataset):
    def __init__(self, original_dataset, grouped_samples=10, drop_remaining=False,
                 shuffle=False, average_grouped_samples=True, state_cache_path=None,
                 balance=False, augment=False):
        """
        Groups n samples from the original dataset by label, with optional data augmentation for MEG signals.

        Parameters:
        - original_dataset: The original dataset to group
        - grouped_samples: Number of samples per group
        - drop_remaining: Drop last incomplete group
        - shuffle: Shuffle dataset before grouping
        - average_grouped_samples: Average grouped samples instead of concatenating
        - state_cache_path: Path to save/load cache
        - balance: Oversample to balance classes
        - augment: Whether to apply data augmentation to each sample
        """
        if (not drop_remaining and not average_grouped_samples):
            raise ValueError(
                "drop_remaining and average_grouped_samples cannot both be False. "
                "Otherwise the dimension of the output will be inconsistent."
            )

        self.original_dataset = original_dataset
        self.average_grouped_samples = average_grouped_samples
        self.grouped_samples = grouped_samples
        self.groups = []  # list of (group_indices, label)
        self.partial_groups = {}
        self.augment = augment

        # Load cache if exists
        if state_cache_path and state_cache_path.exists():
            state = torch.load(state_cache_path)
            self.groups = state['groups']
            self.average_grouped_samples = state['average_grouped_samples']
            self.partial_groups = state['partial_groups']
            self.grouped_samples = state['grouped_samples']
            return

        # Shuffle indices
        if shuffle:
            indices = torch.randperm(len(original_dataset))
        else:
            indices = torch.arange(len(original_dataset))

        # Grouping samples by label
        for i in tqdm(indices, desc="Grouping samples", total=len(original_dataset)):
            label = original_dataset[i][1].item()
            group = self.partial_groups.get(label, [])
            group.append(i.item())
            self.partial_groups[label] = group
            if len(group) == grouped_samples:
                self.groups.append((group, label))
                self.partial_groups[label] = []

        if not drop_remaining:
            for label, group in self.partial_groups.items():
                if group:
                    self.groups.append((group, label))

        # Balance groups
        if balance:
            self._balance_groups()
            self.print_group_counts()

    def _balance_groups(self):
        from collections import defaultdict
        label_to_groups = defaultdict(list)
        for group, label in self.groups:
            label_to_groups[label].append(group)

        max_group_count = max(len(groups)
                              for groups in label_to_groups.values())
        balanced_groups = []
        for label, groups in label_to_groups.items():
            current_count = len(groups)
            if current_count < max_group_count:
                oversample_count = max_group_count - current_count
                oversampled = random.choices(groups, k=oversample_count)
                label_to_groups[label].extend(oversampled)
            balanced_groups.extend((group, label)
                                   for group in label_to_groups[label])

        self.groups = balanced_groups

    def print_group_counts(self):
        from collections import Counter
        label_counter = Counter()
        for _, label in self.groups:
            label_counter[label] += 1
        print("Group counts per label:")
        for label, count in label_counter.items():
            print(f"Label {label}: {count} groups")
        counts = list(label_counter.values())
        print(f"Max group count: {max(counts)}")
        print(f"Min group count: {min(counts)}")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group, label = self.groups[idx]
        samples = [self.original_dataset[i] for i in group]
        samples_data = [sample[0] for sample in samples]

        if self.average_grouped_samples:
            data = torch.stack(samples_data).mean(dim=0)
        else:
            data = torch.concat(samples_data, dim=0)

        if self.augment:
            data = self.apply_augmentations(data)

        return data, torch.tensor(label)

    # data augmentation
    def apply_augmentations(self, data):
        if random.random() < 0.3:
            data = self.add_gaussian_noise(data, noise_std=0.01)
        if random.random() < 0.3:
            data = self.time_shift(data, shift_max=10)
        if random.random() < 0.3:
            data = self.time_mask(data, mask_size=20)
        if random.random() < 0.3:
            data = self.amplitude_scaling(data, scale_range=(0.9, 1.1))
        if random.random() < 0.3:
            data = self.channel_dropout(data, drop_prob=0.1)
        if random.random() < 0.3:
            data = self.frequency_band_perturb(
                data, bands=[(8, 12), (13, 30)], scale_range=(0.8, 1.2))
        return data

    def add_gaussian_noise(self, data, noise_std=0.01):
        return data + torch.randn_like(data) * noise_std

    def time_shift(self, data, shift_max=10):
        shift = random.randint(-shift_max, shift_max)
        return torch.roll(data, shifts=shift, dims=-1)

    def time_mask(self, data, mask_size=20):
        t = data.size(-1)
        start = random.randint(0, max(0, t - mask_size))
        data[:, start:start+mask_size] = 0
        return data

    def amplitude_scaling(self, data, scale_range=(0.9, 1.1)):
        scale = random.uniform(*scale_range)
        return data * scale

    def channel_dropout(self, data, drop_prob=0.1):
        mask = (torch.rand(data.size(0)) > drop_prob).float().unsqueeze(1)
        return data * mask

    def frequency_band_perturb(self, data, bands=[(8, 12)], scale_range=(0.8, 1.2), fs=100):
        """
        Frequency band perturbation: adjust power in certain frequency ranges.
        fs: sampling rate
        """
        # FFT
        freq_data = torch.fft.rfft(data, dim=-1)
        freqs = torch.fft.rfftfreq(data.size(-1), 1/fs)

        for (low, high) in bands:
            scale = random.uniform(*scale_range)
            band_mask = (freqs >= low) & (freqs <= high)
            freq_data[:, band_mask] *= scale

        # Inverse FFT
        data = torch.fft.irfft(freq_data, n=data.size(-1), dim=-1)
        return data


class MyGroupedDatasetV1(torch.utils.data.Dataset):
    def __init__(self, original_dataset, grouped_samples=10, drop_remaining=False, shuffle=False, average_grouped_samples=True, state_cache_path=None):
        """
        Groups n samples from the original dataset by label 

        Parameters:
        - original_dataset: The original dataset to group
        - grouped_samples: The number of samples to group over
        - drop_remaining: Whether to drop the last group if it is incomplete
        - shuffle: Whether to shuffle the samples
        - average_grouped_samples: Whether to average the grouped samples
        - state_cache_path: Path to save/load the grouped dataset state
        - If state_cache_path is provided, it will load the state from the cache if it exists
        """

        if (not drop_remaining and not average_grouped_samples):
            raise ValueError(
                "drop_remaining and average_grouped_samples cannot both be False. Otherwise the dimension of the output will be inconsistent.")

        self.original_dataset = original_dataset
        self.average_grouped_samples = average_grouped_samples
        self.groups = []
        self.partial_groups = {}
        self.grouped_samples = grouped_samples

        if state_cache_path:
            if state_cache_path.exists():
                state = torch.load(state_cache_path)
                self.groups = state['groups']
                self.average_grouped_samples = state['average_grouped_samples']
                self.partial_groups = state['partial_groups']
                self.grouped_samples = state['grouped_samples']
                return
            # else, we will create the cache file later

        if shuffle:
            indices = torch.randperm(len(original_dataset))
        else:
            indices = torch.arange(len(original_dataset))

        for i in tqdm(indices, desc="Grouping samples", total=len(original_dataset)):
            label = original_dataset[i][1].item()
            group = self.partial_groups.get(label, [])
            group.append(i.item())
            self.partial_groups[label] = group
            if (len(group) == grouped_samples):
                self.groups.append(group)
                self.partial_groups[label] = []

        if not drop_remaining:
            for group in self.partial_groups.values():
                if group:
                    self.groups.append(group)

        if state_cache_path:
            state_cache_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                'groups': self.groups,
                'average_grouped_samples': self.average_grouped_samples,
                'partial_groups': self.partial_groups,
                'grouped_samples': self.grouped_samples
            }
            torch.save(state, state_cache_path)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        samples = [self.original_dataset[i] for i in group]
        samples_data = [sample[0] for sample in samples]
        if self.average_grouped_samples:
            data = torch.stack(samples_data)
            data = data.mean(dim=0)
        else:
            data = torch.concat(samples_data, dim=0)
        label = samples[0][1]

        return data, label
