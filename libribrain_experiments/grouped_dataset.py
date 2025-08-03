
from pathlib import Path
import torch
from tqdm import tqdm


class MyGroupedDataset(torch.utils.data.Dataset):
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
