import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class MethaneDataset(Dataset):
    def __init__(
        self,
        input_files,
        target_files,
        input_dir,
        target_dir,
        norm_params=None,
        norm_type="minmax",
        augment=False,
        return_fname=False,
    ):
        self.input_files = list(input_files)
        self.target_files = list(target_files)
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.norm_params = norm_params
        self.norm_type = norm_type
        self.augment = augment
        self.return_fname = return_fname

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        fname = self.input_files[idx]

        x = torch.from_numpy(
            np.load(os.path.join(self.input_dir, fname)).astype(np.float32)
        )
        y = torch.from_numpy(
            np.load(os.path.join(self.target_dir, self.target_files[idx])).astype(np.float32)
        )

        if self.norm_params is not None:
            min_vals, max_vals, mean_vals, std_vals = self.norm_params
            if self.norm_type == "minmax":
                x = (x - min_vals) / (max_vals - min_vals + 1e-8)
            elif self.norm_type == "zscore":
                x = (x - mean_vals) / (std_vals + 1e-8)
            else:
                raise ValueError(f"Unsupported norm_type: {self.norm_type}")

        if self.augment:
            combined = torch.cat([x, y], dim=0)
            if torch.rand(1).item() > 0.5:
                combined = TF.hflip(combined)
            if torch.rand(1).item() > 0.5:
                combined = TF.vflip(combined)
            if torch.rand(1).item() > 0.5:
                angle = 90 * torch.randint(0, 4, (1,)).item()
                combined = TF.rotate(combined, angle)
            # target is always the last channel block of size 1
            x, y = combined[:-1], combined[-1:].contiguous()

        if self.return_fname:
            return x, y, fname
        return x, y