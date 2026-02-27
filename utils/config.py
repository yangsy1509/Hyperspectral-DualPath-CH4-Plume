from pathlib import Path
import numpy as np
import torch
import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_norm_params(norm_param_dir, norm_files):
    min_vals = np.load(Path(norm_param_dir) / norm_files["min"])
    max_vals = np.load(Path(norm_param_dir) / norm_files["max"])
    mean_vals = np.load(Path(norm_param_dir) / norm_files["mean"])
    std_vals = np.load(Path(norm_param_dir) / norm_files["std"])

    min_vals = np.asarray(min_vals, dtype=np.float32)
    max_vals = np.asarray(max_vals, dtype=np.float32)
    mean_vals = np.asarray(mean_vals, dtype=np.float32)
    std_vals = np.asarray(std_vals, dtype=np.float32)

    if min_vals.ndim == 0:
        shape = (1, 1, 1)
    else:
        shape = (len(min_vals), 1, 1)

    return (
        torch.tensor(min_vals, dtype=torch.float32).view(*shape),
        torch.tensor(max_vals, dtype=torch.float32).view(*shape),
        torch.tensor(mean_vals, dtype=torch.float32).view(*shape),
        torch.tensor(std_vals, dtype=torch.float32).view(*shape),
    )
