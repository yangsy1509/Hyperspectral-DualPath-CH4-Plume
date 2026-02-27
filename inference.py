import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.methane_dataset import MethaneDataset
from models.builder import build_model, model_forward
from utils.config import load_yaml, load_norm_params, ensure_dir
from utils.checkpoint import find_best_model_path, load_model_weights
from utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--gpus", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    if args.gpus is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)

    set_seed(int(cfg["train"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(cfg["data"]["fold_csv"])
    norm_params = load_norm_params(cfg["data"]["norm_param_dir"], cfg["data"]["norm_files"])

    arch = cfg["model"]["arch"].lower()
    input_mode = cfg["data"]["input_mode"].lower()
    save_root = Path(cfg["logging"]["save_root"]) / input_mode / arch
    pred_dir = save_root / f"fold{args.fold}" / "predictions"
    ensure_dir(pred_dir)

    val_df = df[df["fold"] == args.fold].reset_index(drop=True)
    val_files = val_df["plm_fname"].tolist()

    dataset = MethaneDataset(
        val_files, val_files,
        cfg["data"]["input_dir"], cfg["data"]["target_dir"],
        norm_params=norm_params,
        norm_type=cfg["data"]["normalization"],
        augment=False,
        return_fname=True,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model, from_logits = build_model(cfg["model"])
    ckpt_path = find_best_model_path(save_root / f"fold{args.fold}", input_mode=input_mode, arch=arch)
    load_model_weights(model, ckpt_path, input_mode=input_mode, arch=arch, map_location=device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for inputs, targets, fname in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            preds = model_forward(model, arch, inputs, targets)
            probs = torch.sigmoid(preds) if from_logits else preds
            prob_np = probs.squeeze().cpu().numpy()

            out_path = pred_dir / fname[0]
            np.save(out_path, prob_np)

    print(f"[DONE] saved predictions to: {pred_dir}")


if __name__ == "__main__":
    main()
