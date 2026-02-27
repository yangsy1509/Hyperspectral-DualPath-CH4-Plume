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
from utils.metrics import binary_metrics_from_probs
from utils.checkpoint import find_best_model_path, load_model_weights
from utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
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
    eval_dir = save_root / "eval"
    ensure_dir(eval_dir)

    rows = []

    for fold in range(int(cfg["train"]["n_folds"])):
        val_df = df[df["fold"] == fold].reset_index(drop=True)
        val_files = val_df["plm_fname"].tolist()

        dataset = MethaneDataset(
            val_files, val_files,
            cfg["data"]["input_dir"], cfg["data"]["target_dir"],
            norm_params=norm_params,
            norm_type=cfg["data"]["normalization"],
            augment=False,
            return_fname=False,
        )

        loader = DataLoader(
            dataset,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["train"]["num_workers"]),
            pin_memory=True,
        )

        model, from_logits = build_model(cfg["model"])
        ckpt_path = find_best_model_path(save_root / f"fold{fold}", input_mode=input_mode, arch=arch)
        load_model_weights(model, ckpt_path, input_mode=input_mode, arch=arch, map_location=device)
        model = model.to(device)
        model.eval()

        all_tgts, all_probs = [], []

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                preds = model_forward(model, arch, inputs, targets)
                probs = torch.sigmoid(preds) if from_logits else preds

                all_tgts.append(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        y_true = np.concatenate(all_tgts, axis=0)
        y_prob = np.concatenate(all_probs, axis=0)
        metrics = binary_metrics_from_probs(y_true, y_prob)

        rows.append({
            "fold": fold,
            "ckpt_path": ckpt_path,
            **metrics,
        })

    df_out = pd.DataFrame(rows)

    mean_row = {"fold": "mean"}
    std_row = {"fold": "std"}
    for col in ["accuracy", "precision", "recall", "f1", "iou", "roc_auc", "prc_auc"]:
        mean_row[col] = df_out[col].mean()
        std_row[col] = df_out[col].std()

    df_out = pd.concat([df_out, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    out_csv = eval_dir / "eval_summary.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"[DONE] saved: {out_csv}")


if __name__ == "__main__":
    main()
