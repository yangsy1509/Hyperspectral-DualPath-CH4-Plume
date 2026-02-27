import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from datasets.methane_dataset import MethaneDataset
from losses.tversky import TverskyLoss
from utils.config import load_yaml, ensure_dir, load_norm_params
from utils.metrics import binary_metrics_from_probs
from utils.seed import set_seed, seed_worker_factory
from utils.checkpoint import uses_full_checkpoint, save_best_checkpoint, save_last_checkpoint
from models.builder import build_model, model_forward

try:
    import wandb
except Exception:
    wandb = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--gpus", type=str, default=None)
    p.add_argument("--disable_wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    if args.gpus is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)

    seed = int(cfg["train"]["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(cfg["data"]["fold_csv"])
    norm_params = load_norm_params(cfg["data"]["norm_param_dir"], cfg["data"]["norm_files"])

    arch = cfg["model"]["arch"].lower()
    input_mode = cfg["data"]["input_mode"].lower()
    n_folds = int(cfg["train"]["n_folds"])
    save_root = Path(cfg["logging"]["save_root"]) / input_mode / arch
    ensure_dir(save_root)

    use_wandb = bool(cfg["logging"].get("use_wandb", False)) and not args.disable_wandb and (wandb is not None)
    use_full_ckpt = uses_full_checkpoint(input_mode, arch)

    cv_rows = []

    for fold in range(n_folds):
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        val_df = df[df["fold"] == fold].reset_index(drop=True)

        train_files = train_df["plm_fname"].tolist()
        val_files = val_df["plm_fname"].tolist()

        print(f"\n==================== {input_mode.upper()} | {arch} | FOLD {fold} ====================")
        print(f"train patches: {len(train_files)} | val patches: {len(val_files)}")

        if "scene_id" in df.columns:
            print(f"train scenes: {train_df['scene_id'].nunique()} | val scenes: {val_df['scene_id'].nunique()}")

        fold_dir = save_root / f"fold{fold}"
        ensure_dir(fold_dir)

        if use_wandb:
            wandb.init(
                project=cfg["project"]["name"],
                name=f"{cfg['project']['experiment_name']}_fold{fold}",
                config={**cfg, "fold": fold},
                reinit=True,
            )

        train_dataset = MethaneDataset(
            train_files, train_files,
            cfg["data"]["input_dir"], cfg["data"]["target_dir"],
            norm_params=norm_params,
            norm_type=cfg["data"]["normalization"],
            augment=True,
        )
        val_dataset = MethaneDataset(
            val_files, val_files,
            cfg["data"]["input_dir"], cfg["data"]["target_dir"],
            norm_params=norm_params,
            norm_type=cfg["data"]["normalization"],
            augment=False,
        )

        g = torch.Generator()
        g.manual_seed(seed + fold)
        seed_worker = seed_worker_factory(seed, fold)

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=True,
            num_workers=int(cfg["train"]["num_workers"]),
            generator=g,
            worker_init_fn=seed_worker,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["train"]["num_workers"]),
            generator=g,
            worker_init_fn=seed_worker,
            pin_memory=True,
        )

        model, from_logits = build_model(cfg["model"])
        model = model.to(device)

        loss_fn = TverskyLoss(alpha=0.3, beta=0.7, from_logits=from_logits)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["train"]["learning_rate"],
            weight_decay=cfg["train"]["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg["train"].get("scheduler_tmax", cfg["train"]["epochs"]))
        )
        scaler = GradScaler(enabled=bool(cfg["train"]["amp"]))

        best_val_loss = float("inf")
        best_epoch = None
        best_model_path = None
        patience_counter = 0
        
        for epoch in range(int(cfg["train"]["epochs"])):
            start_time = time.time()
            model.train()
            train_loss = 0.0
            current_lr = scheduler.get_last_lr()[0]

            for inputs, targets in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=bool(cfg["train"]["amp"])):
                    preds = model_forward(model, arch, inputs, targets)
                    loss = loss_fn(preds, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            train_loss /= max(len(train_loader), 1)

            model.eval()
            val_loss = 0.0
            all_tgts, all_probs = [], []

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    preds = model_forward(model, arch, inputs, targets)
                    loss = loss_fn(preds, targets)
                    val_loss += loss.item()

                    probs = torch.sigmoid(preds) if from_logits else preds
                    all_tgts.append(targets.detach().cpu().numpy())
                    all_probs.append(probs.detach().cpu().numpy())

            val_loss /= max(len(val_loader), 1)

            # save rolling "last" checkpoint if using full checkpoint format
            last_path = fold_dir / ("last.pt" if use_full_ckpt else "last.pth")
            save_last_checkpoint(
                last_path, model, optimizer, scheduler, scaler,
                epoch + 1, best_val_loss, use_full_ckpt
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                best_model_path = str(
                    fold_dir / (f"best_epoch{best_epoch}.pt" if use_full_ckpt else f"best_model_epoch{best_epoch}.pth")
                )
                save_best_checkpoint(
                    best_model_path, model, optimizer, scheduler, scaler,
                    best_epoch, best_val_loss, use_full_ckpt
                )
                print(f"[fold {fold}] Model saved at epoch {best_epoch} with val_loss: {val_loss:.4f}")
            else:
                patience_counter += 1

            scheduler.step()

            y_true = np.concatenate(all_tgts, axis=0)
            y_prob = np.concatenate(all_probs, axis=0)
            metrics = binary_metrics_from_probs(y_true, y_prob)

            log_dict = {
                "fold": fold,
                "Epoch": epoch + 1,
                "Learning_Rate": current_lr,
                "Train_Loss": train_loss,
                "Val_Loss": val_loss,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1_Score": metrics["f1"],
                "IoU": metrics["iou"],
                "ROC_AUC": metrics["roc_auc"],
                "PRC_AUC": metrics["prc_auc"],
                "Epoch_Duration": time.time() - start_time,
                "Best_Val_Loss": best_val_loss,
                "Best_Epoch": best_epoch,
            }

            if use_wandb:
                wandb.log(log_dict)

            print(
                f"[fold {fold}] Epoch [{epoch+1}/{cfg['train']['epochs']}], "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"F1: {metrics['f1']:.4f}, ROC AUC: {metrics['roc_auc']:.4f}, "
                f"PRC AUC: {metrics['prc_auc']:.4f}"
            )

            if patience_counter >= cfg["train"]["patience"]:
                print(f"[fold {fold}] Early stopping at epoch {epoch+1}")
                break

        if use_wandb:
            wandb.finish()

        cv_rows.append({
            "fold": fold,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_patches": len(train_files),
            "val_patches": len(val_files),
            "best_model_path": best_model_path,
            "checkpoint_format": "pt_full_checkpoint" if use_full_ckpt else "pth_state_dict",
        })

    cv_df = pd.DataFrame(cv_rows)
    cv_path = save_root / "cv_summary.csv"
    cv_df.to_csv(cv_path, index=False)
    print(f"\n[DONE] CV summary saved: {cv_path}")


if __name__ == "__main__":
    main()
