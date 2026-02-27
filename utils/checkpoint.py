from pathlib import Path
import torch


def uses_full_checkpoint(input_mode: str, arch: str) -> bool:
    return str(input_mode).lower() == "rad" and str(arch).lower() == "segformer"


def best_pattern(input_mode: str, arch: str) -> str:
    if uses_full_checkpoint(input_mode, arch):
        return "best_epoch*.pt"
    return "best_model_epoch*.pth"


def latest_pattern(input_mode: str, arch: str) -> str:
    if uses_full_checkpoint(input_mode, arch):
        return "last.pt"
    return "last.pth"


def find_best_model_path(fold_dir, input_mode: str, arch: str):
    fold_dir = Path(fold_dir)
    pattern = best_pattern(input_mode, arch)
    ckpts = sorted(fold_dir.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(f"No {pattern} found in {fold_dir}")
    return str(ckpts[-1])


def find_latest_checkpoint_path(fold_dir, input_mode: str, arch: str):
    fold_dir = Path(fold_dir)
    path = fold_dir / latest_pattern(input_mode, arch)
    return str(path) if path.exists() else None


def save_best_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val_loss, use_full_checkpoint: bool):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if use_full_checkpoint:
        torch.save({
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": float(best_val_loss),
        }, path)
    else:
        torch.save(model.state_dict(), path)


def save_last_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val_loss, use_full_checkpoint: bool):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if use_full_checkpoint:
        torch.save({
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": float(best_val_loss),
        }, path)
    else:
        torch.save(model.state_dict(), path)


def load_model_weights(model, ckpt_path, input_mode: str, arch: str, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if uses_full_checkpoint(input_mode, arch):
        model.load_state_dict(ckpt["model"])
        return ckpt
    model.load_state_dict(ckpt)
    return ckpt