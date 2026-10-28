"""Seeding, device selection, and checkpoint helpers."""

from pathlib import Path
import random
import shutil
import numpy as np
import torch

from src.config import CHECKPOINTS_DIR, FIGURES_DIR, SEED


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(device_str: str = "cpu") -> torch.device:
    return torch.device(device_str)


def copy_to_figures(source: Path | str, name: str | None = None) -> Path:
    """Copy an existing figure into reports/figures for the final report."""
    source = Path(source)
    destination = FIGURES_DIR / (name or source.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def save_checkpoint(
    path: Path | str,
    epoch: int,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    history: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "history": history or {},
        },
        path,
    )
    return path


def load_checkpoint(
    path: Path | str,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module | None = None,
    opt_g: torch.optim.Optimizer | None = None,
    opt_d: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    generator.load_state_dict(ckpt["generator"])
    if discriminator is not None and "discriminator" in ckpt:
        discriminator.load_state_dict(ckpt["discriminator"])
    if opt_g is not None and "opt_g" in ckpt:
        opt_g.load_state_dict(ckpt["opt_g"])
    if opt_d is not None and "opt_d" in ckpt:
        opt_d.load_state_dict(ckpt["opt_d"])
    return ckpt
