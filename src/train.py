"""Adversarial training loop for the Fashion-MNIST cGAN.

Usage:
    python -m src.train --epochs 30 --batch-size 128
    python -m src.train --resume outputs/checkpoints/ckpt.pt --epochs 40
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    BETAS,
    CHECKPOINTS_DIR,
    EPOCHS,
    FAKE_LABEL,
    LATENT_DIM,
    LOGS_DIR,
    LR,
    REAL_LABEL,
    SAMPLE_EVERY,
    SAMPLES_DIR,
    SAMPLES_PER_CLASS,
    SEED,
)
from src.data import get_dataloader, make_fixed_grid
from src.models import Discriminator, Generator
from src.plots import save_loss_curves
from src.utils import copy_to_figures, get_device, load_checkpoint, save_checkpoint, set_seed

HISTORY_KEYS = ("loss_d", "loss_g", "acc_d", "d_real", "d_fake")


def train_step(
    generator: nn.Module,
    discriminator: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    criterion: nn.Module,
    real: torch.Tensor,
    labels: torch.Tensor,
    latent_dim: int = LATENT_DIM,
) -> dict[str, float]:
    """Run one discriminator update and one generator update on a batch."""
    n = real.size(0)
    device = real.device
    real_targets = torch.full((n, 1), REAL_LABEL, device=device)
    fake_targets = torch.full((n, 1), FAKE_LABEL, device=device)

    # Discriminator: real images with smoothed targets, fakes with 0.
    z = torch.randn(n, latent_dim, device=device)
    fake = generator(z, labels)
    opt_d.zero_grad(set_to_none=True)
    logits_real = discriminator(real, labels)
    logits_fake = discriminator(fake.detach(), labels)
    loss_d = criterion(logits_real, real_targets) + criterion(logits_fake, fake_targets)
    loss_d.backward()
    opt_d.step()

    # Generator: make the discriminator call fakes real (unsmoothed target 1).
    opt_g.zero_grad(set_to_none=True)
    logits_gen = discriminator(fake, labels)
    loss_g = criterion(logits_gen, torch.ones_like(logits_gen))
    loss_g.backward()
    opt_g.step()

    with torch.no_grad():
        correct = (logits_real > 0).sum() + (logits_fake < 0).sum()
        return {
            "loss_d": loss_d.item(),
            "loss_g": loss_g.item(),
            "acc_d": correct.item() / (2 * n),
            "d_real": torch.sigmoid(logits_real).mean().item(),
            "d_fake": torch.sigmoid(logits_fake).mean().item(),
        }


@torch.no_grad()
def save_sample_grid(
    generator: nn.Module,
    z: torch.Tensor,
    labels: torch.Tensor,
    path: Path,
    samples_per_class: int = SAMPLES_PER_CLASS,
) -> Path:
    """Save the fixed grid: one row per class, values mapped back to [0, 1]."""
    was_training = generator.training
    generator.eval()
    images = generator(z, labels)
    generator.train(was_training)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, path, nrow=samples_per_class, normalize=True, value_range=(-1, 1))
    return path


def train(
    loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = LR,
    device: torch.device | str = "cpu",
    sample_every: int = SAMPLE_EVERY,
    checkpoints_dir: Path = CHECKPOINTS_DIR,
    samples_dir: Path = SAMPLES_DIR,
    logs_dir: Path = LOGS_DIR,
    resume: Path | str | None = None,
    seed: int = SEED,
    max_batches: int | None = None,
    progress: bool = True,
) -> dict[str, list[float]]:
    """Train the cGAN and return per-epoch mean metrics.

    Writes the latest checkpoint to `checkpoints_dir/ckpt.pt` every epoch, an
    epoch-numbered copy whenever a sample grid is saved, and the metric history
    as JSON plus a loss-curve figure in `logs_dir`.
    """
    device = torch.device(device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=BETAS)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=BETAS)
    criterion = nn.BCEWithLogitsLoss()

    history: dict[str, list[float]] = {key: [] for key in HISTORY_KEYS}
    start_epoch = 1
    if resume is not None:
        ckpt = load_checkpoint(resume, generator, discriminator, opt_g, opt_d, device=device)
        start_epoch = ckpt["epoch"] + 1
        history.update(ckpt.get("history") or {})
        print(f"Resumed from {resume} at epoch {ckpt['epoch']}")

    fixed_z, fixed_labels = make_fixed_grid(seed=seed, device=device)
    checkpoints_dir, samples_dir, logs_dir = Path(checkpoints_dir), Path(samples_dir), Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, epochs + 1):
        generator.train()
        discriminator.train()
        sums = dict.fromkeys(HISTORY_KEYS, 0.0)
        steps = 0
        total = len(loader) if max_batches is None else min(len(loader), max_batches)
        bar = tqdm(loader, total=total, desc=f"Epoch {epoch}/{epochs}", leave=False, disable=not progress)
        for real, labels in bar:
            if max_batches is not None and steps >= max_batches:
                break
            metrics = train_step(
                generator, discriminator, opt_g, opt_d, criterion,
                real.to(device), labels.to(device),
            )
            for key in HISTORY_KEYS:
                sums[key] += metrics[key]
            steps += 1
            bar.set_postfix(loss_d=f"{metrics['loss_d']:.3f}", loss_g=f"{metrics['loss_g']:.3f}")
        bar.close()

        for key in HISTORY_KEYS:
            history[key].append(sums[key] / max(steps, 1))
        print(
            f"Epoch {epoch}/{epochs}  loss_d={history['loss_d'][-1]:.4f}  "
            f"loss_g={history['loss_g'][-1]:.4f}  acc_d={history['acc_d'][-1]:.3f}  "
            f"D(x)={history['d_real'][-1]:.3f}  D(G(z))={history['d_fake'][-1]:.3f}"
        )

        save_checkpoint(checkpoints_dir / "ckpt.pt", epoch, generator, discriminator, opt_g, opt_d, history)
        if epoch % sample_every == 0 or epoch == epochs or epoch == start_epoch:
            save_sample_grid(generator, fixed_z, fixed_labels, samples_dir / f"epoch_{epoch:03d}.png")
            save_checkpoint(
                checkpoints_dir / f"ckpt_epoch_{epoch:03d}.pt",
                epoch, generator, discriminator, opt_g, opt_d, history,
            )
        (logs_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_loss_curves(history, logs_dir / "loss_curves.png")

    if history["loss_d"] and logs_dir == LOGS_DIR and samples_dir == SAMPLES_DIR:
        loss_curve = logs_dir / "loss_curves.png"
        if loss_curve.exists():
            copy_to_figures(loss_curve, "loss_curves.png")
        final_epoch = len(history["loss_d"])
        final_grid = samples_dir / f"epoch_{final_epoch:03d}.png"
        if final_grid.exists():
            copy_to_figures(final_grid, "sample_grid_final.png")

    return history


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Fashion-MNIST cGAN.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="last epoch to train (inclusive)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sample-every", type=int, default=SAMPLE_EVERY)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint to continue from")
    parser.add_argument("--max-batches", type=int, default=None, help="limit batches per epoch (smoke test)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device(args.device)
    loader = get_dataloader(batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers)
    train(
        loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        sample_every=args.sample_every,
        resume=args.resume,
        seed=args.seed,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
