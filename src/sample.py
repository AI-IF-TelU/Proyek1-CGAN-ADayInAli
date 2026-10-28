"""Generate a labelled sample grid from a trained generator checkpoint.

Usage:
    python -m src.sample --checkpoint outputs/checkpoints/ckpt.pt
    python -m src.sample --checkpoint outputs/checkpoints/ckpt.pt --classes 5 8 --per-class 16
    python -m src.sample --checkpoint outputs/checkpoints/ckpt.pt --fixed
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision.utils import save_image

from src.config import CHECKPOINTS_DIR, CLASS_NAMES, LATENT_DIM, NUM_CLASSES, SAMPLES_DIR, SAMPLES_PER_CLASS, SEED
from src.data import make_fixed_grid
from src.models import Generator
from src.utils import get_device, load_checkpoint


def load_generator(checkpoint: Path | str, device: torch.device | str = "cpu") -> tuple[Generator, int]:
    """Return the generator from a checkpoint in eval mode, plus its epoch."""
    generator = Generator().to(device)
    ckpt = load_checkpoint(checkpoint, generator, device=device)
    generator.eval()
    return generator, ckpt.get("epoch", 0)


@torch.no_grad()
def generate(
    generator: nn.Module,
    classes: list[int],
    per_class: int,
    seed: int = SEED,
    device: torch.device | str = "cpu",
    batch_size: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate `per_class` images for each class, ordered class by class.

    Returns images in [-1, 1] with shape (len(classes) * per_class, 1, 28, 28)
    and the matching labels.
    """
    rng = torch.Generator().manual_seed(seed)
    labels = torch.tensor(classes, dtype=torch.long).repeat_interleave(per_class)
    z = torch.randn(len(labels), LATENT_DIM, generator=rng)
    images = [
        generator(z[i:i + batch_size].to(device), labels[i:i + batch_size].to(device)).cpu()
        for i in range(0, len(labels), batch_size)
    ]
    return torch.cat(images), labels


def save_grid(images: torch.Tensor, path: Path | str, per_class: int) -> Path:
    """Save images as a grid with one row per class."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, path, nrow=per_class, normalize=True, value_range=(-1, 1))
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a sample grid from a cGAN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_DIR / "ckpt.pt")
    parser.add_argument("--out", type=Path, default=None, help="output PNG (default: outputs/samples/sample_epoch_NNN.png)")
    parser.add_argument("--classes", type=int, nargs="+", default=list(range(NUM_CLASSES)), help="class ids 0-9")
    parser.add_argument("--per-class", type=int, default=SAMPLES_PER_CLASS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fixed", action="store_true", help="use the training fixed grid (all classes)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    bad = [c for c in args.classes if not 0 <= c < NUM_CLASSES]
    if bad:
        parser.error(f"class ids must be 0-{NUM_CLASSES - 1}, got {bad}")
    if args.per_class < 1:
        parser.error("--per-class must be at least 1")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = get_device(args.device)
    generator, epoch = load_generator(args.checkpoint, device)

    if args.fixed:
        z, labels = make_fixed_grid(seed=args.seed, device=device)
        with torch.no_grad():
            images = generator(z, labels).cpu()
        per_class, classes = SAMPLES_PER_CLASS, list(range(NUM_CLASSES))
    else:
        images, _ = generate(generator, args.classes, args.per_class, args.seed, device)
        per_class, classes = args.per_class, args.classes

    out = args.out or SAMPLES_DIR / f"sample_epoch_{epoch:03d}{'_fixed' if args.fixed else ''}.png"
    save_grid(images, out, per_class)
    print(f"Epoch {epoch} generator -> {out}")
    print("Rows:", ", ".join(CLASS_NAMES[c] for c in classes))


if __name__ == "__main__":
    main()
