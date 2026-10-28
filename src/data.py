"""Fashion-MNIST data loaders and deterministic sample-grid inputs."""

from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    LATENT_DIM,
    NUM_CLASSES,
    SAMPLES_PER_CLASS,
    SEED,
)


def _seed_worker(_worker_id: int) -> None:
    """Seed Python and NumPy RNGs in each DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_transform() -> transforms.Compose:
    """Convert PIL grayscale image to tensor normalized to [-1, 1]."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def get_dataset(train: bool = True, data_dir: Path | str = DATA_DIR, download: bool = True) -> datasets.FashionMNIST:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    return datasets.FashionMNIST(
        root=str(data_path),
        train=train,
        download=download,
        transform=get_transform(),
    )


def get_dataloader(
    batch_size: int = BATCH_SIZE,
    train: bool = True,
    data_dir: Path | str = DATA_DIR,
    seed: int = SEED,
    num_workers: int = 0,
    download: bool = True,
) -> DataLoader:
    dataset = get_dataset(train=train, data_dir=data_dir, download=download)
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        generator=gen if train else None,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        drop_last=False,
    )


def make_fixed_grid(
    samples_per_class: int = SAMPLES_PER_CLASS,
    latent_dim: int = LATENT_DIM,
    num_classes: int = NUM_CLASSES,
    seed: int = SEED,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic (z, labels) ordered row-by-row per class."""
    gen = torch.Generator().manual_seed(seed)
    total = num_classes * samples_per_class
    z = torch.randn((total, latent_dim), generator=gen).to(device)
    labels = torch.arange(num_classes, dtype=torch.long, device=device).repeat_interleave(samples_per_class)
    return z, labels
