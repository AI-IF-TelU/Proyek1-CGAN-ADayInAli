"""Evaluate a trained cGAN with conditional accuracy and mode coverage.

A small CNN classifier is trained once on real Fashion-MNIST and cached. The
generator then produces a balanced set of images per class, and we measure:

- conditional accuracy: how often the classifier agrees with the requested label;
- prediction spread: normalized entropy of the predicted-class histogram
  (1.0 = all ten classes equally represented);
- diversity ratio per class: mean pairwise distance between generated images in
  the classifier's feature space, divided by the same value for real test images
  of that class (about 1.0 = as varied as real data; near 0 = mode collapse).

Usage:
    python -m src.evaluate --checkpoint outputs/checkpoints/ckpt.pt
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import CHECKPOINTS_DIR, CLASS_NAMES, LOGS_DIR, NUM_CLASSES, SEED
from src.data import get_dataloader
from src.sample import generate, load_generator
from src.utils import copy_to_figures, get_device, set_seed

CLASSIFIER_PATH = CHECKPOINTS_DIR / "classifier.pt"


class Classifier(nn.Module):
    """Small CNN for 28x28 Fashion-MNIST in [-1, 1]; `embed` gives 128-d features."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, num_classes))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


@torch.no_grad()
def predict(model: nn.Module, images: torch.Tensor, device: torch.device | str = "cpu", batch_size: int = 1000) -> torch.Tensor:
    model.eval()
    return torch.cat([
        model(images[i:i + batch_size].to(device)).argmax(1).cpu()
        for i in range(0, len(images), batch_size)
    ])


@torch.no_grad()
def embed(model: Classifier, images: torch.Tensor, device: torch.device | str = "cpu", batch_size: int = 1000) -> torch.Tensor:
    model.eval()
    return torch.cat([
        model.embed(images[i:i + batch_size].to(device)).cpu()
        for i in range(0, len(images), batch_size)
    ])


def train_classifier(
    train_loader: DataLoader,
    epochs: int = 6,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
) -> Classifier:
    model = Classifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(images)
            loss_fn(logits, labels).backward()
            opt.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
        print(f"Classifier epoch {epoch}/{epochs}  train acc={correct / total:.4f}")
    return model.eval()


def load_or_train_classifier(
    path: Path = CLASSIFIER_PATH,
    retrain: bool = False,
    epochs: int = 6,
    device: torch.device | str = "cpu",
) -> Classifier:
    """Load the cached classifier, or train it on real data and cache it."""
    path = Path(path)
    if path.exists() and not retrain:
        model = Classifier().to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return model.eval()
    model = train_classifier(get_dataloader(train=True), epochs=epochs, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


def load_test_set() -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = zip(*get_dataloader(batch_size=1000, train=False))
    return torch.cat(images), torch.cat(labels)


def confusion_matrix(targets: torch.Tensor, preds: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Rows are requested/true labels, columns are predictions."""
    idx = targets * num_classes + preds
    return torch.bincount(idx, minlength=num_classes ** 2).view(num_classes, num_classes)


def per_class_accuracy(cm: torch.Tensor) -> list[float]:
    return (cm.diag().float() / cm.sum(1).clamp(min=1).float()).tolist()


def normalized_entropy(preds: torch.Tensor, num_classes: int = NUM_CLASSES) -> float:
    """Entropy of the predicted-class histogram divided by log(num_classes)."""
    p = torch.bincount(preds, minlength=num_classes).float()
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * p.log()).sum() / math.log(num_classes))


def mean_pairwise_distance(features: torch.Tensor) -> float:
    n = len(features)
    if n < 2:
        return 0.0
    d = torch.cdist(features, features)
    return float(d.sum() / (n * (n - 1)))


def diversity_ratios(
    gen_feats: torch.Tensor,
    gen_labels: torch.Tensor,
    real_feats: torch.Tensor,
    real_labels: torch.Tensor,
    max_per_class: int = 500,
    num_classes: int = NUM_CLASSES,
) -> list[float]:
    """Generated / real mean pairwise feature distance for each class."""
    ratios = []
    for c in range(num_classes):
        g = gen_feats[gen_labels == c][:max_per_class]
        r = real_feats[real_labels == c][:max_per_class]
        real_spread = mean_pairwise_distance(r)
        ratios.append(mean_pairwise_distance(g) / real_spread if real_spread > 0 else 0.0)
    return ratios


def save_confusion_matrix(cm: torch.Tensor, path: Path, title: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rates = cm.float() / cm.sum(1, keepdim=True).clamp(min=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
    ax.set_xlabel("Classifier prediction")
    ax.set_ylabel("Requested label")
    ax.set_title(title)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = rates[i, j].item()
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def evaluate(
    checkpoint: Path | str,
    per_class: int = 1000,
    seed: int = SEED,
    device: torch.device | str = "cpu",
    retrain_classifier: bool = False,
    classifier_epochs: int = 6,
    logs_dir: Path = LOGS_DIR,
) -> dict:
    """Compute all metrics for one checkpoint and write JSON + confusion matrix."""
    set_seed(seed)
    classifier = load_or_train_classifier(retrain=retrain_classifier, epochs=classifier_epochs, device=device)
    real_images, real_labels = load_test_set()
    real_preds = predict(classifier, real_images, device)
    real_cm = confusion_matrix(real_labels, real_preds)

    generator, epoch = load_generator(checkpoint, device)
    gen_images, gen_labels = generate(generator, list(range(NUM_CLASSES)), per_class, seed, device)
    gen_preds = predict(classifier, gen_images, device)
    gen_cm = confusion_matrix(gen_labels, gen_preds)

    ratios = diversity_ratios(
        embed(classifier, gen_images, device), gen_labels,
        embed(classifier, real_images, device), real_labels,
    )
    gen_acc = per_class_accuracy(gen_cm)
    real_acc = per_class_accuracy(real_cm)
    results = {
        "checkpoint": str(checkpoint),
        "epoch": epoch,
        "samples_per_class": per_class,
        "classifier_test_accuracy": real_cm.diag().sum().item() / real_cm.sum().item(),
        "conditional_accuracy": gen_cm.diag().sum().item() / gen_cm.sum().item(),
        "prediction_entropy": normalized_entropy(gen_preds),
        "mean_diversity_ratio": sum(ratios) / len(ratios),
        "per_class": {
            name: {
                "conditional_accuracy": gen_acc[c],
                "classifier_accuracy_on_real": real_acc[c],
                "diversity_ratio": ratios[c],
            }
            for c, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": gen_cm.tolist(),
    }

    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(checkpoint).stem
    (logs_dir / f"eval_{stem}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_confusion_matrix(gen_cm, logs_dir / f"confusion_{stem}.png", f"Generated samples, epoch {epoch}")
    if stem == "ckpt" and logs_dir == LOGS_DIR:
        copy_to_figures(logs_dir / f"confusion_{stem}.png", "confusion_matrix_final.png")
    return results


def print_report(results: dict) -> None:
    print(f"\nCheckpoint: {results['checkpoint']} (epoch {results['epoch']})")
    print(f"Classifier accuracy on real test set: {results['classifier_test_accuracy']:.4f}")
    print(f"Conditional accuracy on generated:    {results['conditional_accuracy']:.4f}")
    print(f"Prediction entropy (1.0 = balanced):  {results['prediction_entropy']:.4f}")
    print(f"Mean diversity ratio (1.0 = real):    {results['mean_diversity_ratio']:.4f}\n")
    print(f"{'class':<12} {'cond acc':>9} {'real acc':>9} {'diversity':>10}")
    for name, m in results["per_class"].items():
        print(f"{name:<12} {m['conditional_accuracy']:>9.3f} {m['classifier_accuracy_on_real']:>9.3f} {m['diversity_ratio']:>10.3f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a cGAN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_DIR / "ckpt.pt")
    parser.add_argument("--per-class", type=int, default=1000, help="generated images per class")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--retrain-classifier", action="store_true")
    parser.add_argument("--classifier-epochs", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results = evaluate(
        args.checkpoint,
        per_class=args.per_class,
        seed=args.seed,
        device=get_device(args.device),
        retrain_classifier=args.retrain_classifier,
        classifier_epochs=args.classifier_epochs,
    )
    print_report(results)


if __name__ == "__main__":
    main()
