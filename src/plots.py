"""Training-curve figures for the cGAN."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; no display needed
import matplotlib.pyplot as plt


def save_loss_curves(history: dict[str, list[float]], path: Path | str) -> Path:
    """Plot per-epoch G/D loss and discriminator accuracy side by side."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["loss_d"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(epochs, history["loss_d"], label="Discriminator")
    ax_loss.plot(epochs, history["loss_g"], label="Generator")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("BCE loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, history["acc_d"], label="D accuracy")
    ax_acc.plot(epochs, history["d_real"], label="D(x) real", linestyle="--")
    ax_acc.plot(epochs, history["d_fake"], label="D(G(z)) fake", linestyle="--")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_title("Discriminator")
    ax_acc.legend()

    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
