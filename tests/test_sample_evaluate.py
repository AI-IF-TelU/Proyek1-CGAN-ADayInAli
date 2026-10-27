"""Tests for sampling and evaluation helpers (no dataset download needed)."""

import math
import tempfile
import unittest
from pathlib import Path

import torch

from src.config import IMG_CHANNELS, IMG_SIZE, NUM_CLASSES
from src.evaluate import (
    Classifier,
    confusion_matrix,
    diversity_ratios,
    mean_pairwise_distance,
    normalized_entropy,
    per_class_accuracy,
    predict,
)
from src.models import Discriminator, Generator
from src.sample import generate, load_generator, main as sample_main
from src.utils import save_checkpoint


def write_checkpoint(path: Path, epoch: int = 7) -> Path:
    gen, disc = Generator(), Discriminator()
    return save_checkpoint(
        path, epoch, gen, disc,
        torch.optim.Adam(gen.parameters()), torch.optim.Adam(disc.parameters()),
    )


class TestSample(unittest.TestCase):
    def test_generate_orders_labels_and_is_deterministic(self) -> None:
        gen = Generator().eval()
        images, labels = generate(gen, [3, 8], per_class=5, seed=1)
        self.assertEqual(tuple(images.shape), (10, IMG_CHANNELS, IMG_SIZE, IMG_SIZE))
        self.assertEqual(labels.tolist(), [3] * 5 + [8] * 5)
        again, _ = generate(gen, [3, 8], per_class=5, seed=1)
        self.assertTrue(torch.equal(images, again))
        self.assertTrue(images.min() >= -1 and images.max() <= 1)

    def test_cli_writes_grid_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = write_checkpoint(Path(tmp) / "ckpt.pt", epoch=7)
            generator, epoch = load_generator(ckpt)
            self.assertEqual(epoch, 7)
            self.assertFalse(generator.training)
            out = Path(tmp) / "grid.png"
            sample_main(["--checkpoint", str(ckpt), "--out", str(out), "--per-class", "4"])
            self.assertTrue(out.exists())
            fixed = Path(tmp) / "fixed.png"
            sample_main(["--checkpoint", str(ckpt), "--out", str(fixed), "--fixed"])
            self.assertTrue(fixed.exists())

    def test_cli_rejects_bad_class_id(self) -> None:
        with self.assertRaises(SystemExit):
            sample_main(["--classes", "10"])


class TestEvaluateMetrics(unittest.TestCase):
    def test_classifier_shapes(self) -> None:
        model = Classifier()
        x = torch.randn(6, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
        self.assertEqual(tuple(model(x).shape), (6, NUM_CLASSES))
        self.assertEqual(tuple(model.embed(x).shape), (6, 128))
        self.assertEqual(tuple(predict(model, x).shape), (6,))

    def test_confusion_matrix_and_accuracy(self) -> None:
        targets = torch.tensor([0, 0, 1, 1, 2])
        preds = torch.tensor([0, 1, 1, 1, 0])
        cm = confusion_matrix(targets, preds, num_classes=3)
        self.assertEqual(cm.tolist(), [[1, 1, 0], [0, 2, 0], [1, 0, 0]])
        self.assertEqual(per_class_accuracy(cm), [0.5, 1.0, 0.0])

    def test_normalized_entropy_bounds(self) -> None:
        self.assertAlmostEqual(normalized_entropy(torch.arange(10)), 1.0, places=6)
        self.assertAlmostEqual(normalized_entropy(torch.zeros(50, dtype=torch.long)), 0.0, places=6)
        two = normalized_entropy(torch.tensor([0, 1] * 10))
        self.assertAlmostEqual(two, math.log(2) / math.log(10), places=6)

    def test_diversity_detects_collapse(self) -> None:
        torch.manual_seed(0)
        labels = torch.arange(NUM_CLASSES).repeat_interleave(20)
        real = torch.randn(len(labels), 8)
        collapsed = torch.zeros(len(labels), 8)
        self.assertEqual(mean_pairwise_distance(collapsed[:20]), 0.0)
        self.assertTrue(all(r == 0.0 for r in diversity_ratios(collapsed, labels, real, labels)))
        same = diversity_ratios(real, labels, real, labels)
        self.assertTrue(all(abs(r - 1.0) < 1e-6 for r in same))


if __name__ == "__main__":
    unittest.main()
