"""Smoke tests for the training loop on a tiny synthetic dataset."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import IMG_CHANNELS, IMG_SIZE, LATENT_DIM, NUM_CLASSES
from src.models import Discriminator, Generator
from src.train import HISTORY_KEYS, train, train_step


def tiny_loader(n: int = 16, batch_size: int = 8) -> DataLoader:
    images = torch.rand(n, IMG_CHANNELS, IMG_SIZE, IMG_SIZE) * 2 - 1
    labels = torch.randint(0, NUM_CLASSES, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=True)


class TestTrainStep(unittest.TestCase):
    def test_step_updates_both_networks_and_reports_metrics(self) -> None:
        torch.manual_seed(0)
        gen, disc = Generator(), Discriminator()
        opt_g = torch.optim.Adam(gen.parameters(), lr=2e-4)
        opt_d = torch.optim.Adam(disc.parameters(), lr=2e-4)
        g_before = gen.fc.weight.clone()
        d_before = disc.classifier.weight.clone()

        real = torch.rand(8, IMG_CHANNELS, IMG_SIZE, IMG_SIZE) * 2 - 1
        labels = torch.randint(0, NUM_CLASSES, (8,))
        metrics = train_step(gen, disc, opt_g, opt_d, nn.BCEWithLogitsLoss(), real, labels, LATENT_DIM)

        self.assertEqual(set(metrics), set(HISTORY_KEYS))
        self.assertFalse(torch.equal(g_before, gen.fc.weight))
        self.assertFalse(torch.equal(d_before, disc.classifier.weight))
        self.assertTrue(0.0 <= metrics["acc_d"] <= 1.0)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))


class TestTrainLoop(unittest.TestCase):
    def test_train_writes_outputs_and_resumes(self) -> None:
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dirs = dict(checkpoints_dir=tmp / "ckpt", samples_dir=tmp / "samples", logs_dir=tmp / "logs")

            history = train(tiny_loader(), epochs=2, sample_every=2, **dirs)
            for key in HISTORY_KEYS:
                self.assertEqual(len(history[key]), 2)
            self.assertTrue((tmp / "ckpt" / "ckpt.pt").exists())
            self.assertTrue((tmp / "ckpt" / "ckpt_epoch_002.pt").exists())
            self.assertTrue((tmp / "samples" / "epoch_001.png").exists())
            self.assertTrue((tmp / "samples" / "epoch_002.png").exists())
            self.assertTrue((tmp / "logs" / "loss_curves.png").exists())
            saved = json.loads((tmp / "logs" / "history.json").read_text())
            self.assertEqual(saved["loss_d"], history["loss_d"])

            resumed = train(tiny_loader(), epochs=3, resume=tmp / "ckpt" / "ckpt.pt", **dirs)
            self.assertEqual(len(resumed["loss_d"]), 3)
            self.assertEqual(resumed["loss_d"][:2], history["loss_d"])
            self.assertEqual(torch.load(tmp / "ckpt" / "ckpt.pt", weights_only=True)["epoch"], 3)

    def test_max_batches_limits_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            history = train(
                tiny_loader(n=32), epochs=1, max_batches=1,
                checkpoints_dir=tmp / "c", samples_dir=tmp / "s", logs_dir=tmp / "l",
            )
            self.assertEqual(len(history["loss_d"]), 1)


if __name__ == "__main__":
    unittest.main()
