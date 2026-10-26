"""Tests for Fashion-MNIST data pipeline and fixed grid generation."""

import unittest
import torch

from src.config import IMG_CHANNELS, IMG_SIZE, LATENT_DIM, NUM_CLASSES, SAMPLES_PER_CLASS
from src.data import get_dataloader, make_fixed_grid


class TestDataPipeline(unittest.TestCase):
    def test_fixed_grid_shapes_and_determinism(self) -> None:
        z1, y1 = make_fixed_grid()
        z2, y2 = make_fixed_grid()
        total = NUM_CLASSES * SAMPLES_PER_CLASS
        self.assertEqual(tuple(z1.shape), (total, LATENT_DIM))
        self.assertEqual(tuple(y1.shape), (total,))
        self.assertEqual(y1.dtype, torch.long)
        self.assertTrue(torch.equal(z1, z2))
        self.assertTrue(torch.equal(y1, y2))
        self.assertEqual(set(y1.tolist()), set(range(NUM_CLASSES)))

    def test_dataloader_shapes_and_normalization(self) -> None:
        loader = get_dataloader(batch_size=32, train=False)
        images, labels = next(iter(loader))
        self.assertEqual(tuple(images.shape), (32, IMG_CHANNELS, IMG_SIZE, IMG_SIZE))
        self.assertEqual(tuple(labels.shape), (32,))
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.long)
        self.assertGreaterEqual(float(images.min()), -1.0 - 1e-5)
        self.assertLessEqual(float(images.max()), 1.0 + 1e-5)
        self.assertTrue(((labels >= 0) & (labels < NUM_CLASSES)).all())


if __name__ == "__main__":
    unittest.main()
