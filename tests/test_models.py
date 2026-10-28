"""Forward-pass shape, range, and parameter-count tests for cGAN models."""

import unittest
import torch

from src.config import IMG_CHANNELS, IMG_SIZE, LATENT_DIM, NUM_CLASSES
from src.models import Discriminator, Generator, count_parameters


class TestModels(unittest.TestCase):
    def test_generator_forward_and_params(self) -> None:
        gen = Generator()
        z = torch.randn(16, LATENT_DIM)
        labels = torch.randint(0, NUM_CLASSES, (16,))
        out = gen(z, labels)
        self.assertEqual(tuple(out.shape), (16, IMG_CHANNELS, IMG_SIZE, IMG_SIZE))
        self.assertGreaterEqual(out.detach().min().item(), -1.0)
        self.assertLessEqual(out.detach().max().item(), 1.0)
        self.assertEqual(count_parameters(gen), 741_249)

    def test_discriminator_forward_and_params(self) -> None:
        disc = Discriminator()
        images = torch.randn(16, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
        labels = torch.randint(0, NUM_CLASSES, (16,))
        logits = disc(images, labels)
        self.assertEqual(tuple(logits.shape), (16, 1))
        self.assertEqual(count_parameters(disc), 192_881)


if __name__ == "__main__":
    unittest.main()
