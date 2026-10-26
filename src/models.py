"""Conditional Generator and Discriminator for 28x28 Fashion-MNIST."""

import torch
from torch import nn

from src.config import EMBED_DIM, IMG_CHANNELS, IMG_SIZE, LATENT_DIM, NUM_CLASSES


class Generator(nn.Module):
    """Map latent noise z and class label y to a (N, 1, 28, 28) image in [-1, 1]."""

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        num_classes: int = NUM_CLASSES,
        embed_dim: int = EMBED_DIM,
        img_channels: int = IMG_CHANNELS,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.fc = nn.Linear(latent_dim + embed_dim, 128 * 7 * 7)
        self.net = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        y = self.label_emb(labels)
        x = torch.cat([z, y], dim=1)
        x = self.fc(x).view(-1, 128, 7, 7)
        return self.net(x)


class Discriminator(nn.Module):
    """Map (N, 1, 28, 28) image and class label y to a (N, 1) real/fake logit."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        embed_dim: int = EMBED_DIM,
        img_channels: int = IMG_CHANNELS,
        img_size: int = IMG_SIZE,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.label_proj = nn.Linear(embed_dim, img_size * img_size)
        self.features = nn.Sequential(
            nn.Conv2d(img_channels + 1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(128 * 3 * 3, 1)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        y = self.label_proj(self.label_emb(labels)).view(-1, 1, self.img_size, self.img_size)
        x = torch.cat([images, y], dim=1)
        x = self.features(x).flatten(1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
