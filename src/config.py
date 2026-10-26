"""Default paths, seeds, and hyperparameters for Fashion-MNIST cGAN."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUTS_DIR = ROOT_DIR / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
SAMPLES_DIR = OUTPUTS_DIR / "samples"
LOGS_DIR = OUTPUTS_DIR / "logs"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

SEED = 42
NUM_CLASSES = 10
IMG_CHANNELS = 1
IMG_SIZE = 28

LATENT_DIM = 64
EMBED_DIM = 32

BATCH_SIZE = 128
EPOCHS = 30
LR = 2e-4
BETAS = (0.5, 0.999)
REAL_LABEL = 0.9
FAKE_LABEL = 0.0
SAMPLE_EVERY = 5
SAMPLES_PER_CLASS = 8

CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
