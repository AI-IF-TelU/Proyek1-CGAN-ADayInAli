<a id="readme-top"></a>

<div align="center">
  <h1>GENAITUBES</h1>
  <p><strong>A class-conditional GAN built from scratch on Fashion-MNIST</strong></p>
  <p>Course project for <strong>CAK4KBB3 — Vibe Coding Generative AI from Scratch</strong><br>
  Group <strong>A Day in Ali</strong></p>
</div>

[![Python][python-shield]][python-url]
[![PyTorch][pytorch-shield]][pytorch-url]
[![Dataset][dataset-shield]][dataset-url]
[![Status][status-shield]][status-url]

> **Project status:** repository scaffold and documentation only. The data pipeline,
> models, training loop, and evaluation scripts are **not implemented yet**. Every
> command under [Usage](#usage) is a planned entry point, not a working feature.

## Table of Contents

- [About the Project](#about-the-project)
- [Model](#model)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Training and Evaluation Plan](#training-and-evaluation-plan)
- [Scope](#scope)
- [Team](#team)
- [Project Workflow](#project-workflow)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## About the Project

This project implements a **Conditional Generative Adversarial Network (cGAN)**
from scratch in PyTorch. The generator learns to produce 28×28 Fashion-MNIST
images conditioned on one of ten clothing classes, and the discriminator learns
to judge whether an image is real or generated **and** consistent with its label.

The method follows the course's mandatory vibe-coding process: architecture,
code, debugging, and optimization are developed through documented interaction
with an LLM. No tutorial code is copied; library documentation is used only as a
reference for basic functions.

**Why cGAN is the chosen method:** the assignment guide lists *Conditional GAN
Klasik* with Fashion-MNIST as one of its suggested options. The challenge named
in the guide — concatenating image and label tensors — is the core of this
implementation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Model

### Generator

```text
z ~ N(0, 1)  (64)
label y      (10) ── embedding (10 → 32) ──┐
                                           ├── concat (96)
                                           ▼
                        Linear → 7×7×128
                        ConvTranspose2d → 14×14×64
                        ConvTranspose2d → 28×28×1
                        Tanh
```

### Discriminator

```text
image x (1×28×28)
label y (10) ── embedding (10 → 32) ── Linear → 28×28
                                           │
                                           ▼
                        concat on channel dimension (2×28×28)
                        Conv2d 32 → 14×14
                        Conv2d 64 → 7×7
                        Conv2d 128 → 3×3
                        Linear → logit
```

### Training setup

- `BCEWithLogitsLoss` for both networks.
- One-sided label smoothing: real targets are `0.9`, fake targets are `0.0`.
- The generator minimizes the loss of fake images being classified as real.
- Adam with learning rate `2e-4`, betas `(0.5, 0.999)`.
- Batch size `128`.
- `BatchNorm` after the generator's linear/transposed-conv layers and the
  discriminator's conv layers.
- ReLU in the generator; `LeakyReLU(0.2)` in the discriminator.

| Network | Parameters |
| --- | ---: |
| Generator | 741,249 |
| Discriminator | 192,881 |
| **Total** | **934,130** |

The generator returns `(N, 1, 28, 28)` and the discriminator returns `(N, 1)`.
The total is about `0.93M` parameters, which keeps CPU training practical.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Dataset

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) is loaded with
`torchvision.datasets.FashionMNIST`.

| Property | Value |
| --- | --- |
| Training images | 60,000 |
| Test images | 10,000 |
| Image size | 28×28 grayscale |
| Classes | 10 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot) |
| Normalization | `[-1, 1]`, to match the generator's `Tanh` output |

The download lives in `data/raw/`, which is ignored by Git. A fixed noise/label
grid is reused across runs so sample grids are directly comparable.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Repository Structure

```text
.
├── README.md                  ← this file
├── data/
│   └── raw/                   ← Fashion-MNIST download (ignored)
├── notebooks/
│   ├── 01-ad-data-exploration.ipynb      ← planned
│   └── 02-ad-cgan-experiments.ipynb      ← planned
├── src/
│   ├── __init__.py            ← planned
│   ├── config.py              ← planned: paths, seeds, hyperparameters
│   ├── data.py                ← planned: loaders and fixed sample grid
│   ├── models.py              ← planned: Generator and Discriminator
│   ├── train.py               ← planned: adversarial loop and checkpoints
│   ├── sample.py              ← planned: generate grids from a checkpoint
│   ├── evaluate.py            ← planned: conditional accuracy and mode coverage
│   ├── plots.py               ← planned: loss curves and sample figures
│   └── utils.py               ← planned: seeding, device, checkpoint helpers
├── tests/
│   ├── test_data.py           ← planned: shapes, range, label mapping
│   └── test_models.py         ← planned: forward-pass shape checks
├── logs/
│   └── LOG_INTERAKSI_LLM.md   ← required interaction log (to be created)
├── references/
│   └── README.md              ← planned: paper, dataset, and API links
├── docs/
│   ├── architecture.md        ← planned
│   └── experiment_protocol.md ← planned
├── reports/
│   └── figures/               ← curated figures for the final report
└── outputs/                   ← generated artifacts (ignored)
    ├── checkpoints/
    ├── samples/
    └── logs/
```

Tracked empty directories contain a `.gitkeep` placeholder because Git does not
track empty directories. `data/` and `outputs/` are intentionally ignored.

**What lives where:**

- `src/` contains reusable code; notebooks are for exploration only.
- `logs/LOG_INTERAKSI_LLM.md` is the graded LLM interaction log.
- `references/` holds source links and notes, never copied code blocks.
- `reports/figures/` holds the small set of figures used in the final report.
- `outputs/` holds checkpoints, full sample grids, and raw training logs.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

- Python 3.10+ (tested on 3.14)
- [uv](https://docs.astral.sh/uv/) (any recent version)
- Linux, macOS, or Windows
- No GPU required

### Minimum hardware

| Component | Minimum | Recommended |
| --- | --- | --- |
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8 GB |
| Disk | ~3 GB free | 5 GB |
| GPU | Not required | Optional |

Training runs on CPU. Runtime depends on your CPU and batch size; use batch
size `32` on lower-end machines. Checkpoint after every epoch so a slow run
can resume.

### Installation

```bash
git clone https://github.com/AI-IF-TelU/Proyek1-CGAN-ADayInAli.git
cd Proyek1-CGAN-ADayInAli

uv venv
source .venv/bin/activate

# CPU-only wheels: smaller download, no CUDA
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install matplotlib tqdm
```

Verify the environment:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

Dependencies stay inside `.venv`; the system Python is left untouched. On
Windows, activate with `.venv\Scripts\activate` instead.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

> These commands are **planned**. They will fail until `src/` is implemented.

```bash
# Train the conditional GAN
python -m src.train --epochs 30 --batch-size 128

# Generate a sample grid from a checkpoint
python -m src.sample --checkpoint outputs/checkpoints/ckpt.pt

# Evaluate conditional accuracy and mode coverage
python -m src.evaluate --checkpoint outputs/checkpoints/ckpt.pt
```

Recommended workflow: explore in `notebooks/`, move reusable logic into `src/`,
then run the scripts end to end.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Training and Evaluation Plan

### Training

- CPU-only by default; no GPU required.
- Default batch size `128`; use `64` or `32` on lower-end machines if an epoch
  is slow or memory is tight.
- Runtime depends on your CPU. Measure one epoch first, then choose the number
  of epochs.
- 20–30 epochs is the default target; checkpoint after every epoch.
- Save a fixed-grid sample sheet every few epochs.
- Record generator loss, discriminator loss, and discriminator accuracy.

### Evaluation

- **Conditional accuracy:** train a small CNN classifier on real Fashion-MNIST,
  generate a balanced set of images for each class, and measure how often the
  classifier agrees with the requested label.
- **Mode coverage:** inspect whether each class produces varied outputs rather
  than collapsing to a single template; combine classifier-prediction spread
  with visual inspection of the fixed grid.
- **No FID:** Inception is too slow on this CPU-only setup, so FID is not used.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Scope

This project deliberately keeps a narrow scope: one architecture (cGAN), one
dataset (Fashion-MNIST), and at most two hyperparameter variants. The hard part
is GAN training stability, not the amount of code.

**Scope guards:** no FID, no extra datasets, no architecture search. The rubric
puts 45% on the LLM log and teamwork, so the log and report are written
alongside the code rather than at the end.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Team

| Member | Student ID |
| --- | --- |
| Moreno Cello Rhyrhm | 103012340256 |
| Adhyatma Chrysostomos Davu | 103012340479 |
| Maliki Kadafa Tsaqif | 103012340185 |
| Mohamed Ali Ibrahim | 103012350550 |

The four job desks from the assignment guide — prompt engineering/LLM navigation,
data pipeline, training loop/hyperparameter tuning, and documentation/error
handling — are recorded in the LLM interaction log as the project progresses.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Project Workflow

- **Vibe coding:** every code change is produced by directing an LLM, with prompts
  and reasoning captured in the log.
- **Interaction log:** each session records *Inisiasi*, *Debugging*, and
  *Optimasi*; the guide's template omits the last section, but the rubric grades
  it.
- **No copied tutorials:** external sources are cited in `references/`; only
  official API documentation is consulted for basic functions.
- **Git:** one logical change per commit, imperative commit subjects, and
  per-member authorship so contributions are visible.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] Repository scaffold, `.gitignore`, and README
- [ ] Create the project virtual environment and install `torchvision`
- [ ] Implement `src/data.py` and verify Fashion-MNIST shapes/normalization
- [ ] Implement `src/models.py` and shape tests
- [ ] Implement `src/train.py` with checkpointing and loss curves
- [ ] Implement `src/sample.py` with a fixed noise/label grid
- [ ] Implement `src/evaluate.py` (conditional accuracy + mode coverage)
- [ ] Run the first full training and record metrics
- [ ] Complete `logs/LOG_INTERAKSI_LLM.md` and the final report

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

This is an academic course project for CAK4KBB3. No open-source license is
granted; all rights remain with the authors and the course. Third-party datasets
and libraries retain their own licenses.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

- [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) —
  Mirza & Osindero, 2014
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) —
  Zalando Research
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html) — API
  reference for layers, optimizers, and data loading
- CAK4KBB3 project guide — assignment requirements and rubric

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Badge references -->
[python-shield]: https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[pytorch-shield]: https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[pytorch-url]: https://pytorch.org/
[dataset-shield]: https://img.shields.io/badge/Dataset-Fashion--MNIST-FF6F00?style=for-the-badge
[dataset-url]: https://github.com/zalandoresearch/fashion-mnist
[status-shield]: https://img.shields.io/badge/Status-Scaffolding-yellow?style=for-the-badge
[status-url]: #roadmap
