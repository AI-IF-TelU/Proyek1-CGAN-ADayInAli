<a id="readme-top"></a>

<div align="center">
  <h1>GEN AI Project 1</h1>
  <p><strong>A class-conditional GAN built from scratch on Fashion-MNIST</strong></p>
  <p>Course project for <strong>CAK4KBB3 — Vibe Coding Generative AI from Scratch</strong><br>
  Group <strong>A Day in Ali</strong></p>
</div>

[![Python][python-shield]][python-url]
[![PyTorch][pytorch-shield]][pytorch-url]
[![Dataset][dataset-shield]][dataset-url]
[![Status][status-shield]][status-url]

> **Project status:** the data pipeline, models, training loop, sampling,
> evaluation, tests, and report figures are implemented. A 30-epoch baseline has
> been trained and evaluated (see [Results](#results)). Per-member LLM logs exist;
> log consolidation and the final report are in progress.

## Table of Contents

- [About the Project](#about-the-project)
- [Model](#model)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Training and Evaluation Plan](#training-and-evaluation-plan)
- [Results](#results)
- [Scope](#scope)
- [Team](#team)
- [Project Workflow](#project-workflow)
- [Roadmap](#roadmap)
- [References](#references)
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
│   ├── 01-ad-data-exploration.ipynb      ← planned; not implemented
│   └── 02-ad-cgan-experiments.ipynb      ← active: training, figures, evaluation
├── src/
│   ├── __init__.py
│   ├── config.py              ← paths, seeds, hyperparameters
│   ├── data.py                ← loaders and fixed sample grid
│   ├── models.py              ← Generator and Discriminator
│   ├── train.py               ← adversarial loop and checkpoints
│   ├── sample.py              ← generate grids from a checkpoint
│   ├── evaluate.py            ← conditional accuracy and mode coverage
│   ├── plots.py               ← loss curves
│   └── utils.py               ← seeding, device, checkpoint helpers, report copies
├── tests/
│   ├── test_data.py           ← shapes, range, label mapping
│   ├── test_models.py         ← forward-pass shape checks
│   ├── test_train.py          ← training step, outputs, and resume
│   └── test_sample_evaluate.py ← sampling and evaluation metrics
├── logs/
│   ├── ALI_LOG_INTERAKSI_LLM.md
│   ├── DAVU_LOG_INTERAKSI_LLM.md
│   ├── MALIKI_LOG_INTERAKSI_LLM.md
│   └── MORENO_LOG_INTERAKSI_LLM.md
├── references/
│   └── README.md              ← annotated paper, dataset, and API links
├── docs/
│   ├── architecture.md        ← planned
│   └── experiment_protocol.md ← planned
├── reports/
│   └── figures/               ← report copies: loss curve, grids, confusion matrix, metrics
└── outputs/                   ← generated artifacts (ignored)
    ├── checkpoints/
    ├── samples/
    └── logs/
```

Tracked empty directories contain a `.gitkeep` placeholder because Git does not
track empty directories. `data/` and `outputs/` are intentionally ignored.

**What lives where:**

- `src/` contains reusable code; `02` is the active notebook, while `01` remains
  planned and is not implemented.
- `logs/` holds the per-member LLM interaction logs (Ali, Davu, Maliki,
  Moreno); a unified `LOG_INTERAKSI_LLM.md` is not yet created.
- `references/README.md` holds the annotated paper, dataset, and API
  references; no code blocks are copied.
- `reports/figures/` holds report-ready copies of the loss curve, sample grids,
  confusion matrix, and evaluation metrics.
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

Training runs on CPU by default; CUDA is optional. Runtime depends on your
CPU/GPU and batch size; use batch size `32` on lower-end machines. Checkpoint
after every epoch so a slow run can resume.

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

For a CUDA build, install the matching PyTorch/torchvision pair from the
[official selector](https://pytorch.org/get-started/locally/) instead; the
notebook selects CUDA automatically.

Verify the environment:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

Dependencies stay inside `.venv`; the system Python is left untouched. On
Windows, activate with `.venv\Scripts\activate` instead.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### Fresh run vs. resume

`outputs/` is ignored and safe to delete. For a clean run, remove it **before**
starting:

```bash
rm -rf outputs
```

On Windows PowerShell, use `Remove-Item -Recurse -Force outputs` instead.

- **Notebook:** `RESUME=True` loads `outputs/checkpoints/ckpt.pt`. If that file
  exists, training resumes instead of starting over. Clean `outputs/` for a fresh
  run, or raise `EPOCHS` to continue from the checkpoint.
- **CLI:** `src.train` starts fresh unless `--resume` is passed; cleaning
  `outputs/` is still recommended so old epoch checkpoints, samples, and logs do
  not mix with the new run.
- `data/raw/` is separate and can stay; Fashion-MNIST does not need to be
  downloaded again.

```bash
# Clean run from epoch 1
rm -rf outputs
python -m src.train --epochs 30 --batch-size 128

# Continue a stopped run (--epochs must be above the checkpoint epoch)
python -m src.train --resume outputs/checkpoints/ckpt.pt --epochs 40

# Quick smoke test: 1 epoch, 20 batches
python -m src.train --epochs 1 --max-batches 20

# Run all tests (test_data downloads Fashion-MNIST on first run)
python -m unittest discover -s tests
```

Training writes, under `outputs/`:

- `checkpoints/ckpt.pt` — latest checkpoint, overwritten every epoch.
- `checkpoints/ckpt_epoch_NNN.pt` — kept on the first epoch of each run, every
  `--sample-every` epochs (default 5), and the last epoch.
- `samples/epoch_NNN.png` — fixed noise/label grid, one row per class, saved on
  the same epochs.
- `logs/history.json` and `logs/loss_curves.png` — per-epoch mean `loss_d`,
  `loss_g`, discriminator accuracy, `D(x)`, and `D(G(z))`.

Report-ready copies of the existing outputs are also written to
`reports/figures/`:

- `loss_curves.png` — copy of `outputs/logs/loss_curves.png`
- `sample_grid_final.png` — final-epoch grid from `outputs/samples/`
- `confusion_matrix_final.png` — copy of `outputs/logs/confusion_ckpt.png`
- `real_samples.png` — real-data batch grid from the notebook
- `sample_grid_progress.png` — epoch 1/10/20/30 grids from the notebook
- `fresh_samples_per_class.png` — fresh per-class samples from the notebook
- `evaluation_metrics.png` — metrics-by-checkpoint chart from the notebook

The checked-in baseline was run on CUDA (torch `2.14.1+cu130`). The model also
runs on CPU; measure your first epoch and plan the run length accordingly.

```bash
# Generate a sample grid from a checkpoint (one row per class)
python -m src.sample --checkpoint outputs/checkpoints/ckpt.pt

# Only some classes, more samples per row (class ids 0-9, see Dataset)
python -m src.sample --checkpoint outputs/checkpoints/ckpt.pt --classes 5 8 --per-class 16

# Evaluate conditional accuracy and mode coverage
python -m src.evaluate --checkpoint outputs/checkpoints/ckpt.pt
```

`src.sample` writes `outputs/samples/sample_epoch_NNN.png`. `src.evaluate` trains
the reference classifier on first use (about 1.5 minutes, cached in
`outputs/checkpoints/classifier.pt`), then writes `outputs/logs/eval_<checkpoint>.json`
and a confusion-matrix figure `outputs/logs/confusion_<checkpoint>.png`.

Recommended workflow: explore in `notebooks/`, move reusable logic into `src/`,
then run the scripts end to end.

`notebooks/02-ad-cgan-experiments.ipynb` is the active notebook: it runs the same
training from Jupyter, resumes from the latest checkpoint, saves the report
figures, and shows loss curves, fixed-grid progress, fresh samples per class,
and evaluation metrics across checkpoints. Install `ipykernel` in `.venv` to use
it as a kernel. `notebooks/01-ad-data-exploration.ipynb` remains planned and is
not implemented.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Training and Evaluation Plan

### Training

- Runs on CPU by default; the notebook selects CUDA when available. No GPU
  required.
- Default batch size `128`; use `64` or `32` on lower-end machines if an epoch
  is slow or memory is tight.
- Runtime depends on the available CPU/GPU. Measure one epoch first, then choose
  the number of epochs.
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
- **No FID:** Inception is too slow on the available CPU setup, so conditional
  accuracy and mode coverage are used instead.

`src/evaluate.py` implements this with a small CNN (two conv layers, 6 epochs on
the training set) and 1,000 generated images per class. Mode coverage is measured
two ways: the normalized entropy of predicted classes (1.0 = all classes equally
represented) and a per-class diversity ratio, the mean pairwise distance between
generated images in the classifier's feature space divided by the same value for
real test images (about 1.0 = as varied as real data, near 0 = collapse).

### Reproducibility and safety

- Python, NumPy, and PyTorch are seeded; DataLoader workers reseed from the
  worker seed, and one fixed noise/label grid is reused across runs.
- Checkpoints contain state dicts and load with
  `torch.load(..., weights_only=True)`.
- JSON logs use explicit UTF-8 encoding.
- CLI numeric arguments reject zero values; use positive counts.
- Report figures are copies of existing outputs and notebook figures; no new
  plot types are generated.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Results

Baseline run: 30 epochs, batch size 128, lr `2e-4`, CUDA (torch `2.14.1+cu130`).
Full outputs, figures, and notes are in `notebooks/02-ad-cgan-experiments.ipynb`;
report copies are in `reports/figures/`.

| Metric | Value |
| --- | ---: |
| Classifier accuracy on real test images (ceiling) | 0.9129 |
| Conditional accuracy on generated images | **0.9170** |
| Prediction entropy (1.0 = balanced) | 0.9986 |
| Mean diversity ratio (1.0 = real) | 0.9448 |

| Epoch | 1 | 5 | 10 | 20 | 30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conditional accuracy | 0.737 | 0.827 | 0.903 | 0.913 | 0.917 |
| Diversity ratio | 0.648 | 0.844 | 0.898 | 0.940 | 0.945 |

- The generator follows the requested label about as reliably as the classifier
  recognizes real images (0.9170 vs 0.9129); most of the gain happens by epoch 10.
- Trouser, Ankle boot, Bag, Sandal, and Sneaker score 0.97–1.00. Pullover (0.777,
  often read as Coat) and Shirt (0.774) are weakest; the classifier confuses the
  same classes on real data.
- No mode collapse: every class is represented (entropy 0.9986) and within-class
  variety is close to real data (diversity 0.9448).
- The discriminator slowly gains on the generator (D accuracy 0.906 → 0.934,
  `D(G(z))` 0.24 → 0.15), so the planned variants target D: no BatchNorm on its
  first layer, or a lower D learning rate.

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
- [x] Create the project virtual environment and install `torchvision`
- [x] Implement `src/data.py` and verify Fashion-MNIST shapes/normalization
- [x] Implement `src/models.py` and shape tests
- [x] Implement `src/train.py` with checkpointing and loss curves
- [x] Implement `src/sample.py` with a fixed noise/label grid
- [x] Implement `src/evaluate.py` (conditional accuracy + mode coverage)
- [x] Run the first full training and record metrics
- [x] Add annotated references and BibTeX (`references/README.md`)
- [x] Add report-figure copies and notebook `savefig`
- [x] Verify PyTorch APIs, safe checkpoint loading, and reproducibility
- [ ] Consolidate the per-member LLM logs and write the final report

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## References

See [`references/README.md`](references/README.md) for annotated cGAN and GAN
papers, the Fashion-MNIST paper and dataset, official PyTorch/torchvision,
Matplotlib, and Python API links, plus BibTeX. The repository stays link-only;
no tutorial code or PDFs are vendored.

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
- [Annotated reference list](references/README.md) — full sources and BibTeX
- CAK4KBB3 project guide — assignment requirements and rubric

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Badge references -->
[python-shield]: https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[pytorch-shield]: https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[pytorch-url]: https://pytorch.org/
[dataset-shield]: https://img.shields.io/badge/Dataset-Fashion--MNIST-FF6F00?style=for-the-badge
[dataset-url]: https://github.com/zalandoresearch/fashion-mnist
[status-shield]: https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge
[status-url]: #roadmap
