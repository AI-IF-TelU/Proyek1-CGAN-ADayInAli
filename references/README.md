# References & Source Notes

**Project:** GEN AI Project 1 — Conditional GAN on Fashion-MNIST<br>
**Group:** A Day in Ali · CAK4KBB3<br>
**Last link check:** 2026-10-28

> This file records the sources used to design, implement, and verify the
> project. It contains **links and citations only** — no tutorial code,
> notebooks, or vendored PDFs.
>
> The implementation is written from scratch through the team's LLM-guided
> sessions. Official library documentation is used for API reference only.

## 1. Primary sources

| Source | Why it matters | Links |
| --- | --- | --- |
| Mirza & Osindero, *Conditional Generative Adversarial Nets* (2014) | Defines conditioning both G and D on the label `y` and combining `z` with `y` in a joint hidden representation. | [Abstract](https://arxiv.org/abs/1411.1784) · [PDF](https://arxiv.org/pdf/1411.1784) · [DOI](https://doi.org/10.48550/arXiv.1411.1784) |
| Goodfellow et al., *Generative Adversarial Nets* (2014) | Original adversarial objective and the non-saturating generator loss used as context. | [Abstract](https://arxiv.org/abs/1406.2661) · [PDF](https://arxiv.org/pdf/1406.2661) |
| Xiao, Rasul & Vollgraf, *Fashion-MNIST* (2017) | Dataset design: 60,000 train / 10,000 test, 28×28 grayscale, 10 classes. | [Paper](https://arxiv.org/abs/1708.07747) · [Dataset repo](https://github.com/zalandoresearch/fashion-mnist) |

## 2. Official API documentation

Only the functions actually used by the project are listed.

### PyTorch

- [`torch.load`](https://docs.pytorch.org/docs/stable/generated/torch.load.html) — checkpoint loading with `weights_only=True`.
- [`torch.save`](https://docs.pytorch.org/docs/stable/generated/torch.save.html) — state-dict checkpoints.
- [`DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) — batching, shuffling, worker seeding.
- [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) — `manual_seed`, `worker_init_fn`, `generator`.
- [`nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [`nn.ConvTranspose2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)
- [`nn.BCEWithLogitsLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [`optim.Adam`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)
- [`torch.bincount`](https://docs.pytorch.org/docs/stable/generated/torch.bincount.html) and [`torch.cdist`](https://docs.pytorch.org/docs/stable/generated/torch.cdist.html) — evaluation helpers.

### torchvision

- [`FashionMNIST`](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html)
- [Transforms](https://docs.pytorch.org/vision/stable/transforms.html) — `ToTensor`, `Normalize`.
- [`save_image`](https://docs.pytorch.org/vision/stable/generated/torchvision.utils.save_image.html) and [`make_grid`](https://docs.pytorch.org/vision/stable/generated/torchvision.utils.make_grid.html) — `normalize=True, value_range=(-1, 1)`.

### Matplotlib and Python

- [`Axes.set_xticks`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xticks.html) — confusion-matrix labels.
- [`Figure.savefig`](https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.savefig)
- [`pathlib`](https://docs.python.org/3/library/pathlib.html) — paths and UTF-8 text I/O.
- [`shutil.copy2`](https://docs.python.org/3/library/shutil.html#shutil.copy2) — report-figure copies.
- [`argparse`](https://docs.python.org/3/library/argparse.html) — CLI validation.

## 3. How sources were used

| Source | Used for | Code copied? |
| --- | --- | --- |
| cGAN paper | Conditioning mechanism and architecture rationale | No |
| GAN paper | Adversarial objective and training context | No |
| Fashion-MNIST paper/repo | Dataset shape, split, labels, and loading conventions | No |
| PyTorch / torchvision docs | API signatures and documented behavior | No — API calls only |
| Matplotlib / Python docs | Plotting and standard-library behavior | No |

**Not used as a base:** GitHub cGAN notebooks, Medium/LearnOpenCV
walkthroughs, YouTube tutorials, or the PyTorch DCGAN tutorial code.
Those are tutorials, not API references, and the assignment forbids
copying them.

## 4. What is safe to download

| Item | Recommendation |
| --- | --- |
| The three arXiv papers above | Safe to download as PDFs for reading and citation. |
| Official PyTorch / torchvision / Matplotlib / Python docs | Read online; use only as API reference. |
| Fashion-MNIST | Let `torchvision` download it to `data/raw/` (ignored by Git). |
| Tutorial code, notebooks, or cGAN GitHub repositories | **Do not copy.** Not a reference for this project. |

This repository stays **link-only**. If an offline copy of a paper is
needed for the report, keep the PDF outside the repository or in a
local ignored folder; do not vendor tutorial code or datasets.

## 5. Citations

```bibtex
@article{mirza2014conditional,
  title   = {Conditional Generative Adversarial Nets},
  author  = {Mirza, Mehdi and Osindero, Simon},
  journal = {arXiv preprint arXiv:1411.1784},
  year    = {2014},
  doi     = {10.48550/arXiv.1411.1784}
}

@inproceedings{goodfellow2014generative,
  title     = {Generative Adversarial Nets},
  author    = {Goodfellow, Ian and Pouget-Abadie, Jean and Mirza, Mehdi and
               Xu, Bing and Warde-Farley, David and Ozair, Sherjil and
               Courville, Aaron and Bengio, Yoshua},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {27},
  year      = {2014}
}

@article{xiao2017fashion,
  title   = {Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
  author  = {Xiao, Han and Rasul, Kashif and Vollgraf, Roland},
  journal = {arXiv preprint arXiv:1708.07747},
  year    = {2017}
}
```

## 6. Project-local sources

- `README.md` — model, setup, results, and workflow.
- `logs/` — per-member LLM interaction logs.
- `[CAK4KBB3 2026-1] Panduan Proyek 1 - Vibe Coding.md` — course guide
  (local only; not vendored).
