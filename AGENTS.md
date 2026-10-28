# GEN AI Project 1 — Conditional GAN from scratch (Fashion-MNIST)

Course project for CAK4KBB3 "Vibe Coding Generative AI from Scratch", group **A Day in Ali**. The cGAN, tests, notebook, and a 30-epoch baseline are implemented; LLM logs and the final report are still in progress. This file records the decisions and verified limits.

## Project

- Assignment guide: `[CAK4KBB3 2026-1] Panduan Proyek 1 - Vibe Coding.md` (Indonesian, authoritative for deliverables).
- Project overview: `README.md` — directory map, model summary, setup, team, workflow. Tracked and committed.
- Stack: Python 3.14.8 + PyTorch 2.14.1 (CPU), uv 0.13.6; `torchvision` still missing locally (required by data/train/sample/evaluate).
- Entry point (implemented): `python -m src.train`; `src/` code exists, but torchvision is needed to run it.
- Team: four members — see **Team** below.
- **Model (final choice): Conditional GAN (cGAN)** — class-conditional GAN conditioned on the 10 Fashion-MNIST labels, 28x28 inputs, 10 classes. Proposal sent to lecturer; approval pending.
- Method is mandatory: code is produced by directing an LLM, never copied from tutorials. Every session must be logged.
- Grading: vibe-coding originality 30%, LLM log 25%, model/training 25%, teamwork 20%.
- Baseline: 30 epochs on CUDA; README reports 0.9170 conditional accuracy, 0.9986 entropy, and 0.9448 diversity (earlier CPU run: 0.912).
- Repo: `main` tracks `origin/main` (SSH); 32 commits, scaffold `ab994d3`; HEAD is one commit ahead of `origin/main` (push from the user's terminal).
- Per-member commits are graded; every change must be staged in Git as it is made — see **Git workflow** below.
- LLM logs: all four per-member logs exist (`ALI`, `DAVU`, `MALIKI`, `MORENO`); a unified `logs/LOG_INTERAKSI_LLM.md` index is missing.

## Team

| Member | Student ID |
| --- | --- |
| Moreno Cello Rhyrhm | 103012340256 |
| Adhyatma Chrysostomos Davu | 103012340479 |
| Maliki Kadafa Tsaqif | 103012340185 |
| Mohamed Ali Ibrahim | 103012350550 |

Role mapping follows the guide's four job desks and is recorded in the LLM interaction log.

## Commands

Verified locally (2026-10-28):

```bash
python3 --version          # 3.14.8
uv --version               # 0.13.6
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # 2.14.1 False
python3 -c "import torchvision"   # MISSING
```

Environment floor for teammates: Python 3.10+ (tested on 3.14), PyTorch 2.x, no GPU.

`torchvision` is still missing here; install it before data/training/sampling/evaluation:
`uv venv && uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && uv pip install matplotlib tqdm`

Tests use `unittest` (no pytest). The full suite needs torchvision; the model-only tests pass now:
- `python -m unittest discover -s tests -p "test_models.py"` — 2 tests pass.
- `python -m compileall -q src tests` — passes.
- `python -m unittest discover -s tests` — full suite after torchvision setup.

Implemented entry points (need torchvision):

```bash
python -m src.train --epochs 30 --batch-size 128
python -m src.train --resume outputs/checkpoints/ckpt.pt --epochs 30
python -m src.train --epochs 1 --max-batches 20
python -m src.sample --checkpoint outputs/checkpoints/ckpt.pt
python -m src.evaluate --checkpoint outputs/checkpoints/ckpt.pt
```

No `pyproject.toml`, `requirements.txt`, or `Makefile`; dependencies are installed directly.

## Repository layout

As of 2026-10-28, the implementation and a local 30-epoch output set are present; `outputs/` is tracked (small artifacts) and `data/` is ignored by Git.

- `README.md` — overview, model, setup, Results, team, workflow.
- `src/config.py` — seeds, paths, hyperparameters, class names.
- `src/data.py` — Fashion-MNIST loaders, `[-1, 1]` normalization, fixed grid.
- `src/models.py` — Generator (741,249 params) and Discriminator (192,881), 934,130 total.
- `src/train.py` — adversarial loop, checkpoint/resume, sample grids, loss curves.
- `src/sample.py` — sample grids from a checkpoint.
- `src/evaluate.py` — reference CNN, conditional accuracy, entropy, diversity, confusion matrix.
- `src/plots.py`, `src/utils.py` — loss curves; seeding/checkpoint helpers; report-figure copy.
- `tests/` — 4 unittest files; model tests pass locally, the rest need torchvision.
- `notebooks/02-ad-cgan-experiments.ipynb` — 30-epoch baseline with outputs and metrics.
- `logs/` — per-member logs (`ALI`, `DAVU`, `MALIKI`, `MORENO`); no unified `LOG_INTERAKSI_LLM.md`.
- `references/README.md` — annotated paper, dataset, and API links; `docs/` still planned/empty.
- `reports/figures/` — report copies from `outputs/` and the notebook: loss curve, sample grids, confusion matrix, real samples, fresh samples, eval metrics.
- `data/raw/` — Fashion-MNIST download (ignored; populated here).
- `outputs/` — baseline samples/logs tracked; weight files (`*.pt`) ignored.

## Conventions

- Log every LLM interaction in the guide's template **plus a third "Tahap Optimasi" section**; the rubric grades initiation -> debugging -> optimization, but the template omits the last one.
- Never paste tutorial code blocks. Library docs may be referenced for basic functions only.
- Portability floor: 2-core CPU, 4 GB RAM, ~3 GB free disk, no GPU; 4-core/8 GB is the recommended baseline. The 24-thread/32 GB machine is the developer reference, not a requirement. Keep models ~1M params; default batch 128, fall back to 64/32 if speed or memory is tight.
- Measured on the 24-thread reference machine: trimmed cGAN 1.06M params = 136 ms/step (~64 s/epoch); 1.7M-param variant = 312 ms/step (~146 s/epoch); minimal ViT 0.45M = 165 ms/step (~77 s/epoch). A 4-core machine will be several times slower; measure actual epoch time before planning runs.
- Fix seeds and reuse one noise/label grid across runs so sample grids are comparable.
- No FID: Inception is too slow on CPU. Use conditional accuracy + mode coverage instead.
- Defaults: `z=64`, Adam `2e-4` betas `(0.5, 0.999)`, ReLU in G, LeakyReLU 0.2 in D, BatchNorm, one-sided label smoothing.
- Tests use `unittest`; run `python -m unittest discover -s tests` after torchvision setup. Model-only tests and `compileall` pass locally.
- Checkpoints are state_dict-only and load with `weights_only=True`; JSON logs use UTF-8; DataLoader workers seed Python/NumPy RNG.
- README metric claims are backed by notebook outputs; re-run `src.evaluate` before changing them.
- Report figures are copies of existing outputs and notebook figures; do not add new plot types unless asked.
- Delivery window: 3 working days. Keep scope to one architecture, one dataset, and at most two hyperparameter variants; no architecture search. Budget extra wall-clock time on low-core machines (4 cores or fewer).
- Commit messages follow [`GIT_COMMIT_GUIDE.md`](./GIT_COMMIT_GUIDE.md), adapted from <https://cbea.ms/git-commit/>: imperative subject <= 50 characters, no trailing period, blank line, body wrapped at 72 explaining what and why.

## Git workflow

- **Stage every change as you make it.** Any file you create, edit, initialize, or generate for the project must be `git add`-ed in the same session, before you finish. Never leave untracked or unstaged work behind.
- This applies to initialization (`.gitignore`, `AGENTS.md`, dependencies), features (`src/`), documentation (`README.md`, `logs/`, guides), fixes, and config changes.
- Make one logical change per commit and commit at the checkpoints the rubric grades: initiation -> debugging -> optimization. Use the repo's configured member identity so the contribution is attributable.
- Do not push from the sandbox: it cannot access the user's SSH agent/key. Commit locally and let the user run `git push` from their terminal.
- Commit messages must follow [`GIT_COMMIT_GUIDE.md`](./GIT_COMMIT_GUIDE.md); verify the log with `git log --oneline`.
- Keep data and weights out of Git: `data/` and `*.pt`/`*.pth` are ignored; never force-add them. `outputs/` logs and sample grids are tracked on purpose so the baseline artifacts ship with the repo; regenerated runs overwrite them, so commit only intentional updates.
- `README.md` is tracked and must be committed; it is not ignored.
- The user may hold a file back for review (for example `README.md`). Do not stage it until they approve.
- Local-only references — `GIT_COMMIT_GUIDE.md`, the assignment guide, `BLANK_README.md`, and `EXAMPLE_README.md` — are intentionally ignored by `.gitignore`. Do not force-add them; they stay on each member's machine. `AGENTS.md` is tracked so the LLM/harness working notes are visible.

## Notes

- **Current state (2026-10-28):** full implementation, tests, notebook baseline, and 4 member logs are in. Missing: a unified log index and the final report. README documents the per-member logs and notes that the unified log is pending.
- **Artifact policy (2026-10-28):** baseline `outputs/` logs and sample grids are tracked; weight files (`*.pt`) stay ignored by decision, so a clone has every metric and figure but must retrain to sample or evaluate.
- **Report figures (2026-10-28):** `src/train.py` and `src/evaluate.py` copy the existing loss curve, final sample grid, and final confusion matrix into `reports/figures/`; the notebook saves its existing inline figures (`real_samples`, `sample_grid_progress`, `fresh_samples_per_class`, `evaluation_metrics`) there too. Raw artifacts stay under `outputs/`.
- **Committed (2026-10-28):** `4678790` contains the README latest-run update, the notebook run, and the four refreshed report figures; `58ff81a` aligns the notebook markdown and README with those results; the latest three commits add the baseline `outputs/` artifacts plus `AGENTS.md`, expand the Moreno log with Sesi 0 and Sesi 8, and add Sesi 9 with a verbatim raw-prompt appendix. The tree is clean; only the latest commit still needs a push.
- **Re-baseline after pulls:** teammates may push large changes. After `git pull`, treat the pulled tree as authoritative; do not assume the scaffold, README model, folder layout, or commands in this file still match. Check `git status`, `git log --oneline --decorate -5`, and the updated `README.md`, then update this file before writing code.
- **Plan is provisional:** the cGAN architecture, 0.93M parameter target, folder names, and entry points may change. Reconcile before implementing.
- **Notebook workflow (2026-10-28):** the user runs `notebooks/02-ad-cgan-experiments.ipynb` manually in the Jupyter kernel "GenAI +cuda" on their laptop. Do not set up or train in the sandbox; it lacks torchvision and SSH. The committed notebook already contains the baseline outputs.
- **Docs review (2026-10-28):** verified `torch.load(weights_only=True)`, DataLoader worker seeding, `torchvision.utils.save_image(value_range=...)`, Matplotlib `set_xticks`, and UTF-8 JSON against the library docs; CLI numeric counts now reject zero values.
- **Notebook resume (2026-10-28):** with `RESUME=True`, `ckpt.pt` at epoch 30, and `EPOCHS=30`, `train()` loads history and skips training but still refreshes `reports/figures/`; raise `EPOCHS` to continue, or clear `outputs/` to start fresh. The newest run completed with 0 errors and 0.917 conditional accuracy.
- **References (2026-10-28):** `references/README.md` is committed with the paper, dataset, and API links plus BibTeX; the root README reference line is committed too.
- **Observed change:** the pull added the implementation and baseline; README now reflects the latest CUDA run (0.9170 conditional accuracy) and documents fresh-run cleanup; the earlier CPU run was 0.912. Model tests pass; the full suite is blocked locally by missing torchvision.

## Known gaps (verified 2026-10-28)

- **Decision:** the team chose to leave the other members' logs untouched; the log issues below are reported for their owners, not fixed here.
- **Ali log is malformed.** `logs/ALI_LOG_INTERAKSI_LLM.md` starts with chat preamble ("My bad bro ... one single code box"), keeps its header and first ~50 lines inside a Markdown code fence opened at line 3, and ends with an empty code fence.
- **Davu and Maliki logs lack the guide's chat-link field.** Neither has a `Tautan Sesi Chat` line or any AI share link; only Ali (three ChatGPT links) and Moreno (explicit Markdown-export note) address it.
- **No unified log index.** `logs/LOG_INTERAKSI_LLM.md` does not exist; the guide's job desk 4 asks for a compiled log, and the README points at the per-member files instead.
- **No final report.** `reports/` holds only `figures/`; the roadmap's last item is unchecked.
- **`docs/` is empty** (`.gitkeep` only); README marks `architecture.md` and `experiment_protocol.md` as planned. `notebooks/01` is intentionally not implemented.
- **Full suite never run here.** `torchvision` is still absent in this sandbox, so only `test_models.py` (2 tests) and `compileall` are verified locally.
- **Untracked session exports.** `session-20261028-230251.md` and `session-20261028-231607.md` sit in the repo root and are not ignored; they were the sources for the Moreno log, which now quotes the raw prompts verbatim, so the log stands alone. The user plans to delete the exports. Do not commit them.
