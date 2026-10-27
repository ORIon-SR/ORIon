# ORIon: a staged comparison of 4x super-resolution methods on NASA APOD images

ORIon compares seven ways of enlarging astronomical images by a factor of four, all scored by one script on the same held-out data:

| # | Method | Source |
|---|--------|--------|
| 1 | Bicubic interpolation | PyTorch `F.interpolate` |
| 2 | Swin2SR, zero-shot | pretrained `caidas/swin2SR-classical-sr-x4-64` (Hugging Face), not fine-tuned |
| 3 | SRCNN-style residual CNN | ours (`implementations/srcnn.py`) |
| 4 | Conditional DDPM (1000 steps) | ours (`implementations/bicubic_diffusion.py`) |
| 5 | Consistency model (iCT), sampled with 1, 2 and 4 steps | ours (`implementations/consistency_model.py`) |

The report (IEEE format) describes the method, results and limitations. Headline result on 9,902 test patches from 251 unseen images: SRCNN-style 37.22 dB PSNR / 0.880 SSIM, zero-shot Swin2SR 37.12 dB / 0.880, bicubic 33.94 dB / 0.842.

## Repository layout

```
scripts/            data pipeline (download+filter, split, patch extraction) and analysis helpers
implementations/    models, training scripts (*.slurm are the exact commands we ran), evaluation
apod_data/splits/   the train/val/test image lists we used (committed)
models_eval/test/   our evaluation outputs: summary.json and per_patch.csv (committed)
nasa_apod_complete.csv   APOD metadata (Kaggle: ahsanneural/nasa-astronomy-picture-of-the-day-1995-2026)
pyproject.toml, uv.lock  pinned environment
```

## 1. Setup

Requirements: Python >= 3.14, [uv](https://docs.astral.sh/uv/), and for training/evaluation of the generative models NVIDIA GPUs with CUDA (training uses `torchrun` with 2 GPUs and the NCCL backend).

```bash
git clone <this repository> ORIon && cd ORIon
uv sync                      # creates .venv from uv.lock
source .venv/bin/activate
export WANDB_MODE=disabled   # logging goes to Weights & Biases; disable it, or run `wandb login`
```

Do not use `requirements.txt` (a stale export that does not list PyTorch); `pyproject.toml` / `uv.lock` are the source of truth.

## 2. Data

The images come from NASA's Astronomy Picture of the Day (https://apod.nasa.gov). **Licensing:** 66% of the images we kept (1,646 of 2,506) are credited to third-party photographers or observatories, so they are not public domain. Use them for coursework only and do not redistribute the images or the patches.

```bash
# (a) Download and resolution-filter. Keeps images with both sides >= 1024 px (2 x 512). Network-bound; allow a long time.
python scripts/apod_parse_filter.py --csv nasa_apod_complete.csv --output-dir apod_data --hr-patch-size 512

# (b) Split by image (not by patch). The split we used is already committed in apod_data/splits/;
#     regenerating it needs the same 2,506 images. Note the explicit ratios: the script default is 0.8/0.1/0.1.
python scripts/apod_dataset_split.py --input apod_data/images --output-dir apod_data/splits \
    --train 0.85 --val 0.05 --test 0.10 --seed 0 --force

# (c) Extract 512x512 HR patches and their 128x128 LR pairs (x4 bicubic down-sampling).
python scripts/patch_creator.py --downscale-factor 4 --seed 42 --num-workers 8
```

Expected: 2,130 / 125 / 251 train / val / test images, about 90,000 training patches and 9,902 test patches in `apod_data/dataset/{train,val,test}/{HR,LR}/`. Per image, the number of patches is `clip(round(2*H*W/512^2), 4, 100)`, chosen at random positions with a texture filter (grayscale std >= 15).

## 3. Train (from `implementations/`)

```bash
cd implementations
torchrun --nproc-per-node=2 srcnn.py            --batch-size 64 --scaling-factor 4 --seed 42
torchrun --nproc-per-node=2 bicubic_diffusion.py --batch-size 8 --epochs 200 --patience 50 --seed 42
torchrun --nproc-per-node=2 consistency_model.py --batch-size 8 --epochs 200 --patience 50 --seed 42
```

These are the commands in `train_*.slurm` (batch sizes are per GPU). Checkpoints are written to `implementations/checkpoints/{srcnn,diffusion,consistency}_best.pth` (git-ignored) and preview images to `previews/`. All three use Adam with learning rate 1e-4 and early stopping on the validation set.

## 4. Evaluate (from `implementations/`)

```bash
python models_eval.py --split test --diffusion-batch-size 64 --force --seed 0      # all 7 variants; about 26 h in our run, mostly the 1000-step diffusion
python models_eval.py --split test --models bicubic srcnn swin2sr --force          # quick check, no generative sampling
python models_eval.py --split test --limit 200 --force                             # smoke test on 200 patches
```

Outputs go to `implementations/results/models_eval/test/`: `summary.json` (patch means, image means, 95% bootstrap CIs over images) and `per_patch.csv`. Our own run is committed in `models_eval/test/`.
To pin the Swin2SR weights add `--revision <hub commit hash>`.

## 5. Report numbers and figures

```bash
python scripts/paired_stats.py models_eval/test/per_patch.csv             # paired SRCNN-vs-Swin2SR differences, % of patches below bicubic
python scripts/make_report_figures.py models_eval/test/summary.json Figures   # writes Figures/metrics.png (+ per-metric PNGs); needs matplotlib
```

## Seeds

| Step | Seed | Notes |
|------|------|-------|
| Train/val/test split | `--seed 0` | shuffles the sorted image list |
| Patch extraction | `--seed 42` | each image gets its own generator from `(seed, file name)`, so the result does not depend on the number of workers, the process start method or the order |
| SRCNN, diffusion, consistency training | `--seed 42` (+ GPU rank) | `torch.manual_seed`; the data sampler is seeded and re-shuffled per epoch |
| Diffusion / consistency sampling at test time | `--seed 0` | the noise draw depends on the batch size, so use the batch sizes above |
| Bootstrap confidence intervals | 0 | 2,000 resamples of the test images |

## Reproducibility notes and known limits

- **Not bit-exact.** Training uses mixed precision, `torch.compile` and multi-GPU NCCL, which are not fully deterministic. Expect small run-to-run differences; the ranking of methods should hold, but we have not re-run the full pipeline from a clean checkout to check how large the differences are.
- **Patch set.** The patches behind the numbers in the report were generated before per-image seeding was added to `patch_creator.py`, so a fresh run produces a different (but now fixed) random patch set and slightly different metrics. The files used for the report are available from the authors on request.
- **Checkpoints and patches are not in the repository** (large, git-ignored); steps 2c and 3 regenerate them.
- **Swin2SR** weights are downloaded from the Hugging Face Hub; the revision is not pinned unless you pass `--revision`.
- **Live downloads.** Step 2a depends on apod.nasa.gov still serving the same images; the committed split lists record which images we used.
- The `*.slurm` files target our university cluster (node names, paths); the commands inside them are the ones listed above.

## Use of AI tools

See the "Use of AI Tools" section of the report.
