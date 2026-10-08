# AGENTS.md — ORIon

## Project Overview

ORIon (Observing Resolution Improvement) is a super-resolution research project for COMS4036A at Wits University. It implements multiple "rungs" of an SR ladder trained on NASA Astronomy Picture of the Day (APOD) imagery:

1. Bicubic baseline (reference)
2. SRCNN — trained, checkpoint saved
3. Conditional DDPM diffusion — trained, checkpoint saved
4. Consistency model — implemented, not yet trained

**Author:** Liam Binedell (liam.binedell@gmail.com)

## Environment

- **Python:** 3.14
- **Package manager:** uv (`uv.lock` present)
- **Dependencies:** numpy, pillow, torch, torchvision, wandb (+ pandas, jupyter ecosystem for notebooks)
- **Install:** `uv sync` or `pip install -r requirements.txt`
- **W&B project:** `ORIon-SR`

## Directory Layout

```
ORIon/
  implementations/          # Model architectures + training loops (self-contained files)
    dataloader.py           # SRDataset — paired HR/LR PNG loader
    metrics.py              # Gaussian-windowed SSIM from scratch
    srcnn.py                # SRCNN: residual CNN + PixelShuffle 4x upscale
    bicubic_diffusion.py    # Conditional DDPM: UNet + noise prediction
    consistency_model.py    # Consistency Model: 1-step SR via self-consistency training
  scripts/                  # Data pipeline
    apod_parse_filter.py    # Download + filter APOD images by resolution
    apod_dataset_split.py   # Deterministic train/val/test split (85/5/10)
    patch_creator.py        # Extract 512x512 HR patches + optional LR downscale
    apod_estimate_dataset_size.py  # Download size estimator
    discard_reason.py       # Quick diagnostic for filtered-out images
    generate_patches.slurm  # SLURM job for griffin/prime cluster
  src/orion/                # Package stub (empty __init__.py, not yet used)
  apod_data/
    images/                 # 2,506 raw APOD images (gitignored)
    splits/                 # train.csv (2130), val.csv (125), test.csv (251)
    dataset/                # Generated patches: {split}/HR/ and {split}/LR/ (gitignored)
  checkpoints/              # Saved .pth files (gitignored)
  previews/                 # LR|SR|HR comparison images
  wandb/                    # Local W&B logs (gitignored)
  nasa_apod_complete.csv    # Raw Kaggle APOD metadata (20,287 rows)
```

## Data Pipeline

1. `scripts/apod_parse_filter.py` — streams `nasa_apod_complete.csv`, downloads images that pass resolution filter, outputs filtered metadata CSV
2. `scripts/apod_dataset_split.py` — splits filtered images into train/val/test by image (not patch), outputs split CSVs + config JSON
3. `scripts/patch_creator.py` — extracts 512x512 HR patches with texture filter (std >= 15), optionally produces LR patches via bicubic downscale (default factor 16 → 32x32 LR)

**Patch counts scale with image area:** `n = clip(round(alpha * H * W / patch_size^2), 4, 100)` with alpha=2.

**Dataset stats:** 2,506 images → ~90K train patches (HR 512x512, LR 32x32).

## Models

### SRCNN (`implementations/srcnn.py`)
- Shallow conv (3→64 ch) → 8 residual blocks (Conv-ReLU-Conv + skip) → Conv + PixelShuffle(4x) → final conv (64→3)
- Input: 128x128 LR → Output: 512x512 HR
- Loss: MSE. Optimizer: Adam. Early stopping patience=10.
- Checkpoint: `checkpoints/srcnn_best.pth`

### Conditional DDPM (`implementations/bicubic_diffusion.py`)
- UNet: 2-level encoder-decoder, sinusoidal time embeddings, ResBlocks conditioned on timestep, skip connections
- 6 input channels (3 noisy HR + 3 bicubic LR condition), 3 output channels
- Scheduler: 1000 timesteps, linear beta schedule (1e-4 to 0.02)
- Training: predict noise from LR-conditioned input, MSE loss, gradient clipping (max_norm=1.0)
- Inference: iterative denoising from Gaussian noise conditioned on bicubic-interpolated LR
- Checkpoint: `checkpoints/diffusion_best.pth`

### Consistency Model (`implementations/consistency_model.py`)
- Same ConditionalUNet architecture as diffusion model (2-level encoder-decoder, sinusoidal time embeddings)
- 6 input channels (3 noisy HR + 3 bicubic LR condition), 3 output channels
- Training: consistency training (Song & Dhariwal 2023) — sample adjacent noise levels σ_n, σ_{n+1} from discretized schedule, predict x₀ from both, enforce self-consistency with stop-gradient
- Inference: single-step denoising from Gaussian noise conditioned on bicubic-interpolated LR (no iterative sampling needed)
- Noise schedule: 150 discretized levels, σ ∈ [0.002, 80.0], ρ=7.0
- Checkpoint: `checkpoints/consistency_best.pth` (not yet saved — model not yet trained)

## Shared Components

- **SRDataset** (`implementations/dataloader.py`): reads paired HR/LR PNGs from `{root}/{split}/HR/` and `{root}/{split}/LR/`, returns `(lr, hr)` tensors in [0,1] with shape (C,H,W)
- **SSIM** (`implementations/metrics.py`): canonical Gaussian-windowed SSIM, 11x11 window, sigma=1.5
- **Metrics used:** MSE (training loss), MAE, SSIM (validation)

## Conventions

- Each model file in `implementations/` is self-contained with its own training loop and `main()` entry point
- All models log to W&B project `ORIon-SR`
- Preview images (LR|SR|HR side-by-side) saved to `previews/`
- Best models saved to `checkpoints/{model_name}_best.pth`
- Split CSVs have header: `path,width,height` (no split column in per-split files)
- Patch filenames: `{image_stem}_{index:04d}.png`
- Seeds: patch extraction uses `--seed 42`, split uses seed=0

## Running Training

```bash
# SRCNN
python implementations/srcnn.py

# Diffusion
python implementations/bicubic_diffusion.py

# Consistency Model
python implementations/consistency_model.py
```

Both expect the dataset at `apod_data/dataset/{train,val,test}/{HR,LR}/`.

## Important Notes

- The `src/orion/` package is a stub — not used yet. All real code lives in `implementations/` and `scripts/`.
- Data directories (`apod_data/images/`, `apod_data/dataset/`) are gitignored and large.
- `requirements.txt` is UTF-16 encoded (quirk from Windows origin).
- The SLURM script targets the "griffin" node on the "prime" partition.
