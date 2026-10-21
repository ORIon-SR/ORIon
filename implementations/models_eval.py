#!/usr/bin/env python3
"""Rungs 1-5 evaluation -- trained models + bicubic baseline + Swin2SR zero-shot on the APOD test split.

Evaluates the same way as ``swin2sr_eval.py`` so all rungs are directly comparable:

  * images are float tensors in [0, 1], outputs clamped to [0, 1]
  * SSIM  = ``metrics.compute_ssim`` (11x11 Gaussian window, RGB, no border shave)
  * MAE   = mean absolute error
  * PSNR  = 10*log10(1 / MSE) per patch
  * the headline number is the mean over patches, with image-level bootstrap CI

Models evaluated:

  * bicubic          (rung 1 baseline)
  * srcnn            (rung 2, trained checkpoint)
  * diffusion        (rung 3, trained checkpoint, iterative sampling)
  * consistency      (rung 4, trained checkpoint, single-step sampling)
  * swin2sr          (rung 5, pretrained zero-shot)

Run from ``implementations/``:

    python models_eval.py --split test --limit 16              # smoke test
    python models_eval.py --split test --models srcnn diffusion consistency swin2sr
    python models_eval.py --split test --wandb                 # the real run

Checkpoints are loaded from ``checkpoints/`` by default.
Swin2SR downloads from HuggingFace on first use (or pass --model-id for local folder).
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from dataloader import SRDataset
from metrics import compute_ssim

SCALE = 4
METRICS = ("psnr", "ssim", "mae")
CONVENTIONS = (
    "float [0,1], output clamped, no border shave, no uint8 rounding; "
    "SSIM = metrics.compute_ssim; PSNR = 10*log10(1/MSE) per patch; mean over patches"
)


# --------------------------------------------------------------------------- models
class BicubicUpsample:
    """Rung 1. Same call the diffusion scripts use to build their conditioning image."""

    name = "bicubic"

    def __call__(self, lr: torch.Tensor) -> torch.Tensor:
        return F.interpolate(lr, scale_factor=SCALE, mode="bicubic", align_corners=False).clamp(0, 1)


class SRCNNModel:
    """Rung 2. Trained SRCNN with PixelShuffle upscale."""

    name = "srcnn"

    def __init__(self, checkpoint_path: Path, device: torch.device, upscale_factor: int = 4):
        from srcnn import SRCNN

        self.net = SRCNN(upscale_factor=upscale_factor).to(device).eval()
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.net.load_state_dict(state)
        self.device = device

    @torch.inference_mode()
    def __call__(self, lr: torch.Tensor) -> torch.Tensor:
        out = self.net(lr)
        return out.clamp(0, 1)


class DiffusionModel:
    """Rung 3. Trained conditional DDPM with iterative sampling."""

    name = "diffusion"

    def __init__(self, checkpoint_path: Path, device: torch.device, num_timesteps: int = 1000):
        from bicubic_diffusion import ConditionalUNet, DDPMScheduler

        self.net = ConditionalUNet().to(device).eval()
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.net.load_state_dict(state)
        self.scheduler = DDPMScheduler(num_timesteps=num_timesteps, device=str(device))
        self.device = device
        self.num_timesteps = num_timesteps

    @torch.inference_mode()
    def __call__(self, lr: torch.Tensor) -> torch.Tensor:
        batch_size = lr.size(0)
        hr_size = (lr.shape[-2] * SCALE, lr.shape[-1] * SCALE)

        lr_scaled = lr * 2.0 - 1.0
        lr_conditioned = F.interpolate(
            lr_scaled, size=hr_size, mode="bicubic", align_corners=False
        )

        x_t = torch.randn((batch_size, 3, hr_size[0], hr_size[1]), device=self.device)

        for t_idx in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)
            model_input = torch.cat([x_t, lr_conditioned], dim=1)
            pred_noise = self.net(model_input, t).float()

            beta = self.scheduler.betas[t_idx]
            alpha = self.scheduler.alphas[t_idx]
            sqrt_one_minus_alpha_cumprod = self.scheduler.sqrt_one_minus_alphas_cumprod[t_idx]

            mean = (1 / torch.sqrt(alpha)) * (
                x_t - (beta / sqrt_one_minus_alpha_cumprod) * pred_noise
            )

            if t_idx > 0:
                z = torch.randn_like(x_t)
                sigma = torch.sqrt(self.scheduler.posterior_variance[t_idx])
                x_t = mean + sigma * z
            else:
                x_t = mean

        out = (x_t + 1.0) / 2.0
        return out.clamp(0, 1)


class ConsistencyModelWrapper:
    """Rung 4. Trained consistency model with single-step sampling."""

    name = "consistency"

    def __init__(self, checkpoint_path: Path, device: torch.device, sigma_max: float = 80.0):
        from consistency_model import ConsistencyModel

        self.model = ConsistencyModel(sigma_max=sigma_max).to(device).eval()
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state)
        self.device = device
        self.sigma_max = sigma_max

    @torch.inference_mode()
    def __call__(self, lr: torch.Tensor) -> torch.Tensor:
        batch_size = lr.size(0)
        hr_size = (lr.shape[-2] * SCALE, lr.shape[-1] * SCALE)

        lr_scaled = lr * 2.0 - 1.0
        lr_conditioned = F.interpolate(
            lr_scaled, size=hr_size, mode="bicubic", align_corners=False
        )

        x_t = self.sigma_max * torch.randn((batch_size, 3, hr_size[0], hr_size[1]), device=self.device)
        sigma = torch.full((batch_size,), self.sigma_max, device=self.device)

        model_input = torch.cat([x_t, lr_conditioned], dim=1)
        pred_x0 = self.model(model_input, sigma).float()

        out = (pred_x0 + 1.0) / 2.0
        return out.clamp(0, 1)


class Swin2SRZeroShot:
    """Rung 5. Pretrained Swin2SR x4, used as-is."""

    name = "swin2sr"

    def __init__(self, model_id: str, revision: str | None, device: torch.device):
        from transformers import Swin2SRForImageSuperResolution

        self.net = Swin2SRForImageSuperResolution.from_pretrained(model_id, revision=revision).to(device).eval()
        if self.net.config.upscale != SCALE:
            raise SystemExit(f"{model_id} upscales x{self.net.config.upscale}, but the dataset is x{SCALE}")

    @torch.inference_mode()
    def __call__(self, lr: torch.Tensor) -> torch.Tensor:
        return self.net(pixel_values=lr).reconstruction.float().clamp(0, 1)


# --------------------------------------------------------------------------- metrics
def compute_psnr(sr: torch.Tensor, hr: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(sr, hr).item()
    return 10.0 * math.log10(data_range**2 / max(mse, 1e-10))


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile-bootstrap CI of the mean."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return float(v.mean()), float(v.mean())
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def source_of(patch_name: str) -> str:
    """'2012-11-30_0003.png' -> '2012-11-30'."""
    return patch_name.rsplit(".", 1)[0].rsplit("_", 1)[0]


def per_image_means(rows: list[dict], key: str) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for r in rows:
        groups.setdefault(r["source"], []).append(r[key])
    return {s: float(np.mean(v)) for s, v in groups.items()}


def summarise(rows: list[dict], model_names: list[str]) -> dict:
    """Patch-level mean/std + image-level bootstrap CI."""
    out: dict = {}
    for m in model_names:
        sel = [r for r in rows if r["model"] == m]
        out[m] = {"n_patches": len(sel), "n_images": len({r["source"] for r in sel})}
        for k in METRICS:
            v = np.array([r[k] for r in sel])
            img = per_image_means(sel, k)
            lo, hi = bootstrap_ci(list(img.values()))
            out[m][k] = dict(
                patch_mean=float(v.mean()),
                patch_std=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                image_mean=float(np.mean(list(img.values()))),
                image_ci95=[lo, hi],
            )
    return out


# --------------------------------------------------------------------------- previews
def to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.clamp(0, 1).mul(255).round().byte().cpu().numpy()
    return Image.fromarray(np.transpose(arr, (1, 2, 0)))


def save_preview(lr, sr, hr, path: Path) -> Image.Image:
    """LR (bicubic-enlarged) | SR | HR."""
    lr_up = to_pil(lr).resize(to_pil(hr).size, Image.Resampling.BICUBIC)
    sr_p, hr_p = to_pil(sr), to_pil(hr)
    w, h = hr_p.size
    combined = Image.new("RGB", (w * 3, h))
    for i, im in enumerate((lr_up, sr_p, hr_p)):
        combined.paste(im, (w * i, 0))
    combined.save(path)
    return combined


# --------------------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description="Trained model evaluation on the APOD split")
    p.add_argument("--data-dir", type=Path, default=project_root / "apod_data" / "dataset")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--models", nargs="+",
                   choices=["bicubic", "srcnn", "diffusion", "consistency", "swin2sr"],
                   default=["bicubic", "srcnn", "diffusion", "consistency", "swin2sr"])
    p.add_argument("--checkpoint-dir", type=Path, default=Path(__file__).resolve().parent / "checkpoints")
    p.add_argument("--srcnn-checkpoint", type=Path, default=None)
    p.add_argument("--diffusion-checkpoint", type=Path, default=None)
    p.add_argument("--consistency-checkpoint", type=Path, default=None)
    p.add_argument("--srcnn-upscale-factor", type=int, default=4)
    p.add_argument("--diffusion-timesteps", type=int, default=1000)
    p.add_argument("--consistency-sigma-max", type=float, default=80.0)
    p.add_argument("--model-id", default="caidas/swin2SR-classical-sr-x4-64",
                   help="Swin2SR Hub id or local folder")
    p.add_argument("--revision", default=None, help="Pin a Hub commit/tag for Swin2SR reproducibility")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="Only the first N patches (smoke test)")
    p.add_argument("--out-dir", type=Path, default=Path("results/models_eval"))
    p.add_argument("--force", action="store_true", help="Overwrite existing results for this split")
    p.add_argument("--num-previews", type=int, default=5)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="ORIon-SR")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-name", default="Models evaluation")
    return p.parse_args()


def load_model(name: str, args, device: torch.device):
    ckpt_dir = args.checkpoint_dir
    if name == "bicubic":
        return BicubicUpsample()
    elif name == "srcnn":
        path = args.srcnn_checkpoint or (ckpt_dir / "srcnn_best.pth")
        return SRCNNModel(path, device, args.srcnn_upscale_factor)
    elif name == "diffusion":
        path = args.diffusion_checkpoint or (ckpt_dir / "diffusion_best.pth")
        return DiffusionModel(path, device, args.diffusion_timesteps)
    elif name == "consistency":
        path = args.consistency_checkpoint or (ckpt_dir / "consistency_best.pth")
        return ConsistencyModelWrapper(path, device, args.consistency_sigma_max)
    elif name == "swin2sr":
        return Swin2SRZeroShot(args.model_id, args.revision, device)
    else:
        raise ValueError(f"Unknown model: {name}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = args.out_dir / args.split
    csv_path = out_dir / "per_patch.csv"
    if csv_path.exists() and not args.force:
        raise SystemExit(f"{csv_path} exists. Use --force to overwrite.")
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)

    dataset = SRDataset(args.data_dir, split=args.split)
    if args.limit:
        dataset.filenames = dataset.filenames[: args.limit]
    filenames = dataset.filenames
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    models = []
    for name in args.models:
        models.append(load_model(name, args, device))
    names = [m.name for m in models]
    print(f"{dataset} | device={device} | models={names} | {len(filenames)} patches")

    wb = None
    config = dict(
        models=names,
        scale_factor=SCALE,
        dataset="nasa-apod",
        split=args.split,
        n_patches=len(filenames),
        batch_size=args.batch_size,
        metric_conventions=CONVENTIONS,
        srcnn_upscale_factor=args.srcnn_upscale_factor,
        diffusion_timesteps=args.diffusion_timesteps,
        consistency_sigma_max=args.consistency_sigma_max,
        swin2sr_model_id=args.model_id,
        swin2sr_revision=args.revision,
        torch=torch.__version__,
    )
    if args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                        name=args.wandb_name, config=config)

    rows: list[dict] = []
    seen, previews_done, t0 = 0, 0, time.time()
    with torch.inference_mode():
        for lr, hr in loader:
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
            batch_names = filenames[seen: seen + lr.size(0)]
            for m in models:
                sr = m(lr)
                if sr.shape != hr.shape:
                    raise SystemExit(f"{m.name}: output {tuple(sr.shape)} != HR {tuple(hr.shape)}")
                for i, fn in enumerate(batch_names):
                    rows.append(dict(
                        patch=fn, source=source_of(fn), model=m.name,
                        psnr=compute_psnr(sr[i], hr[i]),
                        ssim=compute_ssim(sr[i], hr[i]),
                        mae=F.l1_loss(sr[i], hr[i]).item(),
                    ))
                if m.name != "bicubic" or names == ["bicubic"]:
                    for i, fn in enumerate(batch_names):
                        if previews_done + i >= args.num_previews:
                            break
                        img = save_preview(lr[i], sr[i], hr[i],
                                           out_dir / "previews" / f"{m.name}_{fn}")
                        if wb is not None:
                            import wandb
                            wb.log({f"preview/{m.name}/{previews_done + i}": wandb.Image(
                                img, caption=f"LR | {m.name} | HR - {fn}")})
            previews_done += min(len(batch_names), max(0, args.num_previews - previews_done))
            seen += lr.size(0)
            print(f"  {seen}/{len(filenames)} patches ({time.time() - t0:.0f}s)")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patch", "source", "model", *METRICS])
        w.writeheader()
        w.writerows(rows)

    summary = summarise(rows, names)
    (out_dir / "summary.json").write_text(json.dumps(dict(config=config, results=summary), indent=2))

    print(f"\n{'model':<12} {'PSNR (dB)':>10} {'SSIM':>8} {'MAE':>8}   mean over {len(filenames)} patches "
          f"[image-level 95% CI]")
    for m in names:
        s = summary[m]
        print(f"{m:<12} {s['psnr']['patch_mean']:10.3f} {s['ssim']['patch_mean']:8.4f} {s['mae']['patch_mean']:8.5f}"
              f"   PSNR {s['psnr']['image_ci95'][0]:.2f}-{s['psnr']['image_ci95'][1]:.2f}"
              f" | SSIM {s['ssim']['image_ci95'][0]:.4f}-{s['ssim']['image_ci95'][1]:.4f}")
    print(f"\nResults written to {out_dir}")

    if wb is not None:
        for m in names:
            for k in METRICS:
                wb.summary[f"{args.split}/{m}/{k}"] = summary[m][k]["patch_mean"]
                wb.summary[f"{args.split}/{m}/{k}_image_ci95"] = summary[m][k]["image_ci95"]
        wb.finish()


if __name__ == "__main__":
    sys.exit(main())
