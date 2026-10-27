#!/usr/bin/env python3
"""Rung 2 -- zero-shot off-the-shelf Swin2SR (x4) evaluated on the APOD test split.

Reads the HR/LR patch pairs that ``scripts/patch_creator.py`` already wrote (via the
shared ``SRDataset``), runs a pretrained Swin2SR with NO fine-tuning, and scores it
with the same conventions the other rungs use in their validation loops:

  * images are float tensors in [0, 1], outputs clamped to [0, 1]
  * SSIM  = ``metrics.compute_ssim`` (11x11 Gaussian window, RGB, no border shave)
  * MAE   = mean absolute error
  * PSNR  = 10*log10(1 / MSE) per patch (the other scripts do not report PSNR yet)
  * the headline number is the mean over patches, like ``validate()`` elsewhere

Bicubic upsampling (rung 1) runs through the same code path by default, so both rungs
are scored on identical patches in one run and a paired difference is reported.

Run from ``implementations/`` (it imports ``dataloader`` and ``metrics`` from there):

    python swin2sr_eval.py --split test --limit 16        # smoke test
    python swin2sr_eval.py --split test --wandb           # the real run

Extra dependency: ``transformers`` (``uv add transformers`` / ``pip install transformers``).
Offline / cluster nodes: download once, then pass the folder to ``--model-id``:

    huggingface-cli download caidas/swin2SR-classical-sr-x4-64 --local-dir models/swin2sr-classical-x4
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
DEFAULT_MODEL_ID = "caidas/swin2SR-classical-sr-x4-64"
METRICS = ("psnr", "ssim", "mae")
CONVENTIONS = (
    "float [0,1], output clamped, no border shave, no uint8 rounding; "
    "SSIM = metrics.compute_ssim; PSNR = 10*log10(1/MSE) per patch; mean over patches"
)


# --------------------------------------------------------------------------- models
# Every model is a callable: (B,3,h,w) float in [0,1]  ->  (B,3,h*SCALE,w*SCALE) float in [0,1]
class BicubicUpsample:
    """Rung 1. Same call the diffusion scripts use to build their conditioning image."""

    name = "bicubic"

    def __call__(self, lr: torch.Tensor) -> torch.Tensor:
        return F.interpolate(lr, scale_factor=SCALE, mode="bicubic", align_corners=False).clamp(0, 1)


class Swin2SRZeroShot:
    """Rung 2. Pretrained Swin2SR x4, used as-is."""

    name = "swin2sr"

    def __init__(self, model_id: str, revision: str | None, device: torch.device):
        from transformers import Swin2SRForImageSuperResolution

        self.net = Swin2SRForImageSuperResolution.from_pretrained(model_id, revision=revision).to(device).eval()
        if self.net.config.upscale != SCALE:
            raise SystemExit(f"{model_id} upscales x{self.net.config.upscale}, but the dataset is x{SCALE}")

    @torch.inference_mode()
    def __call__(self, lr: torch.Tensor) -> torch.Tensor:
        # Swin2SR takes [0,1] pixels directly (the HF image processor only rescales by 1/255)
        return self.net(pixel_values=lr).reconstruction.float().clamp(0, 1)


# --------------------------------------------------------------------------- metrics
def compute_psnr(sr: torch.Tensor, hr: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(sr, hr).item()
    return 10.0 * math.log10(data_range**2 / max(mse, 1e-10))  # identical images cap at 100 dB


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile-bootstrap CI of the mean."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return float(v.mean()), float(v.mean())
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def source_of(patch_name: str) -> str:
    """'2012-11-30_0003.png' -> '2012-11-30' (patch_creator names patches <stem>_<idx>.png)."""
    return patch_name.rsplit(".", 1)[0].rsplit("_", 1)[0]


def per_image_means(rows: list[dict], key: str) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for r in rows:
        groups.setdefault(r["source"], []).append(r[key])
    return {s: float(np.mean(v)) for s, v in groups.items()}


def summarise(rows: list[dict], model_names: list[str]) -> dict:
    """Patch-level mean/std (matches the other rungs) + image-level bootstrap CI.

    Patches cut from one APOD image are correlated, so the CI resamples *images*, not patches.
    """
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
    if "swin2sr" in model_names and "bicubic" in model_names:
        by = {m: {r["patch"]: r for r in rows if r["model"] == m} for m in ("swin2sr", "bicubic")}
        common = sorted(set(by["swin2sr"]) & set(by["bicubic"]))
        d_rows = [
            dict(source=source_of(p), **{k: by["swin2sr"][p][k] - by["bicubic"][p][k] for k in METRICS})
            for p in common
        ]
        out["swin2sr_minus_bicubic"] = {"n_patches": len(d_rows)}
        for k in METRICS:
            d = np.array([r[k] for r in d_rows])
            img = per_image_means(d_rows, k)
            lo, hi = bootstrap_ci(list(img.values()))
            out["swin2sr_minus_bicubic"][k] = dict(
                patch_mean=float(d.mean()),
                image_ci95=[lo, hi],
                frac_patches_swin2sr_better=float((d < 0).mean() if k == "mae" else (d > 0).mean()),
            )
    return out


# --------------------------------------------------------------------------- previews
def to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.clamp(0, 1).mul(255).round().byte().cpu().numpy()
    return Image.fromarray(np.transpose(arr, (1, 2, 0)))


def save_preview(lr, sr, hr, path: Path) -> Image.Image:
    """LR (bicubic-enlarged) | SR | HR, same layout as the other scripts' previews."""
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
    p = argparse.ArgumentParser(description="Zero-shot Swin2SR (+ bicubic) evaluation on the APOD split")
    p.add_argument("--data-dir", type=Path, default=project_root / "apod_data" / "dataset")
    p.add_argument("--split", choices=["train", "val", "test"], default="test",
                   help="Final comparison numbers must come from `test`; use `val` for any model-selection decisions.")
    p.add_argument("--models", nargs="+", choices=["swin2sr", "bicubic"], default=["swin2sr", "bicubic"])
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hub id or local folder")
    p.add_argument("--revision", default=None, help="Pin a Hub commit/tag for reproducibility")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="Only the first N patches (smoke test)")
    p.add_argument("--out-dir", type=Path, default=Path("results/swin2sr_zeroshot"))
    p.add_argument("--force", action="store_true", help="Overwrite existing results for this split")
    p.add_argument("--num-previews", type=int, default=5)
    p.add_argument("--wandb", action="store_true", help="Log to W&B (project ORIon-SR, like the other rungs)")
    p.add_argument("--wandb-project", default="ORIon-SR")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-name", default="Swin2SR zero-shot")
    return p.parse_args()


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
        models.append(BicubicUpsample() if name == "bicubic" else Swin2SRZeroShot(args.model_id, args.revision, device))
    names = [m.name for m in models]
    print(f"{dataset} | device={device} | models={names} | {len(filenames)} patches")

    wb = None
    config = dict(
        architecture="swin2sr-classical-x4 (zero-shot)", model_id=args.model_id, revision=args.revision,
        scale_factor=SCALE, dataset="nasa-apod", split=args.split, n_patches=len(filenames),
        batch_size=args.batch_size, metric_conventions=CONVENTIONS, models=names,
        torch=torch.__version__,
    )
    if args.wandb:
        import wandb

        wb = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name, config=config)

    rows: list[dict] = []
    seen, previews_done, t0 = 0, 0, time.time()
    with torch.inference_mode():
        for lr, hr in loader:
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
            if lr.shape[-1] * SCALE != hr.shape[-1] or lr.shape[-2] * SCALE != hr.shape[-2]:
                raise SystemExit(f"LR {tuple(lr.shape[-2:])} x{SCALE} != HR {tuple(hr.shape[-2:])}; "
                                 "was the dataset built with --downscale-factor 4?")
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
                        img = save_preview(lr[i], sr[i], hr[i], out_dir / "previews" / f"{m.name}_{fn}")
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

    print(f"\n{'model':<9} {'PSNR (dB)':>10} {'SSIM':>8} {'MAE':>8}   mean over {len(filenames)} patches "
          f"[image-level 95% CI]")
    for m in names:
        s = summary[m]
        print(f"{m:<9} {s['psnr']['patch_mean']:10.3f} {s['ssim']['patch_mean']:8.4f} {s['mae']['patch_mean']:8.5f}"
              f"   PSNR {s['psnr']['image_ci95'][0]:.2f}-{s['psnr']['image_ci95'][1]:.2f}"
              f" | SSIM {s['ssim']['image_ci95'][0]:.4f}-{s['ssim']['image_ci95'][1]:.4f}")
    if "swin2sr_minus_bicubic" in summary:
        d = summary["swin2sr_minus_bicubic"]
        print(f"\nSwin2SR - bicubic: PSNR {d['psnr']['patch_mean']:+.3f} dB "
              f"(CI {d['psnr']['image_ci95'][0]:+.2f}..{d['psnr']['image_ci95'][1]:+.2f}), "
              f"SSIM {d['ssim']['patch_mean']:+.4f}; better on "
              f"{100 * d['psnr']['frac_patches_swin2sr_better']:.0f}% of patches (PSNR)")
    print(f"\nResults written to {out_dir}")

    if wb is not None:
        for m in names:
            for k in METRICS:
                wb.summary[f"{args.split}/{m}/{k}"] = summary[m][k]["patch_mean"]
                wb.summary[f"{args.split}/{m}/{k}_image_ci95"] = summary[m][k]["image_ci95"]
        wb.finish()


if __name__ == "__main__":
    sys.exit(main())
