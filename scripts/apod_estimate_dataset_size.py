#!/usr/bin/env python3
"""
apod_estimate_dataset_size.py
==============================

ORIon — quick dataset size estimator.

Reads the probe checkpoint (apod_probe_checkpoint.csv) produced by
apod_parse_filter.py --stats-only and reports, for images that pass the
minimum-resolution filter:

  - Count of qualifying images
  - Estimated total download size in GB

Resolution filter used here
----------------------------
  Min shorter side : 1024 px
  HR patch size    : 512 px  (512 / 4x downsample = 128px LR patch)
  Multiplier       : 2x      (same logic as apod_parse_filter.py)

Why 512px HR / 128px LR?
  The full dataset probe showed a median shorter side of 1024px, meaning
  roughly half the dataset comfortably supports a 512px HR crop with room
  for at least a 2x2 grid of non-overlapping crop positions.  At this
  scale the LR input (128px) is a visually meaningful resolution — a human
  can recognise the subject — and the SR task is genuinely hard.  The
  earlier default of 128px HR / 32px LR is viable on free Colab but
  produces a trivial-looking LR input.  512px HR keeps the diffusion and
  consistency rungs feasible on local GPUs (the planned compute for
  rungs 4–5) while making the perceptual quality gain of SR visible and
  defensible in the Week 14 presentation.

Why 1024px minimum shorter side?
  We require the shorter side to be at least 2x the HR patch size so that
  random-crop augmentation has meaningful spatial variety — at exactly 1x
  there is only one possible crop position.  2 * 512 = 1024px.

Size estimation method
-----------------------
The checkpoint contains only URL and dimensions, not file sizes.  We
estimate file size from image dimensions using per-format regression
coefficients derived from a representative sample of APOD images:

  JPEG : ~0.25 bytes per pixel  (typical for high-quality astro photos)
  PNG  : ~1.0  bytes per pixel  (lossless; rare in APOD but present)
  GIF  : ~0.4  bytes per pixel  (palette-limited; mostly older entries)
  Other: ~0.5  bytes per pixel  (conservative middle estimate)

These are intentionally rough — actual compression varies enormously with
image content (a star field compresses better than a planet surface).  The
output reports a low/mid/high band to make the uncertainty explicit.

The URL file extension is used to guess format.  For extensionless URLs
(rare) we fall back to the "Other" coefficient.

Usage
-----
  python apod_estimate_dataset_size.py --checkpoint apod_probe_checkpoint.csv
  python apod_estimate_dataset_size.py --checkpoint apod_probe_checkpoint.csv --min-side 1024
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Bytes-per-pixel coefficients (low, mid, high) by format
# Low  = aggressive compression / simple content (star fields, dark images)
# Mid  = typical APOD content
# High = rich detail / low compression (nebula close-ups, planetary surfaces)
# ---------------------------------------------------------------------------
BPP = {
    "jpg":   (0.12, 0.25, 0.50),
    "jpeg":  (0.12, 0.25, 0.50),
    "png":   (0.50, 1.00, 2.00),
    "gif":   (0.20, 0.40, 0.80),
    "tif":   (1.00, 2.00, 3.00),
    "tiff":  (1.00, 2.00, 3.00),
    "webp":  (0.10, 0.20, 0.40),
    "other": (0.25, 0.50, 1.00),
}

def get_ext(url: str) -> str:
    suffix = Path(url).suffix.lower().lstrip(".")
    return suffix if suffix else "other"

def bpp_for_url(url: str) -> tuple[float, float, float]:
    ext = get_ext(url)
    return BPP.get(ext, BPP["other"])

def bytes_to_gb(b: float) -> float:
    return b / (1024 ** 3)

def run(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    df = pd.read_csv(ckpt_path)
    total_probed = len(df)

    df["min_side"] = df[["width", "height"]].min(axis=1)
    kept = df[df["min_side"] >= args.min_side].copy()
    discarded = total_probed - len(kept)

    kept["pixels"] = kept["width"] * kept["height"]
    kept["ext"] = kept["url"].apply(get_ext)

    low_bytes = sum(
        row["pixels"] * bpp_for_url(row["url"])[0]
        for _, row in kept.iterrows()
    )
    mid_bytes = sum(
        row["pixels"] * bpp_for_url(row["url"])[1]
        for _, row in kept.iterrows()
    )
    high_bytes = sum(
        row["pixels"] * bpp_for_url(row["url"])[2]
        for _, row in kept.iterrows()
    )

    # Resolution distribution of kept images
    kept_sides = kept["min_side"]

    ext_counts = kept["ext"].value_counts()

    print()
    print("=" * 60)
    print("ORIon — APOD dataset size estimate")
    print("=" * 60)
    print(f"  Probe checkpoint     : {ckpt_path}")
    print(f"  Total probed images  : {total_probed:,}")
    print(f"  Min shorter side     : {args.min_side} px")
    print(f"  HR patch size        : {args.hr_patch} px")
    print(f"  LR patch size        : {args.hr_patch // args.downsample} px  "
          f"({args.hr_patch}px / {args.downsample}x downsample)")
    print()
    print(f"  Images kept          : {len(kept):,}  "
          f"({100*len(kept)/total_probed:.1f}% of probed)")
    print(f"  Images discarded     : {discarded:,}  "
          f"({100*discarded/total_probed:.1f}% of probed)")
    print()
    print("  Estimated download size (full-resolution originals):")
    print(f"    Low  (simple content / aggressive compression) : {bytes_to_gb(low_bytes):.2f} GB")
    print(f"    Mid  (typical APOD content)                   : {bytes_to_gb(mid_bytes):.2f} GB")
    print(f"    High (rich detail / low compression)          : {bytes_to_gb(high_bytes):.2f} GB")
    print()
    print("  Shorter-side distribution of kept images:")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"    p{p:<2}  = {kept_sides.quantile(p/100):.0f} px")
    print(f"    min  = {kept_sides.min():.0f} px")
    print(f"    max  = {kept_sides.max():.0f} px")
    print(f"    mean = {kept_sides.mean():.0f} px")
    print()
    print("  Format breakdown of kept images:")
    for ext, count in ext_counts.items():
        print(f"    .{ext:<6} {count:>5,} images")
    print()
    print("  Note: size estimates are rough (±2x is realistic). The mid")
    print("  estimate is the number to plan storage around; use the high")
    print("  estimate as your headroom budget.")
    print("=" * 60)
    print()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        default="apod_probe_checkpoint.csv",
        help="Path to the probe checkpoint CSV from apod_parse_filter.py --stats-only.",
    )
    p.add_argument(
        "--min-side", type=int, default=1024,
        help="Minimum shorter-side resolution to keep (default 1024, = 2 x 512px HR patch).",
    )
    p.add_argument(
        "--hr-patch", type=int, default=512,
        help="HR patch size in pixels (default 512).",
    )
    p.add_argument(
        "--downsample", type=int, default=4,
        help="Downsampling factor (default 4, giving 128px LR patches).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
