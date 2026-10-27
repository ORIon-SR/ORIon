"""Paired statistics quoted in the report (SRCNN vs Swin2SR, diffusion vs bicubic).

Usage (from the repo root, after running models_eval.py):
    python scripts/paired_stats.py [path/to/per_patch.csv]

Bootstrap: 2000 resamples of the test IMAGES (not patches), seed 0.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models_eval/test/per_patch.csv")
d = pd.read_csv(csv)
tabs = {m: d.pivot_table(index="patch", columns="model", values=m) for m in ("psnr", "ssim", "mae")}
src = d.drop_duplicates("patch").set_index("patch")["source"]


def boot(x, seed=0):
    r = np.random.default_rng(seed)
    return np.percentile([x[r.integers(0, len(x), len(x))].mean() for _ in range(2000)], [2.5, 97.5])


print(f"{len(tabs['psnr'])} patches from {src.nunique()} images")
for m, t in tabs.items():
    x = (t["srcnn"] - t["swin2sr"]).groupby(src).mean().values
    print(f"{m}: SRCNN - Swin2SR (mean over images) = {x.mean():.5f}, 95% CI {boot(x).round(5)}")
print("SRCNN higher PSNR on %.1f%% of patches" % (((tabs["psnr"].srcnn - tabs["psnr"].swin2sr) > 0).mean() * 100))
print("SRCNN higher SSIM on %.1f%% of patches" % (((tabs["ssim"].srcnn - tabs["ssim"].swin2sr) > 0).mean() * 100))
print("SRCNN lower MAE on %.1f%% of patches" % (((tabs["mae"].srcnn - tabs["mae"].swin2sr) < 0).mean() * 100))
print("Diffusion below bicubic: PSNR %.1f%%, SSIM %.1f%%" % tuple(((tabs[m].diffusion - tabs[m].bicubic) < 0).mean() * 100 for m in ("psnr", "ssim")))
