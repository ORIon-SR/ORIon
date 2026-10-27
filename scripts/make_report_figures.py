"""Regenerates the metric figures of the report from summary.json.

Writes Figures/metrics.png (PSNR, SSIM and MAE side by side, used in the report) and the three
single-metric versions Figures/PSNR.png, SSIM.png, MAE.png.

Usage (from the repo root; needs matplotlib, e.g. `uv pip install matplotlib`):
    python scripts/make_report_figures.py [summary.json] [output_dir]
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

summary = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models_eval/test/summary.json")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("Figures")
out.mkdir(parents=True, exist_ok=True)
R = json.load(open(summary))["results"]
order = [("bicubic", "Bicubic"), ("srcnn", "SRCNN"), ("diffusion", "Diffusion"),
         ("consistency", "Consistency (1-step)"), ("consistency_2step", "Consistency (2-step)"),
         ("consistency_4step", "Consistency (4-step)"), ("swin2sr", "Swin2SR")]
meta = {"psnr": ("PSNR (dB), higher is better", "PSNR.png"),
        "ssim": ("SSIM, higher is better", "SSIM.png"),
        "mae": ("MAE, lower is better", "MAE.png")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
ys = list(range(len(order)))[::-1]


def draw(ax, m, labels=True):
    for y, (k, _) in zip(ys, order):
        v = R[k][m]
        lo, hi = v["image_ci95"]
        ax.plot([lo, hi], [y - 0.14] * 2, color="#1f5fae", lw=1.4, solid_capstyle="butt")
        ax.plot([v["image_mean"]], [y - 0.14], "s", color="#1f5fae", ms=4.2,
                label="Image average (95% CI)" if y == ys[0] else None)
        ax.plot([v["patch_mean"]], [y + 0.14], "o", color="#7b1a63", ms=4.2,
                label="Patch average" if y == ys[0] else None)
    ax.set_yticks(ys)
    ax.set_yticklabels([n for _, n in order])
    if not labels:
        ax.tick_params(labelleft=False)
    ax.set_xlabel(meta[m][0])
    ax.grid(axis="x", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# combined figure (one row, shared model labels)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), dpi=300, sharey=True)
for i, (ax, m) in enumerate(zip(axes, meta)):
    draw(ax, m, labels=(i == 0))
    ax.set_title("(" + "abc"[i] + ")", fontsize=8, loc="left")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=7.5, handletextpad=0.3, columnspacing=1.5)
fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.92))
fig.savefig(out / "metrics.png")
plt.close(fig)

# single-metric versions
for m, (_, fn) in meta.items():
    fig, ax = plt.subplots(figsize=(3.5, 2.7), dpi=300)
    draw(ax, m)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=7, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.93))
    fig.savefig(out / fn)
    plt.close(fig)
print("wrote", out / "metrics.png", "and PSNR.png, SSIM.png, MAE.png")
