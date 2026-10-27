#!/usr/bin/env python3
"""Upscale a whole image x4 with the trained residual CNN ("SRCNN" rung), using tiled inference.

Place this file in ``implementations/`` (next to ``srcnn.py``) and run from there:

    python upscale.py my_photo.jpg                          # -> my_photo_x4.png
    python upscale.py my_photo.jpg -o out.png --compare     # also writes a bicubic | CNN side-by-side
    python upscale.py my_photo.jpg --verify                 # checks tiling == whole-image forward pass

How the tiling works
--------------------
The CNN is fully convolutional, so each output pixel depends only on input pixels within its
receptive field: 1 (shallow conv) + 16 (8 residual blocks x 2 convs) + 1 (upsample conv) = 18 LR pixels
in each direction (the final 3x3 conv at HR adds under 1 LR pixel).
We therefore cut the image into tiles, give each tile a *halo* of >= 20 LR pixels of real
neighbouring context, run the network, and crop the halo away. Every kept pixel then sees exactly the
context it would see in a whole-image pass, so there are no seams and no blending is needed.
(The generative models cannot do this: each tile would start from different random noise.)
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCALE = 4
RECEPTIVE_RADIUS = 19  # 18 LR px of conv context + 1 for the HR-side final conv, rounded up
MAX_OUTPUT_PIXELS = 100_000_000  # refuse absurd outputs unless --force


def load_model(checkpoint: Path, device: torch.device):
    from srcnn import SRCNN  # the exact class used in training and evaluation

    net = SRCNN(upscale_factor=SCALE).to(device).eval()
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    net.load_state_dict(state)
    return net


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0  # same [0, 1] range as SRDataset
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def to_image(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


@torch.inference_mode()
def upscale_tiled(net, lr: torch.Tensor, tile: int, halo: int, device: torch.device, amp: bool) -> torch.Tensor:
    """lr: (1, 3, H, W) in [0, 1] on CPU. Returns (1, 3, 4H, 4W) on CPU."""
    _, _, H, W = lr.shape
    halo = max(halo, RECEPTIVE_RADIUS + 1)
    # Reflect-pad by the halo so edge tiles also have context (reflect needs pad < dim).
    pad = min(halo, H - 1, W - 1)
    padded = F.pad(lr, (pad, pad, pad, pad), mode="reflect")
    if pad < halo:  # tiny images: fall back to replicate for the remainder
        extra = halo - pad
        padded = F.pad(padded, (extra, extra, extra, extra), mode="replicate")
    out = torch.zeros(1, 3, H * SCALE, W * SCALE)

    amp_dtype = None
    if amp and device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    n_y, n_x = math.ceil(H / tile), math.ceil(W / tile)
    done = 0
    for iy in range(n_y):
        for ix in range(n_x):
            y0, x0 = iy * tile, ix * tile
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            # window in padded coordinates (padded index = original index + halo)
            win = padded[:, :, y0 : y1 + 2 * halo, x0 : x1 + 2 * halo].to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                sr = net(win)
            sr = sr.float()
            core = sr[:, :, halo * SCALE : (halo + (y1 - y0)) * SCALE, halo * SCALE : (halo + (x1 - x0)) * SCALE]
            out[:, :, y0 * SCALE : y1 * SCALE, x0 * SCALE : x1 * SCALE] = core.cpu()
            done += 1
        print(f"  tiles: {done}/{n_y * n_x}", end="\r", flush=True)
    print()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Tiled x4 upscaling with the trained residual CNN")
    p.add_argument("image", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None, help="default: <name>_x4.png next to the input")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/srcnn_best.pth"))
    p.add_argument("--tile", type=int, default=128, help="LR tile size (pixels) kept per pass")
    p.add_argument("--halo", type=int, default=24, help="extra LR context per side; must be > 19")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only); faster, tiny numeric change")
    p.add_argument("--compare", action="store_true", help="also save a bicubic | CNN side-by-side")
    p.add_argument("--verify", action="store_true", help="check tiled output equals a whole-image pass (small images only)")
    p.add_argument("--force", action="store_true", help="allow outputs above 100 megapixels")
    args = p.parse_args()

    if not args.image.exists():
        sys.exit(f"input not found: {args.image}")
    if not args.checkpoint.exists():
        sys.exit(f"checkpoint not found: {args.checkpoint} (pass --checkpoint)")
    if args.halo <= RECEPTIVE_RADIUS:
        sys.exit(f"--halo must exceed the receptive field radius ({RECEPTIVE_RADIUS}), otherwise tiles show seams")

    img = Image.open(args.image)
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        print("note: alpha channel dropped (the model is RGB only)")
    img = img.convert("RGB")
    W, H = img.size
    if H * W * SCALE * SCALE > MAX_OUTPUT_PIXELS and not args.force:
        sys.exit(f"output would be {W * SCALE}x{H * SCALE} px (> 100 MP). Downscale the input or pass --force.")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"input {W}x{H} -> output {W * SCALE}x{H * SCALE} on {device}")
    net = load_model(args.checkpoint, device)
    lr = to_tensor(img)

    sr = upscale_tiled(net, lr, args.tile, args.halo, device, args.amp)

    out_path = args.output or args.image.with_name(args.image.stem + "_x4.png")
    to_image(sr).save(out_path)
    print(f"saved {out_path}")

    if args.compare:
        bic = F.interpolate(lr, scale_factor=SCALE, mode="bicubic", align_corners=False).clamp(0, 1)
        left, right = to_image(bic), to_image(sr)
        both = Image.new("RGB", (left.width * 2, left.height))
        both.paste(left, (0, 0))
        both.paste(right, (left.width, 0))
        cmp_path = out_path.with_name(out_path.stem + "_compare.png")
        both.save(cmp_path)
        print(f"saved {cmp_path}  (left: bicubic, right: CNN)")

    if args.verify:
        if H * W > 600 * 600:
            print("verify skipped: image too large for a whole-image pass; use a crop under 600x600")
        else:
            with torch.inference_mode():
                full = net(lr.to(device)).float().cpu()
            # Edge pixels differ (zero padding vs reflect padding); compare the interior only.
            m = RECEPTIVE_RADIUS * SCALE
            if full.shape[-1] > 2 * m and full.shape[-2] > 2 * m:
                diff = (full - sr)[:, :, m:-m, m:-m].abs().max().item()
                print(f"verify: max |tiled - whole| in the interior = {diff:.2e}  ({'OK' if diff < 1e-3 else 'MISMATCH'})")
            else:
                print("verify skipped: image too small to have an interior")


if __name__ == "__main__":
    main()
