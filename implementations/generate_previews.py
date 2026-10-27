#!/usr/bin/env python3
"""Generate preview images for each model (LR | SR | HR side-by-side).

Optimized for RTX 5060 (Blackwell): torch.compile + bf16 autocast.

Usage:
    python generate_previews.py --split test --num-previews 10
    python generate_previews.py --split val --models srcnn consistency
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataloader import SRDataset
from models_eval import (
    BicubicUpsample,
    ConsistencyFewStep,
    ConsistencyModelWrapper,
    DiffusionModel,
    SRCNNModel,
    Swin2SRZeroShot,
)

SCALE = 4


def to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.clamp(0, 1).mul(255).round().byte().cpu().numpy()
    return Image.fromarray(arr.transpose(1, 2, 0))


def save_preview(lr, sr, hr, path: Path) -> None:
    """LR (bicubic-enlarged) | SR | HR."""
    lr_up = to_pil(lr).resize(to_pil(hr).size, Image.Resampling.BICUBIC)
    sr_p, hr_p = to_pil(sr), to_pil(hr)
    w, h = hr_p.size
    combined = Image.new("RGB", (w * 3, h))
    for i, im in enumerate((lr_up, sr_p, hr_p)):
        combined.paste(im, (w * i, 0))
    combined.save(path)
    print(f"  Saved: {path}")


def load_model(name: str, args, device: torch.device):
    ckpt_dir = args.checkpoint_dir
    if name == "bicubic":
        return BicubicUpsample()
    elif name == "srcnn":
        return SRCNNModel(ckpt_dir / "srcnn_best.pth", device, args.srcnn_upscale_factor)
    elif name == "diffusion":
        return DiffusionModel(
            ckpt_dir / "diffusion_best.pth",
            device,
            args.diffusion_timesteps,
            compile=args.compile,
            amp=args.amp,
        )
    elif name == "consistency":
        return ConsistencyModelWrapper(
            ckpt_dir / "consistency_best.pth",
            device,
            args.consistency_sigma_max,
            compile=args.compile,
            amp=args.amp,
        )
    elif name == "consistency_2step":
        return ConsistencyFewStep(
            ckpt_dir / "consistency_best.pth",
            device,
            num_steps=2,
            sigma_max=args.consistency_sigma_max,
            compile=args.compile,
            amp=args.amp,
        )
    elif name == "consistency_4step":
        return ConsistencyFewStep(
            ckpt_dir / "consistency_best.pth",
            device,
            num_steps=4,
            sigma_max=args.consistency_sigma_max,
            compile=args.compile,
            amp=args.amp,
        )
    elif name == "swin2sr":
        return Swin2SRZeroShot(args.model_id, args.revision, device)
    else:
        raise ValueError(f"Unknown model: {name}")


def source_of(patch_name: str) -> str:
    """'2012-11-30_0003.png' -> '2012-11-30'."""
    return patch_name.rsplit(".", 1)[0].rsplit("_", 1)[0]


def select_preview_patches(filenames: list[str], num_images: int = 5, patches_per_image: int = 2) -> list[str]:
    """Select first N unique images, taking M patches from each."""
    from collections import defaultdict
    patches_by_image = defaultdict(list)
    for fn in filenames:
        img = source_of(fn)
        if len(patches_by_image[img]) < patches_per_image:
            patches_by_image[img].append(fn)
        if len(patches_by_image) >= num_images:
            break
    selected = []
    for img_patches in patches_by_image.values():
        selected.extend(img_patches)
    return selected


def main():
    parser = argparse.ArgumentParser(description="Generate preview images for each model")
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--data-dir", type=Path, default=project_root / "apod_data" / "dataset")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["bicubic", "srcnn", "diffusion", "consistency", "consistency_2step", "consistency_4step", "swin2sr"],
        default=["bicubic", "srcnn", "diffusion", "consistency", "consistency_2step", "consistency_4step", "swin2sr"],
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(__file__).resolve().parent / "checkpoints")
    parser.add_argument("--srcnn-upscale-factor", type=int, default=4)
    parser.add_argument("--diffusion-timesteps", type=int, default=1000)
    parser.add_argument("--consistency-sigma-max", type=float, default=80.0)
    parser.add_argument("--model-id", default="caidas/swin2SR-classical-sr-x4-64")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--num-images", type=int, default=5, help="Number of unique images to preview")
    parser.add_argument("--patches-per-image", type=int, default=2, help="Number of patches per image")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--diffusion-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("previews"))
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="torch.compile models")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Use bf16 autocast")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_dir = args.out_dir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = SRDataset(args.data_dir, split=args.split)
    all_filenames = dataset.filenames

    selected_patches = select_preview_patches(all_filenames, args.num_images, args.patches_per_image)
    num_previews = len(selected_patches)
    selected_set = set(selected_patches)
    selected_indices = {i for i, fn in enumerate(all_filenames) if fn in selected_set}

    print(f"Dataset: {len(all_filenames)} patches from {args.split} split")
    print(f"Selected: {len(selected_patches)} patches from {args.num_images} images ({args.patches_per_image} patches each)")
    print(f"Device: {device}")
    print(f"Models: {args.models}")
    print(f"Optimizations: compile={args.compile}, amp={args.amp}")
    print(f"Output: {out_dir}\n")

    for model_name in args.models:
        print(f"Loading {model_name}...")
        t0 = time.time()
        model = load_model(model_name, args, device)
        print(f"  Loaded in {time.time() - t0:.1f}s")

        batch_size = args.diffusion_batch_size if model_name in ("diffusion", "consistency", "consistency_2step", "consistency_4step") else args.batch_size
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

        previews_saved = 0
        t0 = time.time()

        with torch.inference_mode():
            for batch_idx, (lr, hr) in enumerate(loader):
                lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
                sr = model(lr)

                for i in range(lr.size(0)):
                    global_idx = batch_idx * batch_size + i
                    if global_idx in selected_indices:
                        fn = all_filenames[global_idx]
                        save_preview(lr[i], sr[i], hr[i], out_dir / f"{model_name}_{fn}")
                        previews_saved += 1
                        if previews_saved >= num_previews:
                            break

                if previews_saved >= num_previews:
                    break

                print(f"  Processed {min((batch_idx + 1) * batch_size, len(all_filenames))}/{len(all_filenames)} patches", end="\r")

        print(f"\n  {previews_saved} previews saved in {time.time() - t0:.1f}s\n")
        del model
        torch.cuda.empty_cache()

    print(f"\nDone! Previews saved to {out_dir}")


if __name__ == "__main__":
    main()
