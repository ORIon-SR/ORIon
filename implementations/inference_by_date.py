#!/usr/bin/env python3
"""Run inference on all patches from a specific image date.

Finds all patches matching the given date across dataset splits,
runs inference with each model, and saves results alongside the
LR input and HR ground truth.

Usage:
    python inference_by_date.py 2015-05-31
    python inference_by_date.py 2015-05-31 --split test --models srcnn diffusion
    python inference_by_date.py 2015-05-31 --no-compile --no-amp
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models_eval import (
    BicubicUpsample,
    ConsistencyFewStep,
    ConsistencyModelWrapper,
    DiffusionModel,
    SRCNNModel,
    Swin2SRZeroShot,
)


class DatePatchDataset(Dataset):
    """Dataset that loads specific (split, filename) patches."""

    def __init__(self, data_root: Path, patches: list[tuple[str, str]]):
        self.data_root = data_root
        self.patches = patches  # list of (split, filename)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        split, filename = self.patches[idx]
        lr_path = self.data_root / split / "LR" / filename
        hr_path = self.data_root / split / "HR" / filename

        import numpy as np
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        lr_arr = np.array(lr_img, dtype=np.float32) / 255.0
        hr_arr = np.array(hr_img, dtype=np.float32) / 255.0

        lr_tensor = torch.from_numpy(lr_arr).permute(2, 0, 1)
        hr_tensor = torch.from_numpy(hr_arr).permute(2, 0, 1)

        return lr_tensor, hr_tensor, filename


def find_patches(data_root: Path, date: str, split: str | None = None) -> list[tuple[str, str]]:
    """Find all patches matching the given date.

    Returns list of (split, filename) tuples.
    """
    splits = [split] if split else ["train", "val", "test"]
    patches = []

    for s in splits:
        lr_dir = data_root / s / "LR"
        if not lr_dir.exists():
            continue
        for f in sorted(lr_dir.glob(f"{date}_*.png")):
            hr_path = data_root / s / "HR" / f.name
            if hr_path.exists():
                patches.append((s, f.name))

    return patches


def save_tensor_as_png(tensor: torch.Tensor, path: Path) -> None:
    """Save a [0, 1] float tensor as PNG."""
    arr = tensor.clamp(0, 1).mul(255).round().byte().cpu().numpy()
    img = Image.fromarray(arr.transpose(1, 2, 0))
    img.save(path)


def load_model(name: str, args, device: torch.device):
    """Load a model by name using the wrappers from models_eval."""
    ckpt_dir = args.checkpoint_dir
    compile_model = not args.no_compile
    amp_model = not args.no_amp

    if name == "bicubic":
        return BicubicUpsample()
    elif name == "srcnn":
        return SRCNNModel(ckpt_dir / "srcnn_best.pth", device, args.srcnn_upscale_factor)
    elif name == "diffusion":
        return DiffusionModel(
            ckpt_dir / "diffusion_best.pth",
            device,
            args.diffusion_timesteps,
            compile=compile_model,
            amp=amp_model,
        )
    elif name == "consistency":
        return ConsistencyModelWrapper(
            ckpt_dir / "consistency_best.pth",
            device,
            args.consistency_sigma_max,
            compile=compile_model,
            amp=amp_model,
        )
    elif name == "consistency_2step":
        return ConsistencyFewStep(
            ckpt_dir / "consistency_best.pth",
            device,
            num_steps=2,
            sigma_max=args.consistency_sigma_max,
            compile=compile_model,
            amp=amp_model,
        )
    elif name == "consistency_4step":
        return ConsistencyFewStep(
            ckpt_dir / "consistency_best.pth",
            device,
            num_steps=4,
            sigma_max=args.consistency_sigma_max,
            compile=compile_model,
            amp=amp_model,
        )
    elif name == "swin2sr":
        return Swin2SRZeroShot(args.model_id, args.revision, device)
    else:
        raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on all patches from a specific date"
    )
    project_root = Path(__file__).resolve().parent.parent

    parser.add_argument("date", help="Image date to process (e.g. 2015-05-31)")
    parser.add_argument(
        "--data-dir", type=Path, default=project_root / "apod_data" / "dataset"
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="Only search this split (default: search all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "bicubic", "srcnn", "diffusion", "consistency",
            "consistency_2step", "consistency_4step", "swin2sr",
        ],
        default=[
            "bicubic", "srcnn", "diffusion", "consistency",
            "consistency_2step", "consistency_4step", "swin2sr",
        ],
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints",
        help="Directory containing model checkpoints",
    )
    parser.add_argument("--srcnn-upscale-factor", type=int, default=4)
    parser.add_argument("--diffusion-timesteps", type=int, default=1000)
    parser.add_argument("--consistency-sigma-max", type=float, default=80.0)
    parser.add_argument(
        "--model-id", default="caidas/swin2SR-classical-sr-x4-64"
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--diffusion-batch-size", type=int, default=2,
        help="Batch size for diffusion/consistency models (default: 2)",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--out-dir", type=Path, default=project_root / "results" / "inference"
    )
    parser.add_argument(
        "--no-compile", action="store_true", help="Disable torch.compile"
    )
    parser.add_argument(
        "--no-amp", action="store_true", help="Disable autocast (bf16/fp16)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing output directory"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Find patches for this date
    print(f"Searching for patches from {args.date}...")
    patches = find_patches(args.data_dir, args.date, args.split)

    if not patches:
        split_msg = f" in split '{args.split}'" if args.split else ""
        print(f"No patches found for date {args.date}{split_msg}")
        sys.exit(1)

    # Group by split for reporting
    by_split = {}
    for split, fn in patches:
        by_split.setdefault(split, []).append(fn)

    print(f"Found {len(patches)} patches:")
    for split, fns in sorted(by_split.items()):
        print(f"  {split}: {len(fns)} patches")

    # Set up output directory
    out_dir = args.out_dir / args.date
    if out_dir.exists() and not args.force:
        print(f"Output directory {out_dir} already exists. Use --force to overwrite.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for each model
    for model_name in args.models:
        (out_dir / model_name).mkdir(exist_ok=True)

    # Create LR and HR directories and copy originals
    lr_out_dir = out_dir / "LR"
    hr_out_dir = out_dir / "HR"
    lr_out_dir.mkdir(exist_ok=True)
    hr_out_dir.mkdir(exist_ok=True)

    print(f"\nCopying LR and HR originals to {out_dir}...")
    for split, filename in patches:
        src_lr = args.data_dir / split / "LR" / filename
        src_hr = args.data_dir / split / "HR" / filename
        shutil.copy2(src_lr, lr_out_dir / filename)
        shutil.copy2(src_hr, hr_out_dir / filename)

    print(f"Copied {len(patches)} LR/HR pairs")

    # Create dataset and dataloader
    dataset = DatePatchDataset(args.data_dir, patches)

    print(f"\nDevice: {device}")
    print(f"Models: {args.models}")
    print(f"Optimizations: compile={not args.no_compile}, amp={not args.no_amp}")
    print(f"Output: {out_dir}\n")

    # Run inference for each model
    for model_name in args.models:
        print(f"Loading {model_name}...")
        t0 = time.time()
        model = load_model(model_name, args, device)
        print(f"  Loaded in {time.time() - t0:.1f}s")

        batch_size = (
            args.diffusion_batch_size
            if model_name in ("diffusion", "consistency", "consistency_2step", "consistency_4step")
            else args.batch_size
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        model_out_dir = out_dir / model_name
        saved_count = 0
        t0 = time.time()

        with torch.inference_mode():
            for batch_idx, (lr, hr, filenames) in enumerate(loader):
                lr = lr.to(device, non_blocking=True)
                sr = model(lr)

                for i, filename in enumerate(filenames):
                    save_tensor_as_png(sr[i], model_out_dir / filename)
                    saved_count += 1

                print(
                    f"  {model_name}: {saved_count}/{len(patches)} patches",
                    end="\r",
                )

        elapsed = time.time() - t0
        print(f"\n  Saved {saved_count} images in {elapsed:.1f}s ({saved_count/elapsed:.1f} img/s)\n")

        # Clear GPU memory before loading next model
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    print(f"\nDone! Results saved to {out_dir}")
    print(f"\nDirectory structure:")
    print(f"  {out_dir}/")
    print(f"    LR/          ({len(patches)} input patches)")
    print(f"    HR/          ({len(patches)} ground truth patches)")
    for model_name in args.models:
        print(f"    {model_name}/  ({len(patches)} output patches)")


if __name__ == "__main__":
    main()
