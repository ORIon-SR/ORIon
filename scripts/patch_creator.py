#!/usr/bin/env python3
"""Extract fixed-size HR patches from APOD train/val/test splits.

Patches are sampled randomly per image, with a texture filter (min standard
deviation) to prefer informative regions. The number of patches per image
scales with image area. Images smaller than the patch size are skipped.

Optionally produces LR patches via bicubic downsampling for super-resolution
training pairs.

Usage:
    python scripts/patch_creator.py \
        --input-dir apod_data/images \
        --output-dir apod_data/dataset \
        --patch-size 512 \
        --downscale-factor 4
"""

import argparse
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

SPLIT_NAMES = ["train", "val", "test"]
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
SPLITS_DIR = PROJECT_ROOT / "apod_data" / "splits"


def patch_count(height, width, alpha=2, patch_size=512, n_min=4, n_max=100):
    return int(
        np.clip(np.round((alpha * height * width) / patch_size**2), n_min, n_max)
    )


def read_split_csv(csv_path) -> list[tuple[Path, int, int]]:
    print(csv_path)
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        _ = f.readline().strip().split(",")  # consume header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            raw_path, w, h = parts[0], int(parts[1]), int(parts[2])
            # Normalise Windows backslash separators to forward slashes.
            norm = raw_path.replace("\\", "/")
            records.append((Path(norm), w, h))
    return records


def select_patches(
    img_arr: np.ndarray,
    num_patches: int,
    patch_size: int = 512,
    min_std: float = 15.0,
    max_attempts: int = 500,
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int]]:
    if rng is None:
        rng = np.random.default_rng()
    h, w = img_arr.shape[:2]
    if h < patch_size or w < patch_size:
        return []

    gray = np.mean(img_arr, axis=2) if img_arr.ndim == 3 else img_arr
    max_y = h - patch_size
    max_x = w - patch_size

    coords = []
    seen = set()
    attempts = 0

    while len(coords) < num_patches and attempts < max_attempts:
        attempts += 1
        y = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
        x = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        coord = (y, x)

        if coord in seen:
            continue

        gray_patch = gray[y : y + patch_size, x : x + patch_size]
        if np.std(gray_patch) >= min_std:
            seen.add(coord)
            coords.append(coord)

    fallback_attempts = 0
    max_fallback_attempts = 200
    while len(coords) < num_patches and fallback_attempts < max_fallback_attempts:
        fallback_attempts += 1
        y = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
        x = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        coord = (y, x)

        if coord not in seen:
            seen.add(coord)
            coords.append(coord)

    return coords


def _process_single_image(
    img_path: Path,
    w: int,
    h: int,
    output_dir: Path,
    split_name: str,
    patch_size: int,
    downscale_factor: int | None,
    seed: int = 0,
) -> tuple[int, int]:
    """Process one image: extract patches and save them. Returns (patches_extracted, skipped).

    The RNG is derived from (seed, image file name), so the patches of an image are the same
    for any number of workers, any process start method and any processing order.
    """
    hr_out = output_dir / split_name / "HR"
    lr_out = output_dir / split_name / "LR"

    if not img_path.exists():
        print(f"  warning: {img_path} not found, skipping", file=sys.stderr)
        return 0, 1

    if w < patch_size or h < patch_size:
        print(
            f"  warning: {img_path} ({w}x{h}) smaller than "
            f"{patch_size}x{patch_size}, skipping",
            file=sys.stderr,
        )
        return 0, 1

    try:
        img = Image.open(img_path).convert("RGB")
        img_arr = np.array(img)
        h, w = img_arr.shape[:2]
    except Exception as exc:  # noqa: BLE001
        print(f"  warning: could not open {img_path} ({exc})", file=sys.stderr)
        return 0, 1

    num_patches = patch_count(h, w)
    rng = np.random.default_rng([seed, zlib.crc32(img_path.name.encode("utf-8"))])
    patches = select_patches(img_arr, num_patches, patch_size=patch_size, rng=rng)
    stem = img_path.stem

    count = 0
    for i, (y, x) in enumerate(patches):
        name = f"{stem}_{i:04d}.png"
        hr_patch = img.crop((x, y, x + patch_size, y + patch_size))
        hr_patch.save(hr_out / name)

        if downscale_factor is not None:
            lr_patch_size = patch_size // downscale_factor
            lr_patch = hr_patch.resize(
                (lr_patch_size, lr_patch_size), Image.Resampling.BICUBIC
            )
            lr_patch.save(lr_out / name)

        count += 1

    return count, 0


def process_split(
    split_name: str,
    records: list[tuple[Path, int, int]],
    output_dir: Path,
    patch_size: int,
    downscale_factor: int | None,
    num_workers: int,
    seed: int = 0,
) -> int:
    hr_out = output_dir / split_name / "HR"
    hr_out.mkdir(parents=True, exist_ok=True)

    lr_out = output_dir / split_name / "LR"
    lr_out.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0

    if num_workers == 1:
        for img_path, w, h in records:
            count, skip = _process_single_image(
                img_path, w, h, output_dir, split_name, patch_size, downscale_factor, seed
            )
            total += count
            skipped += skip
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _process_single_image,
                    img_path,
                    w,
                    h,
                    output_dir,
                    split_name,
                    patch_size,
                    downscale_factor,
                    seed,
                )
                for img_path, w, h in records
            ]
            for future in futures:
                count, skip = future.result()
                total += count
                skipped += skip

    print(f"  {split_name}: {total} patches ({skipped} images skipped)")
    return total


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 512x512 HR patches from each split of the APOD dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="apod_data/dataset",
        type=Path,
        help="Where to write {train,val,test}/{HR,LR}/ patch folders.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=512,
        help="HR patch size in pixels (default 512).",
    )
    parser.add_argument(
        "--downscale-factor",
        type=int,
        default=None,
        help="If set, also produce LR patches by bicubic downsampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Set seed for deterministic patch extraction",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: CPU count). Use 1 for single-threaded.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    num_workers = args.num_workers if args.num_workers is not None else os.cpu_count()
    print(f"Using {num_workers} worker process(es)")

    grand_total = 0
    for split_name in SPLIT_NAMES:
        csv_path = SPLITS_DIR / f"{split_name}.csv"
        if not csv_path.exists():
            print(f"warning: {csv_path} not found, skipping split", file=sys.stderr)
            continue
        records = read_split_csv(csv_path)
        n = process_split(
            split_name,
            records,
            args.output_dir,
            args.patch_size,
            args.downscale_factor,
            num_workers,
            args.seed,
        )
        grand_total += n

    print(f"\nTotal: {grand_total} HR patches written to {args.output_dir}")


if __name__ == "__main__":
    main()
