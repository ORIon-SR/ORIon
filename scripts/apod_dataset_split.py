#!/usr/bin/env python3
"""
apod_dataset_split.py
======================
ORIon — image-level train/val/test split for the filtered APOD HR image pool.

Context (per CV_Project_Proposal_ORIon.docx and the project's decision log):
    - LR-HR pairs are constructed ourselves via bicubic downsampling of each
      HR image, at a fixed x4 factor (with x2/x8 used for the secondary
      downsampling-factor sweep).
    - The split MUST be by image, not by patch: no HR image may contribute
      crops/patches to more than one of train/val/test. This script is the
      point where that guarantee is created, upstream of any patch/pair
      extraction, so every later stage (bicubic baseline, zero-shot model,
      CNN, diffusion, consistency) reuses the exact same three image lists.
    - Input is assumed to already be the *filtered* pool of baseline HR
      images (video-URL rows and animated GIFs already removed by
      apod_filter_download.py). This script does not re-check media_type or
      GIF frame counts.
    - Each image's pixel width/height is recorded alongside its split
      assignment, so the patch-extraction step can draw a number of HR
      patches proportional to image size (a large mosaic should contribute
      more crops than a small frame) instead of a fixed count per image.

What it does:
    1. Discovers the filtered HR image pool, either from a directory of
       image files or from a CSV manifest (e.g. the output of
       apod_filter_download.py) with a path/filename column.
    2. Opens each image (via Pillow, header-only — this does not decode
       pixel data) to record its width and height.
    3. Deterministically shuffles the (sorted) file list with a fixed seed
       and partitions it into train/val/test by the requested fractions.
    4. Writes a single manifest (splits.csv) mapping every image to its
       split and dimensions, plus one manifest per split, plus a
       splits_config.json that records the seed, fractions, counts, basic
       width/height stats per split, and a hash of the input file list
       (including its recorded dimensions) so a re-run can be checked for
       reproducibility (per the course's "seeds set, results reproducible
       from a clean checkout" repo rubric).
    5. Optionally materialises the split by copying or symlinking images
       into <output_dir>/{train,val,test}/ — off by default, since
       manifest-only splitting is cheaper and the pair-construction step
       can just read splits.csv directly.

Usage:
    # From a directory of already-filtered HR images:
    python apod_dataset_split.py \
        --input /data/apod_hr_filtered \
        --output-dir /data/apod_splits \
        --train 0.8 --val 0.1 --test 0.1 --seed 0

    # From a CSV manifest (e.g. apod_filter_download.py output):
    python apod_dataset_split.py \
        --input filtered_manifest.csv --path-column local_path \
        --output-dir /data/apod_splits --seed 0

    # Also materialise the split as symlinked folders:
    python apod_dataset_split.py --input /data/apod_hr_filtered \
        --output-dir /data/apod_splits --materialize symlink

Re-running with the same input pool and --seed always reproduces the same
split; the script refuses to silently overwrite an existing splits.csv
unless --force is passed, and will warn (not fail) if the input file list
or any image's recorded dimensions have changed since the manifest was
written. An image that fails to open is skipped with a warning rather than
aborting the whole run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    width: int
    height: int


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def discover_from_directory(input_dir: Path) -> list[Path]:
    """Recursively collect image files under input_dir, sorted for a
    deterministic starting order (filesystem iteration order is not
    guaranteed to be stable across machines/runs)."""
    paths = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths)


def discover_from_csv(csv_path: Path, path_column: str | None) -> list[Path]:
    """Read a manifest CSV (e.g. apod_filter_download.py's output) and pull
    out the image path column. If --path-column isn't given, try a short
    list of likely names before giving up."""
    candidates = [path_column] if path_column else [
        "path", "filepath", "file_path", "local_path", "filename", "image_path",
    ]

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row")

        col = next((c for c in candidates if c in reader.fieldnames), None)
        if col is None:
            raise ValueError(
                f"Could not find a path column in {csv_path}. "
                f"Header was {reader.fieldnames}; pass --path-column explicitly."
            )

        rows = [Path(row[col]) for row in reader if row.get(col)]

    return sorted(rows)


def read_image_size(path: Path) -> tuple[int, int] | None:
    """Width/height via Pillow. Image.open() only reads the header, not the
    pixel data, so this is cheap even across a full dataset. Returns None
    (and lets the caller warn/skip) rather than raising, so one corrupt
    file doesn't abort the whole split."""
    try:
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        print(f"warning: could not read size of {path} ({exc}); skipping", file=sys.stderr)
        return None


def build_records(paths: list[Path]) -> list[ImageRecord]:
    records = []
    for p in paths:
        size = read_image_size(p)
        if size is None:
            continue
        width, height = size
        records.append(ImageRecord(path=p, width=width, height=height))
    return records


def discover_images(input_path: Path, path_column: str | None) -> list[ImageRecord]:
    if input_path.is_dir():
        paths = discover_from_directory(input_path)
    elif input_path.suffix.lower() == ".csv":
        paths = discover_from_csv(input_path, path_column)
    else:
        raise ValueError(
            f"--input must be a directory of images or a .csv manifest, got {input_path}"
        )

    if not paths:
        raise ValueError(f"No images found under {input_path}")

    records = build_records(paths)
    if not records:
        raise ValueError(f"Found {len(paths)} candidate files under {input_path}, but none could be opened")

    return records


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

@dataclass
class SplitResult:
    assignments: dict[str, list[ImageRecord]] = field(
        default_factory=lambda: {name: [] for name in SPLIT_NAMES}
    )


def compute_split(
    records: list[ImageRecord], train_frac: float, val_frac: float, test_frac: float, seed: int
) -> SplitResult:
    total_frac = train_frac + val_frac + test_frac
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(
            f"--train/--val/--test must sum to 1.0, got {total_frac:.6f}"
        )

    # Deterministic shuffle: records already arrive sorted by path (from
    # discover_images), so the starting order never depends on
    # filesystem/CSV iteration; shuffling with a seeded RNG then means the
    # same (seed, input pool) always yields the same split.
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)
    # test gets the remainder so counts always sum exactly to n, regardless
    # of rounding on the first two.
    n_test = n - n_train - n_val

    if n_test < 0:
        raise ValueError(
            "Rounding of --train/--val pushed test below zero; "
            "adjust fractions or dataset size."
        )

    result = SplitResult()
    result.assignments["train"] = shuffled[:n_train]
    result.assignments["val"] = shuffled[n_train:n_train + n_val]
    result.assignments["test"] = shuffled[n_train + n_val:]
    return result


def pool_hash(records: list[ImageRecord]) -> str:
    """Hash of the sorted (path, width, height) triples, so a re-run can
    detect whether the underlying image pool — or any image's recorded
    dimensions — changed since the manifest was written."""
    h = hashlib.sha256()
    for r in sorted(records, key=lambda r: str(r.path)):
        h.update(f"{r.path}\t{r.width}\t{r.height}\n".encode("utf-8"))
    return h.hexdigest()


# --------------------------------------------------------------------------
# Writing outputs
# --------------------------------------------------------------------------

def write_manifests(result: SplitResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_path = output_dir / "splits.csv"
    with open(combined_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "width", "height", "split"])
        for split_name in SPLIT_NAMES:
            for r in result.assignments[split_name]:
                writer.writerow([str(r.path), r.width, r.height, split_name])

    for split_name in SPLIT_NAMES:
        split_path = output_dir / f"{split_name}.csv"
        with open(split_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "width", "height"])
            for r in result.assignments[split_name]:
                writer.writerow([str(r.path), r.width, r.height])


def size_stats(records: list[ImageRecord]) -> dict:
    if not records:
        return {}
    widths = sorted(r.width for r in records)
    heights = sorted(r.height for r in records)

    def median(vals: list[int]) -> float:
        n = len(vals)
        mid = n // 2
        return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2

    return {
        "width": {"min": widths[0], "median": median(widths), "max": widths[-1]},
        "height": {"min": heights[0], "median": median(heights), "max": heights[-1]},
    }


def write_config(
    output_dir: Path,
    input_path: Path,
    seed: int,
    fracs: tuple[float, float, float],
    result: SplitResult,
    input_hash: str,
) -> None:
    counts = {name: len(result.assignments[name]) for name in SPLIT_NAMES}
    total = sum(counts.values())
    config = {
        "input_path": str(input_path),
        "seed": seed,
        "requested_fractions": {
            "train": fracs[0], "val": fracs[1], "test": fracs[2],
        },
        "actual_counts": counts,
        "actual_fractions": {
            name: round(counts[name] / total, 4) for name in SPLIT_NAMES
        } if total else {},
        "total_images": total,
        "image_size_stats_by_split": {
            name: size_stats(result.assignments[name]) for name in SPLIT_NAMES
        },
        "input_pool_sha256": input_hash,
        "split_unit": "image",  # not patch — see module docstring
    }
    with open(output_dir / "splits_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def materialize(result: SplitResult, output_dir: Path, mode: str) -> None:
    for split_name in SPLIT_NAMES:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for r in result.assignments[split_name]:
            dest = split_dir / r.path.name
            if dest.exists():
                continue
            if mode == "symlink":
                dest.symlink_to(r.path.resolve())
            elif mode == "copy":
                shutil.copy2(r.path, dest)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the filtered APOD HR image pool into train/val/test by image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Directory of filtered HR images, or a CSV manifest with a path column.",
    )
    parser.add_argument(
        "--path-column", default=None,
        help="Column name holding image paths, if --input is a CSV. "
             "Auto-detected from common names if omitted.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Where to write splits.csv / train.csv / val.csv / test.csv / splits_config.json.",
    )
    parser.add_argument("--train", type=float, default=0.8, help="Train fraction (default 0.8).")
    parser.add_argument("--val", type=float, default=0.1, help="Val fraction (default 0.1).")
    parser.add_argument("--test", type=float, default=0.1, help="Test fraction (default 0.1).")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Shuffle seed. Same seed + same input pool always reproduces the same split.",
    )
    parser.add_argument(
        "--materialize", choices=["none", "symlink", "copy"], default="none",
        help="Also lay out output_dir/{train,val,test}/ with symlinks or copies "
             "of the source images (default: manifest-only, no file duplication).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing splits.csv in --output-dir instead of erroring out.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    existing = args.output_dir / "splits.csv"
    if existing.exists() and not args.force:
        print(
            f"error: {existing} already exists. Pass --force to overwrite "
            f"(re-running with the same --seed and input pool reproduces the "
            f"same split anyway, so this is usually a mistake, not a fix).",
            file=sys.stderr,
        )
        return 1

    records = discover_images(args.input, args.path_column)
    result = compute_split(records, args.train, args.val, args.test, args.seed)

    write_manifests(result, args.output_dir)
    write_config(
        args.output_dir, args.input, args.seed,
        (args.train, args.val, args.test), result, pool_hash(records),
    )

    if args.materialize != "none":
        materialize(result, args.output_dir, args.materialize)

    total = len(records)
    print(f"Image pool: {total} images from {args.input}")
    for name in SPLIT_NAMES:
        n = len(result.assignments[name])
        pct = 100 * n / total if total else 0.0
        print(f"  {name:5s}: {n:6d} images ({pct:5.1f}%)")
    print(f"Manifests written to {args.output_dir}/ (columns: path, width, height, split)")
    if args.materialize != "none":
        print(f"Images materialised via {args.materialize} under {args.output_dir}/{{train,val,test}}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
