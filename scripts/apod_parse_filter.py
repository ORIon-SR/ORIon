#!/usr/bin/env python3
"""
apod_parse_filter.py
=====================

ORIon (Observing Resolution Improvement) — APOD dataset parsing & filtering.

Purpose
-------
Turns the raw Kaggle APOD CSV (ahsanneural/nasa-astronomy-picture-of-the-day-1995-2026)
into a clean, resolution-filtered metadata table + a local image cache, ready for the
LR/HR patch-pair construction step described in the project proposal.

This script is deliberately built as TWO PHASES so that the group can make the HR/LR
patch-size decision cheaply, before committing to the full ~10.7k-row download:

  Phase A — HEADER PROBE (cheap, no full downloads)
    For every candidate row, stream just enough of each file (a few KB-2MB) to read the
    image header and recover (width, height, format) WITHOUT downloading the whole file.
    This is enough to apply the resolution filter and to report resolution percentiles,
    which is exactly the output the roadmap says we need to lock in the HR crop size.

  Phase B — FULL DOWNLOAD (only for rows that passed Phase A)
    Only images that already cleared the resolution bar are downloaded in full. This is
    needed for two things a header alone can't tell us: (1) whether a GIF is animated
    (n_frames > 1 requires the whole file), and (2) an exact, decoded width/height/format
    to store in the final metadata (header-derived dims are usually exact, but full
    decode is the ground truth we keep).

Why a resolution filter is needed at all
-----------------------------------------
The proposal fixes a ×4 downsampling factor and requires HR ground-truth patches to be
"meaningfully high-resolution" before we bicubic-downsample them to build LR/HR pairs.
An APOD image below the HR patch size can't even supply one valid crop, so it must be
discarded (and the discard count reported, per the proposal's provenance/data section).

How the minimum resolution is chosen (documented, not hard-coded silently)
----------------------------------------------------------------------------
  HR_PATCH_SIZE          : the side length (px) of the HR crop used to build LR/HR pairs.
                            Default 512. The roadmap leaves this open between 128 and 512;
                            512 is the safer default here because rungs (4) and (5) of the
                            SR ladder (diffusion, consistency) are the project's primary
                            focus — a 512x512 HR / 128x128 LR pair keeps every rung,
                            including the generative ones, producing a non-trivial SR task.
                            Pass --hr-patch-size 256 to reduce compute if needed (see
                            roadmap: "Set hard compute budget ceiling before beginning the
                            diffusion rung").
  DOWNSAMPLE_FACTOR       : fixed at 4, per the proposal's headline protocol. (The
                            secondary ×2/×8 sweep re-uses this same script with
                            --downsample-factor overridden, since the resolution filter
                            scales with it.)
  MIN_DIM_MULTIPLIER      : an image's shorter side must be >= MIN_DIM_MULTIPLIER x
                            HR_PATCH_SIZE to be kept. Default 2.0, i.e. minimum accepted
                            resolution is 1024x1024 for the default 512px HR patch. A 1.0x
                            multiplier would only ever allow a single, fixed, centred HR
                            crop per image — no random-crop augmentation is possible, and
                            every rung would train on the exact same patch location every
                            epoch. 2.0x guarantees at least a 2x2 grid of non-overlapping
                            crop positions (more with overlapping/random crops), which is
                            the minimum needed for meaningful spatial augmentation without
                            being so strict that we throw away most of the dataset.
  => Effective minimum kept resolution = MIN_DIM_MULTIPLIER * HR_PATCH_SIZE on the
     shorter side. Corresponding LR patch size = HR_PATCH_SIZE / DOWNSAMPLE_FACTOR.

Known data-quality issues this script accounts for (see project memory / roadmap)
------------------------------------------------------------------------------------
  1. ~14 rows have media_type == 'image' but the URL actually points to a video file
     (e.g. .mp4/.mov/.webm). media_type alone is not a reliable filter — we additionally
     check the URL's file extension against a video-extension blocklist and drop matches.
  2. ~1,130 rows are GIFs; some are animated and must be excluded (a single still HR/LR
     pair only makes sense for a static image). Animated-ness can only be determined from
     the full file (n_frames > 1 via Pillow), so this check happens in Phase B.

Usage
-----
  # 1) Cheap, whole-dataset pass to decide the crop size (no images saved):
  python apod_parse_filter.py --csv apod_raw.csv --stats-only

  # 2) Small test run end-to-end (per roadmap: --limit 200 before the full download):
  python apod_parse_filter.py --csv apod_raw.csv --limit 200

  # 3) Full run once the crop size is locked in:
  python apod_parse_filter.py --csv apod_raw.csv --hr-patch-size 128

Output
------
  <output-dir>/apod_metadata_filtered.csv   Kept rows: all original metadata columns +
                                             file_type, width, height, n_frames,
                                             local_path, discovered via header/full probe.
  <output-dir>/apod_discarded.csv           Discarded rows + a discard_reason column.
  <output-dir>/images/                      Locally cached full-resolution image files
                                             for every kept row (Phase B).
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import warnings

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError

# We intentionally work with very large telescope images from APOD (some exceed
# 100M+ pixels). MAX_IMAGE_PIXELS = None disables the hard DecompressionBombError,
# and the filterwarnings call suppresses the softer warning that fires at half that
# threshold — both are false positives for our use case of fetching known-good data.
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Extensions we recognise as (potentially) still images worth probing.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".bmp", ".webp"}

# Extensions that mean the row is actually a video, regardless of what media_type says.
# This is the direct fix for the "~14 media_type == image rows point to video files" issue.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg"}

# Formats whose animation status can only be known from the full file.
FORMATS_NEEDING_FULL_DOWNLOAD_FOR_ANIMATION_CHECK = {"GIF"}

USER_AGENT = "ORIon-CV-Project/1.0 (Wits COMS4036A super-resolution coursework; contact via course staff)"
REQUEST_TIMEOUT = 20  # seconds
HEADER_PROBE_CAP_BYTES = 2 * 1024 * 1024  # give up on header-only parsing beyond 2MB
HEADER_PROBE_CHUNK_BYTES = 16 * 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("apod_parse_filter")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    ok: bool
    width: Optional[int] = None
    height: Optional[int] = None
    file_format: Optional[str] = None
    n_frames: Optional[int] = None
    file_size_bytes: Optional[int] = None
    reason: Optional[str] = None  # populated when ok is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_url(row: pd.Series) -> Optional[str]:
    """Prefer hdurl (the high-res original) and fall back to url."""
    hdurl = row.get("hdurl")
    url = row.get("url")
    if isinstance(hdurl, str) and hdurl.strip():
        return hdurl.strip()
    if isinstance(url, str) and url.strip():
        return url.strip()
    return None


def get_extension(url: str) -> str:
    path = urlparse(url).path
    return Path(path).suffix.lower()


def is_video_url(url: str) -> bool:
    return get_extension(url) in VIDEO_EXTENSIONS


def is_plausible_image_url(url: str) -> bool:
    ext = get_extension(url)
    # No extension at all (rare, some APOD URLs are extensionless) — still worth probing.
    return ext == "" or ext in IMAGE_EXTENSIONS


def header_probe(url: str, session: requests.Session) -> ProbeResult:
    """
    Phase A: stream the file in small chunks, trying to parse an image header after each
    chunk, until Pillow can report a size or we exceed HEADER_PROBE_CAP_BYTES.
    Does not save anything to disk. Cheap: for JPEG/PNG/GIF the header lives in the
    first few KB, so most requests are aborted (via the context manager) well before
    the full file is transferred.
    """
    try:
        with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as resp:
            resp.raise_for_status()
            buf = io.BytesIO()
            for chunk in resp.iter_content(chunk_size=HEADER_PROBE_CHUNK_BYTES):
                if not chunk:
                    continue
                buf.write(chunk)
                buf.seek(0)
                try:
                    with Image.open(buf) as im:
                        width, height = im.size
                        fmt = im.format
                    return ProbeResult(ok=True, width=width, height=height, file_format=fmt)
                except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
                    buf.seek(0, io.SEEK_END)
                    if buf.tell() >= HEADER_PROBE_CAP_BYTES:
                        return ProbeResult(ok=False, reason="header_unparseable_within_cap")
                    continue
            return ProbeResult(ok=False, reason="stream_ended_before_header_parsed")
    except requests.exceptions.RequestException as exc:
        return ProbeResult(ok=False, reason=f"request_error:{exc.__class__.__name__}")


def full_download(url: str, session: requests.Session, dest_path: Path) -> ProbeResult:
    """
    Phase B: download the complete file, save it to dest_path, and return exact
    dimensions/format/frame-count. This is the only way to know if a GIF is animated.
    """
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.content
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
            fmt = im.format
            n_frames = getattr(im, "n_frames", 1)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(data)
        return ProbeResult(
            ok=True, width=width, height=height, file_format=fmt,
            n_frames=n_frames, file_size_bytes=len(data),
        )
    except requests.exceptions.RequestException as exc:
        return ProbeResult(ok=False, reason=f"request_error:{exc.__class__.__name__}")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        return ProbeResult(ok=False, reason=f"decode_error:{exc.__class__.__name__}")


def min_side(width: int, height: int) -> int:
    return min(width, height)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def load_raw_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_cols = {"date", "explanation", "hdurl", "media_type", "title", "url", "copyright"}
    missing = expected_cols - set(df.columns)
    if missing:
        log.warning("CSV is missing expected columns: %s (continuing anyway)", missing)
    return df


def report_resolution_percentiles(rows: list[dict]) -> None:
    if not rows:
        log.info("No successfully probed rows to report percentiles for.")
        return
    df = pd.DataFrame(rows)
    df["min_side"] = df[["width", "height"]].min(axis=1)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    stats = df["min_side"].describe(percentiles=[p / 100 for p in percentiles])
    log.info("Shorter-side resolution distribution over %d successfully probed images:", len(df))
    for p in percentiles:
        col = f"{p}%"
        log.info("  p%-3d = %.0f px", p, stats[col])
    log.info("  min  = %.0f px, max = %.0f px, mean = %.0f px", stats["min"], stats["max"], stats["mean"])


def run(args: argparse.Namespace) -> None:
    downsample_factor = args.downsample_factor
    hr_patch = args.hr_patch_size
    lr_patch = hr_patch / downsample_factor
    min_required_side = args.min_dim_multiplier * hr_patch

    log.info(
        "Config: downsample=x%d, HR patch=%dpx, LR patch=%.1fpx, "
        "min accepted shorter side=%.0fpx (multiplier=%.1fx)",
        downsample_factor, hr_patch, lr_patch, min_required_side, args.min_dim_multiplier,
    )

    csv_path = Path(args.csv)
    df = load_raw_csv(csv_path)
    if args.limit:
        df = df.head(args.limit)
        log.info("Test run: limited to first %d rows.", args.limit)

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    metadata_out = output_dir / args.metadata_out
    discarded_out = output_dir / args.discarded_out
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: skip URLs already recorded as kept in a prior run.
    already_kept_urls: set[str] = set()
    if args.resume and metadata_out.exists():
        prior = pd.read_csv(metadata_out)
        already_kept_urls = set(prior.get("source_url", pd.Series(dtype=str)).dropna())
        log.info("Resuming: %d rows already recorded in %s will be skipped.", len(already_kept_urls), metadata_out)

    # --report: just read the existing probe checkpoint and print percentiles, then exit.
    probe_checkpoint_path = output_dir / "apod_probe_checkpoint.csv"
    if args.report:
        if not probe_checkpoint_path.exists():
            log.error("--report requires an existing probe checkpoint at %s. Run --stats-only first.", probe_checkpoint_path)
            sys.exit(1)
        ckpt = pd.read_csv(probe_checkpoint_path)
        log.info("Loaded %d probed rows from checkpoint.", len(ckpt))
        report_resolution_percentiles(ckpt[["width", "height"]].to_dict("records"))
        passing = int((ckpt[["width", "height"]].min(axis=1) >= min_required_side).sum())
        log.info(
            "%d / %d probed rows clear the >= %.0fpx shorter-side bar (HR patch=%dpx, multiplier=%.1fx).",
            passing, len(ckpt), min_required_side, hr_patch, args.min_dim_multiplier,
        )
        return

    session = build_session()

    kept_rows: list[dict] = []
    discarded_rows: list[dict] = []

    # --retry: read apod_discarded.csv, pull out the two retryable categories, and attempt
    # them again.  The two categories need different treatment:
    #
    #   not_in_probe_checkpoint  — these were never probed at all (network error during
    #     stats-only, or a pre-probe filter like wrong media_type / video extension that
    #     now needs re-checking).  We re-run the full Phase A header probe on them, then
    #     Phase B if they pass resolution.  Any that fail again stay in discarded.
    #
    #   full_download_failed:*   — these passed the probe and the resolution filter, but
    #     the full download dropped (almost always a transient ConnectionError).  We skip
    #     Phase A entirely and go straight to Phase B.
    #
    # On completion, successful retries are appended to apod_metadata_filtered.csv and
    # removed from apod_discarded.csv (replaced by their updated discard reason if they
    # still fail, or removed entirely if they succeed).
    if args.retry:
        if not discarded_out.exists():
            log.error("--retry requires an existing %s. Run the main download first.", discarded_out)
            sys.exit(1)

        disc_df = pd.read_csv(discarded_out)

        # Also load the probe checkpoint so we can add new probe results to it.
        ckpt_lookup: dict[str, tuple[int, int]] = {}
        if probe_checkpoint_path.exists():
            ckpt = pd.read_csv(probe_checkpoint_path)
            ckpt_lookup = {
                r["url"]: (int(r["width"]), int(r["height"]))
                for _, r in ckpt.iterrows()
                if pd.notna(r.get("url")) and pd.notna(r.get("width")) and pd.notna(r.get("height"))
            }

        import csv as _csv
        _ckpt_file = open(probe_checkpoint_path, "a", newline="")
        _ckpt_writer = _csv.DictWriter(_ckpt_file, fieldnames=["url", "width", "height"])

        retryable_not_probed = disc_df[disc_df["discard_reason"] == "not_in_probe_checkpoint"].copy()
        retryable_failed_dl  = disc_df[disc_df["discard_reason"].str.startswith("full_download_failed", na=False)].copy()

        log.info(
            "--retry: %d not_in_probe_checkpoint rows to re-probe, "
            "%d full_download_failed rows to retry download.",
            len(retryable_not_probed), len(retryable_failed_dl),
        )

        retry_succeeded_urls: set[str] = set()
        updated_discard: dict[str, str] = {}  # url -> new discard_reason for rows that still fail

        # --- Category 1: not_in_probe_checkpoint — full Phase A then Phase B ---
        for i, (_, row) in enumerate(retryable_not_probed.iterrows(), start=1):
            log.info("Re-probing %d / %d", i, len(retryable_not_probed))
            base_record = {c: row[c] for c in disc_df.columns if c != "discard_reason"}
            url = pick_url(pd.Series(base_record))

            if url is None:
                updated_discard[str(i)] = "no_url"
                continue

            media_type = str(base_record.get("media_type", "")).strip().lower()
            if media_type not in ("image", ""):
                updated_discard[url] = f"media_type_not_image:{media_type}"
                continue
            if is_video_url(url):
                updated_discard[url] = "url_extension_is_video"
                continue
            if not is_plausible_image_url(url):
                updated_discard[url] = f"unrecognised_extension:{get_extension(url)}"
                continue

            probe = header_probe(url, session)
            if not probe.ok:
                updated_discard[url] = f"header_probe_failed:{probe.reason}"
                time.sleep(args.sleep)
                continue

            # Save to checkpoint so we don't re-probe next time.
            if url not in ckpt_lookup:
                _ckpt_writer.writerow({"url": url, "width": probe.width, "height": probe.height})
                _ckpt_file.flush()
                ckpt_lookup[url] = (probe.width, probe.height)

            if min_side(probe.width, probe.height) < min_required_side:
                updated_discard[url] = "below_min_resolution"
                time.sleep(args.sleep)
                continue

            ext = get_extension(url) or ".jpg"
            safe_name = f"{base_record.get('date', f'retry_{i}')}{ext}"
            dest_path = images_dir / safe_name
            full = full_download(url, session, dest_path)

            if not full.ok:
                updated_discard[url] = f"full_download_failed:{full.reason}"
                time.sleep(args.sleep)
                continue

            if full.file_format in FORMATS_NEEDING_FULL_DOWNLOAD_FOR_ANIMATION_CHECK and (full.n_frames or 1) > 1:
                updated_discard[url] = "animated_gif"
                dest_path.unlink(missing_ok=True)
                time.sleep(args.sleep)
                continue

            if min_side(full.width, full.height) < min_required_side:
                updated_discard[url] = "below_min_resolution_on_full_decode"
                dest_path.unlink(missing_ok=True)
                time.sleep(args.sleep)
                continue

            kept_rows.append({
                **base_record,
                "source_url": url,
                "file_type": full.file_format,
                "width": full.width,
                "height": full.height,
                "n_frames": full.n_frames,
                "file_size_bytes": full.file_size_bytes,
                "local_path": str(dest_path),
            })
            retry_succeeded_urls.add(url)
            time.sleep(args.sleep)

        # --- Category 2: full_download_failed — skip probe, go straight to Phase B ---
        for i, (_, row) in enumerate(retryable_failed_dl.iterrows(), start=1):
            log.info("Retrying download %d / %d", i, len(retryable_failed_dl))
            base_record = {c: row[c] for c in disc_df.columns if c != "discard_reason"}
            url = pick_url(pd.Series(base_record))

            if url is None:
                continue

            # We trust the checkpoint dims for the resolution check.
            if url in ckpt_lookup:
                w, h = ckpt_lookup[url]
                if min_side(w, h) < min_required_side:
                    updated_discard[url] = "below_min_resolution"
                    continue

            ext = get_extension(url) or ".jpg"
            safe_name = f"{base_record.get('date', f'retry_dl_{i}')}{ext}"
            dest_path = images_dir / safe_name
            full = full_download(url, session, dest_path)

            if not full.ok:
                updated_discard[url] = f"full_download_failed:{full.reason}"
                time.sleep(args.sleep)
                continue

            if full.file_format in FORMATS_NEEDING_FULL_DOWNLOAD_FOR_ANIMATION_CHECK and (full.n_frames or 1) > 1:
                updated_discard[url] = "animated_gif"
                dest_path.unlink(missing_ok=True)
                time.sleep(args.sleep)
                continue

            if min_side(full.width, full.height) < min_required_side:
                updated_discard[url] = "below_min_resolution_on_full_decode"
                dest_path.unlink(missing_ok=True)
                time.sleep(args.sleep)
                continue

            kept_rows.append({
                **base_record,
                "source_url": url,
                "file_type": full.file_format,
                "width": full.width,
                "height": full.height,
                "n_frames": full.n_frames,
                "file_size_bytes": full.file_size_bytes,
                "local_path": str(dest_path),
            })
            retry_succeeded_urls.add(url)
            time.sleep(args.sleep)

        _ckpt_file.close()

        # Merge successes into metadata, rebuild discarded without the rows that succeeded.
        if kept_rows:
            new_kept = pd.DataFrame(kept_rows)
            if metadata_out.exists():
                prior = pd.read_csv(metadata_out)
                new_kept = pd.concat([prior, new_kept], ignore_index=True)
            new_kept.to_csv(metadata_out, index=False)
            log.info("%d retried rows now kept -> %s", len(kept_rows), metadata_out)
        else:
            log.info("No retried rows succeeded.")

        # Rebuild discarded: remove rows that succeeded, update reasons for rows that
        # still failed with a new reason, leave untouched rows (other discard reasons)
        # as-is.
        retried_urls = set(
            pick_url(pd.Series({c: r[c] for c in disc_df.columns}))
            for _, r in pd.concat([retryable_not_probed, retryable_failed_dl]).iterrows()
        ) - {None}

        def updated_reason(row):
            url = pick_url(pd.Series(row.to_dict()))
            if url in retry_succeeded_urls:
                return None  # will be filtered out
            if url in updated_discard:
                return updated_discard[url]
            return row["discard_reason"]

        disc_df["discard_reason"] = disc_df.apply(updated_reason, axis=1)
        disc_df = disc_df[disc_df["discard_reason"].notna()]
        disc_df.to_csv(discarded_out, index=False)
        log.info("Updated discarded log -> %s (%d rows remaining)", discarded_out, len(disc_df))

        log.info(
            "Retry summary: %d succeeded, %d still failing.",
            len(retry_succeeded_urls),
            len(retried_urls) - len(retry_succeeded_urls),
        )
        return

    # --from-checkpoint: use pre-probed dimensions from the checkpoint CSV to skip the
    # header-probe step entirely.  We join the raw CSV (for metadata columns) against the
    # checkpoint (for width/height) on URL, filter to rows that pass the resolution bar,
    # and go straight to Phase B (full download + animated-GIF check) for those rows.
    # Rows in the raw CSV that are absent from the checkpoint were either never probed
    # (e.g. video rows filtered before probing) or failed the probe — we skip them and
    # record a discard reason so nothing is silently lost.
    if args.from_checkpoint:
        if not probe_checkpoint_path.exists():
            log.error(
                "--from-checkpoint requires an existing probe checkpoint at %s. "
                "Run --stats-only first.", probe_checkpoint_path,
            )
            sys.exit(1)
        ckpt = pd.read_csv(probe_checkpoint_path)
        ckpt_lookup: dict[str, tuple[int, int]] = {
            row["url"]: (int(row["width"]), int(row["height"]))
            for _, row in ckpt.iterrows()
            if pd.notna(row.get("url")) and pd.notna(row.get("width")) and pd.notna(row.get("height"))
        }
        log.info(
            "--from-checkpoint: loaded %d URL→dimension entries from %s.",
            len(ckpt_lookup), probe_checkpoint_path,
        )
        qualifying_urls = {
            url for url, (w, h) in ckpt_lookup.items()
            if min_side(w, h) >= min_required_side
        }
        log.info(
            "%d / %d checkpoint entries pass the >= %.0fpx shorter-side filter.",
            len(qualifying_urls), len(ckpt_lookup), min_required_side,
        )

        total = len(df)
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            if i % 200 == 0 or i == total:
                log.info("Progress: %d / %d rows", i, total)

            base_record = row.to_dict()
            url = pick_url(row)

            if url is None:
                discarded_rows.append({**base_record, "discard_reason": "no_url"})
                continue

            if url in already_kept_urls:
                continue

            if url not in ckpt_lookup:
                # Not in checkpoint — was never successfully probed (video, failed request,
                # non-image media_type, etc.).  Discard quietly without a network request.
                discarded_rows.append({**base_record, "discard_reason": "not_in_probe_checkpoint"})
                continue

            w, h = ckpt_lookup[url]
            if min_side(w, h) < min_required_side:
                discarded_rows.append({
                    **base_record,
                    "discard_reason": "below_min_resolution",
                    "probed_width": w,
                    "probed_height": h,
                })
                continue

            if url not in qualifying_urls:
                continue  # already handled above, belt-and-suspenders

            # Phase B: full download.
            ext = get_extension(url) or ".jpg"
            safe_name = f"{row.get('date', f'row{i}')}{ext}"
            dest_path = images_dir / safe_name
            full = full_download(url, session, dest_path)

            if not full.ok:
                discarded_rows.append({**base_record, "discard_reason": f"full_download_failed:{full.reason}"})
                time.sleep(args.sleep)
                continue

            if full.file_format in FORMATS_NEEDING_FULL_DOWNLOAD_FOR_ANIMATION_CHECK and (full.n_frames or 1) > 1:
                discarded_rows.append({
                    **base_record,
                    "discard_reason": "animated_gif",
                    "n_frames": full.n_frames,
                })
                dest_path.unlink(missing_ok=True)
                time.sleep(args.sleep)
                continue

            if min_side(full.width, full.height) < min_required_side:
                discarded_rows.append({
                    **base_record,
                    "discard_reason": "below_min_resolution_on_full_decode",
                    "probed_width": full.width,
                    "probed_height": full.height,
                })
                dest_path.unlink(missing_ok=True)
                time.sleep(args.sleep)
                continue

            kept_rows.append({
                **base_record,
                "source_url": url,
                "file_type": full.file_format,
                "width": full.width,
                "height": full.height,
                "n_frames": full.n_frames,
                "file_size_bytes": full.file_size_bytes,
                "local_path": str(dest_path),
            })
            time.sleep(args.sleep)

        # Write outputs and exit — skip the rest of the normal Phase A/B loop below.
        kept_df = pd.DataFrame(kept_rows)
        discarded_df = pd.DataFrame(discarded_rows)
        if args.resume and metadata_out.exists():
            prior = pd.read_csv(metadata_out)
            kept_df = pd.concat([prior, kept_df], ignore_index=True)
        kept_df.to_csv(metadata_out, index=False)
        discarded_df.to_csv(discarded_out, index=False)
        log.info("Done. Kept %d rows -> %s", len(kept_df), metadata_out)
        log.info("Discarded %d rows -> %s (see discard_reason column)", len(discarded_df), discarded_out)
        if not kept_df.empty:
            report_resolution_percentiles(kept_df[["width", "height"]].to_dict("records"))
        return

    # ---------------------------------------------------------------------------
    # Normal Phase A + Phase B path (no checkpoint available).
    # ---------------------------------------------------------------------------

    # --stats-only resume: load any existing probe checkpoint so already-probed URLs are skipped.
    already_probed_urls: set[str] = set()
    if args.stats_only and probe_checkpoint_path.exists():
        ckpt = pd.read_csv(probe_checkpoint_path)
        already_probed_urls = set(ckpt["url"].dropna())
        log.info(
            "Resuming stats-only run: %d URLs already in checkpoint %s will be skipped.",
            len(already_probed_urls), probe_checkpoint_path,
        )

    # Open the checkpoint CSV for append so every successful probe is persisted immediately.
    # If resuming, we append to the existing file; if starting fresh, we write a header first.
    _probe_ckpt_file = None
    _probe_ckpt_writer = None
    if args.stats_only:
        _probe_ckpt_exists = probe_checkpoint_path.exists() and len(already_probed_urls) > 0
        _probe_ckpt_file = open(probe_checkpoint_path, "a", newline="")
        import csv as _csv
        _probe_ckpt_writer = _csv.DictWriter(_probe_ckpt_file, fieldnames=["url", "width", "height"])
        if not _probe_ckpt_exists:
            _probe_ckpt_writer.writeheader()

    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        if i % 200 == 0 or i == total:
            log.info("Progress: %d / %d rows", i, total)

        base_record = row.to_dict()
        url = pick_url(row)

        if url is None:
            discarded_rows.append({**base_record, "discard_reason": "no_url"})
            continue

        media_type = str(row.get("media_type", "")).strip().lower()
        if media_type not in ("image", ""):
            discarded_rows.append({**base_record, "discard_reason": f"media_type_not_image:{media_type}"})
            continue

        # Fix for the known "media_type says image but URL is a video" issue.
        if is_video_url(url):
            discarded_rows.append({**base_record, "discard_reason": "url_extension_is_video"})
            continue

        if not is_plausible_image_url(url):
            discarded_rows.append({**base_record, "discard_reason": f"unrecognised_extension:{get_extension(url)}"})
            continue

        if url in already_kept_urls or url in already_probed_urls:
            continue

        probe = header_probe(url, session)
        if not probe.ok:
            discarded_rows.append({**base_record, "discard_reason": f"header_probe_failed:{probe.reason}"})
            time.sleep(args.sleep)
            continue

        # Write this probe result to the checkpoint immediately so it survives any interruption.
        if _probe_ckpt_writer is not None:
            _probe_ckpt_writer.writerow({"url": url, "width": probe.width, "height": probe.height})
            _probe_ckpt_file.flush()

        if min_side(probe.width, probe.height) < min_required_side:
            discarded_rows.append({
                **base_record,
                "discard_reason": "below_min_resolution",
                "probed_width": probe.width,
                "probed_height": probe.height,
            })
            time.sleep(args.sleep)
            continue

        if args.stats_only:
            # We only need the header dims for the percentile report; don't download.
            time.sleep(args.sleep)
            continue

        # Phase B: full download, needed for the exact dims + GIF-animation check + local cache.
        ext = get_extension(url) or ".jpg"
        safe_name = f"{row.get('date', f'row{i}')}{ext}"
        dest_path = images_dir / safe_name
        full = full_download(url, session, dest_path)

        if not full.ok:
            discarded_rows.append({**base_record, "discard_reason": f"full_download_failed:{full.reason}"})
            time.sleep(args.sleep)
            continue

        if full.file_format in FORMATS_NEEDING_FULL_DOWNLOAD_FOR_ANIMATION_CHECK and (full.n_frames or 1) > 1:
            discarded_rows.append({
                **base_record,
                "discard_reason": "animated_gif",
                "n_frames": full.n_frames,
            })
            dest_path.unlink(missing_ok=True)
            time.sleep(args.sleep)
            continue

        if min_side(full.width, full.height) < min_required_side:
            # Rare: header probe under-read a truncated/odd file; exact decode disagrees.
            discarded_rows.append({
                **base_record,
                "discard_reason": "below_min_resolution_on_full_decode",
                "probed_width": full.width,
                "probed_height": full.height,
            })
            dest_path.unlink(missing_ok=True)
            time.sleep(args.sleep)
            continue

        kept_rows.append({
            **base_record,
            "source_url": url,
            "file_type": full.file_format,
            "width": full.width,
            "height": full.height,
            "n_frames": full.n_frames,
            "file_size_bytes": full.file_size_bytes,
            "local_path": str(dest_path),
        })
        time.sleep(args.sleep)

    if args.stats_only:
        if _probe_ckpt_file is not None:
            _probe_ckpt_file.close()
        ckpt = pd.read_csv(probe_checkpoint_path)
        report_resolution_percentiles(ckpt[["width", "height"]].to_dict("records"))
        passing = int((ckpt[["width", "height"]].min(axis=1) >= min_required_side).sum())
        log.info(
            "Stats-only run complete. %d / %d probed rows clear the >= %.0fpx shorter-side bar "
            "(HR patch=%dpx, multiplier=%.1fx). Checkpoint saved to %s",
            passing, len(ckpt), min_required_side, hr_patch, args.min_dim_multiplier,
            probe_checkpoint_path,
        )
        return

    kept_df = pd.DataFrame(kept_rows)
    discarded_df = pd.DataFrame(discarded_rows)

    if args.resume and metadata_out.exists():
        prior = pd.read_csv(metadata_out)
        kept_df = pd.concat([prior, kept_df], ignore_index=True)

    kept_df.to_csv(metadata_out, index=False)
    discarded_df.to_csv(discarded_out, index=False)

    log.info("Done. Kept %d rows -> %s", len(kept_df), metadata_out)
    log.info("Discarded %d rows -> %s (see discard_reason column)", len(discarded_df), discarded_out)
    if not kept_df.empty:
        report_resolution_percentiles(kept_df[["width", "height"]].to_dict("records"))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="Path to the raw Kaggle APOD CSV.")
    p.add_argument("--output-dir", default="./apod_data", help="Directory for outputs and cached images.")
    p.add_argument("--metadata-out", default="apod_metadata_filtered.csv", help="Filename for the kept-rows metadata table.")
    p.add_argument("--discarded-out", default="apod_discarded.csv", help="Filename for the discarded-rows log.")
    p.add_argument("--hr-patch-size", type=int, default=512, help="HR crop side length in pixels (default 512; see module docstring).")
    p.add_argument("--downsample-factor", type=int, default=4, help="Fixed SR factor per the proposal (default 4).")
    p.add_argument("--min-dim-multiplier", type=float, default=2.0, help="Minimum shorter-side = multiplier x HR patch size (default 2.0).")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N rows (use for the --limit 200 test run).")
    p.add_argument("--retry", action="store_true", help="Re-attempt rows from apod_discarded.csv that failed due to 'not_in_probe_checkpoint' (re-probes from scratch) or 'full_download_failed' (skips probe, retries download only). Successful retries are appended to apod_metadata_filtered.csv and removed from apod_discarded.csv.")
    p.add_argument("--from-checkpoint", action="store_true", help="Skip the header-probe step entirely. Read image dimensions from the existing probe checkpoint (apod_probe_checkpoint.csv), filter to rows passing the resolution bar, and go straight to full download (Phase B). Much faster than re-probing when --stats-only has already been completed.")
    p.add_argument("--stats-only", action="store_true", help="Only run the cheap header-probe pass and print resolution percentiles; no downloads saved. Resumable: re-run with --stats-only --resume to continue where you left off.")
    p.add_argument("--report", action="store_true", help="Print percentile report from an existing probe checkpoint (apod_probe_checkpoint.csv) without making any network requests. Use this to read results from a partial or complete --stats-only run.")
    p.add_argument("--sleep", type=float, default=0.1, help="Delay (s) between requests, to be polite to apod.nasa.gov.")
    p.add_argument("--resume", action="store_true", help="Skip URLs already present in an existing metadata-out file.")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
