"""
Dataset Curation Script for Electrical Connection Device Detection (ADAPTIVE)
=============================================================================
Filters an image dataset down to a target count of high-quality, diverse images
suitable for object detection labeling.

What's different from the previous version
------------------------------------------
The quality thresholds (sharpness, brightness, contrast, edge density) used to be
fixed numbers tuned for one camera. This version analyzes the dataset first,
then chooses thresholds based on the distribution of values it actually finds.
The hash-distance threshold is also chosen automatically to land the final
count close to your target range.

Pipeline
--------
  1. Compute quality metrics for every image (no filtering yet)
  2. Analyze metric distributions and choose adaptive thresholds
  3. Print a report so you can sanity-check the chosen thresholds
  4. Apply the filters
  5. Search for the hash distance that gives the best target-count match
  6. Deduplicate near-duplicate frames
  7. Temporal-diversity sampling if we're still over target
  8. Copy chosen files (only with --copy flag — default is dry run)

Usage
-----
  pip install opencv-python imagehash Pillow tqdm numpy

  # Dry run — see what it would do
  python curate_dataset.py --input_dir ./frames --output_dir ./curated \
      --target_min 50 --target_max 65

  # Actually copy files
  python curate_dataset.py --input_dir ./frames --output_dir ./curated \
      --target_min 50 --target_max 65 --copy

  # Manual override of any threshold (skips adaptation for that one)
  python curate_dataset.py --input_dir ./frames --output_dir ./curated \
      --target_min 50 --target_max 65 --sharpness_min 100 --copy

A note on the obstruction filter
--------------------------------
The HSV ranges inside compute_obstruction() were tuned for the MVP setup
(oak desk + simple terminal strip). They will mis-fire on different
backgrounds. The default OBSTRUCTION_THRESH = 1.0 effectively DISABLES this
filter. Only enable it if you've re-tuned the HSV ranges for your background.
"""

import os
import shutil
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Absolute floors and ceilings.
#   These are the limits the adaptive thresholds will not cross.
#   They protect against pathological datasets where percentile-based
#   thresholds alone would still pass garbage.
# ─────────────────────────────────────────────────────────────────────────────
SHARPNESS_FLOOR        = 50.0    # below this, image is unambiguously blurry
BRIGHTNESS_FLOOR       = 30.0    # below this, image is unambiguously too dark
BRIGHTNESS_CEILING     = 235.0   # above this, image is unambiguously blown out
CONTRAST_FLOOR         = 15.0    # below this, image is washed out / flat
EDGE_DENSITY_FLOOR     = 0.01    # below this, image is essentially empty

# Percentiles used for adaptive thresholds.
# We use low percentiles so we only reject clear outliers, not "the worst 25%".
QUALITY_LOW_PERCENTILE  = 5      # reject below the 5th percentile
QUALITY_HIGH_PERCENTILE = 95     # for "middle is best" metrics like brightness

# Obstruction filter — disabled by default; see docstring above.
OBSTRUCTION_THRESH_DEFAULT = 1.0

# Hash-distance search range. Smaller = stricter dedup = fewer frames remain.
HASH_DIST_SEARCH_VALUES = [3, 4, 5, 6, 7, 8, 10, 12, 14]


@dataclass
class ImageScore:
    """Holds metrics, filter result, and final score for one image."""
    path: str
    sharpness: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    edge_density: float = 0.0
    obstruction_frac: float = 0.0
    phash: str = ""
    passed_filters: bool = True
    reject_reason: str = ""
    composite_score: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation — these all just describe an image, no filtering yet
# ─────────────────────────────────────────────────────────────────────────────
def compute_sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian. Higher = sharper edges = sharper image."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness_contrast(gray: np.ndarray):
    """Mean and standard deviation of pixel values (0-255)."""
    return float(gray.mean()), float(gray.std())


def compute_edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels classified as edges by Canny detector."""
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.sum() / 255) / edges.size


def compute_obstruction(img_bgr: np.ndarray) -> float:
    """
    Estimate fraction of frame occupied by a 'foreign' uniform blob (e.g.
    an arm or torso intruding into frame). Heuristic — see docstring at top.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    board_mask = cv2.inRange(hsv, (0, 0, 120), (30, 60, 220))
    skin_mask = cv2.inRange(hsv, (0, 20, 70), (25, 200, 255))
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 60, 60))

    combined = cv2.bitwise_or(skin_mask, dark_mask)
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(board_mask))

    kernel = np.ones((30, 30), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    return float(combined.sum() / 255) / combined.size


def compute_phash(pil_img: Image.Image) -> str:
    """Perceptual hash — similar-looking images get similar hashes."""
    return str(imagehash.phash(pil_img, hash_size=8))


def hamming_distance(h1: str, h2: str) -> int:
    """Number of differing bits between two hex-encoded hashes."""
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def compute_metrics_for_image(img_path: str) -> ImageScore:
    """First pass: just describe the image. No filtering decisions yet."""
    s = ImageScore(path=img_path)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        s.passed_filters = False
        s.reject_reason = "unreadable"
        return s

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    s.sharpness = compute_sharpness(gray)
    s.brightness, s.contrast = compute_brightness_contrast(gray)
    s.edge_density = compute_edge_density(gray)
    s.obstruction_frac = compute_obstruction(img_bgr)

    try:
        pil = Image.open(img_path).convert("RGB")
        s.phash = compute_phash(pil)
    except Exception:
        s.phash = "0" * 16

    return s


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive threshold selection
# ─────────────────────────────────────────────────────────────────────────────
def derive_adaptive_thresholds(scores: list[ImageScore], manual_overrides: dict) -> dict:
    """
    Look at the metric distributions for this dataset and pick thresholds.

    Each adaptive threshold = max(absolute_floor, low_percentile_of_data).
    The floor protects against datasets where everything is bad. The
    percentile adapts to datasets where everything is good.

    manual_overrides : keys may include 'sharpness_min', 'brightness_min',
                       'brightness_max', 'contrast_min', 'edge_density_min',
                       'obstruction_thresh'. Any provided value bypasses
                       the adaptive logic for that threshold.
    """
    # Filter out unreadable images before computing percentiles
    readable = [s for s in scores if s.reject_reason != "unreadable"]
    if not readable:
        raise RuntimeError("No readable images in dataset.")

    sharps = np.array([s.sharpness for s in readable])
    brights = np.array([s.brightness for s in readable])
    contrasts = np.array([s.contrast for s in readable])
    edges = np.array([s.edge_density for s in readable])

    p_low = QUALITY_LOW_PERCENTILE
    p_high = QUALITY_HIGH_PERCENTILE

    thresholds = {
        "sharpness_min":      max(SHARPNESS_FLOOR,    np.percentile(sharps, p_low)),
        "brightness_min":     max(BRIGHTNESS_FLOOR,   np.percentile(brights, p_low)),
        "brightness_max":     min(BRIGHTNESS_CEILING, np.percentile(brights, p_high)),
        "contrast_min":       max(CONTRAST_FLOOR,     np.percentile(contrasts, p_low)),
        "edge_density_min":   max(EDGE_DENSITY_FLOOR, np.percentile(edges, p_low)),
        "obstruction_thresh": OBSTRUCTION_THRESH_DEFAULT,
    }

    # Manual overrides win over adaptive choices
    for key, val in manual_overrides.items():
        if val is not None and key in thresholds:
            thresholds[key] = float(val)

    return thresholds


def print_dataset_report(scores: list[ImageScore], thresholds: dict, overrides: dict):
    """Print a human-readable summary of metric distributions and thresholds."""
    readable = [s for s in scores if s.reject_reason != "unreadable"]
    sharps = np.array([s.sharpness for s in readable])
    brights = np.array([s.brightness for s in readable])
    contrasts = np.array([s.contrast for s in readable])
    edges = np.array([s.edge_density for s in readable])

    def line(name, arr):
        return (f"  {name:<14} min={arr.min():>9.2f}  "
                f"p10={np.percentile(arr, 10):>9.2f}  "
                f"p50={np.percentile(arr, 50):>9.2f}  "
                f"p90={np.percentile(arr, 90):>9.2f}  "
                f"max={arr.max():>9.2f}")

    print("\n── Metric distributions across this dataset ──")
    print(line("sharpness",    sharps))
    print(line("brightness",   brights))
    print(line("contrast",     contrasts))
    print(line("edge_density", edges))

    def mark(key):
        return "(manual)" if overrides.get(key) is not None else "(auto)"

    print("\n── Chosen thresholds ──")
    print(f"  sharpness_min      >= {thresholds['sharpness_min']:>8.2f}    {mark('sharpness_min')}")
    print(f"  brightness_min     >= {thresholds['brightness_min']:>8.2f}    {mark('brightness_min')}")
    print(f"  brightness_max     <= {thresholds['brightness_max']:>8.2f}    {mark('brightness_max')}")
    print(f"  contrast_min       >= {thresholds['contrast_min']:>8.2f}    {mark('contrast_min')}")
    print(f"  edge_density_min   >= {thresholds['edge_density_min']:>8.4f}    {mark('edge_density_min')}")
    print(f"  obstruction_thresh <= {thresholds['obstruction_thresh']:>8.2f}    "
          f"{mark('obstruction_thresh')}  (1.0 = disabled)")


# ─────────────────────────────────────────────────────────────────────────────
# Filtering and scoring
# ─────────────────────────────────────────────────────────────────────────────
def apply_filters(s: ImageScore, t: dict) -> ImageScore:
    """Decide whether one image passes the quality bar; set s.passed_filters."""
    if s.reject_reason == "unreadable":
        return s  # already rejected during metric computation

    if s.sharpness < t["sharpness_min"]:
        s.passed_filters = False
        s.reject_reason = f"blurry ({s.sharpness:.1f})"
    elif s.brightness < t["brightness_min"]:
        s.passed_filters = False
        s.reject_reason = f"too_dark ({s.brightness:.1f})"
    elif s.brightness > t["brightness_max"]:
        s.passed_filters = False
        s.reject_reason = f"overexposed ({s.brightness:.1f})"
    elif s.contrast < t["contrast_min"]:
        s.passed_filters = False
        s.reject_reason = f"low_contrast ({s.contrast:.1f})"
    elif s.edge_density < t["edge_density_min"]:
        s.passed_filters = False
        s.reject_reason = f"empty_frame ({s.edge_density:.4f})"
    elif s.obstruction_frac > t["obstruction_thresh"]:
        s.passed_filters = False
        s.reject_reason = f"obstruction ({s.obstruction_frac:.2%})"

    return s


def composite_score(s: ImageScore) -> float:
    """
    Higher = better. Used to choose which image survives in each
    near-duplicate cluster during deduplication.
    """
    sharpness_score   = min(s.sharpness / 500.0, 1.0)
    edge_score        = min(s.edge_density / 0.15, 1.0)
    obstruct_penalty  = s.obstruction_frac
    brightness_score  = 1.0 - abs(s.brightness - 130) / 130

    return (
        0.40 * sharpness_score
        + 0.25 * edge_score
        + 0.20 * brightness_score
        - 0.15 * obstruct_penalty
    )


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication and hash-distance search
# ─────────────────────────────────────────────────────────────────────────────
def deduplicate(scores: list[ImageScore], hash_dist: int) -> list[ImageScore]:
    """
    Greedy deduplication. Sorted by score descending so we always keep the
    best frame from each near-duplicate cluster.
    """
    scores = sorted(scores, key=lambda x: x.composite_score, reverse=True)
    kept = []
    kept_hashes = []

    for s in scores:
        is_dup = any(hamming_distance(s.phash, h) <= hash_dist for h in kept_hashes)
        if not is_dup:
            kept.append(s)
            kept_hashes.append(s.phash)

    return kept


def find_best_hash_distance(scores: list[ImageScore],
                            target_min: int, target_max: int,
                            manual_override: Optional[int] = None) -> int:
    """
    Try several hash distances and pick the one that lands closest to the
    target count range. Prefer SMALLER hash distances on ties (more diversity).

    If a manual override is provided, just return that.
    """
    if manual_override is not None:
        return manual_override

    target_mid = (target_min + target_max) / 2.0
    print("\n── Hash-distance search ──")
    print(f"  Target range: {target_min}–{target_max} (midpoint {target_mid:.0f})")

    best_dist = None
    best_score = float("inf")
    counts = {}

    for d in HASH_DIST_SEARCH_VALUES:
        n = len(deduplicate(scores, d))
        counts[d] = n

        if target_min <= n <= target_max:
            distance_from_mid = abs(n - target_mid)
        else:
            # Penalize misses outside the target range
            distance_from_mid = abs(n - target_mid) + 1000

        if distance_from_mid < best_score:
            best_score = distance_from_mid
            best_dist = d

    for d in HASH_DIST_SEARCH_VALUES:
        marker = " ← chosen" if d == best_dist else ""
        in_range = "  in range" if target_min <= counts[d] <= target_max else ""
        print(f"  hash_dist={d:>2}: {counts[d]:>4} unique frames{in_range}{marker}")

    return best_dist


# ─────────────────────────────────────────────────────────────────────────────
# Temporal sampling
# ─────────────────────────────────────────────────────────────────────────────
def temporal_diversity_sample(scores: list[ImageScore], target_max: int) -> list[ImageScore]:
    """
    If still over target_max after dedup, pick the highest-scoring frame from
    each of target_max evenly-spaced time buckets (sorted by filename).
    """
    scores = sorted(scores, key=lambda x: os.path.basename(x.path))
    n = len(scores)
    if n <= target_max:
        return scores

    bucket_size = n / target_max
    sampled = []
    for i in range(target_max):
        start = int(i * bucket_size)
        end = int((i + 1) * bucket_size)
        bucket = scores[start:end]
        if bucket:
            best = max(bucket, key=lambda x: x.composite_score)
            sampled.append(best)
    return sampled


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestration
# ─────────────────────────────────────────────────────────────────────────────
def run(input_dir: str, output_dir: str,
        target_min: int, target_max: int,
        copy_files: bool,
        report_path: str,
        manual_thresholds: dict,
        manual_hash_dist: Optional[int]):

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    all_images = [str(p) for p in sorted(input_path.iterdir())
                  if p.suffix.lower() in extensions]

    if not all_images:
        print(f"[ERROR] No images found in {input_path}")
        return

    print(f"\n{'=' * 64}")
    print(f"  Input directory : {input_path}")
    print(f"  Image count     : {len(all_images)}")
    print(f"  Target range    : {target_min}–{target_max}")
    print(f"{'=' * 64}")

    # ── Step 1: compute metrics for all images ──
    print("\nStep 1/5  Computing metrics for every image …")
    scored: list[ImageScore] = []
    for img_path in tqdm(all_images, desc="Scoring", unit="img"):
        scored.append(compute_metrics_for_image(img_path))

    # ── Step 2: derive adaptive thresholds and report ──
    print("\nStep 2/5  Analyzing distributions and choosing thresholds …")
    thresholds = derive_adaptive_thresholds(scored, manual_thresholds)
    print_dataset_report(scored, thresholds, manual_thresholds)

    # ── Step 3: apply quality filters ──
    print("\nStep 3/5  Applying quality filters …")
    for s in scored:
        apply_filters(s, thresholds)
        s.composite_score = composite_score(s)

    passed = [s for s in scored if s.passed_filters]
    rejected = [s for s in scored if not s.passed_filters]

    print(f"  Passed  : {len(passed):>5}")
    print(f"  Rejected: {len(rejected):>5}")
    if rejected:
        from collections import Counter
        reasons = Counter(r.reject_reason.split(" ")[0] for r in rejected)
        print("  Rejection breakdown:")
        for reason, count in reasons.most_common():
            print(f"    {reason:<14} {count:>4}")

    if len(passed) < target_min:
        print(f"\n⚠  Only {len(passed)} images survived quality filtering — "
              f"below target_min={target_min}.")
        print("   The dataset may be lower quality than the floors expect, or")
        print("   you may want to override one of the thresholds manually.")

    # ── Step 4: pick hash distance, then dedup ──
    print("\nStep 4/5  Deduplicating near-duplicate frames …")
    hash_dist = find_best_hash_distance(passed, target_min, target_max,
                                         manual_override=manual_hash_dist)
    unique = deduplicate(passed, hash_dist)
    print(f"\n  Using hash_dist = {hash_dist}")
    print(f"  After dedup    : {len(unique)} unique images")

    # ── Step 5: temporal sample if still over target ──
    print("\nStep 5/5  Final selection …")
    final_set = temporal_diversity_sample(unique, target_max)
    print(f"  Final count    : {len(final_set)}")

    # ── Output ──
    if copy_files:
        for s in tqdm(final_set, desc="Copying", unit="img"):
            dst = output_path / os.path.basename(s.path)
            shutil.copy2(s.path, dst)
        print(f"\n  Files copied to: {output_path}")
    else:
        print("\n  Dry-run mode — no files copied (use --copy to copy files)")

    # ── Save report ──
    report = {
        "input_count": len(all_images),
        "thresholds_used": thresholds,
        "hash_dist_used": hash_dist,
        "passed_filters": len(passed),
        "after_dedup": len(unique),
        "final_selected": len(final_set),
        "rejected": [asdict(s) for s in rejected],
        "selected": [asdict(s) for s in final_set],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Full report    : {report_path}")

    print(f"\n{'=' * 64}")
    print(f"  Done.  {len(final_set)} images selected.")
    print(f"{'=' * 64}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Curate an image dataset for object detection. "
                    "Thresholds adapt to the dataset by default; pass any "
                    "threshold flag to override."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Folder containing all raw images")
    parser.add_argument("--output_dir", required=True,
                        help="Folder to copy selected images into")
    parser.add_argument("--target_min", type=int, default=1000,
                        help="Minimum images to select (default: 1000)")
    parser.add_argument("--target_max", type=int, default=1500,
                        help="Maximum images to select (default: 1500)")
    parser.add_argument("--copy", action="store_true",
                        help="Actually copy files (omit for a dry run)")
    parser.add_argument("--report", default="curation_report.json",
                        help="Path for the JSON curation report")

    # Optional manual overrides — leave blank to let the script choose
    parser.add_argument("--sharpness_min",      type=float, default=None)
    parser.add_argument("--brightness_min",     type=float, default=None)
    parser.add_argument("--brightness_max",     type=float, default=None)
    parser.add_argument("--contrast_min",       type=float, default=None)
    parser.add_argument("--edge_density_min",   type=float, default=None)
    parser.add_argument("--obstruction_thresh", type=float, default=None)
    parser.add_argument("--hash_dist",          type=int,   default=None)

    args = parser.parse_args()

    manual_thresholds = {
        "sharpness_min":      args.sharpness_min,
        "brightness_min":     args.brightness_min,
        "brightness_max":     args.brightness_max,
        "contrast_min":       args.contrast_min,
        "edge_density_min":   args.edge_density_min,
        "obstruction_thresh": args.obstruction_thresh,
    }

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_min=args.target_min,
        target_max=args.target_max,
        copy_files=args.copy,
        report_path=args.report,
        manual_thresholds=manual_thresholds,
        manual_hash_dist=args.hash_dist,
    )
