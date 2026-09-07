"""
01_build_and_filter.py

Single-pass, single-source-of-truth replacement for:
    extract_pristine_blot_gel.py + build_pristine_dataset.py
    + extract_bolt_gel.py + filter_pristine_candidates.py (simplified)

WHY those four are being replaced (see README.md "A. What was wrong" for
the full audit) — short version:
  - Two of your scripts defined "pristine" two DIFFERENT, CONTRADICTORY ways
    (ground-truth-flag exclusion vs. BioFors's train/test split, which is
    not a manipulation label at all) and wrote to folder names that only
    differ by case ("pristine_blot_gel" vs "Pristine_Blot_Gel"), so on
    Windows/Mac they silently collide and on Linux they silently diverge.
  - build_pristine_dataset.py never checks edd_gt/idd_gt/cstd_gt at all,
    so its "pristine" set can contain manipulated images — a correctness
    bug that directly undermines a tonal-edit-detection paper.
  - filter_pristine_candidates.py auto-inverts pixel intensities for
    "polarity correction" — i.e. it silently rewrites the photometric
    signal of your clean/negative-class images, which is exactly the kind
    of thing a paper about photometric consistency cannot tolerate hiding.
  - extract_bolt_gel.py writes flat filenames with no doi prefix -> images
    with the same on-disk name in different papers overwrite each other.

This script does ONE thing, simply:
  1. Read classification.json -> every (doi, filename) labeled "Blot/Gel".
  2. Read edd_gt.json / idd_gt.json / cstd_gt.json -> mark GT-flagged pairs.
  3. For every Blot/Gel image NOT GT-flagged, run three cheap, unmodifying
     quality checks (too small / blank / colored). No local-patch analysis,
     no seam detection, no auto-inversion of pixels. Every image is a
     byte-identical copy of the source file, or it is not copied at all.
  4. Copy each image into exactly one place and log every decision:
        clean_candidates/            (passed everything)
        rejected/gt_flagged/         (in edd_gt/idd_gt/cstd_gt)
        rejected/quality_<reason>/   (too_small, blank, colored, unreadable)
     and record it as one row of manifest.csv. Nothing is ever skipped
     without a row explaining why.

Run 03_split_train_test.py afterwards to turn clean_candidates/ into the
final train/test split.
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

from PIL import Image
import numpy as np

from src.config import DATASET_ROOT, OUTPUT_ROOT, ANNOTATION_DIR

# ---- simple, documented thresholds (tune here if you must; keep it to one place) ----
MIN_SHORT_SIDE = 40      # px; below this an image can't carry useful lane structure
BLANK_STD_THRESH = 5.0   # grayscale intensity std-dev below this = near-uniform/blank
COLOR_SAT_THRESH = 12.0  # mean HSV saturation (0-255) above this = not effectively grayscale


def load_annotations(annotation_dir: Path):
    with open(annotation_dir / "classification.json") as f:
        cls = json.load(f)
    with open(annotation_dir / "edd_gt.json") as f:
        edd_gt = json.load(f)
    with open(annotation_dir / "idd_gt.json") as f:
        idd_gt = json.load(f)
    with open(annotation_dir / "cstd_gt.json") as f:
        cstd_gt = json.load(f)
    return cls, edd_gt, idd_gt, cstd_gt


def build_flagged_set(edd_gt, idd_gt, cstd_gt):
    """(doi, filename) pairs appearing in ANY manipulation ground-truth file."""
    flagged = set()
    for doi, pairs in edd_gt.items():
        for pair_key in pairs.keys():
            f1, f2 = pair_key.split()
            flagged.add((doi, f1))
            flagged.add((doi, f2))
    for doi, panels in idd_gt.items():
        for fn in panels.keys():
            flagged.add((doi, fn))
    for doi, panels in cstd_gt.items():
        for fn in panels.keys():
            flagged.add((doi, fn))
    return flagged


def quality_check(path: Path):
    """Returns (passed: bool, reason: str). Never modifies pixel data."""
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        return False, "unreadable"

    w, h = img.size
    if min(w, h) < MIN_SHORT_SIDE:
        return False, "too_small"

    gray = np.array(img.convert("L"), dtype=np.float64)
    if gray.std() < BLANK_STD_THRESH:
        return False, "blank"

    hsv = np.array(img.convert("HSV"))
    mean_sat = float(hsv[:, :, 1].mean())
    if mean_sat > COLOR_SAT_THRESH:
        return False, "colored"

    return True, "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Write manifest.csv only, copy nothing.")
    args = ap.parse_args()

    dataset_root = Path(DATASET_ROOT)
    annotation_dir = Path(ANNOTATION_DIR)
    out_root = Path(OUTPUT_ROOT)

    clean_dir = out_root / "clean_candidates"
    rejected_dir = out_root / "rejected"
    manifest_path = out_root / "manifest.csv"

    if not dataset_root.is_dir():
        sys.exit(f"ERROR: DATASET_ROOT not found: {dataset_root}")
    if not annotation_dir.is_dir():
        sys.exit(f"ERROR: ANNOTATION_DIR not found: {annotation_dir}")

    cls, edd_gt, idd_gt, cstd_gt = load_annotations(annotation_dir)
    flagged = build_flagged_set(edd_gt, idd_gt, cstd_gt)

    if not args.dry_run:
        clean_dir.mkdir(parents=True, exist_ok=True)
        rejected_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    counts = {"accepted": 0, "gt_flagged": 0, "too_small": 0, "blank": 0,
              "colored": 0, "unreadable": 0, "missing_file": 0}

    for doi, panels in cls.items():
        for filename, category in panels.items():
            if category != "Blot/Gel":
                continue

            src = dataset_root / doi / filename
            dest_name = f"{doi}__{filename}"

            if (doi, filename) in flagged:
                status, reason = "rejected", "gt_flagged"
            elif not src.is_file():
                status, reason = "rejected", "missing_file"
            else:
                passed, qreason = quality_check(src)
                status = "accepted" if passed else "rejected"
                reason = "ok" if passed else qreason

            counts[reason if status == "rejected" else "accepted"] += 1

            if status == "accepted":
                dest = clean_dir / dest_name
            elif reason == "missing_file":
                dest = None
            else:
                sub = rejected_dir / reason
                if not args.dry_run:
                    sub.mkdir(parents=True, exist_ok=True)
                dest = sub / dest_name

            if dest is not None and not args.dry_run and src.is_file():
                shutil.copy2(src, dest)

            rows.append([
                doi, filename, str(src), category, status, reason,
                str(dest) if dest else "",
            ])

    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doi", "filename", "source_path", "category",
                    "status", "reason", "dest_path"])
        w.writerows(rows)

    total = len(rows)
    print("========== STAGE 1: build_and_filter ==========")
    print(f"Total Blot/Gel images (classification.json): {total}")
    for k in ["accepted", "gt_flagged", "too_small", "blank", "colored", "unreadable", "missing_file"]:
        print(f"  {k:12s}: {counts.get(k, 0)}")
    print(f"\nclean_candidates/: {clean_dir}")
    print(f"rejected/:         {rejected_dir}")
    print(f"manifest.csv:      {manifest_path}")
    if args.dry_run:
        print("(--dry-run: no files copied)")


if __name__ == "__main__":
    main()