"""
src/preprocessing/extract_pristine_blot_gel.py

Extracts pristine (unflagged) Blot/Gel images from BioFors, using the
project's config.py for paths and the CONFIRMED on-disk layout:

    DATASET_ROOT/
        1/
            010000.png
            010001.png
            ...
        2/
            ...
        ...
        1031/
            ...

(Verified from the user's own screenshot: BioFors folders are named 1..1031,
each containing that paper's panel images directly — this matches the layout
implied by BioFors's own dataloader/monet_dataloader.py, which loads images
as os.path.join(image_dir, doi, panel_filename).)

Definitions used here (all verified by loading the actual annotation JSONs):

- Category source: annotation_files/classification.json
    { doi(str): { filename.png: category(str) } }
    category in {"Blot/Gel", "Microscopy", "Macroscopy", "FACS"}
    Verified: 1031 papers, 47805 labeled images, 27440 labeled "Blot/Gel".

- "Pristine" = category == "Blot/Gel" AND (doi, filename) does not appear in
  ANY of the three ground-truth manipulation files:
    - edd_gt.json  (external duplication):  {doi: {"f1.png f2.png": [bbox,bbox]}}
    - idd_gt.json  (internal duplication):  {doi: {filename: [bbox,...]}}
    - cstd_gt.json (cut/sharp-transition):  {doi: {filename: [bbox,...]}}
  Verified: 876 of 27440 Blot/Gel images are flagged in at least one file,
  leaving 26564 pristine Blot/Gel images.

Sampling strategy (per prior discussion — do NOT just take all 26564 or a
naive first-N): pristine images are heavily imbalanced across papers
(verified: median 20 images/paper, max 158 in a single paper, 12 papers with
only 1). Taking them unfiltered would bias your "normal" population toward
whichever few papers/labs happen to have the most panels. This script caps
the number of images taken per paper (--per-paper-cap, default 4) and then
randomly samples across papers (--seed for reproducibility) until it reaches
--target-total (default 250), so the final set is diverse across labs/
imaging setups rather than deep within a few.

Output layout (flat, not folder-per-paper — provenance is kept in the
manifest CSV instead, which is all you actually need doi-tracking for):

    OUTPUT_ROOT/pristine_blot_gel/
        <doi>__<filename>.png
        ...
        manifest.csv   (doi, orig_filename, dest_filename, source_path)
"""

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

# Project config — adjust this import if config.py lives somewhere else
# in your project (e.g. `from src.config import ...`).
from src.config import DATASET_ROOT, OUTPUT_ROOT, ANNOTATION_DIR


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
    """(doi, filename) pairs that appear in ANY ground-truth manipulation
    annotation file — i.e. images that are NOT pristine."""
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


def get_pristine_blot_gel_by_paper(cls, flagged):
    """Returns {doi: [filename, filename, ...]} for pristine Blot/Gel images,
    grouped by paper so we can cap/sample per paper afterwards."""
    by_paper = {}
    for doi, panels in cls.items():
        pristine_fns = [
            fn for fn, category in panels.items()
            if category == "Blot/Gel" and (doi, fn) not in flagged
        ]
        if pristine_fns:
            by_paper[doi] = pristine_fns
    return by_paper


def sample_capped(by_paper, per_paper_cap, target_total, seed):
    """Cap per paper, then randomly sample across papers until target_total
    is reached (or we run out). Returns list of (doi, filename)."""
    rng = random.Random(seed)

    # Cap within each paper first (choose randomly if a paper has more than the cap)
    capped = {}
    for doi, fns in by_paper.items():
        if len(fns) > per_paper_cap:
            capped[doi] = rng.sample(fns, per_paper_cap)
        else:
            capped[doi] = list(fns)

    pool = [(doi, fn) for doi, fns in capped.items() for fn in fns]
    rng.shuffle(pool)

    if target_total is not None and len(pool) > target_total:
        pool = pool[:target_total]

    return pool


def find_source_path(dataset_root: Path, doi: str, filename: str):
    """Primary layout only, per the confirmed folder structure: dataset_root/doi/filename"""
    candidate = dataset_root / doi / filename
    return candidate if candidate.is_file() else None


def main():
    parser = argparse.ArgumentParser(
        description="Extract a diverse, capped set of pristine Blot/Gel images from BioFors."
    )
    parser.add_argument(
        "--per-paper-cap",
        type=int,
        default=4,
        help="Max images to take from any single paper (avoids lab/paper bias). Default: 4.",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=250,
        help="Target number of images in the final extracted set. Default: 250. "
        "Pass 0 to extract everything (no cap on total, only per-paper cap still applies).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be copied; do not actually copy files.",
    )
    args = parser.parse_args()

    annotation_dir = Path(ANNOTATION_DIR)
    dataset_root = Path(DATASET_ROOT)
    output_dir = Path(OUTPUT_ROOT) / "pristine_blot_gel"

    if not annotation_dir.is_dir():
        sys.exit(f"ERROR: ANNOTATION_DIR not found: {annotation_dir}")
    if not dataset_root.is_dir():
        sys.exit(f"ERROR: DATASET_ROOT not found: {dataset_root}")

    cls, edd_gt, idd_gt, cstd_gt = load_annotations(annotation_dir)
    flagged = build_flagged_set(edd_gt, idd_gt, cstd_gt)
    by_paper = get_pristine_blot_gel_by_paper(cls, flagged)

    total_blot_gel = sum(
        1 for panels in cls.values() for cat in panels.values() if cat == "Blot/Gel"
    )
    total_pristine = sum(len(fns) for fns in by_paper.values())

    print(f"Total Blot/Gel images (classification.json):      {total_blot_gel}")
    print(f"Flagged (non-pristine) Blot/Gel images:            {total_blot_gel - total_pristine}")
    print(f"Pristine Blot/Gel images available:                {total_pristine}")
    print(f"Papers contributing pristine Blot/Gel images:      {len(by_paper)}")

    target_total = None if args.target_total == 0 else args.target_total
    selected = sample_capped(by_paper, args.per_paper_cap, target_total, args.seed)
    print(f"Selected after per-paper cap ({args.per_paper_cap}) and target-total ({args.target_total}): {len(selected)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    found_count = 0
    missing_count = 0
    missing_examples = []

    with open(manifest_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["doi", "orig_filename", "dest_filename", "source_path"])

        for doi, filename in selected:
            src = find_source_path(dataset_root, doi, filename)

            if src is None:
                missing_count += 1
                if len(missing_examples) < 10:
                    missing_examples.append((doi, filename))
                continue

            dest_filename = f"{doi}__{filename}"
            dest_path = output_dir / dest_filename

            if not args.dry_run:
                shutil.copy2(src, dest_path)

            writer.writerow([doi, filename, dest_filename, str(src)])
            found_count += 1

    print(f"\nCopied:  {found_count}")
    print(f"Missing: {missing_count}")
    if missing_examples:
        print("Example missing (doi, filename) pairs (source not found):")
        for doi, fn in missing_examples:
            print(f"  doi={doi}  filename={fn}  expected_path={dataset_root / doi / fn}")

    print(f"\nOutput dir: {output_dir}")
    print(f"Manifest:   {manifest_path}")
    if args.dry_run:
        print("(--dry-run was set: no files were actually copied)")


if __name__ == "__main__":
    main()