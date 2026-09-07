"""
02_split_train_test.py

Reads manifest.csv produced by 01_build_and_filter.py, takes every row with
status == "accepted", and splits by PAPER (doi) — not by image — into
train/test. This is the leakage-safe split: two panels from the same paper
share lighting/scanner/compression characteristics, so an image-level random
split would let near-duplicate context leak between train and test.

Copies accepted images into:
    final_clean_dataset/train/<doi>__<filename>
    final_clean_dataset/test/<doi>__<filename>

and rewrites manifest.csv in place, adding a "split" column for every row
(accepted rows get "train"/"test"; rejected rows get "" — they were never
candidates for a split, and this is recorded, not silently omitted).
"""

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path
from collections import defaultdict

from src.config import OUTPUT_ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-frac", type=float, default=0.2,
                     help="Fraction of PAPERS assigned to the test split. Default: 0.2.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = ap.parse_args()

    out_root = Path(OUTPUT_ROOT)
    manifest_path = out_root / "manifest.csv"
    clean_dir = out_root / "clean_candidates"
    final_dir = out_root / "final_clean_dataset"
    train_dir = final_dir / "train"
    test_dir = final_dir / "test"

    if not manifest_path.is_file():
        sys.exit(f"ERROR: manifest.csv not found at {manifest_path} — run 01_build_and_filter.py first.")

    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if "split" not in fieldnames:
        fieldnames = fieldnames + ["split"]

    accepted_by_doi = defaultdict(list)
    for row in rows:
        row.setdefault("split", "")
        if row["status"] == "accepted":
            accepted_by_doi[row["doi"]].append(row)

    dois = sorted(accepted_by_doi.keys())
    rng = random.Random(args.seed)
    rng.shuffle(dois)

    n_test_papers = max(1, round(len(dois) * args.test_frac)) if dois else 0
    test_dois = set(dois[:n_test_papers])
    train_dois = set(dois[n_test_papers:])

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    n_train_imgs = n_test_imgs = 0
    for row in rows:
        if row["status"] != "accepted":
            continue
        doi = row["doi"]
        split = "test" if doi in test_dois else "train"
        row["split"] = split

        dest_name = f"{doi}__{row['filename']}"
        src = Path(row["dest_path"])  # copy from clean_candidates/, not the raw dataset again
        dest = (test_dir if split == "test" else train_dir) / dest_name

        if not src.is_file():
            sys.exit(f"ERROR: expected accepted file missing from clean_candidates/: {src}")
        shutil.copy2(src, dest)
        row["dest_path"] = str(dest)

        if split == "train":
            n_train_imgs += 1
        else:
            n_test_imgs += 1

    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("========== STAGE 2: split_train_test ==========")
    print(f"Papers  -> train: {len(train_dois)}  test: {len(test_dois)}  total: {len(dois)}")
    print(f"Images  -> train: {n_train_imgs}  test: {n_test_imgs}  total: {n_train_imgs + n_test_imgs}")
    print(f"\nfinal_clean_dataset/train/: {train_dir}")
    print(f"final_clean_dataset/test/:  {test_dir}")
    print(f"manifest.csv updated:       {manifest_path}")


if __name__ == "__main__":
    main()