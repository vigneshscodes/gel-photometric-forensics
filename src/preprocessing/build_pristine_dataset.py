"""
build_pristine_dataset.py

Extracts Blot/Gel images that are BOTH:
  1. Labeled "Blot/Gel" in classification.json
  2. Confirmed pristine (part of the BioFors train split, per train_pdfs.pkl)
Then applies automated quality filtering (size, blank/uniform, fluorescent-color)
and writes a manifest CSV logging every accept/reject decision.
"""

import json
import pickle
import csv
import cv2
import numpy as np
from pathlib import Path

from src.config import DATASET_ROOT, OUTPUT_ROOT, ANNOTATION_DIR

CLASSIFICATION_FILE = ANNOTATION_DIR / "classification.json"
TRAIN_PDFS_FILE = ANNOTATION_DIR / "train_pdfs.pkl"

OUTPUT_DIR = OUTPUT_ROOT / "Pristine_Blot_Gel"
MANIFEST_PATH = OUTPUT_ROOT / "pristine_blot_gel_manifest.csv"

MIN_SIZE = 150          # shorter-side pixel threshold
STD_THRESH = 5.0        # below this = blank/near-uniform
COLOR_DIFF_THRESH = 10  # above this = likely fluorescent/color blot

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Load train_pdfs.pkl and normalize it to a set of folder IDs
# ---------------------------------------------------------------------------
def get_train_folder_ids(path: Path) -> set:
    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"[INFO] train_pdfs.pkl loaded — type: {type(data)}")

    # Defensive handling: the schema wasn't confirmed ahead of time,
    # so support the common shapes and print a sample either way.
    if isinstance(data, dict):
        sample_keys = list(data.keys())[:5]
        print(f"[INFO] Sample keys: {sample_keys}")
        return set(str(k) for k in data.keys())

    if isinstance(data, (list, set, tuple)):
        sample = list(data)[:5]
        print(f"[INFO] Sample entries: {sample}")
        return set(str(x) for x in data)

    raise ValueError(
        f"Unrecognized train_pdfs.pkl structure ({type(data)}). "
        f"Inspect manually with a quick pickle.load + print before rerunning."
    )


# ---------------------------------------------------------------------------
# Step 2: Quality filter for a single image
# ---------------------------------------------------------------------------
def is_usable(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return False, "unreadable"

    h, w = img.shape[:2]
    if min(h, w) < MIN_SIZE:
        return False, "too_small"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if gray.std() < STD_THRESH:
        return False, "blank_or_uniform"

    if img.ndim == 3:
        b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
        max_channel_diff = max(
            np.abs(b - g).mean(), np.abs(g - r).mean(), np.abs(b - r).mean()
        )
        if max_channel_diff > COLOR_DIFF_THRESH:
            return False, "likely_fluorescent_color"

    return True, "ok"


# ---------------------------------------------------------------------------
# Step 3: Main extraction + filtering loop
# ---------------------------------------------------------------------------
def main():
    train_folder_ids = get_train_folder_ids(TRAIN_PDFS_FILE)
    print(f"[INFO] {len(train_folder_ids)} train (pristine) folder IDs loaded.\n")

    with open(CLASSIFICATION_FILE, "r") as f:
        classification = json.load(f)

    manifest_rows = []
    copied, rejected_not_train, rejected_quality, missing = 0, 0, 0, 0

    for folder_id, images in classification.items():

        if folder_id not in train_folder_ids:
            rejected_not_train += len(
                [c for c in images.values() if c == "Blot/Gel"]
            )
            continue

        source_folder = DATASET_ROOT / folder_id
        if not source_folder.exists():
            print(f"[WARNING] Folder not found: {source_folder}")
            continue

        for image_name, category in images.items():
            if category != "Blot/Gel":
                continue

            source_image = source_folder / image_name

            if not source_image.exists():
                print(f"[WARNING] Missing image: {source_image}")
                missing += 1
                manifest_rows.append([folder_id, image_name, "missing_file"])
                continue

            usable, reason = is_usable(source_image)

            if not usable:
                rejected_quality += 1
                manifest_rows.append([folder_id, image_name, reason])
                continue

            destination_image = OUTPUT_DIR / f"{folder_id}_{image_name}"
            shutil = __import__("shutil")
            shutil.copy2(source_image, destination_image)
            copied += 1
            manifest_rows.append([folder_id, image_name, "accepted"])

    # Write manifest
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["folder_id", "image_name", "status"])
        writer.writerows(manifest_rows)

    print("\n========== PRISTINE DATASET BUILD COMPLETE ==========")
    print(f"Accepted (pristine, passed filters) : {copied}")
    print(f"Rejected — not in train split         : {rejected_not_train}")
    print(f"Rejected — failed quality filter      : {rejected_quality}")
    print(f"Missing files                          : {missing}")
    print(f"Saved images to                        : {OUTPUT_DIR}")
    print(f"Manifest written to                    : {MANIFEST_PATH}")


if __name__ == "__main__":
    main()