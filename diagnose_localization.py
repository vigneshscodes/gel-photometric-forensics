
from pathlib import Path
import csv

import cv2
import numpy as np

root = Path("outputs")
case_id = "local_contrast_seed1000_normal_24x24_background"

stats_path = root / "results" / "detector_outputs" / case_id / "block_stats.csv"
mask_path = root / "benchmark" / "cases" / case_id / "mask.png"

mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if mask_img is None:
    raise FileNotFoundError(f"Could not read mask: {mask_path}")

mask = mask_img > 0
edited_scores = []
unedited_scores = []

with open(stats_path, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    required = {"row0", "col0", "row1", "col1", "eligible", "abs_z_k"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError("block_stats.csv is missing required columns.")

    for row in reader:
        if row["eligible"].strip().lower() != "true":
            continue

        try:
            score = float(row["abs_z_k"])
            r0, c0 = int(row["row0"]), int(row["col0"])
            r1, c1 = int(row["row1"]), int(row["col1"])
        except (ValueError, TypeError):
            continue

        if not np.isfinite(score):
            continue

        block_mask = mask[r0:r1, c0:c1]
        if block_mask.size == 0:
            continue

        edited_fraction = float(np.mean(block_mask))

        if edited_fraction > 0:
            edited_scores.append(score)
        else:
            unedited_scores.append(score)


def report(name, scores):
    if not scores:
        print(f"{name}: no valid blocks")
        return

    a = np.asarray(scores)
    print(
        f"{name}: n={len(a)}, "
        f"median={np.median(a):.4f}, "
        f"Q90={np.percentile(a, 90):.4f}, "
        f"max={np.max(a):.4f}, "
        f"fraction_ge_3.5={np.mean(a >= 3.5):.4f}"
    )


print(case_id)
report("Edited blocks", edited_scores)
report("Unedited blocks", unedited_scores)
