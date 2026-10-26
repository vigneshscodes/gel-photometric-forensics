
#!/usr/bin/env python3
"""Exploratory spatial-coherence evaluation for LGPC-BC.

Does not modify the detector or tune its parameters.
Requires: numpy, pandas, opencv-python.
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import cv2
import numpy as np


def read_csv(path):
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, newline="", encoding=encoding) as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Cannot decode {path}")


def truth_label(row):
    label = row.get("image_level_label", "").strip()
    if label in ("local_manipulated", "clean", "global_legitimate"):
        return int(label == "local_manipulated")
    return None


def metrics(y_true, y_pred):
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    fpr = fp / (fp + tn) if fp + tn else 0.0

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": precision, "recall": recall,
        "F1": f1, "false_positive_rate": fpr,
    }


def get_scores(stats_path):
    rows = read_csv(stats_path)
    scores = []

    for row in rows:
        if row.get("eligible", "").strip().lower() != "true":
            continue
        try:
            z = abs(float(row["abs_z_k"]))
            r0 = int(row["row0"])
            c0 = int(row["col0"])
            r1 = int(row["row1"])
            c1 = int(row["col1"])
        except (ValueError, KeyError, TypeError):
            continue

        if math.isfinite(z) and r1 > r0 and c1 > c0:
            scores.append((r0, c0, r1, c1, z))

    return scores


def make_score_map(blocks, height, width):
    """Fill each block's pixel area with its absolute z-score."""
    score_map = np.full((height, width), np.nan, dtype=np.float32)

    for r0, c0, r1, c1, score in blocks:
        r0, c0 = max(0, r0), max(0, c0)
        r1, c1 = min(height, r1), min(width, c1)
        if r1 > r0 and c1 > c0:
            score_map[r0:r1, c0:c1] = score

    return score_map


def connected_region_score(score_map, block_threshold, min_blocks):
    """Return largest connected high-score block count and pixel mask."""
    finite = np.isfinite(score_map)
    high = finite & (score_map >= block_threshold)

    # Connected components are computed on the block grid, not on pixels.
    # Recover one score per block-sized cell using the first finite score.
    block_h = 16
    block_w = 16
    rows = math.ceil(score_map.shape[0] / block_h)
    cols = math.ceil(score_map.shape[1] / block_w)

    grid = np.zeros((rows, cols), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            patch = score_map[
                r * block_h:min((r + 1) * block_h, score_map.shape[0]),
                c * block_w:min((c + 1) * block_w, score_map.shape[1]),
            ]
            vals = patch[np.isfinite(patch)]
            if vals.size and float(np.max(vals)) >= block_threshold:
                grid[r, c] = 1

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        grid, connectivity=8
    )

    component_sizes = [
        int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)
    ]
    largest = max(component_sizes, default=0)

    # Only return regions when the largest component reaches min_blocks.
    pixel_mask = np.zeros(score_map.shape, dtype=bool)
    if largest >= min_blocks:
        largest_label = 1 + int(np.argmax(component_sizes))
        component_grid = labels == largest_label

        for r, c in zip(*np.where(component_grid)):
            r0, c0 = r * block_h, c * block_w
            r1 = min(r0 + block_h, score_map.shape[0])
            c1 = min(c0 + block_w, score_map.shape[1])
            pixel_mask[r0:r1, c0:c1] = True

    return largest, pixel_mask


def calculate_iou(gt, pred):
    union = np.count_nonzero(gt | pred)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(gt & pred) / union)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", default="outputs/benchmark")
    parser.add_argument("--results-dir", default="outputs/results")
    parser.add_argument("--block-threshold", type=float, default=2.0)
    parser.add_argument("--min-blocks", type=int, default=2)
    args = parser.parse_args()

    benchmark_dir = os.path.abspath(args.benchmark_dir)
    results_dir = os.path.abspath(args.results_dir)
    manifest_path = os.path.join(benchmark_dir, "benchmark_manifest.csv")
    result_path = os.path.join(results_dir, "benchmark_results.csv")
    output_path = os.path.join(results_dir, "spatial_coherence_results.csv")

    manifest = {
        r.get("case_id", ""): r
        for r in read_csv(manifest_path)
    }
    results = read_csv(result_path)

    per_case = []
    y_true, y_baseline, y_spatial = [], [], []
    ious = []

    for row in results:
        case_id = row.get("case_id", "")
        label = row.get("image_level_label", "")
        actual = truth_label(row)

        item = {
            "case_id": case_id,
            "label": label,
            "baseline_M_image": row.get("M_image", ""),
            "spatial_largest_component_blocks": None,
            "spatial_flagged": None,
            "localization_iou": None,
            "status": "",
        }

        if row.get("run_status") != "ok" or actual is None:
            item["status"] = "invalid_run_or_label"
            per_case.append(item)
            continue

        stats_path = os.path.join(
            results_dir, "detector_outputs", case_id, "block_stats.csv"
        )
        if not os.path.isfile(stats_path):
            item["status"] = "block_stats_missing"
            per_case.append(item)
            continue

        case_dir = os.path.join(benchmark_dir, "cases", case_id)
        mask_path = os.path.join(case_dir, "mask.png")
        image_path = os.path.join(case_dir, "image.png")

        # Determine dimensions from the ground-truth mask.
        gt_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_img is None:
            item["status"] = "ground_truth_mask_missing"
            per_case.append(item)
            continue

        blocks = get_scores(stats_path)
        score_map = make_score_map(blocks, *gt_img.shape)
        largest, predicted_mask = connected_region_score(
            score_map, args.block_threshold, args.min_blocks
        )

        spatial_flagged = largest >= args.min_blocks
        item["spatial_largest_component_blocks"] = largest
        item["spatial_flagged"] = spatial_flagged
        item["status"] = "ok"

        if label == "local_manipulated":
            gt = gt_img > 0
            iou = calculate_iou(gt, predicted_mask)
            item["localization_iou"] = iou
            ious.append(iou)

        # Use only valid finite baseline scores for fair comparison.
        try:
            baseline_score = float(row["M_image"])
            baseline_valid = math.isfinite(baseline_score)
        except (ValueError, TypeError, KeyError):
            baseline_valid = False
            baseline_score = float("nan")

        if baseline_valid:
            y_true.append(actual)
            y_baseline.append(int(baseline_score >= 3.5))
            y_spatial.append(int(spatial_flagged))

        per_case.append(item)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fields = list(per_case[0].keys()) if per_case else []
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_case)

    print("\n=== Image-level comparison (same finite baseline cases) ===")
    print("Baseline Q95 at threshold 3.5:", metrics(y_true, y_baseline))
    print("Spatial coherence:", metrics(y_true, y_spatial))
    print("\n=== Localization ===")
    print("Cases evaluated:", len(ious))
    print("Mean IoU:", float(np.mean(ious)) if ious else "N/A")
    print("Median IoU:", float(np.median(ious)) if ious else "N/A")
    print("\nSaved:", output_path)
    print(
        "\nExploratory result only. The block threshold and minimum component "
        "size are uncalibrated; this experiment does not establish validity."
    )


if __name__ == "__main__":
    main()
