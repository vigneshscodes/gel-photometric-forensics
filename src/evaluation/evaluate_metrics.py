
#!/usr/bin/env python3
"""
evaluate_metrics.py

Evaluate image-level detection and pixel-level localization using the
existing synthetic benchmark and frozen detector outputs.

Image-level positive class:
    local_manipulated

Image-level negative classes:
    clean, global_legitimate

Localization:
    IoU is calculated only for local-manipulation cases with readable
    ground-truth masks and compatible raw consistency maps.

The localization threshold is an evaluation parameter, not a calibrated
or validated optimum.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Optional

import cv2
import numpy as np


def parse_bool(value: Any) -> Optional[bool]:
    """Convert common CSV boolean values to bool."""
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False

    return None


def parse_float(value: Any) -> Optional[float]:
    """Convert a value to a finite float, or return None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None



def read_csv(path: str) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries with encoding fallback."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required CSV not found: {path}")

    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, "r", newline="", encoding=encoding) as file:
                return list(csv.DictReader(file))
        except UnicodeDecodeError:
            continue

    raise UnicodeError(f"Could not decode CSV: {path}")



def resolve_path(path: str, *base_dirs: str) -> Optional[str]:
    """Resolve an absolute or relative path against candidate directories."""
    if not path:
        return None

    if os.path.isabs(path) and os.path.isfile(path):
        return path

    for base_dir in base_dirs:
        candidate = os.path.abspath(os.path.join(base_dir, path))
        if os.path.isfile(candidate):
            return candidate

    return None


def safe_ratio(numerator: int, denominator: int) -> float:
    """Return zero when a metric's denominator is zero."""
    return numerator / denominator if denominator else 0.0


def calculate_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict[str, Any]:
    """Calculate binary classification metrics."""
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))

    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    fpr = safe_ratio(fp, fp + tn)

    return {
        "n_evaluated": len(y_true),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "false_positive_rate": fpr,
        "note": (
            "Precision, recall, F1, and FPR use local_manipulated as "
            "the positive class. Zero is returned when a denominator "
            "is zero."
        ),
    }


def calculate_iou(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> Optional[float]:
    """Calculate binary intersection over union."""
    if ground_truth.shape != prediction.shape:
        return None

    intersection = int(np.count_nonzero(ground_truth & prediction))
    union = int(np.count_nonzero(ground_truth | prediction))

    if union == 0:
        return 1.0

    return intersection / union


def evaluate_localization(
    case_id: str,
    mask_path: Optional[str],
    raw_map_path: str,
    threshold: float,
) -> dict[str, Any]:
    """Calculate localization IoU for one case."""
    result: dict[str, Any] = {
        "localization_iou": None,
        "localization_status": "",
        "localization_threshold": threshold,
    }

    if not mask_path:
        result["localization_status"] = "ground_truth_mask_not_found"
        return result

    if not os.path.isfile(raw_map_path):
        result["localization_status"] = "raw_map_not_found"
        return result

    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        result["localization_status"] = "ground_truth_mask_unreadable"
        return result

    try:
        raw_map = np.load(raw_map_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        result["localization_status"] = (
            f"raw_map_unreadable: {type(error).__name__}"
        )
        return result

    if raw_map.ndim != 2:
        result["localization_status"] = "raw_map_not_2d"
        return result

    ground_truth = mask_image > 0

    if raw_map.shape != ground_truth.shape:
        result["localization_status"] = "shape_mismatch"
        return result

    # The map stores block-level consistency scores. Use absolute scores
    # and apply the explicitly supplied localization threshold.
    finite = np.isfinite(raw_map)
    prediction = finite & (np.abs(raw_map) >= threshold)

    # Do not treat unavailable map pixels as predicted positives.
    iou = calculate_iou(ground_truth, prediction)

    result["localization_iou"] = iou
    result["localization_status"] = "ok"
    result["ground_truth_pixels"] = int(np.count_nonzero(ground_truth))
    result["predicted_pixels"] = int(np.count_nonzero(prediction))
    result["finite_map_fraction"] = float(np.mean(finite))

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate classification metrics and localization IoU."
    )
    parser.add_argument(
        "--benchmark-dir",
        required=True,
        help="Directory containing benchmark_manifest.csv.",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing benchmark_results.csv and detector_outputs/.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (defaults to results-dir).",
    )
    parser.add_argument(
        "--localization-threshold",
        type=float,
        default=3.5,
        help=(
            "Threshold applied to absolute raw consistency-map values "
            "for localization IoU. Default: 3.5; not automatically calibrated."
        ),
    )
    args = parser.parse_args()

    if not math.isfinite(args.localization_threshold) or (
        args.localization_threshold < 0
    ):
        parser.error("--localization-threshold must be finite and non-negative.")

    benchmark_dir = os.path.abspath(args.benchmark_dir)
    results_dir = os.path.abspath(args.results_dir)
    output_dir = os.path.abspath(args.output_dir or results_dir)
    os.makedirs(output_dir, exist_ok=True)

    manifest_path = os.path.join(benchmark_dir, "benchmark_manifest.csv")
    results_path = os.path.join(results_dir, "benchmark_results.csv")

    manifest_rows = read_csv(manifest_path)
    result_rows = read_csv(results_path)

    manifest_by_id = {
        row["case_id"]: row
        for row in manifest_rows
        if row.get("case_id")
    }

    per_case: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    localization_ious: list[float] = []
    localization_skipped = 0

    for row in result_rows:
        case_id = row.get("case_id", "")
        manifest = manifest_by_id.get(case_id, {})

        status = row.get("run_status", "")
        label = row.get("image_level_label") or manifest.get(
            "image_level_label", ""
        )
        is_local = parse_bool(
            row.get("is_local") or manifest.get("is_local", "")
        )
        flagged = parse_bool(row.get("decision_flagged"))

        case_result: dict[str, Any] = {
            "case_id": case_id,
            "run_status": status,
            "image_level_label": label,
            "is_local": is_local,
            "decision_flagged": flagged,
            "M_image": parse_float(row.get("M_image")),
            "image_ground_truth_positive": None,
            "image_prediction_positive": None,
            "localization_iou": None,
            "localization_status": "not_applicable",
        }

        # Classification metrics use completed detector runs with a
        # recognized ground-truth label and a valid detector decision.
        if (
            status == "ok"
            and label in {"clean", "global_legitimate", "local_manipulated"}
            and flagged is not None
        ):
            truth = int(label == "local_manipulated")
            prediction = int(flagged)

            y_true.append(truth)
            y_pred.append(prediction)

            case_result["image_ground_truth_positive"] = truth
            case_result["image_prediction_positive"] = prediction

        # IoU is meaningful for the known local-edit region, not for
        # all-pixel masks used to describe global tonal transformations.
        if status == "ok" and (
            is_local is True or label == "local_manipulated"
        ):
            mask_path = resolve_path(
                manifest.get("mask_path", ""),
                benchmark_dir,
                os.getcwd(),
            )

            raw_map_path = os.path.join(
                results_dir,
                "detector_outputs",
                case_id,
                "raw_block_consistency_map.npy",
            )

            localization = evaluate_localization(
                case_id=case_id,
                mask_path=mask_path,
                raw_map_path=raw_map_path,
                threshold=args.localization_threshold,
            )
            case_result.update(localization)

            if localization["localization_iou"] is not None:
                localization_ious.append(
                    localization["localization_iou"]
                )
            else:
                localization_skipped += 1

        per_case.append(case_result)

    classification = calculate_classification_metrics(y_true, y_pred)

    localization_summary = {
        "n_localization_cases_with_iou": len(localization_ious),
        "n_localization_cases_skipped": localization_skipped,
        "mean_iou": (
            float(np.mean(localization_ious))
            if localization_ious
            else None
        ),
        "median_iou": (
            float(np.median(localization_ious))
            if localization_ious
            else None
        ),
        "threshold": args.localization_threshold,
        "note": (
            "IoU is calculated only for local-manipulation cases with "
            "usable ground-truth masks and compatible raw maps. The "
            "threshold is an evaluation setting, not a calibrated optimum."
        ),
    }

    summary = {
        "benchmark_dir": benchmark_dir,
        "results_dir": results_dir,
        "classification_metrics": classification,
        "localization_metrics": localization_summary,
        "limitations": [
            "Results reflect the supplied benchmark and detector outputs only.",
            "Synthetic benchmark metrics do not establish performance on real scientific images.",
            "Image-level decisions use the detector's existing decision_flagged output.",
            "Localization IoU uses a separate fixed threshold on the raw map.",
            "No detector parameters are tuned or modified by this script.",
        ],
    }

    per_case_path = os.path.join(output_dir, "per_case_metrics.csv")
    if per_case:
        fieldnames: list[str] = []
        for case in per_case:
            for key in case:
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(per_case_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_case)

    summary_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)

    print("\n=== Image-level classification metrics ===")
    for key, value in classification.items():
        print(f"{key}: {value}")

    print("\n=== Local-edit localization metrics ===")
    for key, value in localization_summary.items():
        print(f"{key}: {value}")

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {per_case_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
