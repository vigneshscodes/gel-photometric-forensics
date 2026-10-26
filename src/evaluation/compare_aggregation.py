
#!/usr/bin/env python3
"""
Compare image-level aggregation methods without modifying the detector.

Methods:
    Q95  : 95th percentile of absolute block z-scores
    Q99  : 99th percentile of absolute block z-scores
    MAX  : Maximum absolute block z-score

Reads:
    outputs/results/benchmark_results.csv
    outputs/results/detector_outputs/<case_id>/block_stats.csv

Writes:
    outputs/results/aggregation_comparison.csv
    outputs/results/aggregation_threshold_metrics.csv

This is an evaluation-only experiment. It does not modify tau,
detector parameters, or detector source code.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Any

import numpy as np


LABELS = {"local_manipulated", "clean", "global_legitimate"}
THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]


def read_csv(path: str) -> list[dict[str, str]]:
    """Read a CSV with common Windows and UTF-8 encodings."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, "r", newline="", encoding=encoding) as file:
                return list(csv.DictReader(file))
        except UnicodeDecodeError:
            continue

    raise UnicodeError(f"Could not decode CSV: {path}")


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def get_block_scores(path: str) -> np.ndarray:
    """Load finite absolute z-scores from eligible blocks."""
    rows = read_csv(path)
    scores: list[float] = []

    if not rows or "abs_z_k" not in rows[0]:
        return np.array([], dtype=float)

    for row in rows:
        # Use eligible blocks only when the column is available.
        if "eligible" in row and not parse_bool(row["eligible"]):
            continue

        try:
            value = float(row["abs_z_k"])
        except (TypeError, ValueError):
            continue

        if math.isfinite(value):
            scores.append(abs(value))

    return np.asarray(scores, dtype=float)


def aggregate_scores(scores: np.ndarray) -> dict[str, float]:
    """Calculate the three candidate image-level scores."""
    return {
        "Q95": float(np.percentile(scores, 95)),
        "Q99": float(np.percentile(scores, 99)),
        "MAX": float(np.max(scores)),
    }


def classification_metrics(
    cases: list[dict[str, Any]],
    score_name: str,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate one aggregation method at one threshold."""
    tp = fp = tn = fn = skipped = 0

    for case in cases:
        label = case["image_level_label"]
        score = case.get(score_name)

        if label not in LABELS or score is None:
            skipped += 1
            continue

        if not math.isfinite(score):
            skipped += 1
            continue

        actual_positive = label == "local_manipulated"
        predicted_positive = score >= threshold

        if actual_positive and predicted_positive:
            tp += 1
        elif actual_positive and not predicted_positive:
            fn += 1
        elif not actual_positive and predicted_positive:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    fpr = fp / (fp + tn) if fp + tn else 0.0

    return {
        "method": score_name,
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "false_positive_rate": fpr,
        "skipped": skipped,
        "evaluated": tp + fp + tn + fn,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Q95, Q99, and MAX image-level aggregation."
    )
    parser.add_argument(
        "--results-dir",
        default="outputs/results",
        help="Existing benchmark results directory.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=THRESHOLDS,
        help="Thresholds to evaluate; default: 1.5 2 2.5 3 3.5 4 5.",
    )
    args = parser.parse_args()

    if not args.thresholds or any(
        not math.isfinite(t) for t in args.thresholds
    ):
        parser.error("Thresholds must be finite numbers.")

    results_dir = os.path.abspath(args.results_dir)
    results_csv = os.path.join(results_dir, "benchmark_results.csv")
    detector_dir = os.path.join(results_dir, "detector_outputs")

    result_rows = read_csv(results_csv)
    cases: list[dict[str, Any]] = []

    missing_block_files = 0
    invalid_scores = 0

    for row in result_rows:
        case_id = row.get("case_id", "").strip()
        label = row.get("image_level_label", "").strip()
        run_status = row.get("run_status", "").strip()

        case: dict[str, Any] = {
            "case_id": case_id,
            "run_status": run_status,
            "image_level_label": label,
            "existing_M_image": None,
            "Q95": None,
            "Q99": None,
            "MAX": None,
            "eligible_finite_blocks": 0,
            "aggregation_status": "",
        }

        try:
            existing = float(row.get("M_image", ""))
            if math.isfinite(existing):
                case["existing_M_image"] = existing
            else:
                invalid_scores += 1
        except (TypeError, ValueError):
            invalid_scores += 1

        if run_status != "ok":
            case["aggregation_status"] = "detector_run_not_ok"
            cases.append(case)
            continue

        block_path = os.path.join(
            detector_dir, case_id, "block_stats.csv"
        )

        if not os.path.isfile(block_path):
            missing_block_files += 1
            case["aggregation_status"] = "block_stats_missing"
            cases.append(case)
            continue

        scores = get_block_scores(block_path)

        if scores.size == 0:
            case["aggregation_status"] = "no_eligible_finite_scores"
            cases.append(case)
            continue

        values = aggregate_scores(scores)
        case.update(values)
        case["eligible_finite_blocks"] = int(scores.size)
        case["aggregation_status"] = "ok"
        cases.append(case)

    # Save scores for every case, including cases that could not be scored.
    per_case_path = os.path.join(
        results_dir, "aggregation_comparison.csv"
    )
    per_case_fields = [
        "case_id",
        "run_status",
        "image_level_label",
        "existing_M_image",
        "Q95",
        "Q99",
        "MAX",
        "eligible_finite_blocks",
        "aggregation_status",
    ]

    with open(
        per_case_path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=per_case_fields)
        writer.writeheader()
        writer.writerows(cases)

    # Evaluate all aggregation methods over the same thresholds.
    metric_rows: list[dict[str, Any]] = []

    for method in ("Q95", "Q99", "MAX"):
        for threshold in args.thresholds:
            metric_rows.append(
                classification_metrics(cases, method, threshold)
            )

    metrics_path = os.path.join(
        results_dir, "aggregation_threshold_metrics.csv"
    )
    metric_fields = [
        "method",
        "threshold",
        "TP",
        "FP",
        "TN",
        "FN",
        "precision",
        "recall",
        "F1",
        "false_positive_rate",
        "evaluated",
        "skipped",
    ]

    with open(
        metrics_path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    print("\n=== Aggregation threshold comparison ===")
    print(
        "Method | Threshold | TP | FP | TN | FN | "
        "Precision | Recall | F1 | FPR | Skipped"
    )

    for row in metric_rows:
        print(
            f"{row['method']:6} | "
            f"{row['threshold']:9.2f} | "
            f"{row['TP']:2} | {row['FP']:2} | "
            f"{row['TN']:2} | {row['FN']:2} | "
            f"{row['precision']:.3f}     | "
            f"{row['recall']:.3f}  | "
            f"{row['F1']:.3f} | "
            f"{row['false_positive_rate']:.3f} | "
            f"{row['skipped']}"
        )

    print("\n=== Data quality ===")
    print(f"Benchmark result rows: {len(result_rows)}")
    print(f"Missing block_stats.csv files: {missing_block_files}")
    print(f"Invalid existing M_image scores: {invalid_scores}")
    print(f"Saved per-case scores: {per_case_path}")
    print(f"Saved threshold metrics: {metrics_path}")

    print(
        "\nNote: These are exploratory synthetic-benchmark results. "
        "Thresholds are not calibrated, and the best observed result "
        "must not be treated as independent validation."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
