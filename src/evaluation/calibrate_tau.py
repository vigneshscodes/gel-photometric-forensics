#!/usr/bin/env python3
"""
calibrate_tau.py -- choose the decision threshold tau from NEGATIVE
(clean + legitimate-global) calibration cases only, for a stated target
false-positive rate. Calibration data must use different seeds than the
test benchmark (see README: calibration seed 5000, test seed 1000).

    python src/evaluation/generate_benchmark.py --output-dir out/cal --seed 5000 --mode calibration
    python src/evaluation/run_benchmark.py --benchmark-dir out/cal --output-dir out/cal_res
    python src/evaluation/calibrate_tau.py --results-dir out/cal_res --target-fpr 0.10
"""
import argparse, csv, json, math, os
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--target-fpr", type=float, default=0.10)
    ap.add_argument("--output", default=None, help="JSON path (default: <results-dir>/calibrated_tau.json)")
    a = ap.parse_args()

    with open(os.path.join(a.results_dir, "benchmark_results.csv"), newline="") as f:
        rows = list(csv.DictReader(f))

    def num(v):
        try:
            x = float(v)
            return x if math.isfinite(x) else None
        except (TypeError, ValueError):
            return None

    neg = [num(r.get("M_image")) for r in rows if r.get("run_status") == "ok" and r["image_level_label"] != "local_manipulated"]
    pos = [num(r.get("M_image")) for r in rows if r.get("run_status") == "ok" and r["image_level_label"] == "local_manipulated"]
    n_undefined = sum(v is None for v in neg + pos)
    neg = np.array([v for v in neg if v is not None])
    pos = np.array([v for v in pos if v is not None])
    if neg.size == 0:
        raise SystemExit("[error] no valid negative calibration scores")

    # Smallest tau such that the fraction of negatives with M >= tau is <= target.
    cand = np.unique(np.concatenate([neg, [neg.max() + 1e-6]]))
    tau = next(float(c) for c in cand if np.mean(neg >= c) <= a.target_fpr)
    out = {
        "tau": tau,
        "target_fpr": a.target_fpr,
        "calibration_fpr": float(np.mean(neg >= tau)),
        "calibration_recall_all_local": float(np.mean(pos >= tau)) if pos.size else None,
        "n_negatives": int(neg.size), "n_positives": int(pos.size),
        "n_undefined_scores_excluded": int(n_undefined),
        "note": "Chosen on calibration seeds only; evaluate on a disjoint test seed.",
    }
    path = a.output or os.path.join(a.results_dir, "calibrated_tau.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
