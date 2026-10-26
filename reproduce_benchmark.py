#!/usr/bin/env python3
"""
One-command, seeded, CPU-only reproduction of the synthetic benchmark.

  1. generate calibration set  (seeds 5000-5007)  -> choose tau on NEGATIVES only
  2. generate held-out test set (seeds 1000-1001) -> evaluate with the calibrated tau

    python reproduce_benchmark.py --work-dir benchmark_runs/repro

Outputs: <work-dir>/test_results/evaluation_metrics.json (+ per-case CSVs).
"""
import argparse, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
EV = os.path.join(ROOT, "src", "evaluation")


def run(*args):
    print("[run]", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=os.path.join(ROOT, "benchmark_runs", "repro"))
    ap.add_argument("--target-fpr", type=float, default=0.10)
    a = ap.parse_args()
    w = os.path.abspath(a.work_dir)
    cal, cal_res = os.path.join(w, "cal"), os.path.join(w, "cal_results")
    test, test_res = os.path.join(w, "test"), os.path.join(w, "test_results")

    run(os.path.join(EV, "generate_benchmark.py"), "--output-dir", cal, "--seed", "5000", "--mode", "calibration")
    run(os.path.join(EV, "run_benchmark.py"), "--benchmark-dir", cal, "--output-dir", cal_res)
    run(os.path.join(EV, "calibrate_tau.py"), "--results-dir", cal_res, "--target-fpr", str(a.target_fpr))
    tau = json.load(open(os.path.join(cal_res, "calibrated_tau.json")))["tau"]

    run(os.path.join(EV, "generate_benchmark.py"), "--output-dir", test, "--seed", "1000", "--mode", "sanity")
    run(os.path.join(EV, "run_benchmark.py"), "--benchmark-dir", test, "--output-dir", test_res, "--tau", repr(tau))
    run(os.path.join(EV, "evaluate_metrics.py"), "--benchmark-dir", test, "--results-dir", test_res)
    print(f"\n[done] calibrated tau={tau:.4f}; metrics: {os.path.join(test_res, 'evaluation_metrics.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
