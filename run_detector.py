#!/usr/bin/env python3
"""
CLI entry point for the LGPC-BC v1.1 detector.

Usage:
    python run_detector.py --input <image> --output <folder>
    python run_detector.py --input <directory> --output <folder>

See README.md / DETECTOR_README.md for parameter documentation and the
scope/limitations this detector does NOT claim to cover.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from detector.config import LGPCBCConfig, default_config
from detector.detector import run_on_image
from detector.preprocessing import InputValidationError

IMAGE_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg", ".pgm"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LGPC-BC v1.1: Local-Global Photometric Consistency (Background "
            "Consistency) detector for gel/blot images. All numeric "
            "parameters below have PROVISIONAL, UNCALIBRATED defaults — see "
            "src/detector/config.py. This tool does not claim empirical "
            "validation until run against the methodology's synthetic "
            "benchmark (Section 16)."
        )
    )
    parser.add_argument("--input", required=True, help="Path to an image file, or a directory of images for batch mode.")
    parser.add_argument("--output", required=True, help="Output directory (created if missing).")

    parser.add_argument("--block-size", type=int, help="B: analysis block side length in pixels.")
    parser.add_argument("--window-size", type=int, help="L: illumination-field median filter window (must be odd, >> B).")
    parser.add_argument("--grad-threshold", type=float, help="theta_grad: gradient magnitude threshold for flat classification.")
    parser.add_argument("--min-flat-pixels", type=int, help="n_min: minimum flat pixels for block eligibility.")
    parser.add_argument("--c-lo", type=int, help="Lower saturation bound.")
    parser.add_argument("--c-hi", type=int, help="Upper saturation bound.")
    parser.add_argument("--n-bins", type=int, help="N_bin: number of intensity bins for the dispersion curve.")
    parser.add_argument("--epsilon", type=float, help="epsilon_0: log-stabilizing constant.")
    parser.add_argument("--beta", type=float, help="Percentile (0-100) used to aggregate block z-scores into M_image.")
    parser.add_argument("--tau", type=float, help="Decision threshold on M_image. UNCALIBRATED placeholder if left default.")

    return parser


def config_from_args(args: argparse.Namespace) -> LGPCBCConfig:
    cfg = default_config()
    if args.block_size is not None:
        cfg.block_size = args.block_size
    if args.window_size is not None:
        cfg.window_size = args.window_size
    if args.grad_threshold is not None:
        cfg.grad_threshold = args.grad_threshold
    if args.min_flat_pixels is not None:
        cfg.min_flat_pixels = args.min_flat_pixels
    if args.c_lo is not None:
        cfg.c_lo = args.c_lo
    if args.c_hi is not None:
        cfg.c_hi = args.c_hi
    if args.n_bins is not None:
        cfg.n_bins = args.n_bins
    if args.epsilon is not None:
        cfg.epsilon_0 = args.epsilon
    if args.beta is not None:
        cfg.beta = args.beta
    if args.tau is not None:
        cfg.tau = args.tau
    cfg.validate()
    return cfg


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in IMAGE_EXTENSIONS


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        config = config_from_args(args)
    except ValueError as e:
        print(f"[config error] {e}", file=sys.stderr)
        return 2

    print("=" * 70)
    print("LGPC-BC v1.1 — active parameters (ALL are provisional/uncalibrated;")
    print("see src/detector/config.py docstring before trusting any result):")
    for k, v in config.as_dict().items():
        print(f"    {k:16s} = {v}")
    print("=" * 70)

    if not os.path.exists(args.input):
        print(f"[error] input path does not exist: {args.input}", file=sys.stderr)
        return 2

    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.input):
        return run_batch(args.input, args.output, config)
    else:
        return run_single(args.input, args.output, config)


def run_single(image_path: str, output_dir: str, config: LGPCBCConfig) -> int:
    try:
        result = run_on_image(image_path, output_dir, config)
    except InputValidationError as e:
        print(f"[input rejected] {image_path}: {e}", file=sys.stderr)
        return 1
    except Exception:
        print(f"[error] unexpected failure processing {image_path}:", file=sys.stderr)
        traceback.print_exc()
        return 1

    print(f"\nImage: {image_path}")
    print(f"  M_image (Q_{config.beta} of |z_k| over eligible blocks): {result.image_level_score}")
    print(f"  tau: {config.tau}")
    print(f"  Decision: {'FLAGGED as inconsistent' if result.decision_flagged else 'not flagged'}")
    print(f"  Outputs written to: {output_dir}")
    return 0


def run_batch(input_dir: str, output_root: str, config: LGPCBCConfig) -> int:
    image_paths = []
    for name in sorted(os.listdir(input_dir)):
        full = os.path.join(input_dir, name)
        if os.path.isfile(full) and is_image_file(full):
            image_paths.append(full)

    if not image_paths:
        print(f"[warning] no image files found in {input_dir} (extensions checked: {sorted(IMAGE_EXTENSIONS)})")
        return 1

    summary = []
    n_errors = 0
    for image_path in image_paths:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        per_image_output = os.path.join(output_root, stem)
        try:
            result = run_on_image(image_path, per_image_output, config)
            summary.append(
                {
                    "image": image_path,
                    "output_dir": per_image_output,
                    "M_image": result.image_level_score,
                    "flagged": result.decision_flagged,
                    "status": "ok",
                }
            )
            flag_str = "FLAGGED" if result.decision_flagged else "not flagged"
            print(f"[ok] {image_path} -> M_image={result.image_level_score:.4f} ({flag_str})")
        except InputValidationError as e:
            n_errors += 1
            summary.append({"image": image_path, "status": "rejected", "reason": str(e)})
            print(f"[rejected] {image_path}: {e}")
        except Exception as e:
            n_errors += 1
            summary.append({"image": image_path, "status": "error", "reason": str(e)})
            print(f"[error] {image_path}: {e}")
            traceback.print_exc()

    with open(os.path.join(output_root, "batch_summary.json"), "w") as f:
        json.dump(
            {
                "num_images": len(image_paths),
                "num_ok": len(image_paths) - n_errors,
                "num_errors_or_rejected": n_errors,
                "active_parameters": config.as_dict(),
                "results": summary,
            },
            f,
            indent=2,
        )

    print(f"\nBatch summary written to: {os.path.join(output_root, 'batch_summary.json')}")
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
