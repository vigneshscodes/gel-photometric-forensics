#!/usr/bin/env python3
"""
generate_benchmark.py — builds the LGPC-BC v1.1 Section 16 synthetic
benchmark dataset: images, pixel-level ground-truth masks, image-level
labels, and exact transformation parameters for every case.

SCOPE (per current instructions)
---------------------------------
Only a SMALL "sanity" case set is implemented and runnable right now
(see build_sanity_case_list below) — NOT the full benchmark. A full-size
generator is intentionally not implemented yet; running one big case
list is out of scope for this step.

This script does not import from or alter src/detector/* in any way; it
only writes new synthetic images/labels to --output-dir.

CASE FAMILIES (methodology Section 16)
----------------------------------------
  1. clean                  - no edit at all
  2. global_brightness      - whole-image brightness shift (legitimate)
  3. global_contrast        - whole-image contrast scaling (legitimate)
  4. global_gamma           - whole-image gamma mapping (legitimate)
  5. global_monotonic_tone  - whole-image smooth monotonic LUT (legitimate)
  6. local_brightness       - localized brightness shift (tamper case)
  7. local_contrast         - localized contrast scaling (tamper case)
  8. local_gamma            - localized gamma mapping (tamper case)
  9. local_nonlinear_tone   - localized smooth monotonic LUT (tamper case)

For families 6-9, each is generated at two region sizes (one >= the
detector's block size B=16, one deliberately SMALLER than B=16 — the
"diluted small edit" case, methodology Section 14) and three placements
(background-only, band-interior-only, crossing a band edge).

OUTPUTS (per case, under --output-dir/cases/<case_id>/)
----------------------------------------------------------
  image.png            uint8 grayscale synthetic image
  mask.png             uint8 {0,255} pixel-level ground-truth manipulation mask
  label.json           image-level label + exact transform parameters

Plus a single top-level manifest:
  --output-dir/benchmark_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from evaluation.synthetic_generator import (
    generate_base_image,
    apply_transform,
    apply_local_transform,
    make_random_monotonic_lut,
    find_region_for_placement,
    make_region_mask,
)


@dataclass
class CaseRecord:
    case_id: str
    category: str
    subtype: str
    is_local: bool
    image_level_label: str  # "clean" | "global_legitimate" | "local_manipulated"
    expected_behavior: str  # documented expectation from the frozen methodology's own disclosed scope, NOT a performance claim
    base_seed: int
    transform_kind: Optional[str]
    transform_params_summary: str
    region: Optional[Dict[str, Any]]
    image_path: str
    mask_path: str
    label_path: str


LOCAL_SIZES = {
    "normal_24x24": (24, 24),  # >= block size B=16, and small enough to fit band interiors
    "small_8x8": (8, 8),  # deliberately smaller than one 16x16 block
}
LOCAL_PLACEMENTS = ["background", "band_interior", "crossing"]


def _save_case(
    output_dir: str,
    case_id: str,
    image: np.ndarray,
    mask: np.ndarray,
    label: Dict[str, Any],
) -> Dict[str, str]:
    case_dir = os.path.join(output_dir, "cases", case_id)
    os.makedirs(case_dir, exist_ok=True)
    image_path = os.path.join(case_dir, "image.png")
    mask_path = os.path.join(case_dir, "mask.png")
    label_path = os.path.join(case_dir, "label.json")

    cv2.imwrite(image_path, image)
    cv2.imwrite(mask_path, (mask.astype(np.uint8) * 255))
    with open(label_path, "w") as f:
        json.dump(label, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)

    return {"image_path": image_path, "mask_path": mask_path, "label_path": label_path}


def build_sanity_case_list(output_dir: str, base_seed: int = 1000) -> List[CaseRecord]:
    """
    Builds and WRITES a small (~40-case) sanity benchmark:
      - 2 base images (different seeds) x clean                     = 2
      - 4 global families x 2 deltas/factors/gammas/curves x 2 bases = 16
      - 4 local families x 2 sizes x 3 placements (1 base each)      = 24
    Total = 42 cases.
    """
    records: List[CaseRecord] = []

    base_seeds = [base_seed, base_seed + 1]
    bases = {s: generate_base_image(seed=s) for s in base_seeds}

    # --- 1. clean ---
    for s in base_seeds:
        base = bases[s]
        case_id = f"clean_seed{s}"
        label = {
            "case_id": case_id,
            "category": "clean",
            "subtype": "none",
            "is_local": False,
            "image_level_label": "clean",
            "expected_behavior": "expect_no_flag (no edit applied at all)",
            "base_seed": s,
            "base_image_params": base.params,
            "transform_kind": None,
            "transform_params": None,
            "region": None,
        }
        paths = _save_case(output_dir, case_id, base.image, np.zeros_like(base.band_mask), label)
        records.append(
            CaseRecord(
                case_id=case_id,
                category="clean",
                subtype="none",
                is_local=False,
                image_level_label="clean",
                expected_behavior=label["expected_behavior"],
                base_seed=s,
                transform_kind=None,
                transform_params_summary="",
                region=None,
                **paths,
            )
        )

    # --- 2-5. global (legitimate) edits ---
    global_specs = {
        "global_brightness": [{"delta": 25.0}, {"delta": -25.0}],
        "global_contrast": [{"factor": 1.4}, {"factor": 0.7}],
        "global_gamma": [{"gamma": 0.7}, {"gamma": 1.4}],
        "global_monotonic_tone": [{"lut_seed": 7001}, {"lut_seed": 7002}],
    }
    transform_kind_for_category = {
        "global_brightness": "brightness",
        "global_contrast": "contrast",
        "global_gamma": "gamma",
        "global_monotonic_tone": "nonlinear_tone",
    }

    for category, param_variants in global_specs.items():
        transform_kind = transform_kind_for_category[category]
        for s in base_seeds:
            base = bases[s]
            for variant_idx, raw_params in enumerate(param_variants):
                params = dict(raw_params)
                if transform_kind == "nonlinear_tone":
                    lut_rng = np.random.default_rng(params.pop("lut_seed"))
                    params["lut"] = make_random_monotonic_lut(lut_rng)

                edited = apply_transform(base.image, transform_kind, params)
                subtype = "_".join(f"{k}={v}" for k, v in raw_params.items())
                case_id = f"{category}_seed{s}_{subtype}"

                # Pixel-level mask for a global edit: every pixel's value
                # was altered by the mapping, so the "edited" mask is the
                # whole image. This is a whole-image transform, not a
                # local tamper, and is expected NOT to be flagged by a
                # method whose model assumes a single global phi
                # (methodology Section 3 / Assumption 1) — recorded via
                # expected_behavior below, not asserted as guaranteed.
                mask = np.ones_like(base.band_mask, dtype=bool)

                label = {
                    "case_id": case_id,
                    "category": category,
                    "subtype": subtype,
                    "is_local": False,
                    "image_level_label": "global_legitimate",
                    "expected_behavior": (
                        "expect_no_flag (single smooth monotonic global mapping; "
                        "matches the method's own 'legitimate global adjustment' "
                        "model, methodology Assumption 1) — not guaranteed, this "
                        "case exists to help measure the false-positive rate, "
                        "methodology Section 16"
                    ),
                    "base_seed": s,
                    "base_image_params": base.params,
                    "transform_kind": transform_kind,
                    "transform_params": {k: v for k, v in params.items() if k != "lut"},
                    "lut_present": "lut" in params,
                    "region": None,
                }
                paths = _save_case(output_dir, case_id, edited, mask, label)
                records.append(
                    CaseRecord(
                        case_id=case_id,
                        category=category,
                        subtype=subtype,
                        is_local=False,
                        image_level_label="global_legitimate",
                        expected_behavior=label["expected_behavior"],
                        base_seed=s,
                        transform_kind=transform_kind,
                        transform_params_summary=subtype,
                        region=None,
                        **paths,
                    )
                )

    # --- 6-9. local (tamper) edits ---
    local_specs = {
        "local_brightness": ("brightness", {"delta": 40.0}),
        "local_contrast": ("contrast", {"factor": 1.8}),
        "local_gamma": ("gamma", {"gamma": 0.4}),
        "local_nonlinear_tone": ("nonlinear_tone", {"lut_seed": 8001}),
    }

    region_rng = np.random.default_rng(base_seed + 999)  # controls region placement search only

    for category, (transform_kind, raw_params) in local_specs.items():
        s = base_seeds[0]  # one base image per local family keeps the sanity set small
        base = bases[s]
        for size_name, (box_h, box_w) in LOCAL_SIZES.items():
            for placement in LOCAL_PLACEMENTS:
                params = dict(raw_params)
                if transform_kind == "nonlinear_tone":
                    lut_rng = np.random.default_rng(params.pop("lut_seed"))
                    params["lut"] = make_random_monotonic_lut(lut_rng)

                r0, c0 = find_region_for_placement(
                    base.band_mask, box_h, box_w, placement, region_rng
                )
                region_mask = make_region_mask(base.height, base.width, r0, c0, box_h, box_w)
                edited = apply_local_transform(base.image, transform_kind, params, region_mask)

                subtype = f"{size_name}_{placement}"
                case_id = f"{category}_seed{s}_{subtype}"

                is_below_block = (box_h < 16) or (box_w < 16)
                if placement == "band_interior":
                    expected = (
                        "expect_undetectable (band-interior edit; ineligible by "
                        "construction, methodology Section 14/17)"
                    )
                elif is_below_block:
                    expected = (
                        "expect_likely_undetectable (region smaller than block size "
                        "B=16; diluted within the block's flat-pixel population, "
                        "methodology Section 14) — not guaranteed either way"
                    )
                else:
                    expected = (
                        "expect_possibly_detectable (background or crossing "
                        "placement, region >= block size) — not guaranteed; no "
                        "threshold has been calibrated yet"
                    )

                label = {
                    "case_id": case_id,
                    "category": category,
                    "subtype": subtype,
                    "is_local": True,
                    "image_level_label": "local_manipulated",
                    "expected_behavior": expected,
                    "base_seed": s,
                    "base_image_params": base.params,
                    "transform_kind": transform_kind,
                    "transform_params": {k: v for k, v in params.items() if k != "lut"},
                    "lut_present": "lut" in params,
                    "region": {
                        "row0": r0,
                        "col0": c0,
                        "height": box_h,
                        "width": box_w,
                        "placement": placement,
                        "smaller_than_block_size_16": is_below_block,
                    },
                }
                paths = _save_case(output_dir, case_id, edited, region_mask, label)
                records.append(
                    CaseRecord(
                        case_id=case_id,
                        category=category,
                        subtype=subtype,
                        is_local=True,
                        image_level_label="local_manipulated",
                        expected_behavior=expected,
                        base_seed=s,
                        transform_kind=transform_kind,
                        transform_params_summary=subtype,
                        region=label["region"],
                        **paths,
                    )
                )

    return records


def write_manifest(output_dir: str, records: List[CaseRecord]) -> str:
    manifest_path = os.path.join(output_dir, "benchmark_manifest.csv")
    fieldnames = [
        "case_id",
        "category",
        "subtype",
        "is_local",
        "image_level_label",
        "expected_behavior",
        "base_seed",
        "transform_kind",
        "transform_params_summary",
        "region_row0",
        "region_col0",
        "region_height",
        "region_width",
        "region_placement",
        "image_path",
        "mask_path",
        "label_path",
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = asdict(r)
            region = row.pop("region") or {}
            row["region_row0"] = region.get("row0", "")
            row["region_col0"] = region.get("col0", "")
            row["region_height"] = region.get("height", "")
            row["region_width"] = region.get("width", "")
            row["region_placement"] = region.get("placement", "")
            writer.writerow(row)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="Directory to write the benchmark dataset into.")
    parser.add_argument("--seed", type=int, default=1000, help="Base RNG seed (default: 1000).")
    parser.add_argument(
        "--mode",
        choices=["sanity"],
        default="sanity",
        help="Only 'sanity' (small, ~40-case) generation is implemented right now.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "sanity":
        records = build_sanity_case_list(args.output_dir, base_seed=args.seed)
    else:
        raise NotImplementedError("Only --mode sanity is implemented; the full benchmark is a later step.")

    manifest_path = write_manifest(args.output_dir, records)

    print(f"[done] generated {len(records)} synthetic cases.")
    print(f"[done] manifest: {manifest_path}")

    by_category: Dict[str, int] = {}
    for r in records:
        by_category[r.category] = by_category.get(r.category, 0) + 1
    for cat, n in sorted(by_category.items()):
        print(f"    {cat:24s}: {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())