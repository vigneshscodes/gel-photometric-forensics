"""
LGPC-BC v1.1 top-level pipeline.

Orchestrates, in order, the frozen methodology's sections:
  1-2  preprocessing.py   : load + validate input, saturation mask
  4    background.py      : illumination field T, residual R
  5    regions.py          : gradient magnitude, flat mask, blocks, eligibility
  4-5  statistics.py       : mu_k, s_k, s_hat(mu) curve fit
  6    consistency.py      : r_k, z_k
  8    visualization.py    : heatmap, eligibility mask, suspicious mask
  9-10 consistency.py      : M_image, decision
  15   (this file)         : required diagnostics

This module does not introduce any new detection logic; it wires the
modules above together and writes the required output artifacts.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from typing import Optional

import cv2
import numpy as np

from .config import LGPCBCConfig
from .preprocessing import load_grayscale_image, compute_saturation_mask, InputValidationError
from .background import estimate_illumination_field, compute_illumination_corrected_residual
from .regions import (
    compute_gradient_magnitude,
    compute_flat_mask,
    partition_blocks,
    mark_eligibility,
)
from .statistics import compute_block_statistics, fit_dispersion_curve
from .consistency import compute_residuals_and_zscores, compute_image_level_score, decide
from .visualization import (
    build_raw_block_map,
    render_heatmap_png,
    build_eligibility_mask,
    build_suspicious_region_mask,
    plot_dispersion_curve,
)


class DetectorResult:
    """Container for everything one run produces, before writing to disk."""

    def __init__(self):
        self.diagnostics: dict = {}
        self.image_level_score: float = float("nan")
        self.decision_flagged: bool = False


def run_on_image(image_path: str, output_dir: str, config: LGPCBCConfig) -> DetectorResult:
    """
    Runs the full LGPC-BC v1.1 pipeline on a single image and writes all
    required output artifacts into output_dir. Returns a DetectorResult
    with the headline score/decision for convenience (e.g. batch summaries).
    """
    config.validate()
    os.makedirs(output_dir, exist_ok=True)

    # --- Sections 1-2: load + validate input, saturation mask ---
    loaded = load_grayscale_image(image_path)
    image = loaded.image
    height, width = loaded.height, loaded.width

    saturation_mask = compute_saturation_mask(image, config.c_lo, config.c_hi)

    # --- Section 4: illumination field + residual ---
    T = estimate_illumination_field(image, config.window_size)
    R = compute_illumination_corrected_residual(image, T)

    # --- Section 5: gradient magnitude, flat mask, blocks, eligibility ---
    G = compute_gradient_magnitude(image)
    flat_mask = compute_flat_mask(G, saturation_mask, config.grad_threshold)
    blocks = partition_blocks(height, width, config.block_size)
    mark_eligibility(blocks, flat_mask, config.min_flat_pixels)

    n_eligible = sum(1 for b in blocks if b.eligible)
    n_ineligible = len(blocks) - n_eligible

    # --- Sections 4-5: per-block statistics + global dispersion curve ---
    block_stats = compute_block_statistics(blocks, image, R, flat_mask)
    curve = fit_dispersion_curve(block_stats, config.n_bins)

    # --- Section 6: residual + robust z-score ---
    block_consistency = compute_residuals_and_zscores(block_stats, curve, config.epsilon_0)

    # goodness-of-fit diagnostic (Section 15, new in v1.1): spread of {r_k}
    valid_r = np.array(
        [bc.r_k for bc in block_consistency if bc.eligible and not np.isnan(bc.r_k)],
        dtype=np.float64,
    )
    from .statistics import median_absolute_deviation  # local import to avoid cycle at module load

    residual_mad_diag = median_absolute_deviation(valid_r) if valid_r.size > 0 else float("nan")

    # --- Sections 9-10: image-level score + decision ---
    m_image = compute_image_level_score(block_consistency, config.beta)
    flagged = decide(m_image, config.tau)

    # --- Section 8: heatmap + masks ---
    raw_map = build_raw_block_map(height, width, blocks, block_consistency)
    eligibility_mask_img = build_eligibility_mask(height, width, blocks)
    suspicious_mask_img = build_suspicious_region_mask(height, width, blocks, block_consistency, config.tau)

    # ---------------- write outputs ----------------
    _write_input_info(output_dir, loaded)
    cv2.imwrite(os.path.join(output_dir, "eligibility_mask.png"), eligibility_mask_img)
    cv2.imwrite(os.path.join(output_dir, "suspicious_region_mask.png"), suspicious_mask_img)
    np.save(os.path.join(output_dir, "raw_block_consistency_map.npy"), raw_map)
    render_heatmap_png(raw_map, os.path.join(output_dir, "normalized_heatmap.png"))
    plot_dispersion_curve(block_stats, curve, os.path.join(output_dir, "dispersion_curve.png"))
    _write_dispersion_curve_csv(output_dir, curve)
    _write_block_stats_csv(output_dir, block_stats, block_consistency)

    per_bin_counts = {
        f"bin_{i}_[{curve.bin_edges[i]:.1f},{curve.bin_edges[i+1]:.1f})": int(curve.bin_counts[i])
        for i in range(len(curve.bin_counts))
    }
    under_populated_bins = [k for k, v in per_bin_counts.items() if v == 0]

    diagnostics = {
        "image_path": image_path,
        "image_height": height,
        "image_width": width,
        "num_blocks_total": len(blocks),
        "num_blocks_eligible": n_eligible,
        "num_blocks_ineligible": n_ineligible,
        "fraction_blocks_eligible": n_eligible / len(blocks) if blocks else float("nan"),
        "fraction_blocks_ineligible": n_ineligible / len(blocks) if blocks else float("nan"),
        "note_ineligible_is_undefined_not_consistent": True,
        "eligible_blocks_per_intensity_bin": per_bin_counts,
        "under_populated_bins_zero_eligible_blocks": under_populated_bins,
        "dispersion_curve_populated_bin_centers": curve.interp_x.tolist(),
        "dispersion_curve_populated_bin_medians": curve.interp_y.tolist(),
        "residual_r_mad_goodness_of_fit": residual_mad_diag,
        "residual_r_mad_interpretation": (
            "Large values indicate the single-noise-level-function assumption "
            "(methodology Assumption 3) is likely untenable for this image; "
            "treat the heatmap as low-confidence, per methodology Section 15."
        ),
        "image_level_score_M_image": m_image,
        "image_level_score_definition": "Q_beta(|z_k|) over eligible blocks (see consistency.py ambiguity #7)",
        "beta_percentile_used": config.beta,
        "tau_threshold_used": config.tau,
        "decision_flagged_inconsistent": flagged,
        "decision_rule": "flagged iff M_image >= tau",
        "active_parameters": config.as_dict(),
        "all_parameters_are_uncalibrated_placeholders": True,
        "scope_limitations": [
            "Does not detect tonal edits confined to band interiors.",
            "Does not detect pure local brightness offsets with no dispersion change.",
            "Does not detect matched-noise-evasion manipulation.",
            "Does not detect edits smaller than one block (B x B).",
            "Not validated on composite/multi-acquisition images (out of scope).",
            "Not empirically validated at all yet; no benchmark has been run.",
        ],
    }
    with open(os.path.join(output_dir, "diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2, default=_json_default)

    with open(os.path.join(output_dir, "image_level_score.json"), "w") as f:
        json.dump(
            {
                "image_level_score_M_image": m_image,
                "tau_threshold_used": config.tau,
                "decision_flagged_inconsistent": flagged,
            },
            f,
            indent=2,
            default=_json_default,
        )

    result = DetectorResult()
    result.diagnostics = diagnostics
    result.image_level_score = m_image
    result.decision_flagged = flagged
    return result


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and np.isnan(o):
        return None
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def _write_input_info(output_dir: str, loaded) -> None:
    info = {
        "input_path": loaded.path,
        "height": loaded.height,
        "width": loaded.width,
        "dtype": "uint8",
        "channels": 1,
        "validated_single_channel_8bit_grayscale": True,
        "resized": False,
    }
    with open(os.path.join(output_dir, "input_info.json"), "w") as f:
        json.dump(info, f, indent=2)


def _write_dispersion_curve_csv(output_dir: str, curve) -> None:
    path = os.path.join(output_dir, "dispersion_curve.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_index", "bin_low", "bin_high", "bin_center", "bin_median_s", "bin_eligible_block_count"])
        for i in range(len(curve.bin_centers)):
            writer.writerow(
                [
                    i,
                    curve.bin_edges[i],
                    curve.bin_edges[i + 1],
                    curve.bin_centers[i],
                    "" if np.isnan(curve.bin_medians[i]) else curve.bin_medians[i],
                    int(curve.bin_counts[i]),
                ]
            )


def _write_block_stats_csv(output_dir: str, block_stats, block_consistency) -> None:
    path = os.path.join(output_dir, "block_stats.csv")
    by_index = {bc.block_index: bc for bc in block_consistency}
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "block_index",
                "row0",
                "col0",
                "row1",
                "col1",
                "eligible",
                "n_flat_pixels",
                "mu_k",
                "s_k",
                "s_hat_mu_k",
                "r_k",
                "z_k",
                "abs_z_k",
            ]
        )
        for s in block_stats:
            bc = by_index[s.block_index]
            writer.writerow(
                [
                    s.block_index,
                    s.row0,
                    s.col0,
                    s.row1,
                    s.col1,
                    s.eligible,
                    s.n_flat_pixels,
                    "" if np.isnan(s.mu_k) else s.mu_k,
                    "" if np.isnan(s.s_k) else s.s_k,
                    "" if np.isnan(bc.s_hat_mu_k) else bc.s_hat_mu_k,
                    "" if np.isnan(bc.r_k) else bc.r_k,
                    "" if np.isnan(bc.z_k) else bc.z_k,
                    "" if np.isnan(bc.z_k) else abs(bc.z_k),
                ]
            )
