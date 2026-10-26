"""
LGPC-BC v1.1 top-level pipeline.

Orchestrates, in order, the frozen methodology's sections:
  1-2  preprocessing.py   : load + validate input, saturation mask
  4    background.py      : illumination field T, residual R
  5    regions.py         : gradient magnitude, flat mask, blocks, eligibility
  4-5  statistics.py      : mu_k, s_k, s_hat(mu) curve fit
  6    consistency.py      : r_k, z_k
  8    visualization.py    : heatmap, eligibility mask, suspicious mask
  9-10 consistency.py     : M_image, decision
  15   (this file)         : required diagnostics

This module does not introduce any new detection logic; it wires the
modules above together and writes the required output artifacts.
"""

from __future__ import annotations

import csv
import json
import os

import cv2
import numpy as np

from .config import LGPCBCConfig

from .preprocessing import (
    load_grayscale_image,
    compute_saturation_mask,
)

from .background import (
    estimate_illumination_field,
    compute_illumination_corrected_residual,
)

from .regions import (
    compute_gradient_magnitude,
    compute_flat_mask,
    compute_adaptive_grad_threshold,
    partition_blocks,
    mark_eligibility,
)

from .statistics import (
    compute_block_statistics,
    fit_dispersion_curve,
    median_absolute_deviation,
)

from .consistency import (
    compute_residuals_and_zscores,
    compute_image_level_score,
    decide,
    estimate_residual_scale,
)

from .features import (
    to_grid,
    pool_signed,
    robust_z,
    level_channel,
    histogram_roughness,
    hist_residuals,
    LOG_SCALE_FLOOR,
    LEVEL_SCALE_FLOOR,
)

from .consistency import BlockConsistency

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


def run_on_image(
    image_path: str,
    output_dir: str,
    config: LGPCBCConfig,
) -> DetectorResult:
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

    saturation_mask = compute_saturation_mask(
        image,
        config.c_lo,
        config.c_hi,
    )

    # --- Section 4: illumination field + residual ---
    T = estimate_illumination_field(
        image,
        config.window_size,
    )

    R = compute_illumination_corrected_residual(
        image,
        T,
    )

    # --- Section 5: gradient magnitude, flat mask, blocks, eligibility ---
    G = compute_gradient_magnitude(image, config.grad_smooth_sigma)

    # theta_grad: manual override if set, otherwise derived from this
    # image's own non-saturated gradient-magnitude distribution.
    if config.grad_threshold is not None:
        theta_grad = config.grad_threshold
        theta_grad_source = "manual_override"
    else:
        theta_grad = compute_adaptive_grad_threshold(
            G,
            saturation_mask,
            config.grad_percentile,
        )
        theta_grad_source = (
            f"adaptive_percentile_{config.grad_percentile}"
        )

    flat_mask = compute_flat_mask(
        G,
        saturation_mask,
        theta_grad,
    )

    blocks = partition_blocks(
        height,
        width,
        config.block_size,
    )

    mark_eligibility(
        blocks,
        flat_mask,
        config.min_flat_pixels,
    )

    n_eligible = sum(
        1
        for b in blocks
        if b.eligible
    )

    n_ineligible = len(blocks) - n_eligible

    # Zero eligible blocks means M_image is mathematically undefined.
    if n_eligible == 0:
        raise RuntimeError(
            f"Zero eligible blocks for '{image_path}' with theta_grad="
            f"{theta_grad} ({theta_grad_source}), min_flat_pixels="
            f"{config.min_flat_pixels}. M_image would be mathematically "
            "undefined, not 'no manipulation detected' -- refusing to "
            "produce a silently-empty result."
        )

    # --- Sections 4-5: per-block statistics + global dispersion curve ---
    block_stats = compute_block_statistics(
        blocks,
        image,
        R,
        flat_mask,
        config.dispersion_estimator,
    )

    curve = fit_dispersion_curve(
        block_stats,
        config.n_bins,
        config.bin_mode,
    )

    # --- Section 6: residual + robust z-score ---
    block_consistency = compute_residuals_and_zscores(
        block_stats,
        curve,
        config.epsilon_0,
    )

    # --- v1.2: complementary channels fused into one block score ---
    # Dispersion alone cannot see additive brightness shifts and is noisy
    # for mild contrast/gamma edits. Each channel is a local-vs-global
    # residual against a model fitted on this same image; the fused score
    # is the max over channels of max(|z_raw|, |z_pooled|). The fused score
    # replaces |z_k| everywhere downstream (M_image, heatmap, mask, CSV).
    extra_cols = {}
    if config.use_extra_channels:
        block_consistency, extra_cols, channel_diag, loc_consistency = _fuse_channels(
            blocks, block_stats, block_consistency, image, flat_mask, G,
            theta_grad, (height, width), config,
        )
    else:
        channel_diag = {"channels": ["dispersion"]}
        loc_consistency = None

    # Goodness-of-fit diagnostic (Section 15):
    # spread of finite residuals {r_k}.
    valid_r = np.array(
        [
            bc.r_k
            for bc in block_consistency
            if bc.eligible and np.isfinite(bc.r_k)
        ],
        dtype=np.float64,
    )

    residual_mad_diag = (
        median_absolute_deviation(valid_r)
        if valid_r.size > 0
        else float("nan")
    )

    residual_scale, residual_scale_source = estimate_residual_scale(valid_r)

    # --- Sections 9-10: image-level score + decision ---
    m_image = compute_image_level_score(
        block_consistency,
        config.beta,
    )

    flagged = decide(
        m_image,
        config.tau,
    )

    # --- Section 8: heatmap + masks ---
    raw_map = build_raw_block_map(
        height,
        width,
        blocks,
        block_consistency,
    )

    eligibility_mask_img = build_eligibility_mask(
        height,
        width,
        blocks,
    )

    suspicious_mask_img = build_suspicious_region_mask(
        height,
        width,
        blocks,
        block_consistency,
        config.tau,
        config.loc_ratio if config.use_extra_channels else 1.0,
        loc_consistency,
    )

    # Localization fallback:
    # If the image is flagged but strict seeds produce an empty mask,
    # recover spatially coherent anomalous blocks using a lower threshold.
    if flagged and not np.any(suspicious_mask_img):
        scores = np.array([
            abs(bc.z_k) if bc.eligible and np.isfinite(bc.z_k) else np.nan
            for bc in block_consistency
        ])

        finite_scores = scores[np.isfinite(scores)]

        if finite_scores.size:
            fallback_tau = max(
                config.loc_ratio * config.tau,
                float(np.percentile(finite_scores, 95))
            )

            fallback_consistency = [
                BlockConsistency(
                    block_index=bc.block_index,
                    eligible=bc.eligible,
                    mu_k=bc.mu_k,
                    s_k=bc.s_k,
                    s_hat_mu_k=bc.s_hat_mu_k,
                    r_k=bc.r_k,
                    z_k=(
                        float(scores[i])
                        if np.isfinite(scores[i])
                        else float("nan")
                    ),
                )
                for i, bc in enumerate(block_consistency)
            ]

            suspicious_mask_img = build_suspicious_region_mask(
                height,
                width,
                blocks,
                fallback_consistency,
                fallback_tau,
                config.loc_ratio,
                loc_consistency,
            )

    # ---------------- write outputs ----------------

    _write_input_info(
        output_dir,
        loaded,
    )

    cv2.imwrite(
        os.path.join(output_dir, "eligibility_mask.png"),
        eligibility_mask_img,
    )

    cv2.imwrite(
        os.path.join(output_dir, "suspicious_region_mask.png"),
        suspicious_mask_img,
    )

    np.save(
        os.path.join(output_dir, "raw_block_consistency_map.npy"),
        raw_map,
    )

    render_heatmap_png(
        raw_map,
        os.path.join(output_dir, "normalized_heatmap.png"),
    )

    plot_dispersion_curve(
        block_stats,
        curve,
        os.path.join(output_dir, "dispersion_curve.png"),
    )

    _write_dispersion_curve_csv(
        output_dir,
        curve,
    )

    _write_block_stats_csv(
        output_dir,
        block_stats,
        block_consistency,
        extra_cols,
    )

    per_bin_counts = {
        f"bin_{i}_[{curve.bin_edges[i]:.1f},"
        f"{curve.bin_edges[i + 1]:.1f})":
            int(curve.bin_counts[i])
        for i in range(len(curve.bin_counts))
    }

    under_populated_bins = [
        k
        for k, v in per_bin_counts.items()
        if v == 0
    ]

    diagnostics = {
        "image_path": image_path,
        "image_height": height,
        "image_width": width,

        "num_blocks_total": len(blocks),
        "num_blocks_eligible": n_eligible,
        "num_blocks_ineligible": n_ineligible,

        "fraction_blocks_eligible": (
            n_eligible / len(blocks)
            if blocks
            else float("nan")
        ),

        "fraction_blocks_ineligible": (
            n_ineligible / len(blocks)
            if blocks
            else float("nan")
        ),

        "theta_grad_used": theta_grad,
        "theta_grad_source": theta_grad_source,

        "note_ineligible_is_undefined_not_consistent": True,

        "eligible_blocks_per_intensity_bin": per_bin_counts,

        "under_populated_bins_zero_eligible_blocks":
            under_populated_bins,

        "dispersion_curve_populated_bin_centers":
            curve.interp_x.tolist(),

        "dispersion_curve_populated_bin_medians":
            curve.interp_y.tolist(),

        "residual_scale_used": residual_scale,
        "residual_scale_source": residual_scale_source,

        "residual_r_mad_goodness_of_fit":
            residual_mad_diag,

        "residual_r_mad_interpretation": (
            "Large values indicate the single-noise-level-function "
            "assumption (methodology Assumption 3) is likely untenable "
            "for this image; treat the heatmap as low-confidence, per "
            "methodology Section 15."
        ),

        "image_level_score_M_image": m_image,

        "image_level_score_definition":
            "Q_beta(fused block score) over eligible blocks; fused score = "
            "max over channels of max(|z_raw|, |z_pooled|) (v1.2). "
            "With use_extra_channels=False it is Q_beta(|z_k|) of v1.1.",

        "channel_diagnostics": channel_diag,

        "beta_percentile_used": config.beta,
        "tau_threshold_used": config.tau,

        "decision_flagged_inconsistent": flagged,

        "decision_rule":
            "flagged iff M_image >= tau",

        "active_parameters":
            config.as_dict(),

        "all_parameters_are_uncalibrated_placeholders":
            True,

        "scope_limitations": [
            "Does not detect tonal edits confined to band interiors.",
            "Brightness offsets are credited only when supported by a sharp step edge and a deviation from the global surface; feathered/soft-edged additive edits and offsets larger than the surface-fit tolerance on smooth band structure may be missed.",
            "Histogram-comb evidence is destroyed by lossy re-compression (JPEG); then only dispersion and level channels remain.",
            "Does not detect matched-noise-evasion manipulation.",
            "Does not detect edits smaller than one block (B x B).",
            "Not validated on composite/multi-acquisition images (out of scope).",
            "Calibrated only on synthetic data; see README for actual benchmark metrics.",
        ],
    }

    with open(
        os.path.join(output_dir, "diagnostics.json"),
        "w",
    ) as f:
        json.dump(
            diagnostics,
            f,
            indent=2,
            default=_json_default,
        )

    with open(
        os.path.join(output_dir, "image_level_score.json"),
        "w",
    ) as f:
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


def _fuse_channels(blocks, block_stats, bcs, image, flat_mask, G, theta_grad, shape, config):
    n = len(blocks)
    ps = config.pool_size
    el = np.array([bc.eligible and np.isfinite(bc.z_k) for bc in bcs])

    def score(r_vec, floor, centre=True):
        """(signed raw z, fused |z|) for a residual vector over blocks."""
        z_raw = robust_z(r_vec, floor) if centre else r_vec
        g = to_grid(r_vec, blocks)
        zp = robust_z(pool_signed(g, ps).ravel(), floor / max(ps, 1)) if ps > 1 else z_raw
        both = np.nanmax(np.vstack([np.abs(z_raw), np.abs(zp)]), axis=0) if n else z_raw
        return z_raw, np.where(np.isfinite(both), both, np.nan), zp

    # channel 1: dispersion (existing residual r_k, already robust-scaled -> z_k)
    r_disp = np.array([bc.r_k for bc in bcs])
    z_disp_raw = np.array([bc.z_k for bc in bcs])
    _, _, zp_disp = score(r_disp, LOG_SCALE_FLOOR)
    s_disp = np.fmax(np.abs(z_disp_raw), np.abs(zp_disp))

    # channel 2: level (global robust surface, edge-sharpness gated)
    d, d_gated, lvl_scale = level_channel(
        blocks, block_stats, G, theta_grad, shape,
        config.level_poly_degree, config.level_step_ratio, image=image,
    )
    zl_raw = np.full(n, np.nan)
    f = np.isfinite(d_gated)
    if f.any() and np.isfinite(lvl_scale):
        zl_raw[f] = d_gated[f] / max(lvl_scale, LEVEL_SCALE_FLOOR)
    if ps > 1:
        pl = pool_signed(to_grid(d_gated, blocks), ps).ravel()
        zl_pool = np.where(np.isfinite(pl), pl / max(lvl_scale, LEVEL_SCALE_FLOOR) if np.isfinite(lvl_scale) else np.nan, np.nan)
    else:
        zl_pool = zl_raw
    s_lvl = np.fmax(np.abs(zl_raw), np.abs(zl_pool))

    # channel 3: histogram roughness (gap / peak fingerprint)
    if config.use_hist_channel:
        chi = histogram_roughness(image, flat_mask, blocks, block_stats,
                                  config.hist_window, config.hist_min_pixels)
        r_hist = hist_residuals(block_stats, chi, config.n_bins, config.bin_mode)
    else:
        chi = np.full(n, np.nan)
        r_hist = np.full(n, np.nan)
    zh_raw, _, zh_pool = score(r_hist, LOG_SCALE_FLOOR)
    s_hist = np.fmax(np.abs(zh_raw), np.abs(zh_pool))

    stack = np.vstack([s_disp, s_lvl, s_hist])
    fused = np.where(np.all(np.isnan(stack), axis=0), np.nan, np.nanmax(np.where(np.isnan(stack), -np.inf, stack), axis=0))
    fused = np.where(np.isfinite(fused), fused, np.nan)

    s_raw = np.vstack([np.abs(z_disp_raw), np.abs(zl_raw), np.abs(zh_raw)])
    raw_fused = np.where(np.all(np.isnan(s_raw), axis=0), np.nan,
                         np.nanmax(np.where(np.isnan(s_raw), -np.inf, s_raw), axis=0))
    raw_fused = np.where(np.isfinite(raw_fused), raw_fused, np.nan)
    loc = []
    for i, bc in enumerate(bcs):
        zr = float(raw_fused[i]) if (bc.eligible and np.isfinite(raw_fused[i])) else float("nan")
        loc.append(BlockConsistency(
            block_index=bc.block_index, eligible=bc.eligible, mu_k=bc.mu_k, s_k=bc.s_k,
            s_hat_mu_k=bc.s_hat_mu_k, r_k=bc.r_k, z_k=zr,
        ))

    out = []
    for i, bc in enumerate(bcs):
        z = float(fused[i]) if (bc.eligible and np.isfinite(fused[i])) else float("nan")
        out.append(BlockConsistency(
            block_index=bc.block_index, eligible=bc.eligible, mu_k=bc.mu_k, s_k=bc.s_k,
            s_hat_mu_k=bc.s_hat_mu_k, r_k=bc.r_k, z_k=z,
        ))

    dom = np.array(["dispersion", "level", "histogram"])
    which = np.full(n, "", dtype=object)
    for i in range(n):
        if np.isfinite(fused[i]):
            which[i] = dom[int(np.argmax(np.where(np.isnan(stack[:, i]), -np.inf, stack[:, i])))]

    extra = {
        "score_dispersion": s_disp, "score_level": s_lvl, "score_histogram": s_hist,
        "level_dev_d_k": d, "hist_roughness_chi": chi, "dominant_channel": which,
    }
    diag = {
        "channels": ["dispersion", "level"] + (["histogram"] if config.use_hist_channel else []),
        "level_surface_scale": lvl_scale,
        "level_blocks_scored": int(np.isfinite(d_gated).sum()),
        "level_blocks_gated_out_as_smooth_structure": int(np.isfinite(d).sum() - np.isfinite(d_gated).sum()),
        "histogram_blocks_scored": int(np.isfinite(chi).sum()),
        "pool_size": ps,
        "block_max_by_channel": {
            "dispersion": float(np.nanmax(s_disp)) if np.isfinite(s_disp).any() else None,
            "level": float(np.nanmax(s_lvl)) if np.isfinite(s_lvl).any() else None,
            "histogram": float(np.nanmax(s_hist)) if np.isfinite(s_hist).any() else None,
        },
    }
    return out, extra, diag, loc


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)

    if isinstance(o, (np.integer,)):
        return int(o)

    if isinstance(o, np.ndarray):
        return o.tolist()

    if isinstance(o, float) and np.isnan(o):
        return None

    raise TypeError(
        f"Object of type {type(o)} is not JSON serializable"
    )


def _write_input_info(
    output_dir: str,
    loaded,
) -> None:
    info = {
        "input_path": loaded.path,
        "height": loaded.height,
        "width": loaded.width,
        "dtype": "uint8",
        "channels": 1,
        "validated_single_channel_8bit_grayscale": True,
        "resized": False,
    }

    with open(
        os.path.join(output_dir, "input_info.json"),
        "w",
    ) as f:
        json.dump(
            info,
            f,
            indent=2,
        )


def _write_dispersion_curve_csv(
    output_dir: str,
    curve,
) -> None:
    path = os.path.join(
        output_dir,
        "dispersion_curve.csv",
    )

    with open(
        path,
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "bin_index",
                "bin_low",
                "bin_high",
                "bin_center",
                "bin_median_s",
                "bin_eligible_block_count",
            ]
        )

        for i in range(len(curve.bin_centers)):
            writer.writerow(
                [
                    i,
                    curve.bin_edges[i],
                    curve.bin_edges[i + 1],
                    curve.bin_centers[i],
                    (
                        ""
                        if np.isnan(curve.bin_medians[i])
                        else curve.bin_medians[i]
                    ),
                    int(curve.bin_counts[i]),
                ]
            )


def _write_block_stats_csv(
    output_dir: str,
    block_stats,
    block_consistency,
    extra_cols=None,
) -> None:
    extra_cols = extra_cols or {}
    extra_names = list(extra_cols.keys())
    index_pos = {s.block_index: i for i, s in enumerate(block_stats)}
    path = os.path.join(
        output_dir,
        "block_stats.csv",
    )

    by_index = {
        bc.block_index: bc
        for bc in block_consistency
    }

    with open(
        path,
        "w",
        newline="",
    ) as f:
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
            ] + extra_names
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

                    (
                        ""
                        if np.isnan(s.mu_k)
                        else s.mu_k
                    ),

                    (
                        ""
                        if np.isnan(s.s_k)
                        else s.s_k
                    ),

                    (
                        ""
                        if np.isnan(bc.s_hat_mu_k)
                        else bc.s_hat_mu_k
                    ),

                    (
                        ""
                        if np.isnan(bc.r_k)
                        else bc.r_k
                    ),

                    (
                        ""
                        if np.isnan(bc.z_k)
                        else bc.z_k
                    ),

                    (
                        ""
                        if np.isnan(bc.z_k)
                        else abs(bc.z_k)
                    ),
                ] + [_fmt(extra_cols[k][index_pos[s.block_index]]) for k in extra_names]
            )


def _fmt(v):
    if isinstance(v, str):
        return v
    try:
        return "" if np.isnan(v) else float(v)
    except TypeError:
        return v