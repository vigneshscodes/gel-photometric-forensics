
"""
Consistency metric, image-level score, and decision rule.

Methodology:
    r_k = log(s_k + epsilon_0) - log(s_hat(mu_k) + epsilon_0)
    z_k = r_k / (1.4826 * MAD(r))
    M_image = Q_beta(abs(z_k))
    Flag iff M_image >= tau

If the residual MAD is zero but residuals are not all zero, the
MAD-normalized score is undefined. This implementation preserves that
fact instead of introducing an arbitrary scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .statistics import (
    BlockStats,
    DispersionCurve,
    median_absolute_deviation,
)


MAD_TO_SIGMA_CONSTANT = 1.4826
IQR_TO_SIGMA_CONSTANT = 1.3490

# Lower bound on the residual scale (log-dispersion units). The log of a
# MAD estimated from ~100 integer-valued pixels has a sampling sd of
# roughly 0.1; a scale far below that is a quantization artefact, not
# evidence of a tight fit, and would turn noise into huge z-scores.
MIN_RESIDUAL_SCALE = 0.05


def estimate_residual_scale(valid_r: np.ndarray):
    """
    Robust scale of the residuals r_k, with an explicit fallback ladder
    instead of silently returning NaN (the original v1.1 behaviour when
    MAD(r) == 0, which happens routinely because block MADs of 8-bit data
    are quantized and many blocks share identical s_k).

    Ladder: 1.4826*MAD -> IQR/1.349 -> std -> undefined.
    A floor of MIN_RESIDUAL_SCALE is applied to any positive estimate.
    Returns (scale_or_nan, source_string).
    """
    valid_r = np.asarray(valid_r, dtype=np.float64)
    valid_r = valid_r[np.isfinite(valid_r)]
    if valid_r.size == 0:
        return float("nan"), "no_valid_residuals"
    if np.all(valid_r == 0.0):
        return 0.0, "all_residuals_zero"

    mad = median_absolute_deviation(valid_r)
    if np.isfinite(mad) and mad > 0.0:
        return max(MAD_TO_SIGMA_CONSTANT * mad, MIN_RESIDUAL_SCALE), "mad"

    q75, q25 = np.percentile(valid_r, [75, 25])
    if q75 - q25 > 0.0:
        return max((q75 - q25) / IQR_TO_SIGMA_CONSTANT, MIN_RESIDUAL_SCALE), "iqr_fallback"

    sd = float(np.std(valid_r))
    if sd > 0.0:
        return max(sd, MIN_RESIDUAL_SCALE), "std_fallback"

    return float("nan"), "undefined"


@dataclass
class BlockConsistency:
    block_index: int
    eligible: bool
    mu_k: float
    s_k: float
    s_hat_mu_k: float
    r_k: float
    z_k: float


def compute_residuals_and_zscores(
    block_stats: List[BlockStats],
    curve: DispersionCurve,
    epsilon_0: float,
) -> List[BlockConsistency]:
    """Calculate per-block residuals and MAD-normalized scores."""
    if not np.isfinite(epsilon_0) or epsilon_0 <= 0:
        raise ValueError("epsilon_0 must be a finite positive number.")

    eligible_stats = [s for s in block_stats if s.eligible]

    # Initialize all results as undefined. Fill only valid eligible blocks.
    results = [
        BlockConsistency(
            block_index=s.block_index,
            eligible=s.eligible,
            mu_k=s.mu_k if s.eligible else float("nan"),
            s_k=s.s_k if s.eligible else float("nan"),
            s_hat_mu_k=float("nan"),
            r_k=float("nan"),
            z_k=float("nan"),
        )
        for s in block_stats
    ]

    if not eligible_stats:
        return results

    mu_arr = np.asarray([s.mu_k for s in eligible_stats], dtype=np.float64)
    s_arr = np.asarray([s.s_k for s in eligible_stats], dtype=np.float64)
    s_hat_arr = curve.evaluate(mu_arr)

    # Dispersion values must be nonnegative before taking logarithms.
    valid = (
        np.isfinite(mu_arr)
        & np.isfinite(s_arr)
        & np.isfinite(s_hat_arr)
        & (s_arr >= 0.0)
        & (s_hat_arr >= 0.0)
    )

    r_arr = np.full(mu_arr.shape, np.nan, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        r_arr[valid] = (
            np.log(s_arr[valid] + epsilon_0)
            - np.log(s_hat_arr[valid] + epsilon_0)
        )

    valid &= np.isfinite(r_arr)
    valid_r = r_arr[valid]

    z_arr = np.full(mu_arr.shape, np.nan, dtype=np.float64)

    if valid_r.size:
        scale, _source = estimate_residual_scale(valid_r)

        if scale == 0.0:
            # All valid residuals match the fitted curve exactly.
            z_arr[valid] = 0.0
        elif np.isfinite(scale):
            z_arr[valid] = r_arr[valid] / scale

    # Map calculations back to the original block order.
    eligible_iter = iter(zip(s_hat_arr, r_arr, z_arr))

    for i, s in enumerate(block_stats):
        if not s.eligible:
            continue

        s_hat, residual, z_score = next(eligible_iter)

        results[i] = BlockConsistency(
            block_index=s.block_index,
            eligible=True,
            mu_k=s.mu_k,
            s_k=s.s_k,
            s_hat_mu_k=float(s_hat),
            r_k=float(residual),
            z_k=float(z_score),
        )

    return results


def compute_image_level_score(
    block_consistency: List[BlockConsistency],
    beta: float,
) -> float:
    """Compute the beta-th percentile of finite absolute z-scores."""
    if not np.isfinite(beta) or not 0.0 <= beta <= 100.0:
        raise ValueError("beta must be between 0 and 100.")

    eligible_abs_z = np.asarray(
        [
            abs(bc.z_k)
            for bc in block_consistency
            if bc.eligible and np.isfinite(bc.z_k)
        ],
        dtype=np.float64,
    )

    if eligible_abs_z.size == 0:
        return float("nan")

    return float(np.percentile(eligible_abs_z, beta))


def decide(m_image: float, tau: float) -> bool:
    """
    Apply the existing threshold rule.

    Note: False for an undefined score means 'not flagged', not
    necessarily 'confirmed clean'. Preserve score validity separately
    in benchmark reports to distinguish these cases.
    """
    if not np.isfinite(tau):
        raise ValueError("tau must be finite.")

    if not np.isfinite(m_image):
        return False

    return bool(m_image >= tau)
