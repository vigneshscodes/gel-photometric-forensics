
"""
Consistency metric, image-level score, and decision rule.
Methodology Sections 6, 9, 10.

    r_k = log(s_k + eps0) - log(s_hat(mu_k) + eps0)
    z_k = r_k / (1.4826 * MAD({r_j : j eligible}))
    M_image = Q_beta({ |z_k| : k eligible })
    Flag iff M_image >= tau
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


# Standard MAD-to-sigma consistency constant.
MAD_TO_SIGMA_CONSTANT = 1.4826


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
    """
    Compute residuals and robust normalized scores for eligible blocks.

    If residual MAD is zero:
      - All finite residuals exactly zero: assign zero z-scores.
      - Any finite residual is nonzero: scores remain undefined.

    Ineligible blocks remain undefined.
    """

    eligible_stats = [s for s in block_stats if s.eligible]

    if not eligible_stats:
        return [
            BlockConsistency(
                block_index=s.block_index,
                eligible=False,
                mu_k=float("nan"),
                s_k=float("nan"),
                s_hat_mu_k=float("nan"),
                r_k=float("nan"),
                z_k=float("nan"),
            )
            for s in block_stats
        ]

    mu_arr = np.array(
        [s.mu_k for s in eligible_stats], dtype=np.float64
    )
    s_arr = np.array(
        [s.s_k for s in eligible_stats], dtype=np.float64
    )

    # Global dispersion-versus-intensity model.
    s_hat_arr = curve.evaluate(mu_arr)

    with np.errstate(divide="ignore", invalid="ignore"):
        r_arr = (
            np.log(s_arr + epsilon_0)
            - np.log(s_hat_arr + epsilon_0)
        )

    valid_mask = np.isfinite(r_arr)
    valid_r = r_arr[valid_mask]

    residual_mad = (
        median_absolute_deviation(valid_r)
        if valid_r.size > 0
        else float("nan")
    )

    z_arr = np.full_like(r_arr, np.nan, dtype=np.float64)

    if np.isfinite(residual_mad) and residual_mad > 0:
        denom = MAD_TO_SIGMA_CONSTANT * residual_mad
        z_arr[valid_mask] = r_arr[valid_mask] / denom

    elif (
        valid_r.size > 0
        and np.all(valid_r == 0.0)
    ):
        # Exact zero residuals mean every finite residual matches
        # the fitted global model. No arbitrary scale is introduced.
        z_arr[valid_mask] = 0.0

    # Otherwise leave scores undefined: zero MAD with nonzero residuals
    # does not provide a defensible MAD-normalized score.

    results: List[BlockConsistency] = []
    eligible_iter = iter(
        zip(mu_arr, s_arr, s_hat_arr, r_arr, z_arr)
    )

    for s in block_stats:
        if not s.eligible:
            results.append(
                BlockConsistency(
                    block_index=s.block_index,
                    eligible=False,
                    mu_k=float("nan"),
                    s_k=float("nan"),
                    s_hat_mu_k=float("nan"),
                    r_k=float("nan"),
                    z_k=float("nan"),
                )
            )
            continue

        mu_k, s_k, s_hat_mu_k, r_k, z_k = next(eligible_iter)

        results.append(
            BlockConsistency(
                block_index=s.block_index,
                eligible=True,
                mu_k=float(mu_k),
                s_k=float(s_k),
                s_hat_mu_k=float(s_hat_mu_k),
                r_k=float(r_k),
                z_k=float(z_k),
            )
        )

    return results


def compute_image_level_score(
    block_consistency: List[BlockConsistency],
    beta: float,
) -> float:
    """
    Compute M_image = Q_beta(|z_k|) over finite eligible scores.

    Returns NaN when no finite eligible scores exist.
    """

    eligible_abs_z = np.array(
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
    Flag an image iff M_image >= tau.

    An undefined score is not classified as clean.
    """

    if not np.isfinite(m_image):
        return False

    return bool(m_image >= tau)
