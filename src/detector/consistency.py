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
    Compute r_k and z_k for every eligible block.

    Ineligible blocks remain undefined and receive NaN values.
    """

    eligible_stats = [
        s for s in block_stats
        if s.eligible
    ]

    # No eligible blocks -> all consistency values are undefined.
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
        [s.mu_k for s in eligible_stats],
        dtype=np.float64,
    )

    s_arr = np.array(
        [s.s_k for s in eligible_stats],
        dtype=np.float64,
    )

    # Evaluate the global dispersion-vs-level model.
    s_hat_arr = curve.evaluate(mu_arr)

    # r_k = log(s_k + eps) - log(s_hat(mu_k) + eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_arr = (
            np.log(s_arr + epsilon_0)
            - np.log(s_hat_arr + epsilon_0)
        )

    # Keep only finite residuals for robust scale estimation.
    valid_r = r_arr[np.isfinite(r_arr)]

    if valid_r.size > 0:
        residual_mad = median_absolute_deviation(valid_r)
    else:
        residual_mad = float("nan")

    # Robust scale of residual population.
    if np.isfinite(residual_mad) and residual_mad > 0:
        denom = MAD_TO_SIGMA_CONSTANT * residual_mad
    else:
        denom = float("nan")

    # A zero/undefined MAD means z-score normalization is mathematically
    # undefined. Do NOT replace it with zero.
    if np.isfinite(denom) and denom > 0:
        z_arr = r_arr / denom
    else:
        z_arr = np.full_like(
            r_arr,
            np.nan,
            dtype=np.float64,
        )

    results: List[BlockConsistency] = []

    # Iterator over eligible-block calculations.
    it = iter(
        zip(
            mu_arr,
            s_arr,
            s_hat_arr,
            r_arr,
            z_arr,
        )
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

        (
            mu_k,
            s_k,
            s_hat_mu_k,
            r_k,
            z_k,
        ) = next(it)

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
    Compute the image-level inconsistency score.

        M_image = Q_beta({ |z_k| : k eligible })

    Only finite eligible z-scores are included.

    If no finite eligible z-scores exist, M_image is undefined
    and NaN is returned.
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

    return float(
        np.percentile(
            eligible_abs_z,
            beta,
        )
    )


def decide(
    m_image: float,
    tau: float,
) -> bool:
    """
    Flag image as inconsistent iff:

        M_image >= tau

    An undefined M_image (NaN) is NOT silently treated as clean.
    """

    if not np.isfinite(m_image):
        return False

    return bool(m_image >= tau)