"""
Consistency metric, image-level score, and decision rule.
Methodology Sections 6, 9, 10.

    r_k = log(s_k + eps0) - log(s_hat(mu_k) + eps0)
    z_k = r_k / (1.4826 * MAD({r_j : j eligible}))
    M_image = Q_beta({ z_k : k eligible })
    Flag iff M_image >= tau
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .statistics import BlockStats, DispersionCurve, median_absolute_deviation

# The standard constant making MAD a consistent estimator of sigma under
# Gaussian residuals (methodology Section 6: "established, not invented").
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
    Computes r_k and z_k for every eligible block. Ineligible blocks pass
    through as eligible=False with NaN r_k/z_k — they are UNDEFINED, never
    scored as "consistent" (see visualization.py for the same distinction
    at pixel level).
    """
    eligible_stats = [s for s in block_stats if s.eligible]
    mu_arr = np.array([s.mu_k for s in eligible_stats], dtype=np.float64)
    s_arr = np.array([s.s_k for s in eligible_stats], dtype=np.float64)

    s_hat_arr = curve.evaluate(mu_arr)
    r_arr = np.log(s_arr + epsilon_0) - np.log(s_hat_arr + epsilon_0)

    # Robust scale of the residual population itself, across all eligible
    # blocks in this image (methodology Section 6).
    valid_r = r_arr[~np.isnan(r_arr)]
    residual_mad = median_absolute_deviation(valid_r) if valid_r.size > 0 else float("nan")
    denom = MAD_TO_SIGMA_CONSTANT * residual_mad if residual_mad > 0 else float("nan")

    if denom and not np.isnan(denom) and denom > 0:
        z_arr = r_arr / denom
    else:
        # Degenerate case: zero or undefined residual spread (e.g. a
        # single eligible block, or all residuals identical). z_k is
        # mathematically undefined here, not zero — zero would falsely
        # assert "no anomaly".
        z_arr = np.full_like(r_arr, np.nan)

    results: List[BlockConsistency] = []
    it = iter(zip(mu_arr, s_arr, s_hat_arr, r_arr, z_arr))
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
        mu_k, s_k, s_hat_mu_k, r_k, z_k = next(it)
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
    block_consistency: List[BlockConsistency], beta: float
) -> float:
    """
    M_image = Q_beta({ |z_k| : k eligible })

    IMPLEMENTATION DECISION (documented ambiguity #7 — the most
    consequential one in this implementation):

    The frozen methodology's Section 9 formula literally reads
        M_image = Q_beta({ z_k : k eligible })
    i.e. a percentile of the SIGNED z_k. But Section 6 defines the
    anomaly score in terms of |z_k| ("large |z_k| means the block's local
    dispersion ... does not match ... the operational definition of
    local-global inconsistency"), and Section 8 assigns |z_k| (not z_k)
    to the heatmap. Taking a high percentile of *signed* z_k would only
    ever be sensitive to blocks whose dispersion is anomalously HIGH
    relative to the fitted curve, and would be blind to blocks whose
    dispersion is anomalously LOW (e.g. local over-smoothing edits),
    which negative z_k values would flag under the |.| definition used
    everywhere else in the document.

    This is a genuine internal inconsistency between Section 6/8's stated
    operational definition and Section 9's literal notation — not a case
    where the methodology is silent. Per instructions, the smallest
    change that resolves it without inventing new science is to make
    Section 9 consistent with Section 6/8's explicit textual definition:
    this implementation computes M_image from |z_k|, matching what the
    heatmap already visualizes, so a user's `M_image >= tau` decision and
    the heatmap they are looking at agree about what counts as anomalous.
    This decision is surfaced in every diagnostics report.
    """
    eligible_z = np.array(
        [abs(bc.z_k) for bc in block_consistency if bc.eligible and not np.isnan(bc.z_k)],
        dtype=np.float64,
    )
    if eligible_z.size == 0:
        return float("nan")
    return float(np.percentile(eligible_z, beta))


def decide(m_image: float, tau: float) -> bool:
    """
    Flag image as inconsistent iff M_image >= tau.

    Returns False (not-flagged) if m_image is NaN (undefined score —
    e.g. no eligible blocks at all) rather than silently flagging or
    clearing the image; caller is responsible for surfacing the NaN case
    as "score undefined", not as a clean negative result.
    """
    if np.isnan(m_image):
        return False
    return bool(m_image >= tau)
