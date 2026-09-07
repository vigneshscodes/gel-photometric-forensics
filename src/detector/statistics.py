"""
Per-block statistics and the global dispersion-vs-level curve.
Methodology Sections 4-5.

    mu_k = Med({ I(u,v) : (u,v) in F_k })     level proxy
    s_k  = MAD({ R(u,v) : (u,v) in F_k })     local dispersion

    s_hat(mu): binned robust median regression across ALL eligible blocks
    in the whole image, then linear interpolation between bin medians.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .regions import Block


def median_absolute_deviation(values: np.ndarray) -> float:
    """
    MAD({x_j}) = median(|x_j - median(x_j)|)

    Raw MAD, with NO 1.4826 consistency-constant scaling applied here.
    That constant is applied later and only where the methodology
    specifies it (Section 6, robust z-score denominator).
    """
    if values.size == 0:
        return float("nan")
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


@dataclass
class BlockStats:
    block_index: int
    row0: int
    col0: int
    row1: int
    col1: int
    eligible: bool
    n_flat_pixels: int
    mu_k: float  # median intensity of flat pixels (from I), NaN if ineligible
    s_k: float  # MAD of R over flat pixels, NaN if ineligible


def compute_block_statistics(
    blocks: List[Block],
    image: np.ndarray,
    R: np.ndarray,
    flat_mask: np.ndarray,
) -> List[BlockStats]:
    """
    Computes mu_k and s_k for every block. Ineligible blocks get NaN for
    both (they are never scored downstream — Section 7).
    """
    stats: List[BlockStats] = []
    for b in blocks:
        if not b.eligible:
            stats.append(
                BlockStats(
                    block_index=b.index,
                    row0=b.row0,
                    col0=b.col0,
                    row1=b.row1,
                    col1=b.col1,
                    eligible=False,
                    n_flat_pixels=b.n_flat_pixels,
                    mu_k=float("nan"),
                    s_k=float("nan"),
                )
            )
            continue

        sub_flat = flat_mask[b.row0 : b.row1, b.col0 : b.col1]
        sub_I = image[b.row0 : b.row1, b.col0 : b.col1].astype(np.float64)
        sub_R = R[b.row0 : b.row1, b.col0 : b.col1]

        flat_I_values = sub_I[sub_flat]
        flat_R_values = sub_R[sub_flat]

        mu_k = float(np.median(flat_I_values))  # level proxy: monotone-preserving, not exact
        s_k = median_absolute_deviation(flat_R_values)

        stats.append(
            BlockStats(
                block_index=b.index,
                row0=b.row0,
                col0=b.col0,
                row1=b.row1,
                col1=b.col1,
                eligible=True,
                n_flat_pixels=b.n_flat_pixels,
                mu_k=mu_k,
                s_k=s_k,
            )
        )
    return stats


@dataclass
class DispersionCurve:
    """
    The fitted global dispersion-vs-level curve s_hat(mu), stored as
    the bin centers/medians used for interpolation, plus per-bin counts
    for the "under-populated bin" diagnostic (Section 15).
    """

    bin_edges: np.ndarray  # length n_bins + 1
    bin_centers: np.ndarray  # length n_bins
    bin_medians: np.ndarray  # length n_bins, NaN where a bin had no eligible blocks
    bin_counts: np.ndarray  # length n_bins, int
    # Only bins with at least one eligible block participate in
    # interpolation; these are the x/y arrays actually passed to interp.
    interp_x: np.ndarray
    interp_y: np.ndarray

    def evaluate(self, mu: np.ndarray) -> np.ndarray:
        """
        s_hat(mu): linear interpolation between populated bin medians.

        IMPLEMENTATION DECISION (documented ambiguity #5): the methodology
        does not say what to do for mu values outside the range of
        populated bins (e.g. an eligible block whose mu_k falls in an
        empty extreme bin). np.interp's default behavior — clamp to the
        nearest edge value for out-of-range x — is used here rather than
        extrapolating, since extrapolating a robust median-regression
        curve beyond its observed support has no basis in the
        methodology and could fabricate confidence in unseen territory.
        This is disclosed, not silent: any block relying on clamped
        extrapolation is exactly a block in an under-populated extreme
        bin, which is already surfaced by the required per-bin diagnostic.
        """
        if self.interp_x.size == 0:
            # No populated bins at all: curve is entirely undefined.
            return np.full_like(mu, np.nan, dtype=np.float64)
        if self.interp_x.size == 1:
            # Cannot interpolate a line from a single point; treat as
            # constant (best-supported, non-extrapolating fallback).
            return np.full_like(mu, self.interp_y[0], dtype=np.float64)
        return np.interp(mu, self.interp_x, self.interp_y)


def fit_dispersion_curve(
    block_stats: List[BlockStats], n_bins: int
) -> DispersionCurve:
    """
    Binned robust median regression (methodology Section 4):
      1. Bin the observed intensity range [0, 255] into n_bins equal bins.
      2. Within each bin, take the median of s_k over eligible blocks
         whose mu_k falls in that bin.
      3. Linearly interpolate between bin medians to get s_hat(mu).

    IMPLEMENTATION DECISION (documented ambiguity #6): "the observed
    intensity range" could mean [0, 255] (the full 8-bit range) or
    [min(mu_k), max(mu_k)] (the range actually observed in this image).
    We use the fixed [0, 255] range so that bin boundaries are stable
    and comparable across images/runs (a per-image adaptive range would
    make bin index k mean a different intensity on every image, which
    would undermine comparing diagnostics across images). This is
    disclosed rather than silently assumed.
    """
    eligible = [s for s in block_stats if s.eligible]
    mu_values = np.array([s.mu_k for s in eligible], dtype=np.float64)
    s_values = np.array([s.s_k for s in eligible], dtype=np.float64)

    bin_edges = np.linspace(0, 255, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_medians = np.full(n_bins, np.nan, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    if mu_values.size > 0:
        # np.digitize with right=False: bin i covers [edges[i], edges[i+1])
        bin_idx = np.digitize(mu_values, bin_edges[1:-1], right=False)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        for i in range(n_bins):
            sel = bin_idx == i
            bin_counts[i] = int(np.count_nonzero(sel))
            if bin_counts[i] > 0:
                bin_medians[i] = float(np.median(s_values[sel]))

    populated = ~np.isnan(bin_medians)
    interp_x = bin_centers[populated]
    interp_y = bin_medians[populated]

    return DispersionCurve(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        bin_medians=bin_medians,
        bin_counts=bin_counts,
        interp_x=interp_x,
        interp_y=interp_y,
    )
