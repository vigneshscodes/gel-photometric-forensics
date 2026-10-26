
"""
Per-block statistics and the global dispersion-vs-level curve.

Methodology:
    mu_k = median(I over flat pixels in block k)
    s_k  = MAD(R over flat pixels in block k)

    s_hat(mu): binned median regression across eligible blocks,
    followed by linear interpolation between populated bin centers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .regions import Block


def median_absolute_deviation(values: np.ndarray) -> float:
    """Raw MAD: median(abs(x - median(x))), without scaling."""
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan")

    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def trimmed_std(values: np.ndarray, trim: float = 0.10) -> float:
    """Std of values after dropping the `trim` fraction with largest |x - median|."""
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    d = np.abs(values - np.median(values))
    kept = values[d <= np.quantile(d, 1.0 - trim)]
    return float(np.std(kept)) if kept.size else float("nan")


@dataclass
class BlockStats:
    block_index: int
    row0: int
    col0: int
    row1: int
    col1: int
    eligible: bool
    n_flat_pixels: int
    mu_k: float
    s_k: float


def compute_block_statistics(
    blocks: List[Block],
    image: np.ndarray,
    R: np.ndarray,
    flat_mask: np.ndarray,
    dispersion_estimator: str = "trimmed_std",
) -> List[BlockStats]:
    """Compute intensity level and local dispersion for each block."""
    image = np.asarray(image)
    R = np.asarray(R)
    flat_mask = np.asarray(flat_mask, dtype=bool)

    if image.ndim != 2 or R.shape != image.shape or flat_mask.shape != image.shape:
        raise ValueError(
            "image, R, and flat_mask must be 2D arrays with identical shapes."
        )

    stats: List[BlockStats] = []

    for b in blocks:
        sub_flat = flat_mask[b.row0:b.row1, b.col0:b.col1]
        sub_I = image[b.row0:b.row1, b.col0:b.col1].astype(np.float64)
        sub_R = R[b.row0:b.row1, b.col0:b.col1].astype(np.float64)

        flat_I = sub_I[sub_flat]
        flat_R = sub_R[sub_flat]

        # An eligible block must have enough valid flat pixels and
        # finite measurements. Preserve the block builder's eligibility
        # decision rather than silently changing the project criteria.
        valid = (
            b.eligible
            and flat_I.size > 0
            and flat_R.size == flat_I.size
            and np.all(np.isfinite(flat_I))
            and np.all(np.isfinite(flat_R))
        )

        if not valid:
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

        stats.append(
            BlockStats(
                block_index=b.index,
                row0=b.row0,
                col0=b.col0,
                row1=b.row1,
                col1=b.col1,
                eligible=True,
                n_flat_pixels=b.n_flat_pixels,
                mu_k=float(np.median(flat_I)),
                s_k=(
                    median_absolute_deviation(flat_R)
                    if dispersion_estimator == "mad"
                    else trimmed_std(flat_R)
                ),
            )
        )

    return stats


@dataclass
class DispersionCurve:
    """Binned median dispersion curve and its diagnostics."""

    bin_edges: np.ndarray
    bin_centers: np.ndarray
    bin_medians: np.ndarray
    bin_counts: np.ndarray
    interp_x: np.ndarray
    interp_y: np.ndarray

    def evaluate(self, mu: np.ndarray) -> np.ndarray:
        """
        Evaluate s_hat(mu).

        Interpolate between populated bin centers and clamp outside
        their support to the nearest populated endpoint.
        """
        mu = np.asarray(mu, dtype=np.float64)
        result = np.full(mu.shape, np.nan, dtype=np.float64)

        if self.interp_x.size == 0:
            return result

        finite = np.isfinite(mu)

        if self.interp_x.size == 1:
            result[finite] = self.interp_y[0]
            return result

        result[finite] = np.interp(
            mu[finite],
            self.interp_x,
            self.interp_y,
        )
        return result


def fit_dispersion_curve(
    block_stats: List[BlockStats],
    n_bins: int,
    bin_mode: str = "quantile",
) -> DispersionCurve:
    """
    Fit the global dispersion-vs-intensity curve s_hat(mu).

    bin_mode="quantile" (default): bins hold equal numbers of eligible
    blocks; each bin's x-position is the median mu_k of its blocks.
    bin_mode="range": legacy equal-width bins over [0, 255] with the
    bin center as x-position.
    The y-value of a populated bin is the median s_k of its blocks;
    evaluation interpolates linearly and clamps outside the support.
    """
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    if bin_mode not in ("quantile", "range"):
        raise ValueError("bin_mode must be 'quantile' or 'range'.")

    eligible = [s for s in block_stats if s.eligible]
    mu_values = np.asarray([s.mu_k for s in eligible], dtype=np.float64)
    s_values = np.asarray([s.s_k for s in eligible], dtype=np.float64)
    ok = (
        np.isfinite(mu_values) & np.isfinite(s_values)
        & (mu_values >= 0.0) & (mu_values <= 255.0) & (s_values >= 0.0)
    )
    mu_values, s_values = mu_values[ok], s_values[ok]

    if bin_mode == "range" or mu_values.size == 0:
        bin_edges = np.linspace(0.0, 255.0, n_bins + 1)
    else:
        bin_edges = np.unique(np.quantile(mu_values, np.linspace(0.0, 1.0, n_bins + 1)))
        if bin_edges.size < 2:
            bin_edges = np.array([bin_edges[0], bin_edges[0] + 1.0])

    n_eff = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_medians = np.full(n_eff, np.nan, dtype=np.float64)
    bin_counts = np.zeros(n_eff, dtype=np.int64)
    x_pos = bin_centers.copy()

    if mu_values.size:
        # Bins are [edge_i, edge_(i+1)); the last bin includes its upper edge.
        bin_idx = np.clip(
            np.searchsorted(bin_edges[1:-1], mu_values, side="right"), 0, n_eff - 1
        )
        for i in range(n_eff):
            sel = bin_idx == i
            count = int(np.count_nonzero(sel))
            bin_counts[i] = count
            if count:
                bin_medians[i] = float(np.median(s_values[sel]))
                if bin_mode == "quantile":
                    x_pos[i] = float(np.median(mu_values[sel]))

    populated = np.isfinite(bin_medians)
    order = np.argsort(x_pos[populated])

    return DispersionCurve(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        bin_medians=bin_medians,
        bin_counts=bin_counts,
        interp_x=x_pos[populated][order],
        interp_y=bin_medians[populated][order],
    )
