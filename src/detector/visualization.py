"""
Heatmap generation and diagnostic plots. Methodology Section 8.

Hard requirement carried through this whole module:
    INELIGIBLE != CONSISTENT.
Ineligible blocks are rendered as an explicit "undefined" color/value,
never as zero (which would falsely claim "confirmed consistent").
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless rendering, no display required
import matplotlib.pyplot as plt

from .regions import Block
from .consistency import BlockConsistency
from .statistics import DispersionCurve, BlockStats

UNDEFINED_VALUE = np.nan  # used in the raw (numeric) map


def build_raw_block_map(
    height: int, width: int, blocks: List[Block], block_consistency: List[BlockConsistency]
) -> np.ndarray:
    """
    Pixel-resolution map of |z_k| for eligible blocks, NaN (undefined)
    for ineligible ones. This is the "raw block consistency map" output
    (values, not yet normalized/colored for display).
    """
    raw = np.full((height, width), UNDEFINED_VALUE, dtype=np.float64)
    by_index = {bc.block_index: bc for bc in block_consistency}
    for b in blocks:
        bc = by_index[b.index]
        if bc.eligible and not np.isnan(bc.z_k):
            raw[b.row0 : b.row1, b.col0 : b.col1] = abs(bc.z_k)
        # else: leave as NaN/undefined — never write 0.0 here.
    return raw


def render_heatmap_png(
    raw_map: np.ndarray,
    output_path: str,
    undefined_color: tuple = (0.5, 0.5, 0.5),  # neutral gray, visually distinct from any colormap value
) -> None:
    """
    Renders the normalized heatmap to a PNG. Eligible-block values are
    colormapped; undefined (ineligible) pixels are painted a flat neutral
    gray that does not appear anywhere in the colormap's data range, so
    they cannot be mistaken for a low/zero anomaly score.
    """
    finite_vals = raw_map[np.isfinite(raw_map)]
    fig, ax = plt.subplots(figsize=(max(6, raw_map.shape[1] / 120), max(5, raw_map.shape[0] / 120) + 0.6))

    cmap = matplotlib.colormaps.get_cmap("inferno").copy()
    cmap.set_bad(color=undefined_color)  # NaN -> undefined_color

    if finite_vals.size > 0:
        vmax = float(np.percentile(finite_vals, 99.5))  # robust display scaling, display-only
        vmax = vmax if vmax > 0 else 1.0
    else:
        vmax = 1.0

    im = ax.imshow(raw_map, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|z_k| (eligible blocks only)")
    ax.set_title(
        "LGPC-BC v1.1 anomaly heatmap\n(gray = ineligible / UNDEFINED, not consistent)",
        fontsize=10,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_eligibility_mask(
    height: int, width: int, blocks: List[Block]
) -> np.ndarray:
    """
    uint8 mask: 255 where a block is eligible, 0 where ineligible.
    This is a separate, explicit artifact so eligibility can be inspected
    independently of the (possibly display-rescaled) heatmap.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for b in blocks:
        if b.eligible:
            mask[b.row0 : b.row1, b.col0 : b.col1] = 255
    return mask


def build_suspicious_region_mask(
    height: int, width: int, blocks: List[Block], block_consistency: List[BlockConsistency], tau: float,
    loc_ratio: float = 1.0,
    loc_consistency: Optional[List[BlockConsistency]] = None,
) -> np.ndarray:
    """
    uint8 mask (255 = suspicious) from the fused block score map.

    v1.1 reused the image-level tau (calibrated on the MAX over all blocks)
    as a per-block cut, so only the single strongest blocks survived and
    the mask lost regions plainly visible in the heatmap. v1.2 uses
    hysteresis on the block grid:
      seeds  : score >= tau            (image-level evidence; empty if unflagged)
      region : blocks with score >= loc_ratio*tau connected (8-neighbour)
               to a seed
      repair : 3x3 closing + hole filling bridges ineligible / gated blocks
               inside an edit (their score is undefined, not 'clean'), limited
               to one block beyond the grown region.
    Seeds use block_consistency (fused, pooled+raw: image-level evidence);
    growth uses loc_consistency (UNPOOLED fused score) when given, because
    pooling smears a high score onto neighbouring clean blocks and would
    enlarge the mask beyond the edit.
    loc_ratio=1.0 reproduces the legacy plain per-block cut.
    Ineligible blocks never seed or grow a region on their own.
    """
    from scipy import ndimage

    nc = sum(1 for b in blocks if b.row0 == 0)
    nr = len(blocks) // nc
    grid = np.full((nr, nc), np.nan)
    by_index = {bc.block_index: bc for bc in block_consistency}
    for b in blocks:
        bc = by_index[b.index]
        if bc.eligible and not np.isnan(bc.z_k):
            grid[b.row0 // (blocks[0].row1 - blocks[0].row0), b.col0 // (blocks[0].col1 - blocks[0].col0)] = abs(bc.z_k)

    lgrid = grid
    if loc_consistency is not None:
        lgrid = np.full((nr, nc), np.nan)
        lby = {bc.block_index: bc for bc in loc_consistency}
        for b in blocks:
            bc = lby[b.index]
            if bc.eligible and not np.isnan(bc.z_k):
                lgrid[b.row0 // (blocks[0].row1 - blocks[0].row0), b.col0 // (blocks[0].col1 - blocks[0].col0)] = abs(bc.z_k)

    with np.errstate(invalid="ignore"):
        seeds = grid >= tau
        low = lgrid >= loc_ratio * tau
    region = ndimage.binary_propagation(seeds, structure=np.ones((3, 3), bool), mask=low | seeds)
    if loc_ratio < 1.0 and region.any():
        closed = ndimage.binary_closing(np.pad(region, 1), structure=np.ones((3, 3), bool))[1:-1, 1:-1]
        region = ndimage.binary_fill_holes(closed | region)

    mask = np.zeros((height, width), dtype=np.uint8)
    bh = blocks[0].row1 - blocks[0].row0
    bw = blocks[0].col1 - blocks[0].col0
    for b in blocks:
        if region[b.row0 // bh, b.col0 // bw]:
            mask[b.row0 : b.row1, b.col0 : b.col1] = 255
    return mask


def plot_dispersion_curve(
    block_stats: List[BlockStats], curve: DispersionCurve, output_path: str
) -> None:
    """
    Plots the fitted s_hat(mu) curve together with the raw per-block
    (mu_k, s_k) scatter it was fit from, so the fit quality is visually
    inspectable, not just asserted.
    """
    eligible = [s for s in block_stats if s.eligible]
    mu_vals = np.array([s.mu_k for s in eligible], dtype=np.float64)
    s_vals = np.array([s.s_k for s in eligible], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mu_vals, s_vals, s=6, alpha=0.35, color="steelblue", label="eligible blocks (mu_k, s_k)")

    if curve.interp_x.size > 0:
        xs = np.linspace(0, 255, 512)
        ys = curve.evaluate(xs)
        ax.plot(xs, ys, color="crimson", linewidth=2, label="s_hat(mu) (binned median regression)")
        ax.scatter(
            curve.interp_x,
            curve.interp_y,
            color="black",
            marker="x",
            s=40,
            label="populated bin medians",
        )

    ax.set_xlabel("mu_k (local intensity level)")
    ax.set_ylabel("s_k (local dispersion, MAD of R)")
    ax.set_title("Global dispersion-vs-level curve")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 255)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)