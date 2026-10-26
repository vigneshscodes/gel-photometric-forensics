"""
Local analysis: gradient/flat masking, block partitioning, eligibility.
Methodology Sections 5 and 7.

Section 7 states eligibility (|F_k| >= n_min) is the ONLY region-selection
rule — no additional handcrafted region logic is added here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import cv2


@dataclass
class Block:
    """One B x B (or smaller at image edges) block."""

    index: int
    row0: int
    col0: int
    row1: int
    col1: int
    n_flat_pixels: int
    eligible: bool


def compute_gradient_magnitude(image: np.ndarray, smooth_sigma: float = 0.0) -> np.ndarray:
    """
    G(u, v): gradient magnitude of I via a standard first-derivative
    operator, computed from I (not R) per methodology Section 5.

    Uses a 3x3 Sobel operator.
    """
    image_f = image.astype(np.float64)
    if smooth_sigma and smooth_sigma > 0:
        image_f = cv2.GaussianBlur(image_f, (0, 0), smooth_sigma, borderType=cv2.BORDER_REFLECT)

    gx = cv2.Sobel(
        image_f,
        cv2.CV_64F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        image_f,
        cv2.CV_64F,
        0,
        1,
        ksize=3,
    )

    G = np.sqrt(gx ** 2 + gy ** 2)

    return G


def compute_adaptive_grad_threshold(
    gradient_magnitude: np.ndarray,
    saturation_mask: np.ndarray,
    percentile: float,
) -> float:
    """
    Compute an image-adaptive gradient threshold from the observed
    non-saturated gradient-magnitude distribution.

    The threshold is the requested percentile of gradient magnitudes
    over non-saturated pixels.

    Flat(u, v) remains:
        G(u, v) < theta_grad and Sat(u, v) == 0

    Eligibility remains governed solely by:
        |F_k| >= n_min
    """
    if gradient_magnitude.shape != saturation_mask.shape:
        raise ValueError(
            "gradient_magnitude and saturation_mask must have the same shape."
        )

    if not (0.0 < percentile < 100.0):
        raise ValueError("percentile must be in (0, 100).")

    non_saturated = gradient_magnitude[~saturation_mask]

    if non_saturated.size == 0:
        raise ValueError(
            "Cannot derive an adaptive gradient threshold: every pixel is "
            "saturated (no non-saturated pixels to estimate a threshold from)."
        )

    return float(np.percentile(non_saturated, percentile))


def compute_flat_mask(
    gradient_magnitude: np.ndarray,
    saturation_mask: np.ndarray,
    theta_grad: float,
) -> np.ndarray:
    """
    Flat(u, v) = 1 if G(u, v) < theta_grad and Sat(u, v) == 0.
    """
    if gradient_magnitude.shape != saturation_mask.shape:
        raise ValueError(
            "gradient_magnitude and saturation_mask must have the same shape."
        )

    flat = (
        (gradient_magnitude < theta_grad)
        & (~saturation_mask)
    )

    return flat.astype(bool)


def partition_blocks(
    height: int,
    width: int,
    block_size: int,
) -> List[Block]:
    """
    Partition an H x W image into non-overlapping B x B blocks in
    row-major order, keeping partial blocks at the bottom/right edge.

    Eligibility is not decided here.
    """
    blocks: List[Block] = []
    idx = 0

    for row0 in range(0, height, block_size):
        row1 = min(row0 + block_size, height)

        for col0 in range(0, width, block_size):
            col1 = min(col0 + block_size, width)

            blocks.append(
                Block(
                    index=idx,
                    row0=row0,
                    col0=col0,
                    row1=row1,
                    col1=col1,
                    n_flat_pixels=0,
                    eligible=False,
                )
            )

            idx += 1

    return blocks


def mark_eligibility(
    blocks: List[Block],
    flat_mask: np.ndarray,
    n_min: int,
) -> None:
    """
    Sets n_flat_pixels and eligible=True/False on each Block.

    Eligibility rule (the ONLY region-selection rule):
        eligible iff |F_k| >= n_min
    """
    for b in blocks:
        sub_flat = flat_mask[
            b.row0:b.row1,
            b.col0:b.col1,
        ]

        n_flat = int(np.count_nonzero(sub_flat))

        b.n_flat_pixels = n_flat
        b.eligible = n_flat >= n_min