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
    """One B x B (or smaller, at image edges — see note below) block."""

    index: int
    row0: int
    col0: int
    row1: int  # exclusive
    col1: int  # exclusive
    n_flat_pixels: int
    eligible: bool


# IMPLEMENTATION DECISION (documented ambiguity #3)
# ---------------------------------------------------
# The methodology says "partition into non-overlapping B x B blocks" but
# does not say what to do when H or W is not an exact multiple of B.
# Two options: (a) crop the image to a multiple of B, discarding a strip
# of pixels, or (b) keep partial blocks at the bottom/right edge.
#
# (a) silently discards real image content and would shrink saturation/
# gradient statistics computed elsewhere in a way the methodology never
# asks for. (b) keeps every pixel but means a small number of blocks are
# smaller than B x B (hence contain fewer possible flat pixels).
#
# Decision taken: (b), partial edge blocks are kept. Since eligibility
# already depends only on the *count* of flat pixels (n_min), a smaller
# edge block is simply less likely to reach n_min — it is not silently
# treated as equivalent to a full block, and the diagnostics report block
# sizes are not otherwise adjusted for this. This does not add any new
# region-selection rule; eligibility is still governed solely by n_min.
def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """
    G(u, v): gradient magnitude of I via a standard first-derivative
    operator, computed from I (not R) per methodology Section 5, to avoid
    amplifying the background filter's own edge artifacts.

    IMPLEMENTATION DECISION (documented ambiguity #4): the methodology
    says "a standard first-derivative operator" without naming one. We use
    a 3x3 Sobel operator (the most common default choice for gradient-
    magnitude flat/edge gating in image forensics), applied to the image
    as float64. This is a fixed, disclosed choice, not a tunable knob.
    """
    image_f = image.astype(np.float64)
    gx = cv2.Sobel(image_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image_f, cv2.CV_64F, 0, 1, ksize=3)
    G = np.sqrt(gx ** 2 + gy ** 2)
    return G


def compute_flat_mask(
    gradient_magnitude: np.ndarray,
    saturation_mask: np.ndarray,
    theta_grad: float,
) -> np.ndarray:
    """
    Flat(u, v) = 1 if G(u, v) < theta_grad and Sat(u, v) == 0, else 0.
    """
    if gradient_magnitude.shape != saturation_mask.shape:
        raise ValueError("gradient_magnitude and saturation_mask must have the same shape.")
    flat = (gradient_magnitude < theta_grad) & (~saturation_mask)
    return flat.astype(bool)


def partition_blocks(height: int, width: int, block_size: int) -> List[Block]:
    """
    Partition an H x W image into non-overlapping B x B blocks in
    row-major order, keeping partial blocks at the bottom/right edge
    (see documented ambiguity #3 above). Eligibility is not decided here.
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


def mark_eligibility(blocks: List[Block], flat_mask: np.ndarray, n_min: int) -> None:
    """
    Sets n_flat_pixels and eligible=True/False on each Block in place.

    Eligibility rule (the ONLY region-selection rule per Section 7):
        eligible iff |F_k| >= n_min
    """
    for b in blocks:
        sub_flat = flat_mask[b.row0 : b.row1, b.col0 : b.col1]
        n_flat = int(np.count_nonzero(sub_flat))
        b.n_flat_pixels = n_flat
        b.eligible = n_flat >= n_min
