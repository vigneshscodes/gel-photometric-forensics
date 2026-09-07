"""
Input handling for LGPC-BC v1.1 (methodology Sections 1-2).

Responsibilities:
  - Load an image and validate it matches the method's stated input scope
    (single-channel, 8-bit grayscale). No resizing, ever.
  - Compute the saturation mask Sat(u, v).

IMPLEMENTATION DECISION (documented ambiguity #1)
--------------------------------------------------
The methodology states color/fluorescent images are "out of scope
(excluded upstream by the dataset pipeline)" — i.e. it assumes a prior
stage already filtered them out. This standalone detector module has no
such upstream guarantee (it may be pointed at an arbitrary image file).

Two options existed:
  (a) Silently convert any color image to grayscale and proceed, or
  (b) Refuse to process non-grayscale / non-8-bit input.

(a) risks silently changing the pixel-noise statistics the entire method
depends on (RGB->gray conversion is itself a nonlinear/weighted mapping,
and fluorescent channel merges are not a "single smooth global phi").
That would be introducing behavior the methodology never specified.

Decision taken: (b). This module REFUSES non-grayscale or non-8-bit
input with a clear error, rather than guessing a conversion. If the
caller truly has a single-channel grayscale image encoded as a 3-channel
file (e.g. all channels identical), they can pre-convert it themselves;
this module will not do it implicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import cv2


class InputValidationError(ValueError):
    """Raised when an input image does not match the method's declared scope."""


@dataclass
class LoadedImage:
    path: str
    image: np.ndarray  # 2D uint8 array, shape (H, W)
    height: int
    width: int


def load_grayscale_image(path: str) -> LoadedImage:
    """
    Load an image from disk and validate it is single-channel 8-bit.

    Raises InputValidationError if the file cannot be read, or is not
    single-channel 8-bit grayscale. Never resizes and never silently
    converts color -> grayscale (see module docstring).
    """
    if not os.path.isfile(path):
        raise InputValidationError(f"Input path does not exist or is not a file: {path}")

    # IMREAD_UNCHANGED preserves channel count and bit depth so we can
    # validate honestly, instead of IMREAD_GRAYSCALE which would silently
    # collapse color images for us.
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise InputValidationError(f"Failed to decode image: {path}")

    if raw.ndim == 3:
        # Could be a color image, or a grayscale image saved with a
        # redundant channel dimension of size 1.
        if raw.shape[2] == 1:
            raw = raw[:, :, 0]
        else:
            raise InputValidationError(
                f"'{path}' has {raw.shape[2]} channels. LGPC-BC v1.1 is defined "
                "only for single-channel 8-bit grayscale images; color/"
                "fluorescent images are explicitly out of scope. Convert "
                "the image to single-channel grayscale yourself before "
                "running the detector — this tool will not guess a "
                "channel-merge for you."
            )

    if raw.dtype != np.uint8:
        raise InputValidationError(
            f"'{path}' has dtype {raw.dtype}, not uint8. LGPC-BC v1.1 is defined "
            "for 8-bit grayscale input. Re-encode as 8-bit before running "
            "the detector — silently rescaling bit depth would alter the "
            "sensor-noise statistics this method depends on."
        )

    h, w = raw.shape
    return LoadedImage(path=path, image=raw, height=h, width=w)


def compute_saturation_mask(image: np.ndarray, c_lo: int, c_hi: int) -> np.ndarray:
    """
    Sat(u, v) = 1 if I(u, v) <= c_lo or I(u, v) >= c_hi, else 0.

    Saturated pixels have truncated/unobservable true dispersion and are
    excluded from every downstream statistic (they are never counted as
    "flat" pixels, regardless of local gradient).
    """
    if image.dtype != np.uint8:
        raise ValueError("compute_saturation_mask expects a uint8 image.")
    sat = (image <= c_lo) | (image >= c_hi)
    return sat.astype(bool)
