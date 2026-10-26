"""
Illumination field estimation, methodology Section 4 (v1.1 bug fix).

v1.0 used a fixed-direction percentile filter to estimate the smooth
illumination field, silently assuming dark-band/light-background polarity.
v1.1 replaces this with a large-window MEDIAN filter, which is
polarity-agnostic: the background is the area-majority pixel class in a
gel/blot image regardless of whether it is the light or dark extreme, so
the median tracks it either way.

This module implements exactly that, and nothing more:
  T(u, v) = median filter of I over an L x L window
  R(u, v) = I(u, v) - T(u, v)

No alpha blending, no directional percentile, no polarity detection.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import median_filter

# IMPLEMENTATION DECISION (documented ambiguity #2)
# ---------------------------------------------------
# The methodology specifies "local median filter over an L x L window" but
# does not specify the filter's boundary handling at image edges. We use
# scipy.ndimage.median_filter with mode="reflect" (mirror padding), which
# is the standard choice for illumination/background estimation: it avoids
# fabricating out-of-image pixel values (as "constant" padding would) and
# avoids wrap-around artifacts (as "wrap" would). This only affects a
# border strip of width ~L/2 pixels around the image edge.
#
# We use scipy.ndimage.median_filter rather than cv2.medianBlur because
# OpenCV's medianBlur only supports large (non-5x5-capped) kernels for
# 8-bit single-channel input via a specific fast path, is restricted for
# other configurations, and always applies a *square* footprint the same
# way scipy does here — scipy gives us explicit, predictable boundary-mode
# control, which the methodology's correctness argument (median as
# area-majority estimator) depends on being handled deliberately rather
# than left to a library default.
_BOUNDARY_MODE = "reflect"


def estimate_illumination_field(image: np.ndarray, window_size: int) -> np.ndarray:
    """
    T(u, v): large-window median filter of I, per methodology Section 4.

    Parameters
    ----------
    image : np.ndarray, uint8, shape (H, W)
    window_size : int (L). Must be odd and >> block size (validated by
        LGPCBCConfig.validate() upstream; not re-validated here).

    Returns
    -------
    np.ndarray, float64, shape (H, W): the estimated illumination field.
    """
    if image.ndim != 2:
        raise ValueError("estimate_illumination_field expects a 2D grayscale image.")
    if window_size % 2 == 0:
        raise ValueError("window_size (L) must be odd.")

    if image.dtype == np.uint8:
        # Exact same result as scipy.ndimage.median_filter(mode="reflect")
        # (cv2.BORDER_REFLECT == scipy "reflect"), but ~1000x faster for
        # large windows on uint8 data.
        pad = window_size // 2
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        T = cv2.medianBlur(padded, window_size)[pad:-pad, pad:-pad]
        return T.astype(np.float64)
    image_f = image.astype(np.float64)
    return median_filter(image_f, size=window_size, mode=_BOUNDARY_MODE)


def compute_illumination_corrected_residual(image: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    R(u, v) = I(u, v) - T(u, v)

    Isolates local content from the smooth illumination field before any
    consistency statistic is computed, so a legitimate shading gradient
    combined with a legitimate global phi cannot be mistaken for a local
    inconsistency.

    Note (methodology-disclosed leakage): T at a pixel is computed from a
    window containing that pixel's own neighborhood, so R is not fully
    independent of block statistics computed from it. For window_size (L)
    >> block_size (B) this leakage is small (bounded roughly by B^2 / L^2)
    and is not corrected for here, per the frozen methodology.
    """
    if image.shape != T.shape:
        raise ValueError("image and illumination field T must have the same shape.")
    R = image.astype(np.float64) - T
    return R
