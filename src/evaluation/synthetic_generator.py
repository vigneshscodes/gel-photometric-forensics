"""
Synthetic gel/blot-like image generator and controlled edit transforms,
for the LGPC-BC v1.1 methodology's Section 16 benchmark requirements.

SCOPE
-----
This module ONLY generates synthetic data (images, ground-truth masks,
transform parameters). It has no dependency on and does not alter
src/detector/*. It does not claim to reproduce real gel/blot acquisition
physics faithfully — it is a controlled testbed for exercising the
frozen method's stated assumptions (methodology Sections 3, 13) and
disclosed failure modes (Section 14), nothing more.

BASE IMAGE MODEL
-----------------
Each base image has:
  - a smooth illumination field (sinusoidal, low frequency)
  - heteroscedastic background noise whose scale depends on local
    intensity level (this deliberately matches LGPC-BC's own core
    modeling assumption, methodology Section 3 / Assumption 3, so the
    benchmark is a fair-but-controlled exercise of that assumption, not
    a real independent validation of it)
  - several Gaussian "bands" (dark blobs), each contribution tracked
    separately so a `band_mask` (True = band-interior pixel) can be
    produced for placing local edits precisely in/out of band regions.

EDIT TRANSFORMS
----------------
Global transforms are applied to every pixel. Local transforms apply the
identical pixel-wise mapping but only inside a rectangular region mask;
outside that region pixels are copied unchanged from the original. Edits
use hard (non-feathered) rectangular boundaries so the ground-truth mask
is exact and unambiguous — a deliberate simplification, disclosed here,
not a claim about how real local edits are made.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from scipy import ndimage


@dataclass
class BaseImage:
    image: np.ndarray  # uint8, (H, W), clean synthetic image
    band_mask: np.ndarray  # bool, (H, W), True = inside a synthetic band
    height: int
    width: int
    seed: int
    params: Dict[str, Any]


def generate_base_image(seed: int, height: int = 256, width: int = 256, n_bands: int = 4) -> BaseImage:
    """
    Deterministic (seed-controlled) synthetic gel/blot-like base image.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]

    illum = 180.0 + 15.0 * np.sin(xx / (width / 2.0)) + 10.0 * np.cos(yy / (height / 2.5))
    noise_scale = 2.5 + 0.02 * (255.0 - illum)
    background = illum + rng.normal(0, 1, size=(height, width)) * noise_scale

    band_contribution = np.zeros((height, width), dtype=np.float64)
    band_positions: List[Dict[str, float]] = []
    for _ in range(n_bands):
        cy = rng.uniform(0.15, 0.85) * height
        cx = rng.uniform(0.15, 0.85) * width
        sigma_y = rng.uniform(0.02, 0.035) * height
        sigma_x = rng.uniform(0.08, 0.15) * width
        amp = rng.uniform(60.0, 100.0)
        yy2 = (yy - cy) ** 2 / (2 * sigma_y ** 2)
        xx2 = (xx - cx) ** 2 / (2 * sigma_x ** 2)
        blob = amp * np.exp(-(yy2 + xx2))
        band_contribution += blob
        band_positions.append(
            {"cy": float(cy), "cx": float(cx), "sigma_y": float(sigma_y), "sigma_x": float(sigma_x), "amp": float(amp)}
        )

    image_f = background - band_contribution
    image = np.clip(image_f, 0, 255).astype(np.uint8)

    # A pixel counts as "band interior" once the band's own contribution
    # is large enough to matter (threshold chosen for a clearly-inside
    # region, not the faint band skirt) — a fixed, disclosed choice, not
    # a tunable benchmark parameter.
    band_mask = band_contribution > 5.0

    return BaseImage(
        image=image,
        band_mask=band_mask,
        height=height,
        width=width,
        seed=seed,
        params={"n_bands": n_bands, "band_positions": band_positions},
    )


def make_random_monotonic_lut(rng: np.random.Generator, n_levels: int = 256) -> np.ndarray:
    """
    A smooth, strictly monotonic non-decreasing 256-entry lookup table
    mapping input level -> output level, spanning ~[0, 255]. Used as the
    "nonparametric monotonic tone curve" transform (methodology's
    legitimate-global-edit family, Section 16) and, applied locally, as
    the "local nonlinear tone edit" tamper family.
    """
    increments = rng.uniform(0.2, 2.0, size=n_levels)
    cum = np.cumsum(increments)
    cum = cum - cum.min()
    cum = cum / cum.max() * 255.0
    return cum  # float64, length n_levels, monotonic non-decreasing


def apply_transform(image: np.ndarray, kind: str, params: Dict[str, Any]) -> np.ndarray:
    """
    Applies one of the four transform families to the WHOLE image.
    (apply_local_transform below restricts the same mapping to a region.)
    """
    img_f = image.astype(np.float64)
    if kind == "brightness":
        out = img_f + params["delta"]
    elif kind == "contrast":
        out = (img_f - 128.0) * params["factor"] + 128.0
    elif kind == "gamma":
        norm = np.clip(img_f, 0, 255) / 255.0
        out = 255.0 * np.power(norm, params["gamma"])
    elif kind == "nonlinear_tone":
        lut = params["lut"]
        idx = np.clip(img_f, 0, 255).astype(np.uint8)
        out = lut[idx]
    else:
        raise ValueError(f"Unknown transform kind: {kind}")
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_local_transform(
    image: np.ndarray, kind: str, params: Dict[str, Any], region_mask: np.ndarray
) -> np.ndarray:
    """
    Applies the same pixel-wise mapping as apply_transform, but only
    inside region_mask (bool, same shape as image). Pixels outside the
    mask are copied unchanged from the original — never resampled,
    blended, or feathered.
    """
    fully_transformed = apply_transform(image, kind, params)
    out = image.copy()
    out[region_mask] = fully_transformed[region_mask]
    return out


def find_region_for_placement(
    band_mask: np.ndarray,
    box_h: int,
    box_w: int,
    placement: str,
    rng: np.random.Generator,
    margin: int = 2,
    max_attempts: int = 5000,
) -> Tuple[int, int]:
    """
    Finds a (row0, col0) top-left corner for a box_h x box_w rectangle
    satisfying the requested placement relative to band_mask:
      - "background"    : box lies entirely outside bands (with a small
                           dilation margin so it isn't hugging a band edge).
      - "band_interior"  : box lies entirely inside a band (with a small
                           erosion margin so it isn't at the band's edge).
      - "crossing"       : box overlaps both band and non-band pixels.

    Raises RuntimeError (not a silently wrong placement) if no valid
    location is found within max_attempts random tries.
    """
    height, width = band_mask.shape
    if box_h > height or box_w > width:
        raise ValueError("Requested region is larger than the image.")

    if placement == "background":
        forbidden = ndimage.binary_dilation(band_mask, iterations=margin)
        allowed = ~forbidden
    elif placement == "band_interior":
        allowed = ndimage.binary_erosion(band_mask, iterations=margin)
    elif placement == "crossing":
        allowed = None
    else:
        raise ValueError(f"Unknown placement: {placement}")

    for _ in range(max_attempts):
        r0 = int(rng.integers(0, height - box_h + 1))
        c0 = int(rng.integers(0, width - box_w + 1))
        box_band = band_mask[r0 : r0 + box_h, c0 : c0 + box_w]

        if placement == "crossing":
            if box_band.any() and (~box_band).any():
                return r0, c0
            continue

        box_allowed = allowed[r0 : r0 + box_h, c0 : c0 + box_w]
        if box_allowed.all():
            return r0, c0

    raise RuntimeError(
        f"Could not find a '{placement}' region of size {box_h}x{box_w} "
        f"after {max_attempts} attempts on a {height}x{width} image with "
        f"this band layout."
    )


def make_region_mask(height: int, width: int, r0: int, c0: int, box_h: int, box_w: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[r0 : r0 + box_h, c0 : c0 + box_w] = True
    return mask
