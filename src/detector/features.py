"""
Complementary local-global photometric channels (LGPC-BC v1.2).

The v1.1 detector scores ONE feature: block noise dispersion s_k against the
global dispersion-vs-level curve s_hat(mu). That cannot see a pure additive
brightness shift (dispersion unchanged) and is weak for mild contrast/gamma
edits (a 16x16 block gives a noisy log-dispersion). This module adds channels
that are all "local value vs. one global model fitted on the same image", so
a legitimate global edit (which moves every block together) stays consistent:

  level : block level mu_k vs a robust global low-order background surface
          (additive / brightness edits). Gated by edge sharpness so that
          smooth band structure is not mistaken for a tonal step.
  hist  : Poisson-normalised histogram roughness of the block neighbourhood
          (gap/peak fingerprint of expansive/compressive tone mappings;
          Stamm & Liu, Cao et al., Bi et al.), compared with the binned
          global curve roughness-vs-level (so a globally enhanced image,
          rough everywhere, remains consistent).
  pooled: every channel is also evaluated after signed 3x3 block pooling,
          which suppresses independent noise but keeps a coherent edit.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
from scipy import ndimage

from .consistency import estimate_residual_scale
from .statistics import BlockStats, fit_dispersion_curve

LEVEL_SCALE_FLOOR = 0.75   # gray levels; block medians of 8-bit data are quantised
LOG_SCALE_FLOOR = 0.03     # log-units, for pooled log-ratio channels


# ----------------------------------------------------------------- helpers
def block_grid_shape(blocks) -> Tuple[int, int]:
    nc = sum(1 for b in blocks if b.row0 == 0)
    nr = len(blocks) // nc
    return nr, nc


def to_grid(values: np.ndarray, blocks) -> np.ndarray:
    nr, nc = block_grid_shape(blocks)
    return np.asarray(values, dtype=np.float64).reshape(nr, nc)


def pool_signed(grid: np.ndarray, size: int = 3, min_valid: int = 3) -> np.ndarray:
    """Mean over a size x size block neighbourhood, ignoring NaN."""
    valid = np.isfinite(grid)
    s = ndimage.uniform_filter(np.where(valid, grid, 0.0), size=size, mode="constant") * size * size
    n = ndimage.uniform_filter(valid.astype(np.float64), size=size, mode="constant") * size * size
    out = np.full(grid.shape, np.nan)
    ok = (n >= min_valid - 1e-9) & valid
    out[ok] = s[ok] / n[ok]
    return out


def robust_z(values: np.ndarray, floor: float) -> np.ndarray:
    """(v - median) / robust scale over finite entries, with a scale floor."""
    v = np.asarray(values, dtype=np.float64)
    z = np.full(v.shape, np.nan)
    f = np.isfinite(v)
    if f.sum() < 4:
        return z
    med = np.median(v[f])
    scale, _ = estimate_residual_scale(v[f] - med)
    if not np.isfinite(scale) or scale <= 0:
        scale = floor
    scale = max(scale, floor)
    z[f] = (v[f] - med) / scale
    return z


# ------------------------------------------------------------ level channel
def _poly_design(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    cols = [(x ** i) * (y ** j) for i in range(degree + 1) for j in range(degree + 1 - i)]
    return np.stack(cols, axis=1)


def fit_robust_surface(row_c, col_c, values, shape, degree: int = 2, n_iter: int = 20):
    """
    Tukey-biweight IRLS low-order polynomial surface through block levels.
    A low-order global surface models legitimate shading; bands and local
    edits are minority outliers and are rejected, so they do not pull the
    reference toward themselves (unlike a window median that adapts to any
    edit covering most of its window).
    Returns (fitted_values_at_points, scale).
    """
    H, W = shape
    x = 2.0 * np.asarray(col_c) / max(W - 1, 1) - 1.0
    y = 2.0 * np.asarray(row_c) / max(H - 1, 1) - 1.0
    v = np.asarray(values, dtype=np.float64)
    X = _poly_design(x, y, degree)
    n_par = X.shape[1]

    fit = np.full(v.shape, np.median(v))
    r = v - fit
    scale = max(1.4826 * np.median(np.abs(r - np.median(r))), LEVEL_SCALE_FLOOR)
    for it in range(n_iter):
        c = max(4.685, 12.0 - 1.5 * it)  # start permissive, tighten to Tukey
        u = r / (c * scale)
        w = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
        if np.count_nonzero(w > 1e-6) < n_par + 3:
            break
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], v * sw, rcond=None)
        fit = X @ beta
        r = v - fit
        scale = max(1.4826 * np.median(np.abs(r - np.median(r))), LEVEL_SCALE_FLOOR)
    return fit, scale


def _sobel_mag(img: np.ndarray, sigma: float) -> np.ndarray:
    f = cv2.GaussianBlur(img.astype(np.float64), (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def level_channel(blocks, block_stats: List[BlockStats], G: np.ndarray, theta_grad: float,
                  shape, degree: int = 2, step_ratio: float = 0.15, min_blocks: int = 12,
                  image: np.ndarray = None, coarse_sigma: float = 3.0, sharp_min: float = 2.0):
    """
    A deviation is credited only if it is a STEP: the fine-scale gradient
    near the block must (a) be consistent with a step of at least
    step_ratio*|d| and (b) shrink strongly when the image is blurred
    (coarse_sigma): a hard-edged patch loses ~4x gradient, whereas a smooth
    Gaussian-profile band keeps ~1x. Small deviations (< 3 robust sigma)
    are always scored.

    Returns (d_k, gated_d_k): signed level deviation of every eligible block
    from the robust global surface (NaN for ineligible), and the same after
    the sharp-step gate (NaN where the deviation is NOT supported by a step
    discontinuity -> smooth band structure / unsupported, so not scored).
    """
    n = len(blocks)
    d = np.full(n, np.nan)
    gated = np.full(n, np.nan)
    el = np.array([s.eligible and np.isfinite(s.mu_k) for s in block_stats])
    if el.sum() < min_blocks:
        return d, gated, float("nan")
    rc = np.array([(b.row0 + b.row1 - 1) / 2.0 for b in blocks])
    cc = np.array([(b.col0 + b.col1 - 1) / 2.0 for b in blocks])
    mu = np.array([s.mu_k for s in block_stats])
    fit, scale = fit_robust_surface(rc[el], cc[el], mu[el], shape, degree)
    d[el] = mu[el] - fit

    # Edge sharpness within the block plus a 4 px margin.
    B = max(b.row1 - b.row0 for b in blocks)
    k = B + 9
    Gmax = cv2.dilate(G.astype(np.float32), np.ones((k, k), np.uint8))
    Gc = _sobel_mag(image, coarse_sigma) if image is not None else None
    Gcmax = cv2.dilate(Gc.astype(np.float32), np.ones((k, k), np.uint8)) if Gc is not None else None
    H, W = shape
    for i in np.flatnonzero(el):
        b = blocks[i]
        g = float(Gmax[min(int(rc[i]), H - 1), min(int(cc[i]), W - 1)])
        # Sobel response of a blurred step of height a is ~ 8 * 0.57 * a / px.
        ratio = max(g - theta_grad, 0.0) / (8.0 * max(abs(d[i]), 2.0))
        sharp = np.inf
        if Gcmax is not None:
            gc = float(Gcmax[min(int(rc[i]), H - 1), min(int(cc[i]), W - 1)])
            sharp = max(g - theta_grad, 0.0) / (gc + 0.25 * theta_grad + 1e-6)
        if (ratio >= step_ratio and sharp >= sharp_min) or abs(d[i]) < 3.0 * scale:
            gated[i] = d[i]
    return d, gated, scale


# ----------------------------------------------------------- hist channel
def histogram_roughness(image: np.ndarray, flat_mask: np.ndarray, blocks,
                        block_stats: List[BlockStats], window: int = 32,
                        min_pixels: int = 256) -> np.ndarray:
    """
    Per-block Poisson-normalised second-difference roughness of the grey-level
    histogram of flat pixels in a window centred on the block:
        chi = mean_i  (h_i - (h_{i-1}+h_{i+1})/2)^2 / (1.5 * mean_loc_i + 1)
    ~1 for a smooth noisy histogram; >>1 for gap/peak (comb) structure.
    """
    H, W = image.shape
    out = np.full(len(blocks), np.nan)
    B = max(b.row1 - b.row0 for b in blocks)
    pad = max((window - B) // 2, 0)
    for i, (b, s) in enumerate(zip(blocks, block_stats)):
        if not s.eligible:
            continue
        r0, r1 = max(0, b.row0 - pad), min(H, b.row1 + pad)
        c0, c1 = max(0, b.col0 - pad), min(W, b.col1 + pad)
        vals = image[r0:r1, c0:c1][flat_mask[r0:r1, c0:c1]]
        if vals.size < min_pixels:
            continue
        lo, hi = np.percentile(vals, [3, 97])
        lo, hi = int(np.floor(lo)), int(np.ceil(hi))
        if hi - lo < 8 or lo < 1 or hi > 254:
            continue
        h = np.bincount(vals, minlength=256).astype(np.float64)
        idx = np.arange(lo + 1, hi)
        dd = h[idx] - 0.5 * (h[idx - 1] + h[idx + 1])
        loc = (h[idx - 1] + h[idx] + h[idx + 1]) / 3.0
        out[i] = float(np.mean(dd * dd / (1.5 * loc + 1.0)))
    return out


def hist_residuals(block_stats: List[BlockStats], chi: np.ndarray, n_bins: int, bin_mode: str):
    """log(chi) minus binned-median global curve of log(chi) vs level."""
    n = len(block_stats)
    r = np.full(n, np.nan)
    ok = np.isfinite(chi)
    if ok.sum() < 12:
        return r
    y = np.log(chi + 0.1)
    fake = [BlockStats(s.block_index, s.row0, s.col0, s.row1, s.col1,
                       bool(ok[i]), s.n_flat_pixels,
                       s.mu_k if ok[i] else np.nan, float(y[i]) + 10.0 if ok[i] else np.nan)
            for i, s in enumerate(block_stats)]  # +10 keeps y >= 0 for the curve helper
    curve = fit_dispersion_curve(fake, min(n_bins, 8), bin_mode)
    mu = np.array([s.mu_k for s in block_stats])
    ref = curve.evaluate(mu) - 10.0
    r[ok] = y[ok] - ref[ok]
    return r