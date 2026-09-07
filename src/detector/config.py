"""
Configuration for the LGPC-BC v1.1 detector.

IMPORTANT - READ BEFORE USE
----------------------------
Every numeric value in DEFAULT_PARAMS below is a PROVISIONAL IMPLEMENTATION
DEFAULT, not a scientifically calibrated or validated value. The frozen
methodology (v1.1, Section 11) explicitly states these parameters are
"uncalibrated placeholders, not defaults". They exist here only so the
program can be executed end-to-end (e.g. for the Stage 4 smoke test).

Before any real use of this detector:
  - `beta` and `tau` MUST be chosen via the synthetic benchmark
    (methodology Section 16), against a stated target false-positive rate.
  - `B`, `L`, `theta_grad`, `n_min`, `c_lo`, `c_hi`, `N_bin`, `epsilon_0`
    are independent, uncalibrated implementation parameters. None of them
    have literature-backed "correct" values for this method (the previously
    used "B >= 64x64" citation, Stamm & Liu 2010, was removed in v1.1 as
    misapplied — it concerns histogram-shape reliability, not a MAD-based
    statistic).

This module intentionally does NOT define `alpha_T`. The v1.0 background
estimator used a directional/percentile blend parameter; v1.1 replaced that
estimator with a plain large-window median filter, which has no blend
parameter. Keeping `alpha_T` around unused would be a dead, misleading knob,
so it has been removed rather than retained as a no-op.
"""

from dataclasses import dataclass, asdict, fields


@dataclass
class LGPCBCConfig:
    """
    All parameters governing one run of the LGPC-BC v1.1 detector.

    Every field maps 1:1 to a symbol in the frozen methodology
    (see docstring of detector.py for the section references).
    """

    # --- Block partitioning ---
    # B: side length (pixels) of the non-overlapping square analysis blocks.
    block_size: int = 16

    # --- Illumination-field estimation ---
    # L: side length (pixels) of the median-filter window used to estimate
    # the smooth illumination field T(u, v). Methodology requires L >> B.
    window_size: int = 129  # kept odd; see preprocessing.py note on parity

    # --- Flat-region gating ---
    # theta_grad: gradient-magnitude threshold below which a pixel is
    # considered "flat" (i.e. plausibly inside a locally-constant-signal
    # region rather than a band edge/interior).
    grad_threshold: float = 8.0

    # n_min: minimum number of flat pixels a block must contain to be
    # eligible for scoring.
    min_flat_pixels: int = 96  # ~37.5% of a 16x16 block by default

    # --- Saturation bounds ---
    # c_lo, c_hi: pixels at or below c_lo, or at or above c_hi, are treated
    # as saturated and excluded from every statistic.
    c_lo: int = 3
    c_hi: int = 252

    # --- Global dispersion-vs-level curve ---
    # N_bin: number of bins used to bin the observed intensity range when
    # fitting ŝ(μ) via binned robust median regression.
    n_bins: int = 16

    # --- Numerical stabilizer ---
    # epsilon_0: additive constant inside the log() calls in the residual
    # r_k, purely to avoid log(0). Not a scientific parameter.
    epsilon_0: float = 1.0e-6

    # --- Aggregation / decision ---
    # beta: percentile (0-100) of eligible-block anomaly scores used to
    # compute the image-level score M_image. NOT fixed by the methodology;
    # must be chosen by calibration (Section 16).
    beta: float = 95.0

    # tau: decision threshold on M_image. NOT fixed by the methodology;
    # must be chosen by calibration against a target false-positive rate.
    # The placeholder below is an arbitrary "z-score-like" cutoff and
    # carries no scientific validation whatsoever.
    tau: float = 3.5

    def validate(self) -> None:
        """Basic structural sanity checks (not scientific calibration)."""
        if self.block_size <= 0:
            raise ValueError("block_size (B) must be a positive integer.")
        if self.window_size <= self.block_size:
            raise ValueError(
                "window_size (L) must be much larger than block_size (B); "
                f"got L={self.window_size}, B={self.block_size}."
            )
        if self.window_size % 2 == 0:
            raise ValueError(
                "window_size (L) must be odd for a well-defined median "
                "filter footprint centered on each pixel."
            )
        if self.grad_threshold < 0:
            raise ValueError("grad_threshold must be non-negative.")
        if self.min_flat_pixels <= 0:
            raise ValueError("min_flat_pixels (n_min) must be positive.")
        if not (0 <= self.c_lo < self.c_hi <= 255):
            raise ValueError("Require 0 <= c_lo < c_hi <= 255.")
        if self.n_bins <= 0:
            raise ValueError("n_bins (N_bin) must be positive.")
        if self.epsilon_0 <= 0:
            raise ValueError("epsilon_0 must be positive.")
        if not (0 < self.beta <= 100):
            raise ValueError("beta must be a percentile in (0, 100].")

    def as_dict(self) -> dict:
        return asdict(self)

    def provisional_flags(self) -> dict:
        """
        Which fields are scientifically uncalibrated placeholders, for
        inclusion in diagnostics output so results are never presented as
        validated defaults.
        """
        return {f.name: True for f in fields(self)}


def default_config() -> LGPCBCConfig:
    """Return a fresh config with provisional implementation defaults."""
    cfg = LGPCBCConfig()
    cfg.validate()
    return cfg
