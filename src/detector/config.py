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
    are independent, uncalibrated implementation parameters.

This module intentionally does NOT define `alpha_T`. The v1.0 background
estimator used a directional/percentile blend parameter; v1.1 replaced that
estimator with a plain large-window median filter, which has no blend
parameter.
"""

from dataclasses import dataclass, asdict, fields
from typing import Optional


@dataclass
class LGPCBCConfig:
    """
    All parameters governing one run of the LGPC-BC v1.1 detector.
    """

    # --- Block partitioning ---
    block_size: int = 16

    # --- Illumination-field estimation ---
    window_size: int = 129

    # --- Flat-region gating ---
    #
    # Optional manual override.
    # If None, detector.py derives theta_grad per image using
    # grad_percentile.
    grad_threshold: Optional[float] = None

    # Percentile used to derive the gradient threshold when
    # grad_threshold is None.
    grad_percentile: float = 40.0

    # --- Block eligibility ---
    min_flat_pixels: int = 96

    # --- Saturation bounds ---
    c_lo: int = 3
    c_hi: int = 252

    # --- Global dispersion-vs-level curve ---
    #
    # Reduced from 16 to 8 because the synthetic images often occupy
    # a limited intensity range.
    n_bins: int = 8

    # --- Numerical stabilizer ---
    epsilon_0: float = 1.0e-6

    # --- Aggregation / decision ---
    #
    # PROVISIONAL ONLY.
    # Must be calibrated using the synthetic benchmark.
    beta: float = 95.0

    # PROVISIONAL ONLY.
    # Must be calibrated against the desired false-positive rate.
    tau: float = 3.5

    def validate(self) -> None:
        """Basic structural sanity checks (not scientific calibration)."""

        if self.block_size <= 0:
            raise ValueError(
                "block_size (B) must be a positive integer."
            )

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

        # Manual gradient threshold is optional.
        if (
            self.grad_threshold is not None
            and self.grad_threshold < 0
        ):
            raise ValueError(
                "grad_threshold must be non-negative."
            )

        if not (0.0 < self.grad_percentile < 100.0):
            raise ValueError(
                "grad_percentile must be in (0, 100)."
            )

        if self.min_flat_pixels <= 0:
            raise ValueError(
                "min_flat_pixels (n_min) must be positive."
            )

        if not (0 <= self.c_lo < self.c_hi <= 255):
            raise ValueError(
                "Require 0 <= c_lo < c_hi <= 255."
            )

        if self.n_bins <= 0:
            raise ValueError(
                "n_bins (N_bin) must be positive."
            )

        if self.epsilon_0 <= 0:
            raise ValueError(
                "epsilon_0 must be positive."
            )

        if not (0 < self.beta <= 100):
            raise ValueError(
                "beta must be a percentile in (0, 100]."
            )

    def as_dict(self) -> dict:
        return asdict(self)

    def provisional_flags(self) -> dict:
        """
        Mark every configuration field as scientifically uncalibrated.
        """
        return {
            f.name: True
            for f in fields(self)
        }


def default_config() -> LGPCBCConfig:
    """Return a fresh config with provisional implementation defaults."""
    cfg = LGPCBCConfig()
    cfg.validate()
    return cfg