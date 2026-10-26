
from pathlib import Path

import numpy as np

from src.detector.config import LGPCBCConfig
from src.detector.preprocessing import (
    load_grayscale_image,
    compute_saturation_mask,
)
from src.detector.background import (
    estimate_illumination_field,
    compute_illumination_corrected_residual,
)
from src.detector.regions import (
    compute_gradient_magnitude,
    compute_adaptive_grad_threshold,
    compute_flat_mask,
    partition_blocks,
    mark_eligibility,
)

ROOT = Path("outputs/benchmark/cases")

targets = [
    "clean_seed1000",
    "local_brightness_seed1000_normal_24x24_background",
    "local_contrast_seed1000_normal_24x24_background",
    "local_gamma_seed1000_normal_24x24_background",
]

cfg = LGPCBCConfig()

for target in targets:
    matches = list(ROOT.glob(f"{target}/image.png"))
    if not matches:
        matches = list(ROOT.glob(f"{target}/*.png"))

    if not matches:
        print(f"\n{target}: IMAGE NOT FOUND")
        continue

    path = matches[0]

    try:
        image = load_grayscale_image(str(path)).image
        sat = compute_saturation_mask(image, cfg.c_lo, cfg.c_hi)

        T = estimate_illumination_field(image, cfg.window_size)
        R = compute_illumination_corrected_residual(image, T)
        G = compute_gradient_magnitude(image)

        theta = compute_adaptive_grad_threshold(
            G, sat, cfg.grad_percentile
        )
        flat = compute_flat_mask(G, sat, theta)

        blocks = partition_blocks(*image.shape, cfg.block_size)
        mark_eligibility(blocks, flat, cfg.min_flat_pixels)

        eligible = [b for b in blocks if b.eligible]

        print(f"\n{target}")
        print(f"  image shape: {image.shape}")
        print(f"  gradient threshold: {theta:.6f}")
        print(f"  zero-gradient fraction: {np.mean(G == 0):.4f}")
        print(f"  flat-pixel fraction: {np.mean(flat):.4f}")
        print(f"  eligible blocks: {len(eligible)}/{len(blocks)}")
        print(f"  residual std: {np.std(R):.4f}")
        print(f"  residual MAD: {np.median(np.abs(R - np.median(R))):.4f}")

    except Exception as exc:
        print(f"\n{target}: ERROR: {exc}")
