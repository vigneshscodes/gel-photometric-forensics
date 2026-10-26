# Gel Photometric Forensics — LGPC-BC detector

CPU-only, classical (no deep learning) detector for **localized tonal edits** in gel / western-blot
images: it fits a global dispersion-vs-intensity curve and flags blocks whose local noise level is
inconsistent with it, while tolerating legitimate *global* brightness/contrast/gamma/tone-curve changes.

## Install and run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# single image or folder (8-bit single-channel input)
python run_detector.py --input path/to/image.png --output out/
# outputs: image_level_score.json, diagnostics.json, normalized_heatmap.png,
#          suspicious_region_mask.png, eligibility_mask.png, raw_block_consistency_map.npy, ...

# reproduce the seeded synthetic benchmark (calibration -> held-out test), ~1 min on CPU
python reproduce_benchmark.py --work-dir benchmark_runs/repro
```

## What was fixed (verified causes)

Baseline: F1 = 0, mean localization IoU ≈ 0.0063, 5 NaN scores.

| Problem | Verified cause | Fix |
|---|---|---|
| Nothing ever flagged | `beta=95` took the 95th percentile of \|z\| over ~170 blocks; a small edit touches ≤4 blocks, so M_image could not reach `tau`. `tau=3.5` was also never calibrated. | `beta=100` (max); `tau` calibrated on **negatives only**, on seeds disjoint from the test set (`calibrate_tau.py`). |
| Edited regions scored as "undefined" | Local contrast/gamma amplifies noise → raw Sobel gradient exceeds the global flat threshold → edited blocks were ineligible (0–3% of in-edit pixels had a score). | Flat-region gate uses a lightly Gaussian-smoothed gradient (`grad_smooth_sigma=0.7`, `grad_percentile=60`). |
| 5 NaN scores | Residual MAD = 0 (block MADs of 8-bit data are quantized, many blocks identical) → all z left undefined. | Scale ladder MAD → IQR → std with a floor (`estimate_residual_scale`); the source is logged in `diagnostics.json`. |
| False alarms on legitimate global tone curves | Raw MAD of integer data is heavily quantized (1.0 vs 2.4 after a compressive LUT); fixed [0,255] bins left only ~2 populated bins. | Trimmed-std dispersion estimator; quantile (data-adaptive) bins, `n_bins=12`. |
| Benchmark ~14 s/image | 129×129 `scipy` median filter. | Exact equivalent via reflect-pad + `cv2.medianBlur` (max abs diff 0.0 vs scipy). |

Legacy behaviour is still selectable (`dispersion_estimator="mad"`, `bin_mode="range"`, `grad_smooth_sigma=0`).
Research-handoff items used selectively: monotone-region opt-out (already the flat/eligibility gate),
MCC/AUC reporting, zero-dispersion guard. Not used (out of scope/not applicable here): Cao-style gap
fingerprints (fragile under JPEG, different problem), consensus gap templates, FBIGEL baseline (not implemented).

## Actual results (synthetic, seeded; `docs/results/`)

Calibration: seeds 5000–5007 (168 cases), τ = 7.549 chosen for ≤10% FPR on 72 negatives.
Held-out test: seeds 1000–1001 (42 cases: 18 negatives, 24 local-edit positives), τ = 7.55.

| Metric (test) | Baseline | Now |
|---|---|---|
| TP / FP / TN / FN | 0 / 3 / 15 / 24 | 8 / 1 / 17 / 16 |
| Precision / Recall | 0 / 0 | 0.889 / 0.333 |
| **F1** | 0.000 | **0.485** |
| MCC | – | 0.335 |
| AUC-ROC | – | 0.661 |
| False-positive rate | 0.167 | 0.056 |
| Mean localization IoU | 0.0063 | **0.087** (median 0.0) |
| NaN scores | 5 | 0 |

Detection by family (test): local_gamma 5/6, local_brightness 3/6, local_contrast 0/6,
local_nonlinear_tone 0/6; false alarms: 1/4 `global_monotonic_tone`, none on clean/brightness/contrast/gamma.
Excluding band-interior edits (documented undetectable): F1 = 0.583, recall 0.438.

## Limitations (please read)

- **Performance is modest.** Test set is tiny (24 positives), so these numbers have wide uncertainty; AUC 0.66 means weak image-level separation. Localization IoU is low (block-level maps, edits only 24×24 or 8×8 px, most edit blocks still ineligible).
- Only **synthetic** images (own noise model, hard-edged rectangles). No BioFors/real-image evaluation was run; the dataset is not in the ZIP. τ is calibrated on synthetic data and must be re-calibrated for real images.
- Not detected by design: edits confined to band interiors, pure brightness offsets without dispersion change, edits smaller than a block, matched-noise edits. That `local_brightness` is flagged 3/6 is **not** an intended mechanism; its cause was not investigated (possibly edge/eligibility side effects).
- Local contrast and local nonlinear-tone edits were not detected at the held-out τ.
- Max-aggregation (`beta=100`) is sensitive to single-block outliers; one legitimate steep global LUT still false-alarms.
- No FBIGEL baseline comparison was implemented. Screening aid only — not proof of misconduct.

## Layout

`src/detector/` (pipeline) · `src/evaluation/` (generator, benchmark, `calibrate_tau.py`, `evaluate_metrics.py`) ·
`src/preprocessing/` (BioFors dataset tooling, unchanged) · `reproduce_benchmark.py` · `docs/results/`
