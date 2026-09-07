# LGPC-BC v1.1 Detector — Integration Notes

## What this is

An implementation of the frozen **LGPC-BC v1.1** methodology (Local-Global
Photometric Consistency — Background Consistency) for flagging local tonal
inconsistencies in single-acquisition gel/blot images. It is a classical,
CPU-only statistical method — no learned models, no deep learning.

**This has NOT been empirically validated.** No benchmark has been run yet.
Every scientifically meaningful parameter (`B`, `L`, `theta_grad`, `n_min`,
`c_lo`, `c_hi`, `N_bin`, `beta`, `tau`) is an uncalibrated placeholder — see
`src/detector/config.py`. Do not treat any flag/no-flag decision this code
produces as validated until the Section 16 synthetic benchmark has been run
and `beta`/`tau` chosen against a stated false-positive target.

## What this does NOT claim to detect

- Tonal edits confined entirely to band interiors (structurally ineligible).
- Pure local brightness offsets with no accompanying dispersion change.
- Matched-noise-evasion manipulation (attacker re-synthesizes noise to
  match the fitted curve).
- Edits smaller than one block (`B x B`).
- Anomalies in images with a very low sensor-noise floor.
- Consistency violations in composite/multi-acquisition figures — those
  are out of scope for this version; exclude them upstream using this
  project's own `composite_panel_suspect`-style flag before running this
  detector, exactly as the methodology's benchmark section specifies.

## Integration into your existing project1

I don't have access to your actual `src/preprocessing/` scripts or your
existing `src/config.py`/`README.md` in this environment (only the folder
structure you described), so **I did not touch or recreate them.** What's
included here is only the new material:

```
src/detector/       <- new package, complete
src/evaluation/      <- empty placeholder (.gitkeep), ready for Section 16 later
run_detector.py       <- new CLI entry point, at project root
requirements.txt      <- new, only detector's own dependencies
DETECTOR_README.md   <- this file
```

Drop these into your existing `project1/` alongside your current
`src/preprocessing/` and `src/config.py`, unmodified. `run_detector.py`
does not import anything from `src/preprocessing/` or your existing
`src/config.py` — the two pipelines are fully decoupled, so nothing in
your frozen preprocessing stage is at risk.

Note: this new `src/detector/config.py` is a **separate file** scoped to
the detector package (`src/detector/config.py`, not `src/config.py`), so
it cannot collide with or shadow your existing top-level `src/config.py`.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Single image:

```bash
python run_detector.py --input path/to/image.png --output path/to/output_dir
```

Batch (directory of images):

```bash
python run_detector.py --input path/to/image_dir --output path/to/output_root
```

Optional parameter overrides (all still uncalibrated placeholders unless
*you* have calibrated them via the Section 16 benchmark):

```bash
python run_detector.py --input img.png --output out/ \
    --block-size 16 --window-size 129 --grad-threshold 25 \
    --min-flat-pixels 96 --c-lo 3 --c-hi 252 --n-bins 16 \
    --beta 95 --tau 3.5
```

## Output structure (per image)

```
output_dir/
├── input_info.json                 image dims/dtype, validated-input record
├── eligibility_mask.png            255 = eligible block, 0 = ineligible
├── raw_block_consistency_map.npy   |z_k| per pixel; NaN = UNDEFINED (ineligible)
├── normalized_heatmap.png          colormapped heatmap; gray = UNDEFINED
├── suspicious_region_mask.png      eligible blocks with |z_k| >= tau
├── dispersion_curve.png            fitted s_hat(mu) vs. raw (mu_k, s_k)
├── dispersion_curve.csv            per-bin data behind the curve
├── block_stats.csv                 per-block mu_k, s_k, s_hat, r_k, z_k
├── diagnostics.json                all required Section 15 diagnostics
└── image_level_score.json          M_image, tau, decision
```

`diagnostics.json` always includes the active parameter values, the
eligible/ineligible block fractions, per-bin eligible-block counts (to
flag under-populated bins), the residual-MAD goodness-of-fit indicator
(new in v1.1), and the full scope-limitation list — so no output can be
read as a validated, unconditional verdict.

## Smoke test performed

Run on synthetic single-channel 8-bit images generated locally (not
provided by you) to confirm the pipeline executes end-to-end without
crashing and that its qualitative behavior matches the specification:

- A "clean" synthetic gel-like image (smooth illumination gradient +
  heteroscedastic background noise + several Gaussian-blob "bands").
- The same image with a local contrast-reduction edit applied to one
  background patch (a case the methodology explicitly says should be
  detectable, since it changes local dispersion without moving the
  block off the fitted curve).

Result: the pipeline ran to completion in single-image, directory-batch,
invalid-input (color image, wrong dtype, missing file), and
invalid-config (even window size) cases, producing all required outputs
each time. On the tampered synthetic image, the two blocks covering the
edited patch received the two highest `|z_k|` values in the entire image
and appear as an isolated bright cluster in `normalized_heatmap.png` at
the correct spatial location. This is a plausibility check on synthetic
data only — it is not evidence of real-world validity and must not be
cited as such; only the Section 16 benchmark can establish that.

One practical observation from smoke testing, not a code defect: with a
low-noise synthetic image and small `n_min`, the per-block dispersion
statistic `s_k` can be discretized enough that many blocks land on
exactly the same bin-median value, driving the residual population's MAD
to exactly zero. The implementation treats that degenerate case as
`z_k` = undefined (NaN) rather than dividing by zero or silently
returning 0 — which is the same "undefined, not consistent" discipline
used for ineligible blocks. `M_image` is then reported as `NaN`/undefined
for that image rather than a fabricated decision. This is expected
behavior of the frozen formulas on such inputs, not something this
implementation "fixed" by adding new logic.

## Methodology ambiguities encountered and resolved (see code comments for full detail)

1. **Non-grayscale/non-8-bit input.** No upstream filtering guarantee
   exists inside this standalone module. Decision: refuse and error
   rather than silently convert (`preprocessing.py`).
2. **Median filter boundary handling.** Not specified. Decision:
   reflect/mirror padding (`background.py`).
3. **Blocks when `H`/`W` not divisible by `B`.** Not specified. Decision:
   keep partial edge blocks; eligibility still governed only by `n_min`
   (`regions.py`).
4. **"Standard first-derivative operator" for `G`.** Not named. Decision:
   3x3 Sobel magnitude (`regions.py`).
5. **`s_hat(mu)` evaluation outside the populated-bin range.** Not
   specified. Decision: clamp (no extrapolation) via `np.interp`
   defaults (`statistics.py`).
6. **Intensity range for binning.** Could be `[0,255]` or per-image
   observed range. Decision: fixed `[0,255]` so bin index has a stable
   meaning across images/runs (`statistics.py`).
7. **Most consequential: `M_image` uses signed `z_k` or `|z_k|`.**
   Section 9's literal formula uses signed `z_k`; Section 6 and 8 both
   define/visualize the anomaly using `|z_k|`. Signed-`z_k` percentile
   would be blind to anomalously *low*-dispersion blocks (e.g. local
   smoothing edits), contradicting Section 6's own stated definition.
   Decision: compute `M_image` from `|z_k|`, matching the heatmap and
   Section 6's text, and documented prominently (`consistency.py`).
8. **Per-block localization threshold for the suspicious-region mask.**
   The methodology only defines `tau` for the single image-level
   decision. Decision: reuse the same `tau` per block for this
   *auxiliary* visualization mask, disclosed as not an independently
   calibrated rule (`visualization.py`).

No mathematical contradiction was found that blocks implementation
outright; ambiguity #7 above was the closest candidate, and was resolved
by following the methodology's own more detailed textual definition
(Sections 6/8) over a literal reading of the Section 9 formula's
notation, rather than inventing new science.

## Stages completed

1. Inspected the stated project structure (no on-disk repo was available
   in this environment; built from the structure you described).
2. Implemented the core algorithm as `src/detector/*`.
3. Implemented `run_detector.py`.
4. Ran a smoke test on synthetic single images (see above).
5. No implementation bugs remained open after the smoke test round
   documented above (the zero-residual-MAD case is expected formula
   behavior on that input, not a bug).
6. Implemented batch (directory) processing.

Stage boundary respected: **no synthetic benchmark (Section 16) and no
large-scale evaluation have been implemented or run.** `src/evaluation/`
is left as an empty placeholder for that future work.
