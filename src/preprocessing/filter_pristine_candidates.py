"""
src/preprocessing/filter_pristine_candidates.py

Second-pass automated triage on the output of extract_pristine_blot_gel.py.

BioFors's "Blot/Gel, unflagged-in-gt-files" label does NOT guarantee an image
is actually a clean, single-channel densitometry-style blot/gel ready for
tone-curve modeling. Manual inspection of a sample turned up:
  - genuine multiplex fluorescent blots / colored stains (2-3 channel signal,
    not true single-channel luminance -> collapsing to grayscale silently
    mixes two different image-formation processes into one "population")
  - near-blank / bad-crop images (mostly flat background, little/no signal)
  - small localized color annotations (hand-drawn arrows, circles) overlaid
    on an otherwise-grayscale image -- these occupy so few pixels that a
    whole-image saturation average barely moves, even though the annotation
    is a real, non-photometric artifact sitting on top of the image data
  - composite multi-panel images: multiple independent blot/gel crops
    stacked or concatenated into a single file, each with its own
    background level / exposure -- this violates the one-image-one-
    tone-mapping assumption the whole pipeline depends on
  - polarity-inverted images: white bands on a black background (a valid,
    common chemiluminescence convention) mixed in with the dominant
    dark-bands-on-light-background population -- these carry the opposite
    photometric sign and will corrupt a pooled tone-curve model unless
    normalized
  - at least one outright mislabeled image (a non-blot photo with a hand-drawn
    annotation) -- this class can only be caught by a human, not by this script

This script does NOT try to catch every problem class (mislabeling can't be
detected automatically here). It catches the classes that ARE cheaply
detectable, so a human only has to eyeball what's flagged instead of all N
images:

  1. COLORED / non-grayscale images  -> flagged as "colored"
     Detected via whole-image mean saturation (HSV) and mean channel-to-channel
     deviation in RGB. True grayscale/near-grayscale images (even if saved as
     RGB) have near-zero saturation and near-identical R=G=B; real color/
     fluorescent images do not.

  2. LOCALIZED colored annotations   -> flagged as "colored_localized"
     Detected via a sliding-patch saturation/channel-spread map: the image's
     saturation and channel-spread are box-filtered over small patches
     (default 9x9), and the MAX patch value is compared against a separate,
     higher threshold. A small colored arrow/circle/dot produces a strong
     local spike even though it barely moves the whole-image mean, so this
     catches what check #1 structurally cannot. Calibrated against synthetic
     test cases (see notes below) so realistic per-channel compression noise
     in a genuinely grayscale image stays well under threshold.

  3. NEAR-BLANK / low-contrast images -> flagged as "blank"
     Detected via overall pixel intensity std-dev. A real blot/gel, even a
     clean negative-control lane, has visible band/background structure and
     non-trivial contrast; a bad crop of mostly white/black space does not.

  4. COMPOSITE / stitched-panel images -> flagged as "composite_panel_suspect"
     (manual review only -- never auto-rejected)
     Detected via a windowed median step-profile along both the row axis and
     the column axis: the per-row (or per-column) mean intensity is smoothed
     with a MEDIAN over a window sized well beyond typical band thickness
     (so individual bands, which are a minority of rows/columns in any given
     window, get suppressed rather than mistaken for a background-level
     shift), and a large step between the median-before and median-after a
     candidate split point indicates a likely panel boundary. This is a
     structural check, not a photometric one -- it does not modify the image,
     it only routes suspect files to the review folder, since a real
     multi-lane gel can legitimately have some background unevenness and a
     human is better placed to make the final call.

  5. POLARITY (normal vs. inverted)  -> recorded for every image; inverted
     images are handled per confidence level
     Detected via Otsu thresholding on the grayscale image: whichever class
     (above/below threshold) contains the majority of pixels is treated as
     "background," and its brightness relative to the minority "foreground"
     class determines polarity (background darker than foreground = white
     bands on black = inverted). HIGH-CONFIDENCE inverted images (background
     class clearly dominant AND clearly separated in intensity from the
     foreground class) are automatically photometrically inverted (255 -
     pixel value) before being written to disk, and the manifest records
     that this happened. LOW-CONFIDENCE cases (ambiguous background/
     foreground split -- e.g. busy multiplex images, or images with no clear
     dominant background) are left untouched and flagged
     "polarity_uncertain" for manual review, exactly like the other
     manual-review-only checks.

Every image is copied into exactly one of two output folders:
  - <clean_dir>  : passed all automatic-reject checks (color, localized
                   color, blank). Composite-panel-suspect and
                   polarity-uncertain images are ALSO routed to review, but
                   are not silently dropped or modified beyond what's
                   documented above.
  - <review_dir> : failed at least one check, needs a human look

If an image was auto-inverted (see #5), the INVERTED pixel data is what gets
written to disk (whichever folder it lands in), not a copy of the original
bytes -- this way, downstream tone-curve modeling never has to redo this
step, and the manifest is the single source of truth for what was changed.
Everything else is a byte-identical copy of the source file.

filter_report.csv lists every image with its measured stats and flag reasons,
regardless of which folder it ended up in, so nothing is silently dropped.

Calibration notes for the localized-saturation check (see dev/test_patch/ for
the synthetic cases these were derived from):
  - An 8x8 solid red dot on an otherwise-clean grayscale image: global mean
    saturation ~0.4 (invisible to check #1), max local-patch saturation
    ~120-230 depending on patch size.
  - A thin 3px-wide diagonal red stroke (worst case for a sliding patch,
    since it's thinner than the patch): global mean saturation ~0.9, max
    local-patch saturation ~95-180.
  - A fully grayscale image with worst-case independent per-channel noise
    (+/-6, simulating JPEG channel decorrelation): max local-patch
    saturation stayed at ~27-30, well under the 50.0 default threshold, and
    well below both annotation cases above.
  This gives a comfortable margin: threshold=50.0 with patch_size=9 catches
  both thick and thin annotations while not tripping on compression noise.

Calibration notes for the composite-panel / seam check (see dev/test_patch2/
for the synthetic cases these were derived from):
  - A single clean panel with several real bands (no seam): max windowed
    median step was ~0.2 on the row axis and ~16 on the column axis (the
    latter from the ordinary whitespace-margin-to-lane-area edge, which is
    normal and NOT a seam).
  - A 3-panel vertically stacked composite with distinct background levels
    per panel: max windowed median step was ~85 on the row axis, and ~39 on
    the column axis (still lower than the row-axis signal, but well above
    the single-panel column baseline of ~16).
  Defaults (row_step_thresh=25.0, col_step_thresh=30.0) sit with margin
  above the single-panel baselines and below the composite-panel signal.

Calibration notes for the polarity check:
  - Normal (dark bands on light background) and inverted (light bands on
    dark background) synthetic test images both produced bg_fraction~0.78
    and mean_separation~170-190 -- clearly confident in both directions,
    validating that the Otsu-based background/foreground split and the
    brightness-comparison rule correctly identify polarity either way.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import OUTPUT_ROOT


def load_as_arrays(path: Path):
    """Returns (rgb_array uint8 HxWx3, gray_array uint8 HxW)."""
    img = Image.open(path).convert("RGB")
    rgb = np.array(img)
    gray = np.array(img.convert("L"))
    return rgb, gray


def compute_saturation_metrics(rgb: np.ndarray):
    """Mean HSV saturation (0-255 scale via PIL) and mean per-pixel channel
    spread (max(R,G,B) - min(R,G,B)), both good cheap colorfulness indicators
    for detecting true grayscale vs. colored/fluorescent images. Also returns
    the full per-pixel saturation and channel-spread maps so callers can run
    localized checks on top of the whole-image means."""
    hsv = np.array(Image.fromarray(rgb).convert("HSV"))
    sat_map = hsv[:, :, 1].astype(np.float32)
    mean_saturation = float(sat_map.mean())

    rgb_f = rgb.astype(np.int16)
    spread_map = (rgb_f.max(axis=2) - rgb_f.min(axis=2)).astype(np.float32)
    mean_channel_spread = float(spread_map.mean())

    return mean_saturation, mean_channel_spread, sat_map, spread_map


def box_filter_max(arr: np.ndarray, patch_size: int) -> float:
    """Max patch-mean of `arr` over all patch_size x patch_size windows,
    computed via an integral image so cost is O(H*W) regardless of patch
    size. Pure numpy (no scipy dependency) to keep this script's footprint
    identical to before. Edge-padded so windows near the border are still
    full patch_size x patch_size (avoids under-counting edge patches, which
    would otherwise bias the max downward right where hand-drawn marks near
    a crop edge are common)."""
    p = patch_size
    if p < 1:
        raise ValueError("patch_size must be >= 1")

    pad_top = p // 2
    pad_bottom = p - 1 - pad_top
    pad_left = p // 2
    pad_right = p - 1 - pad_left
    padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")

    ii = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = padded.cumsum(0).cumsum(1)
    sums = ii[p:, p:] - ii[:-p, p:] - ii[p:, :-p] + ii[:-p, :-p]
    patch_means = sums / (p * p)
    return float(patch_means.max())


def compute_contrast_metric(gray: np.ndarray):
    return float(gray.std())


# --------------------------------------------------------------------------
# Composite / stitched-panel (seam) detection
# --------------------------------------------------------------------------

def _windowed_median_step_profile(profile: np.ndarray, window: int) -> np.ndarray:
    """For each interior position p, compares median(profile[p-window:p])
    against median(profile[p:p+window]) and returns the signed difference.
    Using the median (not the mean) over a window sized well beyond typical
    band thickness means a handful of band rows/columns inside the window
    are outvoted by the surrounding background rows/columns, so real bands
    don't masquerade as background-level steps -- only a sustained shift
    (i.e. an actual panel boundary) produces a large value here."""
    n = len(profile)
    steps = np.full(n, np.nan)
    for p in range(window, n - window):
        before = np.median(profile[p - window:p])
        after = np.median(profile[p:p + window])
        steps[p] = after - before
    return steps


def _extract_seam_peaks(steps: np.ndarray, thresh: float, min_gap: int):
    """Non-max-suppressed list of (position, |step|) for every position
    where the step magnitude clears `thresh`, keeping at most one peak per
    `min_gap`-wide neighborhood (highest magnitude wins)."""
    candidates = [
        (i, abs(float(v)))
        for i, v in enumerate(steps)
        if not np.isnan(v) and abs(float(v)) >= thresh
    ]
    candidates.sort(key=lambda c: -c[1])
    kept = []
    for pos, mag in candidates:
        if all(abs(pos - kp) >= min_gap for kp, _ in kept):
            kept.append((pos, round(mag, 2)))
    return kept


def detect_composite_panels(
    gray: np.ndarray,
    row_window_frac: float = 0.12,
    col_window_frac: float = 0.12,
    row_step_thresh: float = 25.0,
    col_step_thresh: float = 30.0,
    min_window: int = 25,
):
    """Looks for hard background-level discontinuities along the row axis
    (stacked panels) and the column axis (side-by-side panels). Returns
    (row_seams, col_seams), each a list of (position, step_magnitude).
    Images smaller than ~2x the window size on a given axis skip that axis's
    check entirely (too little data to distinguish a real seam from noise)."""
    h, w = gray.shape

    row_seams = []
    row_window = max(min_window, int(h * row_window_frac))
    if h >= 2 * row_window + 10:
        row_profile = gray.mean(axis=1).astype(np.float64)
        row_steps = _windowed_median_step_profile(row_profile, row_window)
        row_seams = _extract_seam_peaks(row_steps, row_step_thresh, min_gap=row_window)

    col_seams = []
    col_window = max(min_window, int(w * col_window_frac))
    if w >= 2 * col_window + 10:
        col_profile = gray.mean(axis=0).astype(np.float64)
        col_steps = _windowed_median_step_profile(col_profile, col_window)
        col_seams = _extract_seam_peaks(col_steps, col_step_thresh, min_gap=col_window)

    return row_seams, col_seams


# --------------------------------------------------------------------------
# Polarity detection
# --------------------------------------------------------------------------

def _otsu_threshold(gray: np.ndarray) -> int:
    """Standard Otsu's method (pure numpy): the intensity threshold that
    maximizes between-class variance of the resulting two-class split."""
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    sum_all = np.dot(np.arange(256), hist)

    sum_bg, weight_bg = 0.0, 0.0
    best_thresh, best_var = 0, -1.0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between_var > best_var:
            best_var = between_var
            best_thresh = t
    return best_thresh


def detect_polarity(gray: np.ndarray, bg_fraction_thresh: float, mean_sep_thresh: float):
    """Splits the image into two classes via Otsu thresholding. The larger
    class (by pixel count) is treated as background; if that background
    class is the DARKER of the two, the image is polarity-inverted (light
    bands on a dark background) relative to the standard dark-bands-on-
    light-background convention. Confidence requires both a clearly
    dominant background class (bg_fraction) and a clear brightness
    separation between the two classes (mean_separation) -- busy/ambiguous
    images (e.g. multiplex fluorescent, or anything without one obviously
    dominant background level) fail one or both and are reported as
    'uncertain' rather than guessed at."""
    thresh = _otsu_threshold(gray)
    low_mask = gray <= thresh
    high_mask = ~low_mask
    low_count, high_count = int(low_mask.sum()), int(high_mask.sum())
    total = gray.size
    low_mean = float(gray[low_mask].mean()) if low_count else 0.0
    high_mean = float(gray[high_mask].mean()) if high_count else 0.0

    if low_count >= high_count:
        bg_count, bg_mean, fg_mean, bg_is_dark = low_count, low_mean, high_mean, True
    else:
        bg_count, bg_mean, fg_mean, bg_is_dark = high_count, high_mean, low_mean, False

    bg_fraction = bg_count / total
    mean_separation = abs(bg_mean - fg_mean)
    raw_polarity = "inverted" if bg_is_dark else "normal"
    confident = bg_fraction >= bg_fraction_thresh and mean_separation >= mean_sep_thresh

    return {
        "polarity": raw_polarity if confident else "uncertain",
        "raw_polarity_guess": raw_polarity,
        "bg_fraction": round(bg_fraction, 3),
        "mean_separation": round(mean_separation, 2),
        "confident": confident,
    }


def save_image_array(rgb_array: np.ndarray, dest_path: Path):
    """Writes an (possibly modified) RGB array to disk, preserving the
    destination's file extension/format. JPEGs are saved at high quality to
    minimize additional generation loss on top of the original compression."""
    img = Image.fromarray(rgb_array.astype(np.uint8))
    if dest_path.suffix.lower() in (".jpg", ".jpeg"):
        img.save(dest_path, quality=95)
    else:
        img.save(dest_path)


def classify_image(
    path: Path,
    sat_thresh: float,
    spread_thresh: float,
    contrast_thresh: float,
    local_sat_thresh: float,
    local_spread_thresh: float,
    local_patch_size: int,
    row_step_thresh: float,
    col_step_thresh: float,
    seam_window_frac: float,
    polarity_bg_fraction_thresh: float,
    polarity_mean_sep_thresh: float,
    auto_invert: bool,
):
    rgb, gray = load_as_arrays(path)

    mean_sat, mean_spread, sat_map, spread_map = compute_saturation_metrics(rgb)
    contrast = compute_contrast_metric(gray)
    max_local_sat = box_filter_max(sat_map, local_patch_size)
    max_local_spread = box_filter_max(spread_map, local_patch_size)

    row_seams, col_seams = detect_composite_panels(
        gray,
        row_window_frac=seam_window_frac,
        col_window_frac=seam_window_frac,
        row_step_thresh=row_step_thresh,
        col_step_thresh=col_step_thresh,
    )

    polarity_result = detect_polarity(
        gray, polarity_bg_fraction_thresh, polarity_mean_sep_thresh
    )

    reasons = []
    if mean_sat > sat_thresh or mean_spread > spread_thresh:
        reasons.append("colored")
    if max_local_sat > local_sat_thresh or max_local_spread > local_spread_thresh:
        reasons.append("colored_localized")
    if contrast < contrast_thresh:
        reasons.append("blank")
    if row_seams or col_seams:
        reasons.append("composite_panel_suspect")
    if polarity_result["polarity"] == "uncertain":
        reasons.append("polarity_uncertain")

    invert_applied = False
    output_rgb = None
    if polarity_result["polarity"] == "inverted" and auto_invert:
        output_rgb = 255 - rgb
        invert_applied = True

    return {
        "mean_saturation": round(mean_sat, 2),
        "mean_channel_spread": round(mean_spread, 2),
        "max_local_saturation": round(max_local_sat, 2),
        "max_local_channel_spread": round(max_local_spread, 2),
        "contrast_std": round(contrast, 2),
        "seam_rows": row_seams,
        "seam_cols": col_seams,
        "polarity": polarity_result["polarity"],
        "polarity_bg_fraction": polarity_result["bg_fraction"],
        "polarity_mean_separation": polarity_result["mean_separation"],
        "polarity_inverted_applied": invert_applied,
        "output_rgb": output_rgb,
        "flags": ";".join(reasons) if reasons else "",
        "passed": len(reasons) == 0,
    }


def _format_seam_list(seams):
    return ";".join(f"{pos}:{mag}" for pos, mag in seams)


def main():
    parser = argparse.ArgumentParser(
        description="Flag colored/non-grayscale, localized color annotations, near-blank images, "
        "and composite/stitched panels, and detect+correct polarity, among extracted pristine "
        "Blot/Gel candidates."
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Folder of extracted images to filter. Default: OUTPUT_ROOT/pristine_blot_gel (from extract_pristine_blot_gel.py).",
    )
    parser.add_argument(
        "--sat-thresh",
        type=float,
        default=12.0,
        help="Whole-image mean HSV saturation (0-255) above which an image is flagged 'colored'. Default: 12.0.",
    )
    parser.add_argument(
        "--spread-thresh",
        type=float,
        default=10.0,
        help="Whole-image mean per-pixel max(R,G,B)-min(R,G,B) above which an image is flagged 'colored'. Default: 10.0.",
    )
    parser.add_argument(
        "--contrast-thresh",
        type=float,
        default=8.0,
        help="Grayscale intensity std-dev below which an image is flagged 'blank'. Default: 8.0.",
    )
    parser.add_argument(
        "--local-sat-thresh",
        type=float,
        default=50.0,
        help="Max sliding-patch-mean HSV saturation above which an image is flagged 'colored_localized' "
        "(catches small colored annotations that don't move the whole-image mean). Default: 50.0.",
    )
    parser.add_argument(
        "--local-spread-thresh",
        type=float,
        default=40.0,
        help="Max sliding-patch-mean channel spread above which an image is flagged 'colored_localized'. Default: 40.0.",
    )
    parser.add_argument(
        "--local-patch-size",
        type=int,
        default=9,
        help="Side length (px) of the square sliding patch used for the localized-color check. Default: 9.",
    )
    parser.add_argument(
        "--row-step-thresh",
        type=float,
        default=25.0,
        help="Windowed median background-level step (row axis) above which an image is flagged "
        "'composite_panel_suspect'. Default: 25.0.",
    )
    parser.add_argument(
        "--col-step-thresh",
        type=float,
        default=30.0,
        help="Windowed median background-level step (column axis) above which an image is flagged "
        "'composite_panel_suspect'. Default: 30.0.",
    )
    parser.add_argument(
        "--seam-window-frac",
        type=float,
        default=0.12,
        help="Seam-detection window size as a fraction of image height/width (min 25px). Default: 0.12.",
    )
    parser.add_argument(
        "--polarity-bg-fraction-thresh",
        type=float,
        default=0.65,
        help="Minimum fraction of pixels that must belong to the dominant (background) Otsu class "
        "for a polarity call to be considered confident. Default: 0.65.",
    )
    parser.add_argument(
        "--polarity-mean-sep-thresh",
        type=float,
        default=15.0,
        help="Minimum intensity separation between background and foreground Otsu classes for a "
        "polarity call to be considered confident. Default: 15.0.",
    )
    parser.add_argument(
        "--disable-auto-invert",
        action="store_true",
        help="Do not automatically invert high-confidence polarity-inverted images; leave pixels "
        "untouched and only record the detected polarity in the manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write filter_report.csv; do not copy files into clean/review folders.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else Path(OUTPUT_ROOT) / "pristine_blot_gel"
    clean_dir = input_dir.parent / "pristine_blot_gel_clean"
    review_dir = input_dir.parent / "pristine_blot_gel_review"
    report_path = input_dir.parent / "filter_report.csv"

    if not input_dir.is_dir():
        sys.exit(f"ERROR: input dir not found: {input_dir}")

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    )
    if not image_paths:
        sys.exit(f"ERROR: no image files found in {input_dir}")

    if not args.dry_run:
        clean_dir.mkdir(parents=True, exist_ok=True)
        review_dir.mkdir(parents=True, exist_ok=True)

    passed_count = 0
    flagged_count = 0
    flag_tally = {
        "colored": 0,
        "colored_localized": 0,
        "blank": 0,
        "composite_panel_suspect": 0,
        "polarity_uncertain": 0,
    }
    inverted_count = 0
    unreadable = []

    with open(report_path, "w", newline="") as rf:
        writer = csv.writer(rf)
        writer.writerow(
            [
                "filename",
                "mean_saturation",
                "mean_channel_spread",
                "max_local_saturation",
                "max_local_channel_spread",
                "contrast_std",
                "seam_rows",
                "seam_cols",
                "polarity",
                "polarity_bg_fraction",
                "polarity_mean_separation",
                "polarity_inverted_applied",
                "flags",
                "passed",
            ]
        )

        for path in image_paths:
            try:
                result = classify_image(
                    path,
                    args.sat_thresh,
                    args.spread_thresh,
                    args.contrast_thresh,
                    args.local_sat_thresh,
                    args.local_spread_thresh,
                    args.local_patch_size,
                    args.row_step_thresh,
                    args.col_step_thresh,
                    args.seam_window_frac,
                    args.polarity_bg_fraction_thresh,
                    args.polarity_mean_sep_thresh,
                    auto_invert=not args.disable_auto_invert,
                )
            except Exception as e:
                unreadable.append((path.name, str(e)))
                writer.writerow(
                    [path.name, "", "", "", "", "", "", "", "", "", "", "", "unreadable", False]
                )
                continue

            writer.writerow(
                [
                    path.name,
                    result["mean_saturation"],
                    result["mean_channel_spread"],
                    result["max_local_saturation"],
                    result["max_local_channel_spread"],
                    result["contrast_std"],
                    _format_seam_list(result["seam_rows"]),
                    _format_seam_list(result["seam_cols"]),
                    result["polarity"],
                    result["polarity_bg_fraction"],
                    result["polarity_mean_separation"],
                    result["polarity_inverted_applied"],
                    result["flags"],
                    result["passed"],
                ]
            )

            if result["polarity_inverted_applied"]:
                inverted_count += 1

            dest_dir = clean_dir if result["passed"] else review_dir
            if not result["passed"]:
                flagged_count += 1
                for reason in result["flags"].split(";"):
                    if reason:
                        flag_tally[reason] = flag_tally.get(reason, 0) + 1
            else:
                passed_count += 1

            if not args.dry_run:
                dest_path = dest_dir / path.name
                if result["polarity_inverted_applied"]:
                    save_image_array(result["output_rgb"], dest_path)
                else:
                    shutil.copy2(path, dest_path)

    print(f"Total images checked: {len(image_paths)}")
    print(f"Passed automatically (-> {clean_dir.name}/): {passed_count}")
    print(f"Flagged for manual review (-> {review_dir.name}/): {flagged_count}")
    print(f"  - colored (global):        {flag_tally.get('colored', 0)}")
    print(f"  - colored_localized:       {flag_tally.get('colored_localized', 0)}")
    print(f"  - blank:                   {flag_tally.get('blank', 0)}")
    print(f"  - composite_panel_suspect: {flag_tally.get('composite_panel_suspect', 0)}")
    print(f"  - polarity_uncertain:      {flag_tally.get('polarity_uncertain', 0)}")
    print(f"Auto-inverted (high-confidence polarity fix applied): {inverted_count}")
    if unreadable:
        print(f"Unreadable files (check format/corruption): {len(unreadable)}")
        for name, err in unreadable[:5]:
            print(f"  {name}: {err}")
    print(f"\nReport: {report_path}")
    if args.dry_run:
        print("(--dry-run set: no files copied, report-only)")
    else:
        print(
            "\nNext step: manually eyeball the review folder, delete anything that's genuinely "
            "unusable, and drag anything that's a false-positive flag (e.g. a legitimately faint "
            "but valid grayscale blot, a single-hue Coomassie/silver-stained gel you've decided to "
            "keep as a modeling simplification, or a composite_panel_suspect that's actually just "
            "one gel with uneven lighting) into the clean folder yourself. Also do a quick skim of "
            "the CLEAN folder for hand-drawn marks with no color (e.g. dark pencil/pen annotations) "
            "-- that failure mode has no saturation signature at all, so no threshold here will ever "
            "catch it."
        )


if __name__ == "__main__":
    main()