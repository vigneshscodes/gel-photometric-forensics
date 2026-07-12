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

Every image is copied into exactly one of two output folders:
  - <clean_dir>  : passed all checks automatically
  - <review_dir> : failed at least one check, needs a human look

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


def classify_image(
    path: Path,
    sat_thresh: float,
    spread_thresh: float,
    contrast_thresh: float,
    local_sat_thresh: float,
    local_spread_thresh: float,
    local_patch_size: int,
):
    rgb, gray = load_as_arrays(path)

    mean_sat, mean_spread, sat_map, spread_map = compute_saturation_metrics(rgb)
    contrast = compute_contrast_metric(gray)
    max_local_sat = box_filter_max(sat_map, local_patch_size)
    max_local_spread = box_filter_max(spread_map, local_patch_size)

    reasons = []
    if mean_sat > sat_thresh or mean_spread > spread_thresh:
        reasons.append("colored")
    if max_local_sat > local_sat_thresh or max_local_spread > local_spread_thresh:
        reasons.append("colored_localized")
    if contrast < contrast_thresh:
        reasons.append("blank")

    return {
        "mean_saturation": round(mean_sat, 2),
        "mean_channel_spread": round(mean_spread, 2),
        "max_local_saturation": round(max_local_sat, 2),
        "max_local_channel_spread": round(max_local_spread, 2),
        "contrast_std": round(contrast, 2),
        "flags": ";".join(reasons) if reasons else "",
        "passed": len(reasons) == 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Flag colored/non-grayscale, localized color annotations, and near-blank images among extracted pristine Blot/Gel candidates."
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
    flag_tally = {"colored": 0, "colored_localized": 0, "blank": 0}
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
                )
            except Exception as e:
                unreadable.append((path.name, str(e)))
                writer.writerow([path.name, "", "", "", "", "", "unreadable", False])
                continue

            writer.writerow(
                [
                    path.name,
                    result["mean_saturation"],
                    result["mean_channel_spread"],
                    result["max_local_saturation"],
                    result["max_local_channel_spread"],
                    result["contrast_std"],
                    result["flags"],
                    result["passed"],
                ]
            )

            if result["passed"]:
                passed_count += 1
                if not args.dry_run:
                    shutil.copy2(path, clean_dir / path.name)
            else:
                flagged_count += 1
                for reason in result["flags"].split(";"):
                    if reason:
                        flag_tally[reason] = flag_tally.get(reason, 0) + 1
                if not args.dry_run:
                    shutil.copy2(path, review_dir / path.name)

    print(f"Total images checked: {len(image_paths)}")
    print(f"Passed automatically (-> {clean_dir.name}/): {passed_count}")
    print(f"Flagged for manual review (-> {review_dir.name}/): {flagged_count}")
    print(f"  - colored (global):     {flag_tally.get('colored', 0)}")
    print(f"  - colored_localized:    {flag_tally.get('colored_localized', 0)}")
    print(f"  - blank:                {flag_tally.get('blank', 0)}")
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
            "but valid grayscale blot, or a single-hue Coomassie/silver-stained gel you've decided "
            "to keep as a modeling simplification) into the clean folder yourself. Also do a quick "
            "skim of the CLEAN folder for hand-drawn marks with no color (e.g. dark pencil/pen "
            "annotations) -- that failure mode has no saturation signature at all, so no threshold "
            "here will ever catch it."
        )


if __name__ == "__main__":
    main()