#!/usr/bin/env python3
"""
04_classify_channel_mode.py — new, additive dataset-audit stage.

WHAT THIS IS
------------
A standalone step that runs AFTER the frozen `01_build_and_filter.py`
pipeline has produced `clean_candidates/` and `manifest.csv`. It classifies
every clean-candidate image by channel mode and derives a reproducible
"grayscale-compatible" subset for the LGPC-BC v1.1 detector.

WHAT THIS IS NOT
----------------
This script does NOT modify or redesign `01_build_and_filter.py`,
`02_split_train_test.py`, `03_dataset_stats.py`, or the `manifest.csv`
they produce. It only READS the existing manifest and images. It writes
brand-new, separate output files; nothing existing is overwritten. It
also does not touch any LGPC-BC detector code (`src/detector/*`), and it
never converts, resamples, or otherwise alters a single pixel value —
classification is read-only.

CLASSIFICATION
--------------
For each image referenced in the manifest:
  - "single_channel" : already 1 channel -> directly usable as grayscale.
  - "rgb_identical"  : 3 channels, R == G == B at every pixel exactly ->
                       losslessly usable as grayscale by taking one
                       channel (matches the identical-channel case the
                       detector's own input validation already accepts).
  - "rgb_differing"  : 3 channels with genuinely different channel
                       content -> out of scope for LGPC-BC v1.1.
  - "other_channel_count" : any other channel count (e.g. 4 channels /
                       RGBA) -> out of scope; recorded, not silently
                       merged into another bucket.
  - "unreadable"     : the file could not be decoded -> recorded, not
                       silently dropped.

The grayscale-compatible subset = "single_channel" + "rgb_identical" rows.

OUTPUTS (all new files, written to --output-dir)
-------------------------------------------------
  manifest_with_channel_mode.csv     every manifest row + a channel_mode column
  manifest_grayscale_compatible.csv  subset manifest: grayscale-compatible rows only
  channel_mode_audit.json            counts/percentages for this run (the
                                      reproducible record of the
                                      classification, per the task)

USAGE
-----
    python src/preprocessing/04_classify_channel_mode.py \\
        --manifest path/to/manifest.csv \\
        --output-dir path/to/output_dir \\
        [--images-root path/to/prepend/if/manifest/paths/are/relative] \\
        [--image-column <column name>]

If --image-column is omitted, the script no longer guesses a column purely
by name. It resolves each CANDIDATE_IMAGE_COLUMNS column's values against
--images-root, checks what fraction of a sample actually exist on disk,
and picks the column with the best real-file match — never inventing or
reconstructing a filename. If nothing resolves well, it reports the
per-column match rates and sample values and stops rather than guessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import cv2

CANDIDATE_IMAGE_COLUMNS = [
    "filepath",
    "file_path",
    "path",
    "image_path",
    "filename",
    "file",
]

# How many manifest rows to sample when verifying a candidate column
# actually points at real files. Keeps column selection fast even on a
# 17,932-row manifest, without needing to touch every file up front.
_COLUMN_CHECK_SAMPLE_SIZE = 100

# A candidate column must resolve at least this fraction of its sample
# to existing files to be accepted automatically.
_MIN_ACCEPTABLE_MATCH_RATE = 0.9


def resolve_path(row_value: str, images_root: str | None) -> str:
    """
    Resolve a manifest row's raw path/filename value against images_root.

    This ONLY joins the manifest's own string value with images_root (or
    returns it unchanged if already absolute) — it never reconstructs,
    zero-pads, or otherwise synthesizes a filename from an ID or partial
    value. If the manifest's value is wrong or points at the wrong
    column, that is surfaced by the existence check below, not silently
    patched over here.
    """
    if images_root and not os.path.isabs(row_value):
        return os.path.join(images_root, row_value)
    return row_value


def _match_rate(column: str, rows: list, images_root: str | None, sample_size: int) -> tuple:
    """
    Fraction of a sample of rows for which resolve_path(row[column], ...)
    points at a file that actually exists. Returns (rate, sample_values).
    Never modifies row_value — existence is checked on the verbatim
    manifest value only.
    """
    sample = rows[:sample_size] if sample_size < len(rows) else rows
    if not sample:
        return 0.0, []
    n_exist = 0
    sample_values = []
    for row in sample:
        raw_value = row.get(column, "")
        resolved = resolve_path(raw_value, images_root)
        if os.path.isfile(resolved):
            n_exist += 1
        if len(sample_values) < 3:
            sample_values.append(raw_value)
    return n_exist / len(sample), sample_values


def detect_image_column(fieldnames: list, rows: list, images_root: str | None, explicit: str | None) -> str:
    if explicit:
        if explicit not in fieldnames:
            raise SystemExit(
                f"[error] --image-column '{explicit}' is not a column in this manifest. "
                f"Available columns: {fieldnames}"
            )
        rate, samples = _match_rate(explicit, rows, images_root, _COLUMN_CHECK_SAMPLE_SIZE)
        print(f"[info] --image-column '{explicit}' resolves {rate:.0%} of a sample to existing files "
              f"(sample values: {samples}).")
        if rate < _MIN_ACCEPTABLE_MATCH_RATE:
            print(
                "[warning] that match rate is low — double-check --images-root and this column "
                "before trusting the results. Proceeding anyway because --image-column was explicit."
            )
        return explicit

    # Try each candidate name that actually exists as a manifest column,
    # and ONLY accept it if its values verifiably resolve to real files
    # under --images-root. This is what fixes the reported bug: a
    # same-named-but-wrong column (e.g. an internal numeric ID) will fail
    # this check and be skipped, instead of being silently accepted.
    candidates_present = [c for c in CANDIDATE_IMAGE_COLUMNS if c in fieldnames]
    rates = {}
    for candidate in candidates_present:
        rate, samples = _match_rate(candidate, rows, images_root, _COLUMN_CHECK_SAMPLE_SIZE)
        rates[candidate] = (rate, samples)
        if rate >= _MIN_ACCEPTABLE_MATCH_RATE:
            print(f"[info] auto-detected image column '{candidate}' ({rate:.0%} of sample resolves to real files).")
            return candidate

    # None of the name-based candidates checked out. Fall back to
    # checking EVERY column in the manifest for the best real-file match,
    # still using only each row's own verbatim value — no invented names.
    for column in fieldnames:
        if column in rates:
            continue
        rate, samples = _match_rate(column, rows, images_root, _COLUMN_CHECK_SAMPLE_SIZE)
        rates[column] = (rate, samples)

    best_column, (best_rate, best_samples) = max(rates.items(), key=lambda kv: kv[1][0]) if rates else (None, (0.0, []))

    if best_column is not None and best_rate >= _MIN_ACCEPTABLE_MATCH_RATE:
        print(f"[info] auto-detected image column '{best_column}' by existence check "
              f"({best_rate:.0%} of sample resolves to real files; not one of the common names, "
              f"so this was found by scanning all columns).")
        return best_column

    report_lines = [
        f"    {col}: {rate:.0%} match, sample values={samples}" for col, (rate, samples) in rates.items()
    ]
    raise SystemExit(
        "[error] Could not find a manifest column whose values resolve to real files under "
        f"--images-root ({images_root!r}). Checked columns:\n" + "\n".join(report_lines) +
        "\nThis will not guess or reconstruct filenames. Re-run with --image-column <name> "
        "once you've identified the correct column above, and/or check --images-root."
    )


def classify_channel_mode(image_path: str) -> str:
    """
    Read-only classification. Never writes back to the image file, never
    alters pixel values. Mirrors the identical-channel check used by the
    detector's own input validation (src/detector/preprocessing.py), but
    implemented independently here so this dataset-audit tool has no
    import dependency on the detector package.
    """
    raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return "unreadable"

    if raw.ndim == 2:
        return "single_channel"

    if raw.ndim == 3:
        n_channels = raw.shape[2]
        if n_channels == 1:
            return "single_channel"
        if n_channels == 3:
            identical = np.array_equal(raw[:, :, 0], raw[:, :, 1]) and np.array_equal(
                raw[:, :, 1], raw[:, :, 2]
            )
            return "rgb_identical" if identical else "rgb_differing"
        return "other_channel_count"

    return "other_channel_count"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="Path to the existing (frozen) manifest.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory for the new audit/subset files (created if missing).")
    parser.add_argument("--images-root", default=None, help="Optional root to prepend to relative manifest paths.")
    parser.add_argument("--image-column", default=None, help="Manifest column holding each image's file path.")
    args = parser.parse_args()

    if not os.path.isfile(args.manifest):
        raise SystemExit(f"[error] manifest not found: {args.manifest}")

    with open(args.manifest, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    image_column = detect_image_column(fieldnames, rows, args.images_root, args.image_column)
    print(f"[info] using manifest column '{image_column}' as the image path.")
    print(f"[info] manifest rows: {len(rows)}")

    os.makedirs(args.output_dir, exist_ok=True)

    counts = {
        "single_channel": 0,
        "rgb_identical": 0,
        "rgb_differing": 0,
        "other_channel_count": 0,
        "unreadable": 0,
    }

    grayscale_compatible_modes = {"single_channel", "rgb_identical"}
    out_fieldnames = list(fieldnames) + ["channel_mode"]

    with_mode_path = os.path.join(args.output_dir, "manifest_with_channel_mode.csv")
    subset_path = os.path.join(args.output_dir, "manifest_grayscale_compatible.csv")

    with open(with_mode_path, "w", newline="") as f_all, open(subset_path, "w", newline="") as f_subset:
        writer_all = csv.DictWriter(f_all, fieldnames=out_fieldnames)
        writer_all.writeheader()
        writer_subset = csv.DictWriter(f_subset, fieldnames=out_fieldnames)
        writer_subset.writeheader()

        for i, row in enumerate(rows):
            resolved_path = resolve_path(row[image_column], args.images_root)
            mode = classify_channel_mode(resolved_path)
            counts[mode] += 1

            out_row = dict(row)
            out_row["channel_mode"] = mode
            writer_all.writerow(out_row)

            if mode in grayscale_compatible_modes:
                writer_subset.writerow(out_row)

            if (i + 1) % 2000 == 0:
                print(f"[progress] classified {i + 1}/{len(rows)}")

    total = len(rows)
    n_gray_compatible = counts["single_channel"] + counts["rgb_identical"]
    n_out_of_scope = counts["rgb_differing"] + counts["other_channel_count"]

    audit = {
        "manifest_input": os.path.abspath(args.manifest),
        "image_column_used": image_column,
        "total_images": total,
        "counts_by_channel_mode": counts,
        "grayscale_compatible_count": n_gray_compatible,
        "grayscale_compatible_fraction": (n_gray_compatible / total) if total else None,
        "out_of_scope_count": n_out_of_scope,
        "out_of_scope_fraction": (n_out_of_scope / total) if total else None,
        "outputs": {
            "manifest_with_channel_mode": os.path.abspath(with_mode_path),
            "manifest_grayscale_compatible_subset": os.path.abspath(subset_path),
        },
        "note": (
            "Classification is read-only; no pixel values were converted, "
            "resampled, or otherwise modified. The frozen preprocessing "
            "scripts (01/02/03) and the original manifest.csv were not "
            "modified by this step."
        ),
    }
    audit_path = os.path.join(args.output_dir, "channel_mode_audit.json")
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)

    print("\n[summary]")
    print(f"  total images                 : {total}")
    print(f"  single_channel               : {counts['single_channel']}")
    print(f"  rgb_identical                : {counts['rgb_identical']}")
    print(f"  rgb_differing                : {counts['rgb_differing']}")
    print(f"  other_channel_count          : {counts['other_channel_count']}")
    print(f"  unreadable                   : {counts['unreadable']}")
    print(f"  grayscale-compatible subset  : {n_gray_compatible} ({audit['grayscale_compatible_fraction']:.1%})" if total else "")
    print(f"\nWrote: {with_mode_path}")
    print(f"Wrote: {subset_path}")
    print(f"Wrote: {audit_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
