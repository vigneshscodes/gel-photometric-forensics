"""
triage_pristine_dataset.py

Auto-sorts images into three buckets so manual review only touches the
genuinely ambiguous ones:
  - likely_keep/     : looks like a standard lane/band blot, no text detected
  - likely_reject/    : text detected, OR no lane-like structure found
  - needs_review/     : borderline cases — only THIS folder needs your eyes
"""

import cv2
import numpy as np
import pytesseract
import shutil
from pathlib import Path

from src.config import OUTPUT_ROOT

# --- set this to wherever you installed Tesseract ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

INPUT_DIR = OUTPUT_ROOT / "Pristine_Blot_Gel"
LIKELY_KEEP = OUTPUT_ROOT / "Triage" / "likely_keep"
LIKELY_REJECT = OUTPUT_ROOT / "Triage" / "likely_reject"
NEEDS_REVIEW = OUTPUT_ROOT / "Triage" / "needs_review"

for d in (LIKELY_KEEP, LIKELY_REJECT, NEEDS_REVIEW):
    d.mkdir(parents=True, exist_ok=True)


def has_text(gray_img):
    """Return True if OCR finds any confident text in the image."""
    text = pytesseract.image_to_string(gray_img, config="--psm 11").strip()
    # require at least 2 alphanumeric characters to avoid noise-triggered false positives
    alnum_chars = sum(c.isalnum() for c in text)
    return alnum_chars >= 2


def lane_structure_score(gray_img):
    """
    Rough heuristic: blots have strong, spatially concentrated vertical
    darkness variation (lanes). Scatter plots / 2D gels tend to have either
    very uniform column variance or scattered, non-periodic peaks.
    Returns a score; higher = more "blot-like".
    """
    col_means = gray_img.mean(axis=0)
    # normalize
    col_means = (col_means - col_means.min()) / (col_means.max() - col_means.min() + 1e-6)
    # count distinct "dips" (lanes are darker than background => local minima)
    inverted = 1 - col_means
    threshold = inverted.mean() + inverted.std()
    above = inverted > threshold
    # count transitions from below->above threshold = rough lane count
    transitions = np.sum(np.diff(above.astype(int)) == 1)
    return transitions


def classify(path):
    img = cv2.imread(str(path))
    if img is None:
        return "likely_reject"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    if has_text(gray):
        return "likely_reject"

    lanes = lane_structure_score(gray)

    if lanes >= 2:      # at least 2 lane-like transitions found -> looks like a blot
        return "likely_keep"
    elif lanes == 0:    # no lane structure at all -> probably not a blot
        return "likely_reject"
    else:               # exactly 1 -> ambiguous, could be single-lane blot or noise
        return "needs_review"


def main():
    all_images = sorted(INPUT_DIR.iterdir())
    counts = {"likely_keep": 0, "likely_reject": 0, "needs_review": 0}

    for i, img_path in enumerate(all_images):
        if not img_path.is_file():
            continue

        try:
            bucket = classify(img_path)
        except Exception as e:
            print(f"[WARNING] Failed on {img_path.name}: {e}")
            bucket = "needs_review"

        dest_dir = {"likely_keep": LIKELY_KEEP,
                    "likely_reject": LIKELY_REJECT,
                    "needs_review": NEEDS_REVIEW}[bucket]

        shutil.copy2(img_path, dest_dir / img_path.name)
        counts[bucket] += 1

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(all_images)}...")

    print("\n========== TRIAGE COMPLETE ==========")
    for bucket, count in counts.items():
        print(f"{bucket:15s}: {count}")
    print(f"\nOnly '{NEEDS_REVIEW}' and a spot-check of '{LIKELY_KEEP}'/'{LIKELY_REJECT}' need your eyes.")


if __name__ == "__main__":
    main()
    