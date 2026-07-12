"""
diagnose_lane_scores.py — run once before re-triaging, to pick better cutoffs
"""

import cv2
import numpy as np
from pathlib import Path
from src.config import OUTPUT_ROOT

INPUT_DIR = OUTPUT_ROOT / "Pristine_Blot_Gel"


def lane_structure_score(gray_img):
    col_means = gray_img.mean(axis=0)
    col_means = (col_means - col_means.min()) / (col_means.max() - col_means.min() + 1e-6)
    inverted = 1 - col_means
    threshold = inverted.mean() + inverted.std()
    above = inverted > threshold
    transitions = np.sum(np.diff(above.astype(int)) == 1)
    return transitions


scores = []
for img_path in sorted(INPUT_DIR.iterdir()):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    scores.append(lane_structure_score(gray))

scores = np.array(scores)
print("Min:", scores.min(), "Max:", scores.max())
print("Mean:", scores.mean(), "Median:", np.median(scores))
for pct in [10, 25, 50, 75, 90]:
    print(f"{pct}th percentile: {np.percentile(scores, pct):.1f}")

# rough histogram in text form
hist, edges = np.histogram(scores, bins=10)
for count, edge in zip(hist, edges):
    print(f"{edge:6.1f} : {'#' * (count // max(1, len(scores)//200))} ({count})")
