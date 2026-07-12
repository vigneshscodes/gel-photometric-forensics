"""
manual_review_tool.py

Fast keep/reject pass over the automatically-filtered pristine folder.
Controls:
  k  -> keep (no visible text/arrows/scale bars/lane-number overlays)
  r  -> reject (annotation artifacts present, or anything else wrong)
  s  -> skip for now (decide later)
  q  -> quit and save progress

Resumable: re-running skips images already decided in the CSV.
"""

import cv2
import csv
from pathlib import Path

from src.config import OUTPUT_ROOT

INPUT_DIR = OUTPUT_ROOT / "Pristine_Blot_Gel"
REVIEW_CSV = OUTPUT_ROOT / "manual_review_decisions.csv"
FINAL_DIR = OUTPUT_ROOT / "Pristine_Blot_Gel_Final"

FINAL_DIR.mkdir(parents=True, exist_ok=True)

# Load existing decisions so the tool is resumable
decided = {}
if REVIEW_CSV.exists():
    with open(REVIEW_CSV, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            decided[row[0]] = row[1]

all_images = sorted(INPUT_DIR.iterdir())
remaining = [p for p in all_images if p.name not in decided]

print(f"Total images: {len(all_images)} | Already decided: {len(decided)} | Remaining: {len(remaining)}")

for img_path in remaining:
    img = cv2.imread(str(img_path))
    if img is None:
        decided[img_path.name] = "reject_unreadable"
        continue

    h, w = img.shape[:2]
    scale = min(900 / w, 900 / h, 1.0)
    disp = cv2.resize(img, (int(w * scale), int(h * scale)))

    cv2.imshow("Review [k=keep, r=reject, s=skip, q=quit]", disp)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('k'):
        decided[img_path.name] = "keep"
    elif key == ord('r'):
        decided[img_path.name] = "reject_artifact"
    else:  # 's' or anything else = skip
        continue

cv2.destroyAllWindows()

# Save decisions
with open(REVIEW_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "decision"])
    for name, dec in decided.items():
        writer.writerow([name, dec])

# Copy kept images into final folder
import shutil
kept = 0
for name, dec in decided.items():
    if dec == "keep":
        shutil.copy2(INPUT_DIR / name, FINAL_DIR / name)
        kept += 1

print(f"\nReview session saved. Kept so far: {kept}")
print(f"Decisions log: {REVIEW_CSV}")
print(f"Final kept images copied to: {FINAL_DIR}")