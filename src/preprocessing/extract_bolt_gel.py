import json
import shutil

from src.config import DATASET_ROOT, OUTPUT_ROOT, ANNOTATION_DIR

CLASSIFICATION_FILE = ANNOTATION_DIR / "classification.json"
OUTPUT_DIR = OUTPUT_ROOT / "Blot_Gel"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CLASSIFICATION_FILE, "r") as f:
    classification = json.load(f)

copied = 0
missing = 0

for folder_id, images in classification.items():

    source_folder = DATASET_ROOT / folder_id

    if not source_folder.exists():
        print(f"[WARNING] Folder not found: {source_folder}")
        continue

    for image_name, category in images.items():

        if category != "Blot/Gel":
            continue

        source_image = source_folder / image_name
        destination_image = OUTPUT_DIR / image_name

        if source_image.exists():
            shutil.copy2(source_image, destination_image)
            copied += 1
        else:
            print(f"[WARNING] Missing image: {source_image}")
            missing += 1

print("\n========== EXTRACTION COMPLETE ==========")
print(f"Blot/Gel images extracted : {copied}")
print(f"Missing images            : {missing}")
print(f"Saved to                  : {OUTPUT_DIR}")