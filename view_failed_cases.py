
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

root = Path("out/test_res2/detector_outputs")
benchmark = Path("out/test")

cases = [
    "local_contrast_seed1000_normal_24x24_background",
    "local_nonlinear_tone_seed1000_normal_24x24_background"
]

outputs = [
    ("Synthesized Image", ["image.png", "input.png", "synthetic.png"]),
    ("Anomaly Heatmap", ["normalized_heatmap.png"]),
    ("Suspicious Mask", ["suspicious_region_mask.png"])
]

for case in cases:
    matches = list(root.rglob(f"*{case}*"))

    if not matches:
        print(f"Output not found: {case}")
        continue

    folder = matches[0]

    # Find the synthesized input image in the benchmark folders
    input_matches = list(benchmark.rglob(f"*{case}*"))
    input_image = None

    for match in input_matches:
        if match.is_file() and match.suffix.lower() in [
            ".png", ".jpg", ".jpeg", ".tif", ".tiff"
        ]:
            if not any(word in match.stem.lower()
                       for word in ["mask", "label", "heatmap"]):
                input_image = match
                break

        elif match.is_dir():
            for file in match.iterdir():
                if file.suffix.lower() in [
                    ".png", ".jpg", ".jpeg", ".tif", ".tiff"
                ] and not any(word in file.stem.lower()
                              for word in ["mask", "label", "heatmap"]):
                    input_image = file
                    break

        if input_image:
            break

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(case)

    for ax, (title, names) in zip(axes, outputs):
        path = None

        if title == "Synthesized Image" and input_image:
            path = input_image
        else:
            for name in names:
                candidate = folder / name
                if candidate.exists():
                    path = candidate
                    break

        if path and path.exists():
            ax.imshow(Image.open(path), cmap="gray")
            ax.set_title(title)
        else:
            ax.set_title(f"{title}\nNot found")
            print(f"{title} not found for {case}")

        ax.axis("off")

    plt.tight_layout()
    plt.show()
