#!/usr/bin/env python3
"""Create and display a visual dashboard for one LGPC-BC detector run.

Example (Windows PowerShell, from the project root):
  python view_detector_results.py --image "path\\to\\image.png" --results "benchmark_runs\\repro\\test_results\\case_name"

The --results folder must be the per-image output folder produced by run_detector.py
(or a per-case folder containing normalized_heatmap.png and related artifacts).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image


def load_score(results: Path) -> dict:
    p = results / "image_level_score.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Original image path")
    parser.add_argument("--results", required=True, help="Per-image detector output directory")
    parser.add_argument("--save", default=None, help="Dashboard PNG path (default: <results>/detector_dashboard.png)")
    parser.add_argument("--no-show", action="store_true", help="Save the dashboard without opening a separate Matplotlib window")
    args = parser.parse_args()

    original = Path(args.image).expanduser()
    results = Path(args.results).expanduser()
    if not original.is_file():
        parser.error(f"Original image does not exist: {original}")
    if not results.is_dir():
        parser.error(f"Results directory does not exist: {results}")

    score = load_score(results)
    items = [
        ("Original image", original),
        ("Normalized anomaly heatmap", results / "normalized_heatmap.png"),
        ("Suspicious-region mask", results / "suspicious_region_mask.png"),
        ("Eligibility mask", results / "eligibility_mask.png"),
        ("Dispersion curve", results / "dispersion_curve.png"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    for ax, (title, path) in zip(axes, items):
        if path.is_file():
            try:
                with Image.open(path) as im:
                    img = im.copy()
                ax.imshow(img, cmap="gray" if img.mode in ("L", "1", "I", "F") else None)
                ax.set_title(title)
            except Exception as exc:
                ax.set_title(f"Could not read: {title}\n{exc}")
        else:
            ax.set_title(f"Missing output: {title}")
        ax.axis("off")

    summary = "LGPC-BC detector result"
    if score:
        value = score.get("image_level_score_M_image", "N/A")
        tau = score.get("tau_threshold_used", "N/A")
        flagged = score.get("decision_flagged_inconsistent", None)
        decision = "FLAGGED as inconsistent" if flagged is True else "Not flagged" if flagged is False else "N/A"
        summary += f"\nM_image={value}   tau={tau}   decision={decision}"
    else:
        summary += "\nScore JSON not found; showing available image artifacts only"
    axes[5].text(0.02, 0.8, summary, va="top", wrap=True, fontsize=12)
    axes[5].axis("off")
    fig.suptitle(original.name, fontsize=13)
    fig.tight_layout()

    dashboard = Path(args.save).expanduser() if args.save else results / "detector_dashboard.png"
    dashboard.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dashboard, dpi=180, bbox_inches="tight")
    print(f"Dashboard saved to: {dashboard.resolve()}")
    if score:
        print(f"Image score: {score.get('image_level_score_M_image', 'N/A')}")
        print(f"Threshold: {score.get('tau_threshold_used', 'N/A')}")
        print(f"Decision: {summary.split('decision=')[-1] if 'decision=' in summary else 'see dashboard'}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
