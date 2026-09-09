"""
Fast Review 2 Demo
------------------

Run from project root:

    python review2_demo.py

This does NOT rerun the 42-case benchmark.

It:
1. Uses the already-generated benchmark images.
2. Runs the detector only on 2 representative images.
3. Opens one visual window showing:
      - Clean image
      - Clean consistency heatmap
      - Clean suspicious mask
      - Local-edit image
      - Local-edit consistency heatmap
      - Local-edit suspicious mask
4. Prints a short narration for the recording.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from src.detector.detector import run_on_image
from src.detector.config import LGPCBCConfig


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

BENCHMARK_DIR = (
    PROJECT_ROOT
    / "benchmark_runs"
    / "sanity_v1"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "benchmark_runs"
    / "sanity_v1_results"
)

DEMO_OUTPUT_DIR = RESULTS_DIR / "demo_outputs"


# ================================================================
# REPRESENTATIVE CASES
# ================================================================

CLEAN_CASE = "clean_seed1000"

LOCAL_CASE = "global_gamma_seed1000_gamma=1.4"


# ================================================================
# FIND BENCHMARK IMAGE
# ================================================================

def find_case_image(case_id: str) -> Path | None:
    """Find the benchmark image from benchmark_manifest.csv."""

    manifest = BENCHMARK_DIR / "benchmark_manifest.csv"

    with manifest.open(
        newline="",
        encoding="cp1252",
    ) as f:

        for row in csv.DictReader(f):

            if row.get("case_id") != case_id:
                continue

            image_path = row.get("image_path")

            if not image_path:
                return None

            path = Path(image_path)

            if not path.is_absolute():
                path = PROJECT_ROOT / path

            if path.exists():
                return path

    return None


# ================================================================
# CREATE UNIQUE OUTPUT DIRECTORY
# ================================================================

def get_output_dir(case_id: str) -> Path:
    """
    Create a fresh output directory without deleting existing
    OneDrive files/folders.
    """

    base_dir = DEMO_OUTPUT_DIR / case_id

    if not base_dir.exists():
        base_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        return base_dir

    counter = 1

    while True:

        candidate = DEMO_OUTPUT_DIR / (
            f"{case_id}_run{counter}"
        )

        if not candidate.exists():

            candidate.mkdir(
                parents=True,
                exist_ok=True,
            )

            return candidate

        counter += 1


# ================================================================
# RUN DETECTOR
# ================================================================

def run_detector(case_id: str) -> Path:
    """
    Run the existing detector on one representative image.

    The 42-case benchmark is NOT rerun.
    """

    image_path = find_case_image(case_id)

    if image_path is None:

        raise SystemExit(
            f"[error] Could not find image for {case_id}"
        )

    output_dir = get_output_dir(case_id)

    print(
        f"\n[demo] Running detector: {case_id}"
    )

    print(
        f"[demo] Input : {image_path}"
    )

    print(
        f"[demo] Output: {output_dir}"
    )

    # ------------------------------------------------------------
    # Use the SAME detector configuration used by the project.
    # ------------------------------------------------------------

    config = LGPCBCConfig()

    try:

        result = run_on_image(
            str(image_path),
            str(output_dir),
            config,
        )

    except Exception as exc:

        raise SystemExit(
            f"[error] Detector failed for {case_id}:\n{exc}"
        ) from exc

    # ------------------------------------------------------------
    # Print headline result
    # ------------------------------------------------------------

    print(
        f"[demo] M_image = {result.image_level_score}"
    )

    print(
        f"[demo] Flagged  = {result.decision_flagged}"
    )

    # ------------------------------------------------------------
    # Verify visualization outputs
    # ------------------------------------------------------------

    heatmap = (
        output_dir
        / "normalized_heatmap.png"
    )

    mask = (
        output_dir
        / "suspicious_region_mask.png"
    )

    if not heatmap.exists() or not mask.exists():

        print(
            "\n[error] Expected visualization files "
            "were not created."
        )

        print(
            f"[error] Output directory:\n{output_dir}"
        )

        generated = list(
            output_dir.rglob("*")
        )

        if generated:

            print("\n[error] Generated files:")

            for path in generated:
                print(
                    f"    {path}"
                )

        else:

            print(
                "\n[error] Output directory is empty."
            )

        raise SystemExit(1)

    print(
        "[demo] Detector output ready."
    )

    return output_dir


# ================================================================
# IMAGE LOADING
# ================================================================

def load_image(path: Path) -> np.ndarray:
    """Load grayscale image and convert to BGR."""

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise FileNotFoundError(path)

    return cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR,
    )


def load_heatmap(path: Path) -> np.ndarray:
    """Load detector heatmap."""

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise FileNotFoundError(path)

    return image


def load_mask(path: Path) -> np.ndarray:
    """Load suspicious-region mask."""

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise FileNotFoundError(path)

    return cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR,
    )


# ================================================================
# DISPLAY HELPERS
# ================================================================

def resize_for_display(
    image: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:

    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )


def add_title(
    image: np.ndarray,
    title: str,
) -> np.ndarray:

    output = image.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 35),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        output,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


# ================================================================
# VISUAL WINDOW
# ================================================================

def make_visual_window(
    clean_dir: Path,
    local_dir: Path,
) -> None:
    """
    Create one 2x3 comparison window.

    Top row:
        Clean Input | Clean Heatmap | Clean Mask

    Bottom row:
        Local Input | Local Heatmap | Local Mask
    """

    clean_source = find_case_image(
        CLEAN_CASE
    )

    local_source = find_case_image(
        LOCAL_CASE
    )

    if clean_source is None:

        raise FileNotFoundError(
            f"Clean image not found: {CLEAN_CASE}"
        )

    if local_source is None:

        raise FileNotFoundError(
            f"Local image not found: {LOCAL_CASE}"
        )

    files = {

        "clean_input":
            clean_source,

        "clean_heatmap":
            clean_dir
            / "normalized_heatmap.png",

        "clean_mask":
            clean_dir
            / "suspicious_region_mask.png",

        "local_input":
            local_source,

        "local_heatmap":
            local_dir
            / "normalized_heatmap.png",

        "local_mask":
            local_dir
            / "suspicious_region_mask.png",
    }

    images = [

        (
            "CLEAN - Input",
            load_image(
                files["clean_input"]
            ),
        ),

        (
            "CLEAN - Consistency Heatmap",
            load_heatmap(
                files["clean_heatmap"]
            ),
        ),

        (
            "CLEAN - Suspicious Mask",
            load_mask(
                files["clean_mask"]
            ),
        ),

        (
            "LOCAL EDIT - Input",
            load_image(
                files["local_input"]
            ),
        ),

        (
            "LOCAL EDIT - Consistency Heatmap",
            load_heatmap(
                files["local_heatmap"]
            ),
        ),

        (
            "LOCAL EDIT - Suspicious Mask",
            load_mask(
                files["local_mask"]
            ),
        ),
    ]

    # ------------------------------------------------------------
    # Tile size
    # ------------------------------------------------------------

    tile_width = 420
    tile_height = 300

    tiles = []

    for title, image in images:

        image = resize_for_display(
            image,
            tile_width,
            tile_height,
        )

        image = add_title(
            image,
            title,
        )

        tiles.append(image)

    # ------------------------------------------------------------
    # 2 x 3 montage
    # ------------------------------------------------------------

    row1 = np.hstack(
        tiles[0:3]
    )

    row2 = np.hstack(
        tiles[3:6]
    )

    montage = np.vstack(
        [
            row1,
            row2,
        ]
    )

    # ------------------------------------------------------------
    # Display
    # ------------------------------------------------------------

    window_name = (
        "Review 2 - Detector Visual Results"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        window_name,
        1260,
        640,
    )

    cv2.imshow(
        window_name,
        montage,
    )

    print(
        "\n" + "=" * 65
    )

    print(
        "VISUAL DEMO WINDOW"
    )

    print(
        "=" * 65
    )

    print(
        "Top row    = Clean image"
    )

    print(
        "Bottom row = Local edited image"
    )

    print(
        "\nPress any key inside the image window "
        "to close it."
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# ================================================================
# RECORDING SCRIPT
# ================================================================

def print_recording_script() -> None:

    print(
        "\n" + "=" * 65
    )

    print(
        "1-MINUTE RECORDING SCRIPT"
    )

    print(
        "=" * 65
    )

    print(
        """
"This is my Local-Global Photometric Consistency
detection pipeline for gel and blot images.

First, I demonstrate the detector on a clean image.
The system processes the image and generates the
consistency heatmap and suspicious-region mask.

Next, I run the same detector on an image containing
a localized tonal edit. The same pipeline generates
the corresponding heatmap and suspicious-region mask.

The complete implementation has also been evaluated
using a 42-case synthetic benchmark containing clean
images, global transformations, and local edits.

The core detection and evaluation pipeline is completed.
The remaining work is calibration, further analysis,
and validation on real scientific images."
"""
    )


# ================================================================
# MAIN
# ================================================================

def main() -> int:

    print(
        "=" * 65
    )

    print(
        "FAST REVIEW 2 DEMO"
    )

    print(
        "=" * 65
    )

    manifest = (
        BENCHMARK_DIR
        / "benchmark_manifest.csv"
    )

    if not manifest.exists():

        print(
            f"[error] Manifest not found:\n{manifest}"
        )

        return 1

    print(
        "\n[info] Existing 42-case benchmark "
        "will NOT be rerun."
    )

    print(
        "[info] Only 2 representative detector "
        "runs will be executed."
    )

    # ------------------------------------------------------------
    # Clean case
    # ------------------------------------------------------------

    clean_dir = run_detector(
        CLEAN_CASE
    )

    # ------------------------------------------------------------
    # Local-edit case
    # ------------------------------------------------------------

    local_dir = run_detector(
        LOCAL_CASE
    )

    # ------------------------------------------------------------
    # Visual comparison
    # ------------------------------------------------------------

    make_visual_window(
        clean_dir,
        local_dir,
    )

    # ------------------------------------------------------------
    # Narration
    # ------------------------------------------------------------

    print_recording_script()

    print(
        "\n" + "=" * 65
    )

    print(
        "DONE"
    )

    print(
        "=" * 65
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())