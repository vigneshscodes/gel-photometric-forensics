# Local-Global Photometric Consistency Framework

A Python-based computer vision prototype for detecting and localizing suspicious tonal edits in scientific gel electrophoresis and Western blot images.

## Features

- **Image preprocessing:** Prepares input images and estimates background characteristics.
- **Local feature extraction:** Analyses image blocks to capture tonal variations.
- **Photometric consistency analysis:** Compares local image characteristics against a global photometric model.
- **Anomaly detection:** Generates an image-level anomaly score and detection decision.
- **Region localization:** Produces anomaly heatmaps and suspicious-region masks.
- **Visualization:** Provides a graphical interface for examining detection results.

## Tech Stack

Python, OpenCV, NumPy, SciPy, scikit-image, Pillow, Matplotlib, Tkinter, Git.

## Project Structure

```text
gel-photometric-forensics/
├── src/
│   ├── detector/
│   │   ├── features.py
│   │   ├── consistency.py
│   │   ├── detector.py
│   │   ├── config.py
│   │   └── visualization.py
│   └── evaluation/
│       ├── generate_benchmark.py
│       └── run_benchmark.py
├── run_detector.py
├── gui_detector.py
├── working_cases_viewer.py
└── README.md
```

*This is an overview of the main files. The repository may contain additional files.*

## Installation

Requires Python 3.10 or later and pip.

Clone the repository:

```bash
git clone https://github.com/vigneshscodes/gel-photometric-forensics.git
cd gel-photometric-forensics
```

Create and activate a virtual environment:

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS**
```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install numpy opencv-python scipy scikit-image pillow matplotlib
```

Install any additional dependencies required by the selected entry point or listed in the project configuration.

## Usage

Run the detector:

```bash
python run_detector.py
```

Launch the graphical interface:

```bash
python gui_detector.py
```

Open the benchmark results viewer:

```bash
python working_cases_viewer.py
```

Follow the on-screen prompts to select the input image or previously generated benchmark results and their corresponding directory.

## Methodology

The framework follows these main stages:

1. Image preprocessing and background estimation.
2. Local block-based feature extraction.
3. Global photometric modelling.
4. Local-global consistency analysis.
5. Image-level anomaly scoring.
6. Suspicious-region localization and visualization.

## Limitations

- Detection performance may vary across tonal-edit types.
- Suspicious-region masks may be inaccurate or empty.
- Evaluation on synthetic images does not establish performance on real biomedical images.
- A flagged image indicates a potential anomaly and does not establish research misconduct.

## Intended Use

This project is an academic research prototype designed to support the screening of scientific images for potential tonal manipulation. Its outputs should be interpreted alongside expert review.

## Author

**Vignesh S.**  
B.Tech. Computer Science and Engineering  
Vellore Institute of Technology, Vellore, India

## License

No license has been specified. Contact the author before redistributing or reusing the code.
