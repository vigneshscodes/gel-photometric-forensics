#!/usr/bin/env python3
"""LGPC-BC Image Forensics Workbench."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = ImageTk = None

ROOT = Path(__file__).resolve().parent
DETECTOR = ROOT / "run_detector.py"
DEFAULT_INPUT = ROOT / "benchmark_runs" / "repro"

IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff",
    ".bmp", ".pgm", ".webp"
}


class LGPCWorkbench:
    def __init__(self, root):
        self.root = root
        root.title("LGPC-BC | Image Forensics Workbench")
        root.geometry("1420x900")
        root.minsize(1050, 700)

        self.input_dir = tk.StringVar(value=str(DEFAULT_INPUT))
        self.output_dir = tk.StringVar()
        self.status = tk.StringVar(value="Choose a dataset folder and select an image.")

        self.image_paths = []
        self.selected_image = None
        self.running = False
        self.photo = None

        self.setup_style()
        self.build_ui()
        self.update_output_path()
        self.scan_images()

    def setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Header.TLabel",
            font=("Segoe UI", 17, "bold")
        )
        style.configure(
            "Run.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=8
        )
        style.configure(
            "TLabelframe.Label",
            font=("Segoe UI", 10, "bold")
        )

    def build_ui(self):
        shell = ttk.Frame(self.root, padding=12)
        shell.pack(fill="both", expand=True)

        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(2, weight=1)

        # Header
        header = ttk.Frame(shell)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(
            header,
            text="LGPC-BC Image Forensics Workbench",
            style="Header.TLabel"
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="Select an image, run the detector, and inspect its results."
        ).pack(anchor="w", pady=(3, 0))

        # Folder selection
        paths = ttk.LabelFrame(
            shell, text="Dataset and results", padding=10
        )
        paths.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        paths.columnconfigure(1, weight=1)

        ttk.Label(paths, text="Input folder").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )

        ttk.Entry(
            paths, textvariable=self.input_dir
        ).grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Button(
            paths, text="Browse folder...",
            command=self.choose_input
        ).grid(row=0, column=2, padx=5)

        ttk.Button(
            paths, text="Scan images",
            command=self.scan_images
        ).grid(row=0, column=3, padx=5)

        ttk.Label(paths, text="Output folder").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=4
        )

        ttk.Entry(
            paths, textvariable=self.output_dir
        ).grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Button(
            paths, text="Browse...",
            command=self.choose_output
        ).grid(row=1, column=2, padx=5)

        ttk.Label(
            paths,
            text="Defaults to <input-folder>_results"
        ).grid(row=1, column=3, padx=5)

        # Main content: image list + central dashboard
        body = ttk.Panedwindow(shell, orient="horizontal")
        body.grid(row=2, column=0, sticky="nsew")

        left = ttk.Frame(body, padding=(0, 0, 10, 0))
        center = ttk.Frame(body)

        body.add(left, weight=1)
        body.add(center, weight=4)

        # Left panel
        files_box = ttk.LabelFrame(
            left, text="Available images", padding=8
        )
        files_box.pack(fill="both", expand=True)

        self.count_label = ttk.Label(files_box, text="0 images")
        self.count_label.pack(anchor="w", pady=(0, 6))

        list_frame = ttk.Frame(files_box)
        list_frame.pack(fill="both", expand=True)

        self.image_list = tk.Listbox(
            list_frame,
            font=("Consolas", 9),
            selectmode="browse",
            exportselection=False
        )
        scrollbar = ttk.Scrollbar(
            list_frame,
            orient="vertical",
            command=self.image_list.yview
        )

        self.image_list.configure(
            yscrollcommand=scrollbar.set
        )

        self.image_list.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.image_list.bind(
            "<<ListboxSelect>>", self.on_select_image
        )

        self.run_button = ttk.Button(
            left,
            text="▶  Run detector",
            style="Run.TButton",
            command=self.run_detector
        )
        self.run_button.pack(fill="x", pady=(8, 0))

        ttk.Button(
            left,
            text="Open results folder",
            command=self.open_output
        ).pack(fill="x", pady=(6, 0))

        # Centre panel
        viewer = ttk.LabelFrame(
            center, text="Visual results", padding=8
        )
        viewer.pack(fill="both", expand=True)

        self.selected_label = ttk.Label(
            viewer,
            text="No image selected",
            font=("Segoe UI", 10, "bold")
        )
        self.selected_label.pack(anchor="w", pady=(0, 6))

        self.dashboard_label = ttk.Label(
            viewer,
            text="Select an image and click Run detector.",
            anchor="center",
            justify="center"
        )
        self.dashboard_label.pack(fill="both", expand=True)

        # Bottom log
        log_box = ttk.LabelFrame(
            shell, text="Run log", padding=8
        )
        log_box.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        log_box.columnconfigure(0, weight=1)

        self.log = tk.Text(
            log_box,
            height=8,
            wrap="word",
            font=("Consolas", 9),
            bg="#111827",
            fg="#e5e7eb",
            state="disabled"
        )
        log_scroll = ttk.Scrollbar(
            log_box,
            orient="vertical",
            command=self.log.yview
        )
        self.log.configure(yscrollcommand=log_scroll.set)

        self.log.grid(row=0, column=0, sticky="ew")
        log_scroll.grid(row=0, column=1, sticky="ns")

        footer = ttk.Frame(shell)
        footer.grid(row=4, column=0, sticky="ew", pady=(7, 0))

        ttk.Label(
            footer, textvariable=self.status
        ).pack(side="left", fill="x", expand=True)

        ttk.Button(
            footer, text="Clear log",
            command=self.clear_log
        ).pack(side="right")

    def log_message(self, message):
        self.log.configure(state="normal")
        self.log.insert("end", str(message) + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def update_output_path(self):
        folder = Path(self.input_dir.get()).expanduser()
        self.output_dir.set(
            str(folder.parent / f"{folder.name}_results")
        )

    def choose_input(self):
        current = self.input_dir.get()

        path = filedialog.askdirectory(
            title="Choose dataset folder",
            initialdir=current if Path(current).is_dir() else str(ROOT)
        )

        if path:
            self.input_dir.set(path)
            self.update_output_path()
            self.scan_images()

    def choose_output(self):
        path = filedialog.askdirectory(
            title="Choose output folder",
            initialdir=self.output_dir.get()
        )

        if path:
            self.output_dir.set(path)

    def scan_images(self):
        folder = Path(self.input_dir.get()).expanduser()

        self.image_list.delete(0, "end")
        self.image_paths = []

        if not folder.is_dir():
            self.count_label.configure(text="Folder not found")
            self.status.set("Choose a valid input folder.")
            return

        excluded_dirs = {
            "mask", "masks", "label", "labels",
            "ground_truth", "ground-truth", "gt"
        }

        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue

            if path.suffix.lower() not in IMAGE_EXTS:
                continue

            try:
                relative = path.relative_to(folder)
            except ValueError:
                continue

            if any(
                part.lower() in excluded_dirs
                for part in relative.parts[:-1]
            ):
                continue

            stem = path.stem.lower()
            if any(
                token in stem
                for token in ("_mask", "mask_", "_label", "label_")
            ):
                continue

            self.image_paths.append(path)

        for path in self.image_paths:
            self.image_list.insert(
                "end", str(path.relative_to(folder))
            )

        count = len(self.image_paths)
        self.count_label.configure(text=f"{count} image(s) found")
        self.status.set(f"Found {count} images.")

        if count:
            self.image_list.selection_set(0)
            self.on_select_image()

    def on_select_image(self, _event=None):
        selection = self.image_list.curselection()

        if not selection:
            return

        self.selected_image = self.image_paths[selection[0]]

        self.selected_label.configure(
            text=str(self.selected_image)
        )

        if Image is not None:
            try:
                with Image.open(self.selected_image) as im:
                    preview = im.convert("RGB")

                self.show_image(preview)
            except Exception as exc:
                self.log_message(f"Preview error: {exc}")

    def show_image(self, image):
        if Image is None or ImageTk is None:
            self.dashboard_label.configure(
                text="Install Pillow to preview images."
            )
            return

        self.root.update_idletasks()

        width = max(500, self.dashboard_label.winfo_width() - 20)
        height = max(320, self.dashboard_label.winfo_height() - 20)

        image.thumbnail(
            (width, height),
            Image.Resampling.LANCZOS
        )

        self.photo = ImageTk.PhotoImage(image)
        self.dashboard_label.configure(
            image=self.photo, text=""
        )

    def run_detector(self):
        if self.running:
            return

        selection = self.image_list.curselection()

        if selection:
            self.selected_image = self.image_paths[selection[0]]

        image = self.selected_image

        if image is None or not image.is_file():
            messagebox.showerror(
                "Choose an image",
                "Select an image from the list first."
            )
            return

        if not DETECTOR.is_file():
            messagebox.showerror(
                "Detector not found",
                f"Expected detector at:\n{DETECTOR}"
            )
            return

        input_folder = Path(self.input_dir.get()).expanduser()
        output_root = Path(self.output_dir.get()).expanduser()

        try:
            relative = image.relative_to(input_folder)
        except ValueError:
            relative = Path(image.name)

        # Separate result directory for every input image.
        output = output_root / relative.parent / image.stem

        self.running = True
        self.run_button.configure(state="disabled")
        self.status.set("Running detector...")

        self.log_message("=" * 65)
        self.log_message(f"Input:  {image}")
        self.log_message(f"Output: {output}")

        threading.Thread(
            target=self.worker,
            args=(image, output),
            daemon=True
        ).start()

    def worker(self, image, output):
        try:
            output.mkdir(parents=True, exist_ok=True)

            command = [
                sys.executable,
                str(DETECTOR),
                "--input", str(image),
                "--output", str(output)
            ]

            self.root.after(
                0, self.log_message,
                "$ " + subprocess.list2cmdline(command)
            )

            process = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                errors="replace"
            )

            output_text = (
                (process.stdout or "") +
                (process.stderr or "")
            )

            if output_text:
                self.root.after(
                    0, self.log_message, output_text
                )

            if process.returncode != 0:
                self.root.after(
                    0, self.finish_error,
                    f"Detector failed with exit code {process.returncode}."
                )
                return

            dashboard = self.create_dashboard(image, output)

            self.root.after(
                0, self.show_dashboard,
                dashboard, output
            )

        except Exception as exc:
            self.root.after(
                0, self.log_message, f"ERROR: {exc}"
            )
            self.root.after(
                0, self.finish_error, "Run failed. Check the log."
            )

    def create_dashboard(self, image, output):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        score_file = output / "image_level_score.json"

        try:
            score = (
                json.loads(score_file.read_text(encoding="utf-8"))
                if score_file.exists() else {}
            )
        except (OSError, json.JSONDecodeError):
            score = {}

        panels = [
            ("Original image", image),
            ("Normalized anomaly heatmap",
             output / "normalized_heatmap.png"),
            ("Suspicious-region mask",
             output / "suspicious_region_mask.png"),
            ("Eligibility mask",
             output / "eligibility_mask.png"),
            ("Dispersion curve",
             output / "dispersion_curve.png")
        ]

        fig, axes = plt.subplots(
            2, 3, figsize=(15, 8)
        )
        axes = axes.ravel()

        for ax, (title, path) in zip(axes, panels):
            if path.is_file():
                try:
                    with Image.open(path) as im:
                        picture = im.copy()

                    ax.imshow(
                        picture,
                        cmap="gray" if picture.mode in ("L", "1", "I", "F") else None
                    )
                    ax.set_title(title)

                except Exception as exc:
                    ax.set_title(f"Could not read {title}: {exc}")
            else:
                ax.text(
                    0.5, 0.5,
                    f"Missing output:\n{path.name}",
                    ha="center", va="center", wrap=True
                )
                ax.set_title(title)

            ax.axis("off")

        value = score.get("image_level_score_M_image", "N/A")
        tau = score.get("tau_threshold_used", "N/A")
        flagged = score.get("decision_flagged_inconsistent")

        if flagged is True:
            decision = "FLAGGED as inconsistent"
        elif flagged is False:
            decision = "Not flagged"
        else:
            decision = "N/A"

        axes[5].axis("off")
        axes[5].text(
            0.03, 0.9,
            "LGPC-BC detector result",
            fontsize=14, weight="bold", va="top"
        )
        axes[5].text(
            0.03, 0.7,
            f"Image score: {value}\n"
            f"Threshold: {tau}\n"
            f"Decision: {decision}",
            fontsize=12, va="top", wrap=True
        )

        fig.suptitle(image.name, fontsize=13)
        fig.tight_layout()

        dashboard = output / "detector_dashboard.png"
        fig.savefig(
            dashboard, dpi=160, bbox_inches="tight"
        )
        plt.close(fig)

        return dashboard

    def show_dashboard(self, dashboard, output):
        try:
            if Image is not None:
                with Image.open(dashboard) as im:
                    preview = im.convert("RGB")
                self.show_image(preview)

            self.status.set(f"Complete. Results saved to: {output}")
            self.log_message(f"Dashboard saved: {dashboard}")

        except Exception as exc:
            self.status.set("Run completed, but dashboard preview failed.")
            self.log_message(f"Preview error: {exc}")

        finally:
            self.running = False
            self.run_button.configure(state="normal")

    def finish_error(self, message):
        self.status.set(message)
        self.running = False
        self.run_button.configure(state="normal")

    def open_output(self):
        folder = Path(self.output_dir.get()).expanduser()

        if not folder.is_dir():
            messagebox.showinfo(
                "No results yet",
                "Run the detector first."
            )
            return

        try:
            if os.name == "nt":
                os.startfile(str(folder))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])

        except Exception as exc:
            messagebox.showerror(
                "Could not open folder", str(exc)
            )


def main():
    root = tk.Tk()
    LGPCWorkbench(root)
    root.mainloop()


if __name__ == "__main__":
    main()