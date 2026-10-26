#!/usr/bin/env python3
"""LGPC-BC viewer for currently flagged local synthetic benchmark cases.

This is a results viewer: it does not rerun or change the detector.
Select benchmark_results.csv, then the benchmark folder containing cases/.
Only local cases with decision_flagged=True are listed.
"""

from __future__ import annotations

import csv
import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = ImageTk = None

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pgm", ".webp"}


class WorkingCasesViewer:
    def __init__(self, root):
        self.root = root
        root.title("LGPC-BC | Flagged Local Test Cases")
        root.geometry("1500x920")
        root.minsize(1050, 700)

        self.csv_path = None
        self.benchmark_dir = None
        self.results_dir = None
        self.cases = []
        self.current_case = None
        self.photos = []

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="LGPC-BC — Currently Detected Local Test Cases",
                  font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(top, text="Displays existing benchmark results only; it does not rerun the detector.").pack(anchor="w", pady=(3, 8))

        controls = ttk.Frame(top)
        controls.pack(fill="x")
        ttk.Button(controls, text="1. Choose benchmark_results.csv", command=self.choose_csv).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="2. Choose benchmark folder…", command=self.choose_benchmark).pack(side="left")
        self.status = tk.StringVar(value="Choose the results CSV and the benchmark folder containing cases/.")
        ttk.Label(root, textvariable=self.status, padding=(10, 0, 10, 8)).pack(anchor="w")

        body = ttk.Panedwindow(root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        left = ttk.Frame(body, padding=(0, 0, 10, 0))
        right = ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=4)

        listbox_frame = ttk.LabelFrame(left, text="Flagged local cases", padding=8)
        listbox_frame.pack(fill="both", expand=True)
        self.count = ttk.Label(listbox_frame, text="0 cases")
        self.count.pack(anchor="w", pady=(0, 6))
        inner = ttk.Frame(listbox_frame)
        inner.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(inner, font=("Consolas", 9), exportselection=False)
        scroll = ttk.Scrollbar(inner, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scroll.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.listbox.bind("<<ListboxSelect>>", self.select_case)

        self.case_title = ttk.Label(right, text="No case selected", font=("Segoe UI", 11, "bold"))
        self.case_title.pack(anchor="w", pady=(0, 4))
        self.case_info = ttk.Label(right, text="Load the benchmark files to begin.", justify="left")
        self.case_info.pack(anchor="w", pady=(0, 8))
        self.panel = ttk.Frame(right)
        self.panel.pack(fill="both", expand=True)
        for c in range(2):
            self.panel.columnconfigure(c, weight=1)
            self.panel.rowconfigure(c, weight=1)
        self.image_labels = []
        for i, title in enumerate(("Original image", "Anomaly heatmap", "Suspicious-region mask", "Eligibility mask")):
            box = ttk.LabelFrame(self.panel, text=title, padding=5)
            box.grid(row=i // 2, column=i % 2, sticky="nsew", padx=4, pady=4)
            label = ttk.Label(box, text="No image loaded", anchor="center")
            label.pack(fill="both", expand=True)
            self.image_labels.append(label)

    def choose_csv(self):
        path = filedialog.askopenfilename(
            title="Choose benchmark_results.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        self.csv_path = Path(path)
        self.results_dir = self.csv_path.parent
        self.load_cases()

    def choose_benchmark(self):
        path = filedialog.askdirectory(title="Choose benchmark folder containing cases/")
        if not path:
            return
        self.benchmark_dir = Path(path)
        self.load_cases()

    def load_cases(self):
        if self.csv_path is None or self.benchmark_dir is None:
            self.status.set("Choose both the results CSV and benchmark folder.")
            return
        try:
            with self.csv_path.open("r", encoding="cp1252", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception as exc:
            messagebox.showerror("Could not read results CSV", str(exc))
            return

        # Keep only local edits the existing benchmark marked as detected.
        selected = []
        for row in rows:
            is_local = str(row.get("is_local", "")).strip().lower() == "true"
            flagged = str(row.get("decision_flagged", "")).strip().lower() == "true"
            if not (is_local and flagged):
                continue
            case_id = row.get("case_id", "").strip()
            case_dir = self.benchmark_dir / "cases" / case_id
            image_path = case_dir / "image.png"
            if not image_path.is_file():
                # Permit selecting the cases directory itself.
                if self.benchmark_dir.name.lower() == "cases":
                    case_dir = self.benchmark_dir / case_id
                    image_path = case_dir / "image.png"
            if not image_path.is_file():
                continue
            out_dir = self.results_dir / "detector_outputs" / case_id
            selected.append({
                "row": row,
                "case_id": case_id,
                "image": image_path,
                "output": out_dir,
            })

        self.cases = selected
        self.listbox.delete(0, "end")
        for item in self.cases:
            row = item["row"]
            score = self.as_float(row.get("M_image"))
            score_text = f"{score:.2f}" if score is not None else "N/A"
            self.listbox.insert("end", f'{row.get("category", "?"):20.20} M={score_text:>6} | {item["case_id"]}')
        self.count.configure(text=f"{len(self.cases)} detected local case(s)")
        self.status.set(
            f"Loaded {len(self.cases)} currently flagged local case(s). "
            "This is a filtered view, not a measure of overall detector accuracy."
        )
        if self.cases:
            self.listbox.selection_set(0)
            self.show_case(0)
        else:
            self.case_title.configure(text="No matching cases found")
            self.case_info.configure(text="Check that the CSV belongs to this benchmark and that the selected folder contains cases/<case_id>/image.png.")

    @staticmethod
    def as_float(value):
        try:
            v = float(value)
            return v if v == v and abs(v) != float("inf") else None
        except (TypeError, ValueError):
            return None

    def select_case(self, _event=None):
        selection = self.listbox.curselection()
        if selection:
            self.show_case(selection[0])

    def show_case(self, index):
        item = self.cases[index]
        row = item["row"]
        self.current_case = item
        score = self.as_float(row.get("M_image"))
        tau = self.as_float(row.get("tau_used"))
        self.case_title.configure(text=item["case_id"])
        self.case_info.configure(
            text=f'Category: {row.get("category", "N/A")}   |   Subtype: {row.get("subtype", "N/A")}\n'
                 f'Image score M: {score:.4f}' if score is not None else f'Category: {row.get("category", "N/A")}'
        )
        # Append threshold and note after configuring the score line.
        info = self.case_info.cget("text")
        if tau is not None:
            info += f"   |   Threshold tau: {tau:.4f}"
        info += "\nStatus: flagged by the current synthetic benchmark run; mask correctness is not implied."
        self.case_info.configure(text=info)

        output = item["output"]
        paths = [
            item["image"],
            output / "normalized_heatmap.png",
            output / "suspicious_region_mask.png",
            output / "eligibility_mask.png",
        ]
        self.photos = []
        for label, path in zip(self.image_labels, paths):
            if Image is None or ImageTk is None:
                label.configure(text="Install Pillow: pip install pillow", image="")
                continue
            if not path.is_file():
                label.configure(text=f"Not found:\n{path.name}", image="")
                continue
            try:
                with Image.open(path) as im:
                    picture = im.convert("RGB")
                picture.thumbnail((520, 300), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(picture)
                self.photos.append(photo)
                label.configure(image=photo, text="")
            except Exception as exc:
                label.configure(text=f"Could not load:\n{path.name}\n{exc}", image="")


def main():
    root = tk.Tk()
    WorkingCasesViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
