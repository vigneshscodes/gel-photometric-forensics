"""
03_dataset_stats.py

Reads the final manifest.csv and prints the numbers you need for the
paper's dataset/experimental-setup section. Read-only — does not touch
any images.
"""

import csv
import sys
from pathlib import Path
from collections import Counter, defaultdict

from src.config import OUTPUT_ROOT


def main():
    manifest_path = Path(OUTPUT_ROOT) / "manifest.csv"
    if not manifest_path.is_file():
        sys.exit(f"ERROR: manifest.csv not found at {manifest_path}")

    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    status_counts = Counter(r["status"] for r in rows)
    reason_counts = Counter(r["reason"] for r in rows if r["status"] == "rejected")
    split_counts = Counter(r.get("split", "") for r in rows if r["status"] == "accepted")

    papers_total = len(set(r["doi"] for r in rows))
    papers_accepted = len(set(r["doi"] for r in rows if r["status"] == "accepted"))
    papers_train = len(set(r["doi"] for r in rows if r.get("split") == "train"))
    papers_test = len(set(r["doi"] for r in rows if r.get("split") == "test"))

    per_paper = defaultdict(int)
    for r in rows:
        if r["status"] == "accepted":
            per_paper[r["doi"]] += 1
    counts_per_paper = sorted(per_paper.values())

    print("========== DATASET STATISTICS ==========")
    print(f"Total Blot/Gel images considered:      {total}")
    print(f"  Accepted (final clean dataset):      {status_counts.get('accepted', 0)}")
    print(f"  Rejected (total):                    {status_counts.get('rejected', 0)}")
    for reason, n in reason_counts.most_common():
        print(f"    - {reason:12s}: {n}")

    print(f"\nPapers touched:                         {papers_total}")
    print(f"Papers contributing to final dataset:   {papers_accepted}")
    print(f"  Train papers: {papers_train}   Test papers: {papers_test}")

    print(f"\nFinal split (images):")
    print(f"  Train: {split_counts.get('train', 0)}")
    print(f"  Test:  {split_counts.get('test', 0)}")

    if counts_per_paper:
        print(f"\nImages per paper (accepted only):")
        print(f"  min={min(counts_per_paper)}  median={counts_per_paper[len(counts_per_paper)//2]}  "
              f"max={max(counts_per_paper)}  mean={sum(counts_per_paper)/len(counts_per_paper):.1f}")


if __name__ == "__main__":
    main()