"""Step 4: summarise the scene-aware split - per-split scene counts, class
balance, and coastal vs open-water balance.

Reads processed/splits/{train,val,test}.csv, DARTIS_2019.tab, and the label
files under processed/yolo/labels/ (read-only). Writes ONLY to
results/tables/split_summary.csv and results/figures/*.png.

Usage:
    python3 scripts/analyze_metadata.py [--dataset-root PATH]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _common import TAB_COLUMNS, coast_or_water, discover_classes, per_image_table, repo_root_default

SPLITS = ("train", "val", "test")

# Okabe-Ito colorblind-safe categorical palette, assigned in a fixed order
# per chart (never cycled / re-assigned by rank).
SPLIT_COLOR = {"train": "#0072B2", "val": "#E69F00", "test": "#009E73"}
OIL_COLOR = {"oil": "#D55E00", "no_oil": "#56B4E9"}
TERRAIN_COLOR = {"coast": "#CC79A7", "water": "#56B4E9"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, default=repo_root_default(),
                    help="DARTIS_2019 repo root (default: parent of scripts/)")
    return p.parse_args()


def bar_chart(out_path: Path, categories: list[str], values: list[int], colors: list[str],
              title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(categories, values, color=colors, width=0.6)
    for rect, v in zip(bars, values):
        ax.annotate(str(v), (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def grouped_bar_chart(out_path: Path, group_labels: list[str], series: dict[str, list[int]],
                       series_colors: dict[str, str], title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    n_series = len(series)
    width = 0.8 / n_series
    x = range(len(group_labels))
    for i, (name, vals) in enumerate(series.items()):
        offsets = [xi + (i - (n_series - 1) / 2) * width for xi in x]
        bars = ax.bar(offsets, vals, width=width, label=name, color=series_colors[name])
        for rect, v in zip(bars, vals):
            ax.annotate(str(v), (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(group_labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = args.dataset_root
    tab_path = root / "DARTIS_2019.tab"
    ann_dir = root / "annotations"
    splits_dir = root / "processed" / "splits"
    labels_dir = root / "processed" / "yolo" / "labels"
    tables_dir = root / "results" / "tables"
    figures_dir = root / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        if not (splits_dir / f"{split}.csv").exists():
            raise RuntimeError(f"{splits_dir / f'{split}.csv'} missing - run scripts/create_split.py first")

    print(f"Dataset root: {root}")
    classes = discover_classes(ann_dir)
    print(f"Classes: {classes}")

    per_image = per_image_table(tab_path)
    per_image["patch_id"] = per_image[TAB_COLUMNS["jpg_file"]].apply(lambda f: Path(f).stem)
    per_image["terrain"] = per_image[TAB_COLUMNS["subset"]].apply(coast_or_water)

    split_assignment = pd.concat([pd.read_csv(splits_dir / f"{s}.csv") for s in SPLITS], ignore_index=True)
    merged = per_image.merge(split_assignment[["patch_id", "split"]], on="patch_id", how="inner")
    assert len(merged) == len(per_image), "every patch must appear in exactly one split"

    def n_objects_for(split: str, patch_id: str) -> int:
        label_path = labels_dir / split / f"{patch_id}.txt"
        if not label_path.exists():
            return 0
        text = label_path.read_text(encoding="utf-8").strip()
        return len(text.splitlines()) if text else 0

    merged["n_objects"] = [n_objects_for(row.split, row.patch_id) for row in merged.itertuples()]
    merged["has_oil"] = merged["n_objects"] > 0

    rows = []
    for split in SPLITS:
        sub = merged[merged["split"] == split]
        rows.append({
            "split": split,
            "n_patches": len(sub),
            "n_scenes": sub[TAB_COLUMNS["scene_id"]].nunique(),
            "n_oil_patches": int(sub["has_oil"].sum()),
            "n_no_oil_patches": int((~sub["has_oil"]).sum()),
            "n_objects": int(sub["n_objects"].sum()),
            "n_coast": int((sub["terrain"] == "coast").sum()),
            "n_water": int((sub["terrain"] == "water").sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(tables_dir / "split_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {tables_dir / 'split_summary.csv'}")

    bar_chart(
        figures_dir / "patches_per_split.png",
        SPLITS, [int(summary.set_index("split").loc[s, "n_patches"]) for s in SPLITS],
        [SPLIT_COLOR[s] for s in SPLITS], "Patches per split", "# patches",
    )
    bar_chart(
        figures_dir / "scenes_per_split.png",
        SPLITS, [int(summary.set_index("split").loc[s, "n_scenes"]) for s in SPLITS],
        [SPLIT_COLOR[s] for s in SPLITS], "Source scenes per split", "# unique scenes",
    )
    grouped_bar_chart(
        figures_dir / "oil_vs_nooil_per_split.png",
        list(SPLITS),
        {
            "oil": [int(summary.set_index("split").loc[s, "n_oil_patches"]) for s in SPLITS],
            "no_oil": [int(summary.set_index("split").loc[s, "n_no_oil_patches"]) for s in SPLITS],
        },
        OIL_COLOR, "Oil vs. no-oil (look-alike) patches per split", "# patches",
    )
    grouped_bar_chart(
        figures_dir / "coastal_vs_water_per_split.png",
        list(SPLITS),
        {
            "coast": [int(summary.set_index("split").loc[s, "n_coast"]) for s in SPLITS],
            "water": [int(summary.set_index("split").loc[s, "n_water"]) for s in SPLITS],
        },
        TERRAIN_COLOR, "Coastal vs. open-water patches per split", "# patches",
    )
    print(f"Wrote 4 figures to {figures_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
