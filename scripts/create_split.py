"""Step 3: SCENE-AWARE train/val/test split.

All patches from the same source scene (DARTIS_2019.tab column
"ID (Sentinel_ID)", the full Sentinel-1 product ID) go to exactly ONE
split - never split a scene across sets. This is the single most important
correctness property of this pipeline: it is what makes downstream
evaluation scene-independent.

Reads DARTIS_2019.tab directly (read-only) plus the flat
processed/yolo/images/ + labels/ built by convert_voc_to_yolo.py. Writes
ONLY to:
  - processed/splits/{train,val,test}.csv  (patch_id, source_scene, split)
  - processed/yolo/images/{train,val,test}/  (reorganized from flat)
  - processed/yolo/labels/{train,val,test}/  (reorganized from flat)
  - processed/yolo/dataset.yaml

Split ratio is by SCENE count (not by patch count), using
sklearn.model_selection.GroupShuffleSplit grouped on scene_id, which splits
groups (not samples) according to the given ratios - exactly matching the
brief. Default 70/15/15, seed=42, both configurable via CLI.

Idempotent: on every run, any existing train/val/test subfolders under
processed/yolo/images|labels are first flattened back to the root (so a
rerun always starts from the same flat state) before being re-split.

Usage:
    python3 scripts/create_split.py [--dataset-root PATH]
        [--train-ratio 0.70] [--val-ratio 0.15] [--test-ratio 0.15] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import yaml
from sklearn.model_selection import GroupShuffleSplit

from _common import TAB_COLUMNS, discover_classes, per_image_table, repo_root_default

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, default=repo_root_default(),
                    help="DARTIS_2019 repo root (default: parent of scripts/)")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def relocate_symlink(link: Path, new_path: Path):
    """Move a (possibly relative) symlink to new_path, recomputing its
    relative target fresh from the resolved real source - never derived
    algebraically from the old relative string, which is fragile and easy
    to get wrong across a directory-depth change."""
    real_target = link.resolve(strict=True)
    link.unlink()
    new_path.symlink_to(os.path.relpath(real_target, new_path.parent))


def flatten_existing_split(images_dir: Path, labels_dir: Path):
    """Undo a previous split materialization so this run starts from a
    clean flat state, regardless of what a prior run left behind."""
    for split in SPLITS:
        split_dir = images_dir / split
        if not split_dir.exists():
            continue
        for f in split_dir.iterdir():
            relocate_symlink(f, images_dir / f.name)
        split_dir.rmdir()
    for split in SPLITS:
        split_dir = labels_dir / split
        if not split_dir.exists():
            continue
        for f in split_dir.iterdir():
            f.rename(labels_dir / f.name)
        split_dir.rmdir()


def scene_group_split(patch_ids: list[str], scene_ids: list[str],
                       train_ratio: float, val_ratio: float, test_ratio: float,
                       seed: int) -> dict[str, str]:
    """Assign every patch to train/val/test such that all patches sharing a
    scene_id land in the same split, at approximately the requested ratio
    of SCENES (GroupShuffleSplit's test_size operates on unique groups)."""
    import numpy as np
    patch_ids = np.array(patch_ids)
    scene_ids = np.array(scene_ids)
    X = np.zeros(len(patch_ids))

    # Stage A: hold out test scenes.
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(X, groups=scene_ids))

    # Stage B: split remaining scenes into train/val at the equivalent
    # relative ratio (val_ratio of the *original* whole, so the final val
    # fraction still matches the requested val_ratio of all scenes).
    remaining_frac = train_ratio + val_ratio
    val_size_within_remaining = val_ratio / remaining_frac
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_within_remaining, random_state=seed)
    trainval_scene_ids = scene_ids[trainval_idx]
    train_sub_idx, val_sub_idx = next(gss_val.split(X[trainval_idx], groups=trainval_scene_ids))
    train_idx = trainval_idx[train_sub_idx]
    val_idx = trainval_idx[val_sub_idx]

    assignment = {}
    for idx in train_idx:
        assignment[patch_ids[idx]] = "train"
    for idx in val_idx:
        assignment[patch_ids[idx]] = "val"
    for idx in test_idx:
        assignment[patch_ids[idx]] = "test"
    return assignment


def main() -> int:
    args = parse_args()
    root = args.dataset_root
    ann_dir = root / "annotations"
    tab_path = root / "DARTIS_2019.tab"
    yolo_images_dir = root / "processed" / "yolo" / "images"
    yolo_labels_dir = root / "processed" / "yolo" / "labels"
    splits_dir = root / "processed" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    assert abs(sum(ratios.values()) - 1.0) < 1e-6, f"ratios must sum to 1.0, got {ratios}"
    print(f"Dataset root: {root}")
    print(f"Split ratios (by scene): {ratios} | seed={args.seed}")

    if not yolo_images_dir.exists() or not any(yolo_images_dir.iterdir()):
        raise RuntimeError(
            f"{yolo_images_dir} is empty - run scripts/convert_voc_to_yolo.py first"
        )

    # Undo any split from a previous run first, so the "flat images exist"
    # check below is reliable whether this is a fresh conversion or a rerun.
    flatten_existing_split(yolo_images_dir, yolo_labels_dir)

    if not any(yolo_images_dir.glob("*.jpg")):
        raise RuntimeError(
            f"{yolo_images_dir} has no flat images after flattening - run scripts/convert_voc_to_yolo.py first"
        )

    per_image = per_image_table(tab_path)
    patch_ids = [Path(f).stem for f in per_image[TAB_COLUMNS["jpg_file"]]]
    scene_ids = list(per_image[TAB_COLUMNS["scene_id"]])
    assert all(isinstance(s, str) and s.strip() for s in scene_ids), \
        "every patch must have a non-empty scene_id - aborting rather than falling back to a random split"

    assignment = scene_group_split(patch_ids, scene_ids, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

    # --- THE hard invariant: no scene straddles more than one split -----
    scene_to_splits: dict[str, set[str]] = {}
    for pid, scene in zip(patch_ids, scene_ids):
        scene_to_splits.setdefault(scene, set()).add(assignment[pid])
    leaking = {s: v for s, v in scene_to_splits.items() if len(v) > 1}
    leakage_ok = len(leaking) == 0
    print(f"\nScene leakage check: {'PASS' if leakage_ok else 'FAIL'}")
    if not leakage_ok:
        print(f"  {len(leaking)} scene(s) span multiple splits: {list(leaking.items())[:5]}")
        return 1

    # --- write processed/splits/{train,val,test}.csv ---------------------
    rows_by_split = {s: [] for s in SPLITS}
    for pid, scene in zip(patch_ids, scene_ids):
        split = assignment[pid]
        rows_by_split[split].append({"patch_id": pid, "source_scene": scene, "split": split})

    for split in SPLITS:
        with open(splits_dir / f"{split}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["patch_id", "source_scene", "split"])
            writer.writeheader()
            writer.writerows(rows_by_split[split])

    # --- reorganize processed/yolo/{images,labels}/ into train/val/test --
    jpg_by_patch = dict(zip(patch_ids, per_image[TAB_COLUMNS["jpg_file"]]))
    for split in SPLITS:
        (yolo_images_dir / split).mkdir(exist_ok=True)
        (yolo_labels_dir / split).mkdir(exist_ok=True)

    for pid in patch_ids:
        split = assignment[pid]
        jpg_name = jpg_by_patch[pid]
        relocate_symlink(yolo_images_dir / jpg_name, yolo_images_dir / split / jpg_name)
        label_name = f"{pid}.txt"
        (yolo_labels_dir / label_name).rename(yolo_labels_dir / split / label_name)

    # --- dataset.yaml ------------------------------------------------------
    classes = discover_classes(ann_dir)
    dataset_yaml = {
        "path": str(root / "processed" / "yolo"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": classes,
    }
    with open(root / "processed" / "yolo" / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    for split in SPLITS:
        sub = rows_by_split[split]
        n_scenes = len({r["source_scene"] for r in sub})
        print(f"  {split}: {len(sub)} patches, {n_scenes} scenes "
              f"({len(sub) / len(patch_ids):.1%} of {len(patch_ids)} total)")

    print(f"\nWrote {splits_dir}/{{train,val,test}}.csv")
    print(f"Reorganized {yolo_images_dir} and {yolo_labels_dir} into train/val/test")
    print(f"Wrote {root / 'processed' / 'yolo' / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
