#!/usr/bin/env python3
"""Create deterministic, group-exclusive train/validation/test manifests."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from _common import project_root, read_csv, stable_fraction, write_csv


def group_key(row):
    return row["source_product"].strip() or row["source_scene"].strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=project_root())
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    manifest = args.root / "processed/manifests/dataset_manifest.csv"
    rows = read_csv(manifest)
    missing = [r["image_id"] for r in rows if not group_key(r)]
    if missing:
        raise RuntimeError(f"SPLIT FAILED: {len(missing)} images lack a reliable source scene/product; refusing patch-level split")
    groups = defaultdict(list)
    for row in rows: groups[group_key(row)].append(row)
    # Largest groups first; stable hash breaks ties. Greedy assignment minimizes target error.
    targets = {"train": .70 * len(rows), "val": .15 * len(rows), "test": .15 * len(rows)}
    assigned = {k: [] for k in targets}
    ordered = sorted(groups.items(), key=lambda x: (-len(x[1]), stable_fraction(x[0], args.seed)))
    for _, group in ordered:
        split = min(targets, key=lambda s: (len(assigned[s]) + len(group) - targets[s]) ** 2 - (len(assigned[s]) - targets[s]) ** 2)
        assigned[split].extend(group)
    scene_sets = {}
    for split, split_rows in assigned.items():
        write_csv(args.root / f"processed/manifests/{split}.csv", split_rows, rows[0].keys())
        scene_sets[split] = {group_key(r) for r in split_rows}
        c = Counter(r["subset"] for r in split_rows)
        print(f"\n{split.upper()}\nnumber images: {len(split_rows)}")
        print(f"number oil images: {sum(int(r['contains_oil']) for r in split_rows)}")
        print(f"number no-oil images: {sum(not int(r['contains_oil']) for r in split_rows)}")
        for subset in ("oc", "ow", "nc", "nw"): print(f"{subset}: {c[subset]}")
        print(f"oil objects: {sum(int(r['num_objects']) for r in split_rows)}")
        print(f"unique source scenes: {len(scene_sets[split])}")
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for a, b in pairs:
        overlap = scene_sets[a] & scene_sets[b]
        print(f"{a}-scenes intersection {b}-scenes = {len(overlap)}")
        if overlap: raise RuntimeError("SPLIT FAILED: source leakage detected")
    print("SPLIT VALIDATION PASSED")


if __name__ == "__main__": main()
