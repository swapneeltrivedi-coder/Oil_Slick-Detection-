"""Step 2: convert Pascal VOC annotations to YOLO txt labels.

Read-only against images/, annotations/, DARTIS_2019.tab. Writes ONLY to
processed/yolo/labels/ and processed/yolo/images/ (a flat layout - the
train/val/test split into subfolders is create_split.py's job, which runs
after this script and knows the scene-aware assignment).

- Class list is discovered from annotations/ (never hardcoded) and reused
  verbatim by create_split.py when it writes dataset.yaml, so both stages
  agree on the same class-id mapping.
- oil-set images (have an XML) -> one label line per object:
    "<class_id> <cx> <cy> <w> <h>" (normalized, class ids from the
    discovered, alphabetically-sorted class list)
- no-oil-set images (no XML) -> an EMPTY .txt file is written. This marks
  them as labeled-background (0 objects), the correct YOLO convention for
  negative/look-alike training images, rather than "unlabeled".
- Images are attached via relative symlinks (not copies), avoiding a ~500MB
  duplication of source imagery.

Idempotent: processed/yolo/images/ and processed/yolo/labels/ are fully
cleared and rebuilt flat on every run (this also resets any prior
train/val/test split materialized by create_split.py - rerun that script
afterward).

Usage:
    python3 scripts/convert_voc_to_yolo.py [--dataset-root PATH]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

from _common import TAB_COLUMNS, discover_classes, per_image_table, parse_voc_xml, repo_root_default, voc_bbox_to_yolo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, default=repo_root_default(),
                    help="DARTIS_2019 repo root (default: parent of scripts/)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.dataset_root
    images_dir, ann_dir, tab_path = root / "images", root / "annotations", root / "DARTIS_2019.tab"
    yolo_images_dir = root / "processed" / "yolo" / "images"
    yolo_labels_dir = root / "processed" / "yolo" / "labels"

    print(f"Dataset root: {root}")
    classes = discover_classes(ann_dir)
    class_to_id = {name: i for i, name in enumerate(classes)}
    print(f"Discovered classes (id order): {classes}")

    # Clear and rebuild flat - idempotent, and resets any prior split layout.
    for d in (yolo_images_dir, yolo_labels_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    per_image = per_image_table(tab_path)

    n_with_labels, n_empty, n_objects = 0, 0, 0
    for _, row in tqdm(per_image.iterrows(), total=len(per_image), desc="converting VOC->YOLO"):
        jpg_name = row[TAB_COLUMNS["jpg_file"]]
        stem = Path(jpg_name).stem

        img_dst = yolo_images_dir / jpg_name
        img_dst.symlink_to(os.path.relpath(images_dir / jpg_name, img_dst.parent))

        label_path = yolo_labels_dir / f"{stem}.txt"
        xml_ref = row[TAB_COLUMNS["xml_file"]]

        if isinstance(xml_ref, str) and xml_ref.strip():
            parsed = parse_voc_xml(ann_dir / xml_ref)
            lines = []
            for obj in parsed["objects"]:
                cx, cy, w, h = voc_bbox_to_yolo(
                    obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"],
                    parsed["width"], parsed["height"],
                )
                lines.append(f"{class_to_id[obj['name']]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            n_objects += len(lines)
            n_with_labels += 1
        else:
            label_path.write_text("", encoding="utf-8")
            n_empty += 1

    print(f"\nConverted {n_with_labels} oil-set annotations -> {n_objects} YOLO objects")
    print(f"Wrote {n_empty} empty labels for no-oil (background) images")
    print(f"Wrote {n_with_labels + n_empty} images (symlinks) to {yolo_images_dir}")
    print(f"Wrote {n_with_labels + n_empty} label files to {yolo_labels_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
