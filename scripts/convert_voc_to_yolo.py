#!/usr/bin/env python3
"""Materialize split images as symlinks and convert VOC boxes to YOLO."""
import argparse
import shutil
from pathlib import Path

from _common import annotation_for, parse_voc, project_root, read_csv


def convert(box, width, height):
    xmin, ymin, xmax, ymax = box
    if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
        raise ValueError(f"Invalid VOC box {box} for {width}x{height}")
    return ((xmin + xmax) / (2 * width), (ymin + ymax) / (2 * height),
            (xmax - xmin) / width, (ymax - ymin) / height)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=project_root())
    p.add_argument("--copy-images", action="store_true", help="copy rather than symlink images")
    args = p.parse_args()
    root = args.root.resolve(); yolo = root / "processed/yolo"
    total = boxes_total = 0
    for split in ("train", "val", "test"):
        rows = read_csv(root / f"processed/manifests/{split}.csv")
        image_dir = yolo / "images" / split; label_dir = yolo / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True); label_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            source = root / row["image_path"]; destination = image_dir / row["image_filename"]
            if destination.exists() or destination.is_symlink(): destination.unlink()
            shutil.copy2(source, destination) if args.copy_images else destination.symlink_to(source)
            boxes = []
            if int(row["contains_oil"]):
                xml = annotation_for(root, source, row["subset"])
                width, height, voc_boxes = parse_voc(xml)
                if (width, height) != (int(row["width"]), int(row["height"])):
                    raise RuntimeError(f"Dimension mismatch for {source}")
                boxes = [convert(box, width, height) for box in voc_boxes]
            (label_dir / f"{source.stem}.txt").write_text("".join("0 " + " ".join(f"{v:.8f}" for v in box) + "\n" for box in boxes))
            total += 1; boxes_total += len(boxes)
    yolo.joinpath("dataset.yaml").write_text(
        f"path: {yolo}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: oil\n")
    if total != 3655 or boxes_total != 3225: raise RuntimeError(f"YOLO CONVERSION FAILED: {total} images, {boxes_total} boxes")
    print(f"images: {total}\nboxes: {boxes_total}\nYOLO CONVERSION PASSED")


if __name__ == "__main__": main()
