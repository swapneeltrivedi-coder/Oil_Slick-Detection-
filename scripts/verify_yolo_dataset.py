#!/usr/bin/env python3
"""Fail-fast integrity validation for the generated YOLO dataset."""
import argparse
from pathlib import Path

from _common import project_root, read_csv


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, default=project_root()); args = p.parse_args()
    errors = []; images = boxes = 0; yolo = args.root / "processed/yolo"
    for split in ("train", "val", "test"):
        for row in read_csv(args.root / f"processed/manifests/{split}.csv"):
            image = yolo / "images" / split / row["image_filename"]
            label = yolo / "labels" / split / f"{Path(row['image_filename']).stem}.txt"
            if not image.exists(): errors.append(f"missing image: {image}")
            if not label.exists(): errors.append(f"missing label: {label}"); continue
            lines = [x for x in label.read_text().splitlines() if x.strip()]
            expected_positive = bool(int(row["contains_oil"]))
            if expected_positive != bool(lines): errors.append(f"oil/label mismatch: {image}")
            for line in lines:
                try: cls, xc, yc, w, h = line.split(); values = tuple(map(float, (xc, yc, w, h)))
                except ValueError: errors.append(f"malformed label: {label}: {line}"); continue
                if cls != "0": errors.append(f"invalid class: {label}: {cls}")
                if not (0 <= values[0] <= 1 and 0 <= values[1] <= 1 and 0 < values[2] <= 1 and 0 < values[3] <= 1):
                    errors.append(f"out-of-range box: {label}: {line}")
            images += 1; boxes += len(lines)
    if images != 3655: errors.append(f"expected 3655 processed images, got {images}")
    if boxes != 3225: errors.append(f"expected 3225 boxes, got {boxes}")
    if errors: raise RuntimeError("YOLO DATASET VALIDATION FAILED\n" + "\n".join(errors[:50]))
    print(f"processed images: {images}\npositive boxes: {boxes}\nYOLO DATASET VALIDATION PASSED")


if __name__ == "__main__": main()
