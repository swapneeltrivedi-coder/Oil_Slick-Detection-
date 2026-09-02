#!/usr/bin/env python3
"""Run immutable raw-data counts before manifest generation."""
from collections import Counter

from _common import POSITIVE, annotation_for, image_files, parse_voc, project_root, subset_for


def main():
    root = project_root(); images = image_files(root); counts = Counter(); objects = 0
    for image in images:
        subset = subset_for(image); counts[subset] += 1
        xml = annotation_for(root, image, subset)
        if subset in POSITIVE:
            width, height, boxes = parse_voc(xml)
            if width <= 0 or height <= 0: raise RuntimeError(f"Invalid dimensions: {xml}")
            objects += len(boxes)
    print(f"images: {len(images)}\nsubsets: {dict(counts)}\nobjects: {objects}")
    if len(images) != 3655 or objects != 3225: raise RuntimeError("RAW DATA AUDIT FAILED")
    print("RAW DATA AUDIT PASSED")


if __name__ == "__main__": main()
