#!/usr/bin/env python3
"""Build the canonical one-row-per-image DARTIS manifest."""
import argparse
import csv
from collections import Counter
from pathlib import Path

from PIL import Image

from _common import POSITIVE, annotation_for, image_files, parse_voc, project_root, subset_for, write_csv


def norm(text):
    return "".join(c.lower() for c in str(text) if c.isalnum())


def pick(row, terms):
    normalized = {norm(k): v for k, v in row.items()}
    for term in terms:
        for key, value in normalized.items():
            if term in key and value:
                return value.strip()
    return ""


def metadata_index(path):
    if not path.exists():
        raise FileNotFoundError(f"Required metadata missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    index = {}
    for row in rows:
        identity = pick(row, ("imagefilename", "filename", "imagename", "patch", "imageid"))
        if identity:
            for key in {norm(identity), norm(Path(identity).name), norm(Path(identity).stem)}:
                index.setdefault(key, []).append(row)
    return index


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=project_root())
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    output = args.output or root / "processed/manifests/dataset_manifest.csv"
    meta = metadata_index(root / "DARTIS_2019.tab")
    rows = []
    for image in image_files(root):
        subset = subset_for(image)
        xml = annotation_for(root, image, subset)
        if xml:
            width, height, boxes = parse_voc(xml)
        else:
            with Image.open(image) as im:
                width, height = im.size
            boxes = []
        matches = meta.get(norm(image.name), meta.get(norm(image.stem), []))
        m = matches[0] if matches else {}
        scene = pick(m, ("sourcescene", "sceneid", "scene"))
        product = pick(m, ("sourceproduct", "productid", "product"))
        timestamp = pick(m, ("timestamp", "acquisitiondate", "datetime", "date"))
        rows.append({
            "image_id": image.stem, "image_filename": image.name,
            "image_path": str(image.relative_to(root)), "subset": subset,
            "contains_oil": int(subset in POSITIVE),
            "environment": "coast" if subset in {"oc", "nc"} else "water",
            "annotation_filename": xml.name if xml else "", "num_objects": len(boxes),
            "width": width, "height": height, "source_scene": scene,
            "source_product": product, "timestamp": timestamp,
        })
    if len({r["image_path"] for r in rows}) != len(rows):
        raise RuntimeError("Duplicate images in manifest")
    write_csv(output, rows)
    counts = Counter(r["subset"] for r in rows)
    print(f"total images: {len(rows)}\noil images: {sum(int(r['contains_oil']) for r in rows)}")
    print(f"no-oil images: {sum(not int(r['contains_oil']) for r in rows)}")
    for subset in ("oc", "ow", "nc", "nw"): print(f"{subset}: {counts[subset]}")
    print(f"total objects: {sum(int(r['num_objects']) for r in rows)}")
    print(f"unique source scenes/products: {len({r['source_scene'] or r['source_product'] for r in rows if r['source_scene'] or r['source_product']})}")
    if len(rows) != 3655 or sum(int(r["num_objects"]) for r in rows) != 3225:
        raise RuntimeError("MANIFEST FAILED: expected 3655 images and 3225 objects")
    print("MANIFEST PASSED")


if __name__ == "__main__": main()
