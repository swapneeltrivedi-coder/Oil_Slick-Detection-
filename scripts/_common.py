"""Shared, dependency-light helpers for the DARTIS baseline pipeline."""
from __future__ import annotations

import csv
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

SUBSETS = ("oc", "ow", "nc", "nw")
POSITIVE = {"oc", "ow"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def image_files(root: Path):
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    return sorted(p for p in (root / "images").rglob("*") if p.suffix.lower() in extensions)


def subset_for(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    found = [s for s in SUBSETS if s in parts or path.stem.lower().startswith(s + "_")]
    if len(found) != 1:
        raise ValueError(f"Cannot determine exactly one subset for {path}")
    return found[0]


def annotation_for(root: Path, image: Path, subset: str) -> Path | None:
    if subset not in POSITIVE:
        return None
    candidates = list((root / "annotations").rglob(image.stem + ".xml"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one XML for {image}, found {len(candidates)}")
    return candidates[0]


def parse_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    width = int(float(root.findtext("size/width", "0")))
    height = int(float(root.findtext("size/height", "0")))
    boxes = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        coords = tuple(float(box.findtext(k, "nan")) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append(coords)
    return width, height, boxes


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fieldnames=None):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stable_fraction(value: str, seed: int = 42) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64
