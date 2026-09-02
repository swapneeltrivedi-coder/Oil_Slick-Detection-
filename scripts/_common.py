"""Internal helper module shared by the DARTIS_2019 data-prep scripts.

Not a pipeline stage itself - just the VOC/tab parsing logic that
audit_dataset.py, convert_voc_to_yolo.py, create_split.py, and
analyze_metadata.py all need, kept in one place instead of duplicated four
times. Every function here only ever reads source data (images/,
annotations/, DARTIS_2019.tab); nothing here writes anything.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path

import pandas as pd

# Column layout of DARTIS_2019.tab, confirmed by manual inspection of the
# header block and cross-checked against the annotation XML files.
TAB_COLUMNS = {
    "subset": "Image set (subset; oc : oil/coast; ow : ...)",
    "jpg_file": "IMAGE (jpg_file)",
    "xml_file": "Binary (xml_file)",
    "patch_name": "ID (patch_name)",
    "scene_id": "ID (Sentinel_ID)",
    "width": "Width [pixel] (patch_width)",
    "height": "Height [pixel] (patch_height)",
}


def repo_root_default() -> Path:
    """Default dataset root: the DARTIS_2019/ directory containing scripts/."""
    return Path(__file__).resolve().parent.parent


def read_tab_file(tab_path: Path) -> pd.DataFrame:
    """Read DARTIS_2019.tab, skipping the PANGAEA header comment block.

    The header block ends at the line containing a bare '*/'; the next
    line is the tab-separated column header.
    """
    with open(tab_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    header_end = next(i for i, line in enumerate(lines) if line.strip() == "*/")
    df = pd.read_csv(StringIO("".join(lines[header_end + 1:])), sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df


def per_image_table(tab_path: Path) -> pd.DataFrame:
    """One row per unique image (the .tab has one row per annotated object
    for the oil set, so oc/ow rows are de-duplicated on jpg_file)."""
    df = read_tab_file(tab_path)
    return df.drop_duplicates(subset=[TAB_COLUMNS["jpg_file"]]).reset_index(drop=True)


def parse_voc_xml(xml_path: Path) -> dict:
    """Parse a Pascal VOC annotation file into a plain dict:
    {"width": int, "height": int, "objects": [{"name", "xmin", "ymin", "xmax", "ymax"}, ...]}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    width, height = int(size.find("width").text), int(size.find("height").text)
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bnd = obj.find("bndbox")
        objects.append({
            "name": name,
            "xmin": int(float(bnd.find("xmin").text)),
            "ymin": int(float(bnd.find("ymin").text)),
            "xmax": int(float(bnd.find("xmax").text)),
            "ymax": int(float(bnd.find("ymax").text)),
        })
    return {"width": width, "height": height, "objects": objects}


def discover_classes(annotations_dir: Path) -> list[str]:
    """Enumerate every distinct <name> label across all annotation XML files.

    Never hardcode the class list - always derive it from the source data,
    sorted for a deterministic, reproducible class-id assignment.
    """
    names = set()
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        parsed = parse_voc_xml(xml_path)
        for obj in parsed["objects"]:
            names.add(obj["name"])
    return sorted(names)


def voc_bbox_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float,
                      img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert an absolute-pixel VOC bbox to normalized YOLO (cx, cy, w, h)."""
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return cx, cy, w, h


def coast_or_water(subset: str) -> str:
    """oc/nc -> 'coast', ow/nw -> 'water', derived from the subset tag."""
    return "coast" if subset in ("oc", "nc") else "water"
