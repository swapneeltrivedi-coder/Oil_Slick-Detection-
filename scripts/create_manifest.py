#!/usr/bin/env python3

import csv
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

IMAGES_DIR = ROOT / "images"
ANNOTATIONS_DIR = ROOT / "annotations"
METADATA_FILE = ROOT / "DARTIS_2019.tab"

OUTPUT_DIR = ROOT / "processed" / "manifests"
OUTPUT_FILE = OUTPUT_DIR / "dataset_manifest.csv"

POSITIVE = {"oc", "ow"}
SUBSETS = {"oc", "ow", "nc", "nw"}


def parse_voc(xml_path):
    root = ET.parse(xml_path).getroot()

    objects = root.findall("object")

    return len(objects)


def get_subset(filename):
    subset = filename.split("-")[0].lower()

    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset: {filename}")

    return subset


def load_metadata():
    """
    Build one metadata entry per image.

    DARTIS_2019.tab fields observed in the dataset:
      0 = subset
      1 = image filename
      2 = annotation filename
      3 = object id
      4 = Sentinel-1 patch/source scene
      5 = timestamp start
      6 = timestamp end
      7 = Sentinel-1 SAFE source product
    """

    lookup = {}

    with METADATA_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.reader(f, delimiter="\t")

        for row in reader:
            if len(row) < 8:
                continue

            image_filename = row[1].strip()

            if not image_filename.lower().endswith(".jpg"):
                continue

            # Metadata may contain multiple rows for the
            # same image because one image can have
            # multiple annotated oil objects.
            if image_filename not in lookup:
                lookup[image_filename] = {
                    "source_scene": row[4].strip(),
                    "timestamp": row[5].strip(),
                    "source_product": row[7].strip(),
                }

    return lookup


def main():

    print("=" * 60)
    print("DARTIS_2019 DATASET MANIFEST GENERATOR")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata()

    print(f"Metadata image entries found : {len(metadata)}")

    rows = []

    images = sorted(IMAGES_DIR.glob("*.jpg"))

    for i, image_path in enumerate(images, start=1):

        filename = image_path.name
        image_id = image_path.stem

        subset = get_subset(filename)

        contains_oil = int(subset in POSITIVE)

        environment = (
            "coast"
            if subset in {"oc", "nc"}
            else "water"
        )

        if contains_oil:
            xml_path = ANNOTATIONS_DIR / f"{image_id}.xml"

            if not xml_path.exists():
                raise FileNotFoundError(
                    f"Missing annotation: {xml_path}"
                )

            num_objects = parse_voc(xml_path)
            annotation_filename = xml_path.name
        else:
            num_objects = 0
            annotation_filename = ""

        with Image.open(image_path) as image:
            width, height = image.size

        meta = metadata.get(
            filename,
            {
                "source_scene": "",
                "source_product": "",
                "timestamp": "",
            }
        )

        rows.append(
            {
                "image_id": image_id,
                "image_filename": filename,
                "image_path": str(
                    image_path.relative_to(ROOT)
                ),
                "subset": subset,
                "contains_oil": contains_oil,
                "environment": environment,
                "annotation_filename": annotation_filename,
                "num_objects": num_objects,
                "width": width,
                "height": height,
                "source_scene": meta["source_scene"],
                "source_product": meta["source_product"],
                "timestamp": meta["timestamp"],
            }
        )

        if i % 500 == 0:
            print(f"Processed {i}/{len(images)} images")

    fieldnames = [
        "image_id",
        "image_filename",
        "image_path",
        "subset",
        "contains_oil",
        "environment",
        "annotation_filename",
        "num_objects",
        "width",
        "height",
        "source_scene",
        "source_product",
        "timestamp",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(
        row["subset"]
        for row in rows
    )

    total_objects = sum(
        row["num_objects"]
        for row in rows
    )

    oil_images = sum(
        row["contains_oil"]
        for row in rows
    )

    no_oil_images = len(rows) - oil_images

    unique_scenes = {
        row["source_scene"]
        for row in rows
        if row["source_scene"]
    }

    unique_products = {
        row["source_product"]
        for row in rows
        if row["source_product"]
    }

    missing_scene = sum(
        not row["source_scene"]
        for row in rows
    )

    missing_product = sum(
        not row["source_product"]
        for row in rows
    )

    missing_both = sum(
        not row["source_scene"]
        and not row["source_product"]
        for row in rows
    )

    print()
    print("=" * 60)
    print("MANIFEST SUMMARY")
    print("=" * 60)

    print(f"Total images          : {len(rows)}")
    print(f"Oil images            : {oil_images}")
    print(f"No-oil images         : {no_oil_images}")

    print()
    print(f"oc                    : {counts['oc']}")
    print(f"ow                    : {counts['ow']}")
    print(f"nc                    : {counts['nc']}")
    print(f"nw                    : {counts['nw']}")

    print()
    print(f"Total objects         : {total_objects}")

    print()
    print(f"Unique source scenes  : {len(unique_scenes)}")
    print(f"Unique products       : {len(unique_products)}")

    print()
    print(f"Missing source scene  : {missing_scene}")
    print(f"Missing product       : {missing_product}")
    print(f"Missing BOTH          : {missing_both}")

    errors = []

    expected = {
        "oc": 375,
        "ow": 990,
        "nc": 351,
        "nw": 1939,
    }

    if len(rows) != 3655:
        errors.append(
            f"Expected 3655 images, got {len(rows)}"
        )

    if total_objects != 3225:
        errors.append(
            f"Expected 3225 objects, got {total_objects}"
        )

    for subset, expected_count in expected.items():

        if counts[subset] != expected_count:

            errors.append(
                f"{subset}: expected "
                f"{expected_count}, "
                f"got {counts[subset]}"
            )

    if errors:

        print()
        print("❌ MANIFEST VALIDATION FAILED")

        for error in errors:
            print(" -", error)

        raise SystemExit(1)

    print()
    print("✅ MANIFEST PASSED")
    print()
    print(f"Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
