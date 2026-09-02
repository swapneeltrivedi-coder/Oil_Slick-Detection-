"""Step 1: audit the DARTIS_2019 source data before any conversion.

Read-only against images/, annotations/, DARTIS_2019.tab. Writes only to
results/tables/audit_summary.csv and results/tables/audit_issues.csv.

Checks:
  - every jpg_file referenced in the .tab exists in images/, and vice versa
  - every non-empty xml_file referenced in the .tab exists in annotations/
    and parses as valid Pascal VOC (flags corrupt/unparseable XML)
  - every bbox is within image bounds and non-degenerate (xmax>xmin, ymax>ymin)
  - prints the discovered class list and per-class object counts

Missing/orphan/corrupt files are ERRORs (exit 1). Bbox-out-of-bounds and
image/tab dimension mismatches are WARNINGs (non-fatal, still reported).

Usage:
    python3 scripts/audit_dataset.py [--dataset-root PATH]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from _common import TAB_COLUMNS, discover_classes, per_image_table, parse_voc_xml, repo_root_default


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, default=repo_root_default(),
                    help="DARTIS_2019 repo root (default: parent of scripts/)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.dataset_root
    images_dir, ann_dir, tab_path = root / "images", root / "annotations", root / "DARTIS_2019.tab"
    tables_dir = root / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root: {root}")
    print("Discovering class list from annotations/ (not hardcoded)...")
    classes = discover_classes(ann_dir)
    print(f"Discovered classes: {classes}")

    print("Reading DARTIS_2019.tab ...")
    per_image = per_image_table(tab_path)

    issues: list[dict] = []

    def flag(severity: str, kind: str, ref: str, msg: str):
        issues.append({"severity": severity, "kind": kind, "ref": ref, "msg": msg})

    tab_jpg = set(per_image[TAB_COLUMNS["jpg_file"]])
    disk_jpg = {p.name for p in images_dir.glob("*.jpg")}
    disk_xml = {p.name for p in ann_dir.glob("*.xml")}

    for missing in sorted(tab_jpg - disk_jpg):
        flag("ERROR", "missing_image", missing, "referenced in .tab but not found in images/")
    for orphan in sorted(disk_jpg - tab_jpg):
        flag("WARNING", "orphan_image", orphan, "present in images/ but not referenced in .tab")

    tab_xml = {x for x in per_image[TAB_COLUMNS["xml_file"]] if isinstance(x, str) and x.strip()}
    for missing in sorted(tab_xml - disk_xml):
        flag("ERROR", "missing_annotation", missing, "referenced in .tab but not found in annotations/")
    for orphan in sorted(disk_xml - tab_xml):
        flag("WARNING", "orphan_annotation", orphan, "present in annotations/ but not referenced in .tab")

    class_counts: Counter[str] = Counter()
    n_objects = 0
    n_checked = 0

    # DataFrame.iterrows() (not itertuples()) because .tab column names
    # contain spaces/parens that itertuples() would mangle.
    for _, row in tqdm(per_image.iterrows(), total=len(per_image), desc="auditing images"):
        jpg_name = row[TAB_COLUMNS["jpg_file"]]
        jpg_path = images_dir / jpg_name
        if not jpg_path.exists():
            continue  # already flagged above

        try:
            with Image.open(jpg_path) as im:
                im.verify()
            with Image.open(jpg_path) as im:
                actual_w, actual_h = im.size
        except Exception as e:
            flag("ERROR", "corrupt_image", jpg_name, f"PIL failed to open/verify: {e}")
            continue

        tab_w, tab_h = int(row[TAB_COLUMNS["width"]]), int(row[TAB_COLUMNS["height"]])
        if (actual_w, actual_h) != (tab_w, tab_h):
            flag("WARNING", "dim_mismatch_tab", jpg_name, f"image={actual_w}x{actual_h} tab={tab_w}x{tab_h}")

        xml_ref = row[TAB_COLUMNS["xml_file"]]
        if isinstance(xml_ref, str) and xml_ref.strip():
            xml_path = ann_dir / xml_ref
            if not xml_path.exists():
                continue  # already flagged above
            try:
                parsed = parse_voc_xml(xml_path)
            except Exception as e:
                flag("ERROR", "corrupt_xml", xml_ref, f"failed to parse: {e}")
                continue

            if (parsed["width"], parsed["height"]) != (actual_w, actual_h):
                flag("WARNING", "dim_mismatch_xml", xml_ref,
                     f"xml={parsed['width']}x{parsed['height']} image={actual_w}x{actual_h}")

            for obj in parsed["objects"]:
                class_counts[obj["name"]] += 1
                n_objects += 1
                if obj["name"] not in classes:
                    flag("ERROR", "unknown_class", xml_ref, f"class '{obj['name']}' not in discovered set")
                if obj["xmax"] <= obj["xmin"] or obj["ymax"] <= obj["ymin"]:
                    flag("ERROR", "degenerate_bbox", xml_ref, f"bbox {obj}")
                elif not (0 <= obj["xmin"] < obj["xmax"] <= actual_w and 0 <= obj["ymin"] < obj["ymax"] <= actual_h):
                    flag("WARNING", "bbox_out_of_bounds", xml_ref, f"bbox {obj} vs image {actual_w}x{actual_h}")

        n_checked += 1

    errors = [i for i in issues if i["severity"] == "ERROR"]
    warnings = [i for i in issues if i["severity"] == "WARNING"]

    with open(tables_dir / "audit_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["n_tab_rows_deduped_to_images", len(per_image)])
        writer.writerow(["n_images_on_disk", len(disk_jpg)])
        writer.writerow(["n_xml_on_disk", len(disk_xml)])
        writer.writerow(["n_images_checked", n_checked])
        writer.writerow(["discovered_classes", "|".join(classes)])
        writer.writerow(["n_objects_total", n_objects])
        for cls, count in sorted(class_counts.items()):
            writer.writerow([f"class_count[{cls}]", count])
        writer.writerow(["n_errors", len(errors)])
        writer.writerow(["n_warnings", len(warnings)])

    with open(tables_dir / "audit_issues.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["severity", "kind", "ref", "msg"])
        writer.writeheader()
        writer.writerows(issues)

    print(f"\nDiscovered classes: {classes}")
    print(f"Class object counts: {dict(class_counts)}")
    print(f"Images checked: {n_checked} | Errors: {len(errors)} | Warnings: {len(warnings)}")
    print(f"Wrote {tables_dir / 'audit_summary.csv'}")
    print(f"Wrote {tables_dir / 'audit_issues.csv'}")

    if errors:
        print(f"\nAUDIT FAILED: {len(errors)} critical issue(s). See results/tables/audit_issues.csv")
        return 1
    print(f"\nAUDIT PASSED ({len(warnings)} non-blocking warning(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
