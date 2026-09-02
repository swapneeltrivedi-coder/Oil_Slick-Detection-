import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime

IMAGE_DIR = "images"
XML_DIR = "annotations"
TAB_FILE = "DARTIS_2019.tab"


# ============================================================
# 1. BASIC FILE COUNTS
# ============================================================

images = sorted(
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".jpg")
)

xmls = sorted(
    f for f in os.listdir(XML_DIR)
    if f.lower().endswith(".xml")
)

print("=" * 70)
print("DARTIS-2019 DATASET AUDIT")
print("=" * 70)

print("\n[1] FILE COUNTS")
print("-" * 70)

print(f"JPG images        : {len(images)}")
print(f"XML annotations   : {len(xmls)}")


# ============================================================
# 2. PREFIX DISTRIBUTION
# ============================================================

prefix_counts = Counter()

for filename in images:
    prefix = filename.split("-")[0]
    prefix_counts[prefix] += 1

print("\n[2] IMAGE PREFIX DISTRIBUTION")
print("-" * 70)

for prefix in ["oc", "ow", "nc", "nw"]:
    print(f"{prefix}: {prefix_counts[prefix]}")

print(f"Total: {sum(prefix_counts.values())}")


# ============================================================
# 3. IMAGE <-> XML MATCHING
# ============================================================

image_stems = {
    os.path.splitext(f)[0]
    for f in images
}

xml_stems = {
    os.path.splitext(f)[0]
    for f in xmls
}

matched = image_stems & xml_stems
images_without_xml = image_stems - xml_stems
xml_without_image = xml_stems - image_stems

print("\n[3] IMAGE / XML CORRESPONDENCE")
print("-" * 70)

print(f"Images                  : {len(image_stems)}")
print(f"XML files               : {len(xml_stems)}")
print(f"Matching image/XML      : {len(matched)}")
print(f"Images without XML      : {len(images_without_xml)}")
print(f"XML without image       : {len(xml_without_image)}")


# ============================================================
# 4. XML ANNOTATION ANALYSIS
# ============================================================

total_objects = 0
objects_per_image = Counter()

object_names = Counter()

bbox_widths = []
bbox_heights = []
bbox_areas = []

xml_image_names = {}

bad_xml = []
empty_xml = []

for xml_file in xmls:

    path = os.path.join(XML_DIR, xml_file)

    try:

        tree = ET.parse(path)
        root = tree.getroot()

        filename_node = root.find(".//filename")

        if filename_node is not None:
            xml_image_names[xml_file] = filename_node.text

        objects = root.findall(".//object")

        if len(objects) == 0:
            empty_xml.append(xml_file)

        objects_per_image[xml_file] = len(objects)

        for obj in objects:

            name = obj.find("name")

            if name is not None:
                object_names[name.text.strip()] += 1

            bbox = obj.find("bndbox")

            if bbox is not None:

                xmin = int(float(bbox.findtext("xmin")))
                ymin = int(float(bbox.findtext("ymin")))
                xmax = int(float(bbox.findtext("xmax")))
                ymax = int(float(bbox.findtext("ymax")))

                width = xmax - xmin
                height = ymax - ymin
                area = width * height

                bbox_widths.append(width)
                bbox_heights.append(height)
                bbox_areas.append(area)

                total_objects += 1

    except Exception as e:

        bad_xml.append((xml_file, str(e)))


print("\n[4] XML / OIL OBJECT ANALYSIS")
print("-" * 70)

print(f"XML files parsed          : {len(xmls) - len(bad_xml)}")
print(f"Bad XML files             : {len(bad_xml)}")
print(f"Empty XML files           : {len(empty_xml)}")
print(f"Total annotated objects   : {total_objects}")

print("\nObject labels:")

for label, count in object_names.most_common():
    print(f"  {label}: {count}")


# ============================================================
# 5. OBJECTS PER IMAGE
# ============================================================

object_count_distribution = Counter(
    objects_per_image.values()
)

print("\n[5] OBJECTS PER ANNOTATED IMAGE")
print("-" * 70)

print(f"Minimum objects/image : {min(objects_per_image.values())}")
print(f"Maximum objects/image : {max(objects_per_image.values())}")

average_objects = (
    total_objects / len(objects_per_image)
    if objects_per_image
    else 0
)

print(f"Average objects/image : {average_objects:.2f}")

print("\nDistribution:")

for count in sorted(object_count_distribution):
    print(
        f"  {count} object(s): "
        f"{object_count_distribution[count]} images"
    )


# ============================================================
# 6. BOUNDING BOX STATISTICS
# ============================================================

def stats(values):

    if not values:
        return None

    values = sorted(values)

    n = len(values)

    if n % 2 == 0:
        median = (values[n//2 - 1] + values[n//2]) / 2
    else:
        median = values[n//2]

    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / n,
        "median": median
    }


width_stats = stats(bbox_widths)
height_stats = stats(bbox_heights)
area_stats = stats(bbox_areas)

print("\n[6] BOUNDING BOX STATISTICS")
print("-" * 70)

print("Bounding-box width:")
print(f"  min    : {width_stats['min']}")
print(f"  max    : {width_stats['max']}")
print(f"  mean   : {width_stats['mean']:.2f}")
print(f"  median : {width_stats['median']:.2f}")

print("\nBounding-box height:")
print(f"  min    : {height_stats['min']}")
print(f"  max    : {height_stats['max']}")
print(f"  mean   : {height_stats['mean']:.2f}")
print(f"  median : {height_stats['median']:.2f}")

print("\nBounding-box area:")
print(f"  min    : {area_stats['min']}")
print(f"  max    : {area_stats['max']}")
print(f"  mean   : {area_stats['mean']:.2f}")
print(f"  median : {area_stats['median']:.2f}")


# ============================================================
# 7. OIL IMAGE DISTRIBUTION
# ============================================================

oil_prefix_counts = Counter()

for filename in images:

    prefix = filename.split("-")[0]

    if prefix in ["oc", "ow"]:
        oil_prefix_counts[prefix] += 1

print("\n[7] OIL / NO-OIL DISTRIBUTION")
print("-" * 70)

oil_total = oil_prefix_counts["oc"] + oil_prefix_counts["ow"]

no_oil_total = (
    prefix_counts["nc"] +
    prefix_counts["nw"]
)

print(f"Oil images       : {oil_total}")
print(f"No-oil images    : {no_oil_total}")

print("\nOil:")
print(f"  Coast (oc)     : {oil_prefix_counts['oc']}")
print(f"  Water (ow)     : {oil_prefix_counts['ow']}")

print("\nNo-oil:")
print(f"  Coast (nc)     : {prefix_counts['nc']}")
print(f"  Water (nw)     : {prefix_counts['nw']}")


# ============================================================
# 8. CLUSTER DISTRIBUTION
# ============================================================

cluster_counts = defaultdict(Counter)

for filename in images:

    parts = filename.replace(".jpg", "").split("-")

    if len(parts) >= 4:

        prefix = parts[0]
        cluster = parts[1]

        if prefix in ["nc", "nw"]:
            cluster_counts[prefix][cluster] += 1

print("\n[8] NO-OIL CLUSTER DISTRIBUTION")
print("-" * 70)

for prefix in ["nc", "nw"]:

    print(f"\n{prefix} clusters:")

    for cluster, count in sorted(
        cluster_counts[prefix].items()
    ):
        print(f"  {cluster}: {count}")


# ============================================================
# 9. TAB FILE ANALYSIS
# ============================================================

print("\n[9] TAB FILE")
print("-" * 70)

if os.path.exists(TAB_FILE):

    rows = []

    with open(
        TAB_FILE,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        for line in f:

            line = line.rstrip("\n")

            if not line.strip():
                continue

            parts = line.split("\t")

            if len(parts) >= 10:
                rows.append(parts)

    print(f"Rows read: {len(rows)}")

    if rows:

        print(f"Columns detected in first row: {len(rows[0])}")

        # Expected fields based on README / observed .tab structure

        dates = []
        patches = []
        products = []
        widths = []
        heights = []

        for row in rows:

            if len(row) > 7:

                patches.append(row[4])
                dates.append(row[5])
                products.append(row[7])

            if len(row) > 9:

                try:
                    widths.append(int(row[8]))
                    heights.append(int(row[9]))
                except ValueError:
                    pass

        print(f"Unique patch names: {len(set(patches))}")
        print(f"Unique Sentinel products: {len(set(products))}")

        if widths:
            print(
                "Patch dimensions: "
                f"{min(widths)}-{max(widths)} × "
                f"{min(heights)}-{max(heights)}"
            )

        # Date range

        clean_dates = []

        for d in dates:

            try:
                clean_dates.append(
                    datetime.fromisoformat(d)
                )
            except Exception:
                pass

        if clean_dates:

            print(
                "Acquisition date range: "
                f"{min(clean_dates).date()} → "
                f"{max(clean_dates).date()}"
            )


# ============================================================
# 10. DUPLICATE PATCH NAMES
# ============================================================

print("\n[10] DUPLICATE PATCH-NAME CHECK")
print("-" * 70)

patch_counter = Counter()

if os.path.exists(TAB_FILE):

    with open(
        TAB_FILE,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        for line in f:

            parts = line.rstrip("\n").split("\t")

            if len(parts) > 4:
                patch_counter[parts[4]] += 1

duplicates = {
    name: count
    for name, count in patch_counter.items()
    if count > 1
}

print(f"Unique patch names : {len(patch_counter)}")
print(f"Duplicated names   : {len(duplicates)}")

if duplicates:

    print("\nExamples of duplicated patch names:")

    for name, count in list(duplicates.items())[:10]:
        print(f"  {name}: {count} records")


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)

print(f"Total JPGs                 : {len(images)}")
print(f"Total XMLs                 : {len(xmls)}")
print(f"Oil images                 : {oil_total}")
print(f"No-oil images              : {no_oil_total}")
print(f"Total oil objects          : {total_objects}")
print(f"Image/XML matches          : {len(matched)}")
print(f"XML without image          : {len(xml_without_image)}")
print(f"Bad XML files              : {len(bad_xml)}")
print(f"Empty XML files            : {len(empty_xml)}")
print(f"Duplicated patch names     : {len(duplicates)}")

print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
