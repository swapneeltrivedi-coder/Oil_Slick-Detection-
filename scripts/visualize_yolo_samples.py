#!/usr/bin/env python3

import csv
import random
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "processed" / "splits"
YOLO = ROOT / "processed" / "yolo"
OUTPUT = ROOT / "results" / "figures" / "dataset_audit"

random.seed(42)
OUTPUT.mkdir(parents=True, exist_ok=True)

rows = []

for split in ("train", "val", "test"):
    with (SPLITS / f"{split}.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            row["split"] = split
            rows.append(row)

for subset in ("oc", "ow", "nc", "nw"):
    candidates = [
        row for row in rows
        if row["patch_id"].startswith(subset + "-")
    ]

    samples = random.sample(
        candidates,
        min(5, len(candidates))
    )

    for row in samples:
        patch_id = row["patch_id"]
        split = row["split"]

        image_candidates = list(
            (YOLO / "images" / split).glob(f"{patch_id}*.jpg")
        )

        if not image_candidates:
            raise FileNotFoundError(
                f"No image found for {patch_id}"
            )

        image_path = image_candidates[0]

        label_path = (
            YOLO
            / "labels"
            / split
            / f"{image_path.stem}.txt"
        )

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        W, H = image.size

        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue

            cls, xc, yc, bw, bh = map(float, line.split())

            x1 = (xc - bw / 2) * W
            y1 = (yc - bh / 2) * H
            x2 = (xc + bw / 2) * W
            y2 = (yc + bh / 2) * H

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=max(2, W // 320)
            )

        output_name = f"{subset}_{split}_{image_path.name}"

        image.save(OUTPUT / output_name)

print(f"Saved visual examples to: {OUTPUT}")
print("VISUAL AUDIT GENERATED")
