#!/usr/bin/env python3
"""Save deterministic random annotated samples without touching raw images."""
import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

from _common import project_root, read_csv


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, default=project_root()); p.add_argument("--per-subset", type=int, default=5); p.add_argument("--seed", type=int, default=42); args = p.parse_args()
    rows = sum((read_csv(args.root / f"processed/manifests/{s}.csv") for s in ("train", "val", "test")), [])
    rng = random.Random(args.seed); output = args.root / "results/figures/dataset_audit"; output.mkdir(parents=True, exist_ok=True)
    for subset in ("oc", "ow", "nc", "nw"):
        candidates = [r for r in rows if r["subset"] == subset]
        for row in rng.sample(candidates, min(args.per_subset, len(candidates))):
            image = Image.open(args.root / row["image_path"]).convert("RGB"); draw = ImageDraw.Draw(image); w, h = image.size
            label = args.root / "processed/yolo/labels" / next(s for s in ("train", "val", "test") if row in read_csv(args.root / f"processed/manifests/{s}.csv")) / f"{Path(row['image_filename']).stem}.txt"
            for line in label.read_text().splitlines():
                _, xc, yc, bw, bh = map(float, line.split()); x1=(xc-bw/2)*w; y1=(yc-bh/2)*h; x2=(xc+bw/2)*w; y2=(yc+bh/2)*h
                draw.rectangle((x1,y1,x2,y2), outline="red", width=max(2, w//320))
            image.save(output / f"{subset}_{row['image_filename']}.jpg", quality=92)
    print(f"VISUAL AUDIT PASSED: examples saved to {output}")


if __name__ == "__main__": main()
