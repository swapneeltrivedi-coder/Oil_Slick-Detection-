#!/usr/bin/env python3
"""Train the reproducible YOLOv8n smoke test or the prepared 30-epoch run."""
import argparse
import csv
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from _common import project_root, read_csv


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, default=project_root()); p.add_argument("--run", choices=("smoke_test", "run_01"), default="smoke_test"); p.add_argument("--device"); args = p.parse_args()
    epochs = 3 if args.run == "smoke_test" else 30
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    project = args.root / "experiments/baseline_yolov8n"
    result = YOLO("yolov8n.pt").train(data=str(args.root / "processed/yolo/dataset.yaml"), epochs=epochs, imgsz=640, batch=-1 if device != "cpu" else 8, device=device, seed=42, workers=2, patience=10, pretrained=True, project=str(project), name=args.run, exist_ok=False)
    metrics = result.results_dict
    out = args.root / "results"; table = out / "tables/baseline_yolov8n_metrics.csv"; table.parent.mkdir(parents=True, exist_ok=True)
    counts = {s: len(read_csv(args.root / f"processed/manifests/{s}.csv")) for s in ("train", "val", "test")}
    row = {"model":"yolov8n.pt", "epochs":epochs, **{f"{s}_images":n for s,n in counts.items()}, "precision":metrics.get("metrics/precision(B)"), "recall":metrics.get("metrics/recall(B)"), "mAP50":metrics.get("metrics/mAP50(B)"), "mAP50_95":metrics.get("metrics/mAP50-95(B)")}
    with table.open("w", newline="") as f: writer=csv.DictWriter(f, fieldnames=row); writer.writeheader(); writer.writerow(row)
    figures = out / "figures/baseline_yolov8n"; figures.mkdir(parents=True, exist_ok=True)
    for name in ("results.png", "PR_curve.png", "F1_curve.png", "confusion_matrix.png", "val_batch0_pred.jpg"):
        source = Path(result.save_dir) / name
        if source.exists(): shutil.copy2(source, figures / name)
    summary = out / "baseline_yolov8n_summary.md"
    summary.write_text(f"# YOLOv8n baseline\n\n## Dataset\nDARTIS 2019 single-class oil detection: {sum(counts.values())} images.\n\n## Model\nYOLOv8n pretrained checkpoint.\n\n## Experimental configuration\n{epochs} epochs, image size 640, seed 42, device `{device}`.\n\n## Train/val/test split\n{counts['train']} / {counts['val']} / {counts['test']} images, grouped by source product/scene.\n\n## Metrics\n- Precision: {row['precision']}\n- Recall: {row['recall']}\n- mAP50: {row['mAP50']}\n- mAP50-95: {row['mAP50_95']}\n\n## Initial observations\nNo qualitative claims have been made; inspect generated predictions.\n\n## Known limitations\nThis is a small YOLOv8n baseline without hyperparameter tuning.\n\n## Next experiment\nRun the 30-epoch configuration after reviewing smoke-test outputs.\n")
    print(f"training completed\nvalidation completed\nbest.pt: {result.save_dir}/weights/best.pt\nlast.pt: {result.save_dir}/weights/last.pt")
    print(row); print("SMOKE TEST PASSED" if args.run == "smoke_test" else "BASELINE RUN PASSED")


if __name__ == "__main__": main()
