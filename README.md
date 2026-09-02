# DARTIS 2019 oil-slick detection baseline

Minimal, leakage-resistant YOLOv8n research pipeline for single-class (`oil`)
object detection. Raw `images/`, `annotations/`, and `DARTIS_2019.tab` are read
only and are never modified.

## Reproduce

Run from the repository root, with the three raw-data inputs present:

```bash
python3 scripts/audit_dataset.py
python3 scripts/create_manifest.py
python3 scripts/create_split.py
python3 scripts/convert_voc_to_yolo.py
python3 scripts/verify_yolo_dataset.py
python3 scripts/visualize_dataset_audit.py

python3 --version
python3 -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available()); print(torch.backends.mps.is_built())"
python3 -c "import ultralytics; print(ultralytics.__version__)"
python3 scripts/train_baseline.py --run smoke_test
```

The split command refuses to proceed when any image lacks source scene/product
metadata. It never silently falls back to a random patch split. Images in the
YOLO tree are absolute symlinks by default; use `--copy-images` only when the
training environment cannot follow symlinks.

After all integrity checks and the smoke test pass, start the first meaningful
experiment without overwriting the smoke test:

```bash
python3 scripts/train_baseline.py --run run_01
```
