# DARTIS_2019 data-prep pipeline

Oil-slick object-detection dataset (Yang & Singha, 2025, PANGAEA
[doi:10.1594/PANGAEA.980773](https://doi.org/10.1594/PANGAEA.980773)) prepared
for scene-independent YOLO training/evaluation. This repo only prepares data -
no model is trained here.

## Source data (read-only, never modified by these scripts)

- `images/` - 3655 SAR patches (640x640 or larger, JPEG)
- `annotations/` - 1365 Pascal VOC XML files (oil-set patches only)
- `DARTIS_2019.tab` - PANGAEA metadata table, one row per annotated object
  (or per patch for the no-oil set); column `ID (Sentinel_ID)` is the full
  Sentinel-1 product ID and is the scene-grouping key used for splitting.
- `download_dartis.py` - original download script

Single class: **`oil`** (discovered from the XML at runtime, never hardcoded).
No-oil patches (`nc`/`nw` subset tags) are look-alikes/background with zero
annotated objects.

## Pipeline order

Run from the `DARTIS_2019/` repo root, in this order:

```bash
python3 scripts/audit_dataset.py
python3 scripts/convert_voc_to_yolo.py
python3 scripts/create_split.py
python3 scripts/analyze_metadata.py
```

`evaluate_lookalikes.py` is a scaffold for a later, model-dependent task (see
below) and is not part of this data-prep sequence.

Every script accepts `--dataset-root PATH` (default: the parent of
`scripts/`) so the pipeline can run against a copy of the repo elsewhere
without editing any code.

### 1. `audit_dataset.py`

Validates image<->XML pairing (missing/orphan/corrupt files), checks every
bbox is within image bounds, and reports the discovered class list and
per-class object counts. Read-only against the source data. Writes:
- `results/tables/audit_summary.csv`
- `results/tables/audit_issues.csv`

### 2. `convert_voc_to_yolo.py`

Converts Pascal VOC XML to YOLO-format labels using the class list
discovered by scanning `annotations/` (never a hardcoded guess). No-oil
images get an explicit empty label file (labeled background, not
"unlabeled"). Images are attached via relative symlinks, not copies, to
avoid duplicating ~500MB of source imagery. Writes a **flat** layout:
- `processed/yolo/images/*.jpg` (symlinks)
- `processed/yolo/labels/*.txt`

### 3. `create_split.py` - the correctness-critical stage

**Scene-aware** train/val/test split: every patch from the same source
scene (`Sentinel_ID`) goes to exactly one split. Uses
`sklearn.model_selection.GroupShuffleSplit` grouped on scene id, so the
ratio is applied to the number of *scenes*, not the number of patches, per
the thesis requirement. Default ratio 70/15/15, seed 42; both configurable:

```bash
python3 scripts/create_split.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

Explicitly prints `Scene leakage check: PASS/FAIL` and aborts (non-zero
exit) if any scene ever ends up split across train/val/test - this must
never happen; a scene-independent split is the entire methodological point
of this preprocessing pipeline. Writes:
- `processed/splits/{train,val,test}.csv` (columns: `patch_id`,
  `source_scene`, `split`)
- reorganizes `processed/yolo/images/` and `processed/yolo/labels/` from
  flat into `{train,val,test}/` subfolders
- `processed/yolo/dataset.yaml`

Idempotent: reruns first flatten any prior split back to the flat layout
convert_voc_to_yolo.py produces, so results are reproducible from a fixed
seed regardless of how many times it's rerun.

### 4. `analyze_metadata.py`

Summarises, per split: patch count, scene count, oil vs. no-oil (look-alike)
patch balance, object count, and coastal vs. open-water balance (derived
from the `oc`/`ow`/`nc`/`nw` subset tag). Writes:
- `results/tables/split_summary.csv`
- `results/figures/patches_per_split.png`
- `results/figures/scenes_per_split.png`
- `results/figures/oil_vs_nooil_per_split.png`
- `results/figures/coastal_vs_water_per_split.png`

### 5. `evaluate_lookalikes.py` - scaffold only

Defines the `LookalikeEvaluator` interface for false-positive / look-alike
cluster analysis against a trained model's predictions. Every method is a
`TODO` stub raising `NotImplementedError` - deliberately not implemented
here, since it requires model inference, which is out of scope for this
data-prep task.

## Directory layout produced by this pipeline

```
DARTIS_2019/
├── images/, annotations/, DARTIS_2019.tab, download_dartis.py   (source, read-only)
├── scripts/
│   ├── _common.py                 (shared VOC/tab parsing helpers, not a pipeline stage)
│   ├── audit_dataset.py
│   ├── convert_voc_to_yolo.py
│   ├── create_split.py
│   ├── analyze_metadata.py
│   └── evaluate_lookalikes.py     (scaffold)
├── processed/
│   ├── yolo/
│   │   ├── images/{train,val,test}/
│   │   ├── labels/{train,val,test}/
│   │   └── dataset.yaml
│   └── splits/{train,val,test}.csv
├── experiments/
│   ├── baseline_yolov8n/{config.yaml, metrics.csv, predictions/}   (empty scaffolding)
│   └── baseline_yolov8s/{config.yaml, metrics.csv, predictions/}   (empty scaffolding)
└── results/{figures/, tables/, false_positives/, false_negatives/}
```

## Dependencies

Python 3.11, standard library + `numpy`, `pandas`, `Pillow`, `pyyaml`,
`scikit-learn`, `matplotlib`, `tqdm`. No `opencv-python` currently in use
(Pillow covers all image I/O this pipeline needs). No `ultralytics` -
intentionally not installed; this repo does not train models.
