"""Step 5: SCAFFOLD ONLY - look-alike cluster / false-positive analysis.

This defines the interface a later, model-dependent task will implement. It
does NOT run any model inference and does NOT read any prediction files yet
- every method below is a stub that raises NotImplementedError with a TODO
describing what it must do.

Rationale for keeping this a scaffold: look-alike/false-positive analysis
requires trained-model predictions (see experiments/<name>/predictions/),
and this data-prep task explicitly excludes training or running any model.

Intended eventual usage:
    python3 scripts/evaluate_lookalikes.py \\
        --dataset-root PATH --experiment experiments/baseline_yolov8n \\
        --predictions experiments/baseline_yolov8n/predictions
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import repo_root_default


class LookalikeEvaluator:
    """Interface for analyzing false positives against the no-oil
    (look-alike) patches, and clustering look-alike false positives by
    SAR signature type (e.g. wind streaks, current shear, biogenic slicks).

    Every method is a TODO stub - fill these in once a trained model's
    predictions exist under experiments/<name>/predictions/.
    """

    def __init__(self, dataset_root: Path, predictions_dir: Path):
        self.dataset_root = dataset_root
        self.predictions_dir = predictions_dir

    def load_predictions(self):
        """TODO: load per-image YOLO prediction files (class, conf, bbox)
        from self.predictions_dir. Format depends on which framework wrote
        them (e.g. Ultralytics .txt or .json export) - not decided yet,
        since no model has been trained in this repo."""
        raise NotImplementedError(
            "TODO: implement once experiments/<name>/predictions/ contains real "
            "model output. Out of scope for the data-prep pipeline."
        )

    def match_predictions_to_ground_truth(self, predictions, iou_threshold: float = 0.5):
        """TODO: match predicted boxes to processed/yolo/labels/test/*.txt
        ground truth via IoU, to classify each prediction as TP/FP/FN."""
        raise NotImplementedError("TODO: implement IoU-based TP/FP/FN matching.")

    def cluster_lookalike_false_positives(self, false_positives):
        """TODO: cluster false positives that fire on no-oil (nc/nw) patches
        by SAR signature characteristics, to characterize which look-alike
        phenomena the model confuses for oil (e.g. via unsupervised
        clustering on patch/crop features with scikit-learn)."""
        raise NotImplementedError("TODO: implement look-alike clustering.")

    def write_false_positive_negative_reports(self, matched):
        """TODO: write per-image crops/records to results/false_positives/
        and results/false_negatives/, plus a summary table to
        results/tables/."""
        raise NotImplementedError("TODO: implement report writing.")

    def run(self):
        self.load_predictions()  # will raise NotImplementedError today


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, default=repo_root_default(),
                    help="DARTIS_2019 repo root (default: parent of scripts/)")
    p.add_argument("--predictions", type=Path, required=False,
                    help="Path to an experiments/<name>/predictions/ directory (not required yet - scaffold only)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("evaluate_lookalikes.py is a SCAFFOLD - no model inference is implemented here.")
    print("This data-prep task explicitly excludes training/running any model.")
    print("Interface defined: LookalikeEvaluator(dataset_root, predictions_dir).run()")
    if args.predictions is not None:
        evaluator = LookalikeEvaluator(args.dataset_root, args.predictions)
        evaluator.run()  # intentionally raises NotImplementedError
    return 0


if __name__ == "__main__":
    sys.exit(main())
