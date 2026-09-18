#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_context

from hawk_derm.evaluation.metrics import multilabel_metrics, per_class_metrics
from hawk_derm.evaluation.predictions import arrays_from_prediction_frame
from hawk_derm.io import write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a raw case-level prediction file.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, default=None)
    args = parser.parse_args()
    config = load_context(args.config)
    frame = pd.read_csv(args.predictions)
    truth, probability = arrays_from_prediction_frame(frame, config["labels"])
    thresholds: np.ndarray | float = 0.5
    if args.thresholds:
        selected = pd.read_csv(args.thresholds).set_index("label")
        thresholds = selected.loc[config["labels"], "threshold"].to_numpy(dtype=float)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "overall_metrics.csv", pd.DataFrame([multilabel_metrics(truth, probability, thresholds)]))
    write_csv(args.output_dir / "per_class_metrics.csv", per_class_metrics(truth, probability, config["labels"], thresholds))
    positive = truth.sum(axis=1) > 0
    if positive.any():
        write_csv(
            args.output_dir / "target_positive_only_overall_metrics.csv",
            pd.DataFrame([multilabel_metrics(truth[positive], probability[positive], thresholds)]),
        )
    write_json(
        args.output_dir / "evaluation_manifest.json",
        {"predictions": str(args.predictions), "thresholds": str(args.thresholds) if args.thresholds else 0.5, "complete": True},
    )


if __name__ == "__main__":
    main()
