#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import load_context

from hawk_derm.config import path_from
from hawk_derm.evaluation.metrics import multilabel_metrics, per_class_metrics
from hawk_derm.evaluation.predictions import arrays_from_prediction_frame
from hawk_derm.io import write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize archived closed-model case predictions into figure plot data.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--prediction", type=Path, action="append", required=True, help="Repeat for each normalized prediction CSV.")
    parser.add_argument("--model-name", action="append", required=True, help="Repeat in the same order as --prediction.")
    args = parser.parse_args()
    if len(args.prediction) != len(args.model_name):
        raise ValueError("Provide one --model-name for every --prediction")
    config = load_context(args.config)
    per_class_frames, overall_rows, source_rows = [], [], []
    for path, model_name in zip(args.prediction, args.model_name, strict=True):
        frame = pd.read_csv(path)
        truth, probability = arrays_from_prediction_frame(frame, config["labels"])
        per_class = per_class_metrics(truth, probability, config["labels"])[["label", "auc", "pr_auc"]]
        per_class.insert(0, "model", model_name)
        per_class_frames.append(per_class)
        overall_rows.append({"model": model_name, **multilabel_metrics(truth, probability)})
        source_rows.append({"model": model_name, "prediction_file": str(path), "case_count": len(frame)})
    plot_dir = path_from(config, "reports_dir") / "plot_data"
    write_csv(plot_dir / "closed_models_per_class.csv", pd.concat(per_class_frames, ignore_index=True))
    write_csv(plot_dir / "closed_models_overall.csv", pd.DataFrame(overall_rows))
    write_json(plot_dir / "closed_models_manifest.json", {"sources": source_rows, "complete": True})


if __name__ == "__main__":
    main()
