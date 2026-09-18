#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_context

from hawk_derm.config import load_yaml, path_from
from hawk_derm.evaluation.metrics import per_class_metrics
from hawk_derm.evaluation.predictions import arrays_from_prediction_frame
from hawk_derm.evaluation.thresholds import select_thresholds, threshold_sweep
from hawk_derm.io import write_csv, write_json
from hawk_derm.statistics.inference import wilson_interval


def mean_seed_predictions(paths: list[Path], labels: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frames = [pd.read_csv(path).sort_values("case_id").reset_index(drop=True) for path in paths]
    if not frames:
        raise FileNotFoundError("No repeated-seed prediction files were found")
    case_ids = frames[0]["case_id"].astype(str)
    truth, _ = arrays_from_prediction_frame(frames[0], labels)
    probabilities = []
    for frame in frames:
        if not case_ids.equals(frame["case_id"].astype(str)):
            raise ValueError("Repeated-seed prediction files do not contain identical ordered case IDs")
        candidate_truth, candidate_probability = arrays_from_prediction_frame(frame, labels)
        if not np.array_equal(truth, candidate_truth):
            raise ValueError("Repeated-seed ground truths disagree")
        probabilities.append(candidate_probability)
    return frames[0][["case_id"]], truth, np.mean(probabilities, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select thresholds on validation and apply them unchanged to locked test predictions.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--rule", choices=["balance", "youden", "f1"], default=None)
    args = parser.parse_args()
    config = load_context(args.config)
    mil_config = load_yaml("configs/mil.yaml")
    root = path_from(config, "artifacts_dir") / "models" / "mil" / args.encoder
    validation_paths = sorted(root.glob("seed_*/validation_predictions.csv"))
    test_paths = sorted(root.glob("seed_*/test_predictions.csv"))
    _, validation_truth, validation_probability = mean_seed_predictions(validation_paths, config["labels"])
    _, test_truth, test_probability = mean_seed_predictions(test_paths, config["labels"])
    threshold_config = mil_config["thresholds"]
    grid = np.arange(
        float(threshold_config["grid_start"]),
        float(threshold_config["grid_stop"]) + float(threshold_config["grid_step"]) / 2,
        float(threshold_config["grid_step"]),
    )
    sweep = threshold_sweep(validation_truth, validation_probability, config["labels"], grid)
    selected = select_thresholds(sweep, args.rule or threshold_config["rule"])
    selected_thresholds = selected.set_index("label").loc[config["labels"], "threshold"].to_numpy()
    sweep["selected"] = False
    for label, threshold in zip(config["labels"], selected_thresholds, strict=True):
        sweep.loc[sweep["label"].eq(label) & np.isclose(sweep["threshold"], threshold), "selected"] = True
    test_metrics = per_class_metrics(test_truth, test_probability, config["labels"], selected_thresholds)
    default_metrics = per_class_metrics(test_truth, test_probability, config["labels"], 0.5)
    default_metrics = default_metrics[["label", "sensitivity", "specificity", "balanced_accuracy"]].rename(
        columns={
            "sensitivity": "default_sensitivity",
            "specificity": "default_specificity",
            "balanced_accuracy": "default_balanced_accuracy",
        }
    )
    test_metrics = test_metrics.merge(default_metrics, on="label", validate="one_to_one")
    intervals = []
    for row in test_metrics.itertuples(index=False):
        sensitivity = wilson_interval(int(row.tp), int(row.tp + row.fn))
        specificity = wilson_interval(int(row.tn), int(row.tn + row.fp))
        intervals.append(
            {
                "sensitivity_ci_lower": sensitivity[0],
                "sensitivity_ci_upper": sensitivity[1],
                "specificity_ci_lower": specificity[0],
                "specificity_ci_upper": specificity[1],
            }
        )
    test_metrics = pd.concat([test_metrics, pd.DataFrame(intervals)], axis=1)
    output = root / "thresholds"
    write_csv(output / "validation_threshold_sweep.csv", sweep)
    write_csv(output / "validation_selected_thresholds.csv", selected)
    write_csv(output / "test_operating_points.csv", test_metrics)
    write_csv(output / "test_default_operating_points.csv", per_class_metrics(test_truth, test_probability, config["labels"], 0.5))
    write_json(
        output / "threshold_manifest.json",
        {
            "encoder": args.encoder,
            "selection_split": "validation",
            "application_split": "test",
            "rule": args.rule or threshold_config["rule"],
            "validation_seed_files": [str(path) for path in validation_paths],
            "test_seed_files": [str(path) for path in test_paths],
            "complete": True,
        },
    )


if __name__ == "__main__":
    main()
