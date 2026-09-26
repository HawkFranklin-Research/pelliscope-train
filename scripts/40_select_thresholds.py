#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from _common import load_context, load_selected_mil_config, mil_run_root

from hawk_derm.config import path_from
from hawk_derm.evaluation.metrics import per_class_metrics
from hawk_derm.evaluation.predictions import arrays_from_prediction_frame
from hawk_derm.evaluation.thresholds import select_thresholds, threshold_sweep
from hawk_derm.io import sha256_file, write_csv, write_json
from hawk_derm.statistics.inference import wilson_interval


def main() -> None:
    parser = argparse.ArgumentParser(description="Select thresholds on validation and apply them unchanged to locked test predictions.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--rule", choices=["balance", "youden", "f1"], default=None)
    parser.add_argument("--mil-config", default=None)
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()
    config = load_context(args.config)
    mil_config, mil_config_path = load_selected_mil_config(config, args.encoder, args.run_tag, args.mil_config)
    root = mil_run_root(config, args.encoder, args.run_tag)
    validation_path = root / "ensemble" / "validation_predictions.csv"
    test_path = root / "ensemble" / "test_predictions.csv"
    validation_truth, validation_probability = arrays_from_prediction_frame(pd.read_csv(validation_path), config["labels"])
    test_truth, test_probability = arrays_from_prediction_frame(pd.read_csv(test_path), config["labels"])
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
    # External cohorts: validation-selected thresholds applied unchanged, never re-selected.
    external_outputs = {}
    for external_path in sorted((root / "ensemble").glob("external_*_predictions.csv")):
        name = external_path.name.removesuffix("_predictions.csv")
        truth, probability = arrays_from_prediction_frame(pd.read_csv(external_path), config["labels"])
        points = per_class_metrics(truth, probability, config["labels"], selected_thresholds)
        bounds = []
        for row in points.itertuples(index=False):
            sensitivity = wilson_interval(int(row.tp), int(row.tp + row.fn)) if row.tp + row.fn else (float("nan"), float("nan"))
            specificity = wilson_interval(int(row.tn), int(row.tn + row.fp)) if row.tn + row.fp else (float("nan"), float("nan"))
            bounds.append(
                {
                    "sensitivity_ci_lower": sensitivity[0],
                    "sensitivity_ci_upper": sensitivity[1],
                    "specificity_ci_lower": specificity[0],
                    "specificity_ci_upper": specificity[1],
                }
            )
        points = pd.concat([points, pd.DataFrame(bounds)], axis=1)
        external_outputs[name] = str(write_csv(output / f"{name}_operating_points.csv", points))
    write_json(
        output / "threshold_manifest.json",
        {
            "encoder": args.encoder,
            "run_tag": args.run_tag,
            "run_mode": config["study"]["run_mode"],
            "selection_split": "validation",
            "application_split": "test",
            "rule": args.rule or threshold_config["rule"],
            "validation_ensemble_predictions": str(validation_path),
            "test_ensemble_predictions": str(test_path),
            "external_operating_points": external_outputs,
            "selected_mil_config": str(mil_config_path),
            "selected_mil_config_sha256": sha256_file(mil_config_path),
            "case_manifest_sha256": sha256_file(path_from(config, "case_manifest")),
            "split_manifest_sha256": sha256_file(path_from(config, "split_manifest")),
            "complete": True,
        },
    )


if __name__ == "__main__":
    main()
