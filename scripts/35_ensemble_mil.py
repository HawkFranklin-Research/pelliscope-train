#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_context, mil_run_root, selected_mil_config_path

from hawk_derm.config import load_yaml, path_from
from hawk_derm.evaluation.calibration import apply_platt, fit_platt
from hawk_derm.evaluation.metrics import multilabel_metrics, per_class_metrics
from hawk_derm.evaluation.predictions import arrays_from_prediction_frame, prediction_frame, save_predictions
from hawk_derm.io import sha256_file, write_csv, write_json


def load_seed_predictions(paths: list[Path], labels: list[str]) -> tuple[pd.DataFrame, np.ndarray, list[np.ndarray]]:
    if not paths:
        raise FileNotFoundError("No repeated-seed prediction files were found")
    frames = [pd.read_csv(path).sort_values("case_id").reset_index(drop=True) for path in paths]
    reference_ids = frames[0]["case_id"].astype(str)
    reference_truth, _ = arrays_from_prediction_frame(frames[0], labels)
    probabilities: list[np.ndarray] = []
    for path, frame in zip(paths, frames, strict=True):
        truth, probability = arrays_from_prediction_frame(frame, labels)
        if not reference_ids.equals(frame["case_id"].astype(str)):
            raise ValueError(f"Seed predictions have different case IDs or ordering: {path}")
        if not np.array_equal(reference_truth, truth):
            raise ValueError(f"Seed predictions have different ground-truth labels: {path}")
        probabilities.append(probability)
    return frames[0], reference_truth, probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="Average fixed-split MIL probabilities across seeds.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--mil-config", default=None)
    args = parser.parse_args()

    config = load_context(args.config)
    root = mil_run_root(config, args.encoder, args.run_tag)
    output = root / "ensemble"
    output.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.mil_config) if args.mil_config else selected_mil_config_path(config, args.encoder, args.run_tag)
    if not config_path.is_file():
        raise FileNotFoundError(f"Selected MIL configuration is missing: {config_path}")

    calibration = load_yaml(config_path).get("calibration", {}) or {}
    calibration_method = str(calibration.get("method", "none"))
    if calibration_method not in {"none", "platt"}:
        raise ValueError(f"Unknown MIL calibration method: {calibration_method}")
    calibration_parameters = None

    overall_rows: list[dict[str, object]] = []
    per_class_rows: list[pd.DataFrame] = []
    source_files: dict[str, list[str]] = {}
    expected_seed_count = len(config[config["study"]["run_mode"]]["seeds"])
    for split in ("validation", "test"):
        paths = sorted(root.glob(f"seed_*/{split}_predictions.csv"))
        if config["study"]["run_mode"] == "full" and len(paths) != expected_seed_count:
            raise ValueError(f"Expected {expected_seed_count} {split} seed files, found {len(paths)}")
        reference, truth, seed_probabilities = load_seed_predictions(paths, config["labels"])
        ensemble_probability = np.mean(seed_probabilities, axis=0)
        if calibration_method == "platt":
            # Fitted once on the validation ensemble (the loop visits validation first), then
            # applied unchanged to test. Per-label AP and ROC-AUC are unaffected by construction.
            if split == "validation":
                calibration_parameters = fit_platt(
                    truth, ensemble_probability, config["labels"], l2=float(calibration.get("l2", 1e-3))
                )
                write_csv(output / "calibration_parameters.csv", calibration_parameters)
            uncalibrated = prediction_frame(
                reference["case_id"].astype(str),
                truth,
                ensemble_probability,
                config["labels"],
                split=split,
                model=f"{args.encoder}+mil:probability_ensemble_uncalibrated",
            )
            save_predictions(
                output / f"{split}_predictions_uncalibrated.csv",
                uncalibrated,
                config["labels"],
                {"aggregation": "mean_probability_across_seeds", "calibration": "none"},
            )
            ensemble_probability = apply_platt(ensemble_probability, calibration_parameters, config["labels"])
        ensemble_frame = prediction_frame(
            reference["case_id"].astype(str),
            truth,
            ensemble_probability,
            config["labels"],
            split=split,
            model=f"{args.encoder}+mil:probability_ensemble",
        )
        save_predictions(
            output / f"{split}_predictions.csv",
            ensemble_frame,
            config["labels"],
            {
                "aggregation": "mean_probability_across_seeds",
                "calibration": calibration_method,
                "seed_count": len(paths),
                "selected_mil_config_sha256": sha256_file(config_path),
            },
        )
        ensemble_metrics = multilabel_metrics(truth, ensemble_probability)
        per_seed_metrics = [multilabel_metrics(truth, probability) for probability in seed_probabilities]
        overall_rows.append({"split": split, "aggregation": "probability_ensemble", **ensemble_metrics})
        overall_rows.append(
            {
                "split": split,
                "aggregation": "mean_per_seed_metric",
                **{
                    key: float(np.nanmean([row[key] for row in per_seed_metrics]))
                    for key in ensemble_metrics
                },
            }
        )
        per_class = per_class_metrics(truth, ensemble_probability, config["labels"])
        per_class.insert(0, "aggregation", "probability_ensemble")
        per_class.insert(0, "split", split)
        per_class_rows.append(per_class)
        source_files[split] = [str(path) for path in paths]

    write_csv(output / "overall_metrics.csv", pd.DataFrame(overall_rows))
    write_csv(output / "per_class_metrics.csv", pd.concat(per_class_rows, ignore_index=True))
    write_json(
        output / "ensemble_manifest.json",
        {
            "encoder": args.encoder,
            "run_tag": args.run_tag,
            "run_mode": config["study"]["run_mode"],
            "aggregation": "mean_probability_across_seeds",
            "mean_per_seed_auc_is_not_ensemble_auc": True,
            "calibration": {
                "method": calibration_method,
                "fitted_on": "validation_ensemble" if calibration_method == "platt" else None,
                "parameters": str(output / "calibration_parameters.csv") if calibration_method == "platt" else None,
            },
            "selected_mil_config": str(config_path),
            "selected_mil_config_sha256": sha256_file(config_path),
            "case_manifest_sha256": sha256_file(path_from(config, "case_manifest")),
            "split_manifest_sha256": sha256_file(path_from(config, "split_manifest")),
            "source_files": source_files,
            "complete": True,
        },
    )


if __name__ == "__main__":
    main()
