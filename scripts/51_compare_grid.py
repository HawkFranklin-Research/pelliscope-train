#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_context

from hawk_derm.config import path_from
from hawk_derm.evaluation.predictions import arrays_from_prediction_frame, prediction_frame, save_predictions
from hawk_derm.io import write_csv, write_json
from hawk_derm.statistics.comparison import compare_prediction_files


def ensemble_predictions(paths: list[Path], labels: list[str], model_name: str, output: Path) -> Path:
    frames = [pd.read_csv(path).sort_values("case_id").reset_index(drop=True) for path in paths]
    if not frames:
        raise FileNotFoundError(f"No prediction files supplied for {model_name}")
    truth, _ = arrays_from_prediction_frame(frames[0], labels)
    probabilities = []
    for frame in frames:
        candidate_truth, candidate_probability = arrays_from_prediction_frame(frame, labels)
        if not frames[0]["case_id"].astype(str).equals(frame["case_id"].astype(str)) or not np.array_equal(truth, candidate_truth):
            raise ValueError(f"Repeated predictions for {model_name} are not paired")
        probabilities.append(candidate_probability)
    result = prediction_frame(
        frames[0]["case_id"].astype(str),
        truth,
        np.mean(probabilities, axis=0),
        labels,
        split="test",
        model=model_name,
    )
    return save_predictions(output, result, labels, {"aggregation": "mean_probability_across_seeds", "seed_files": [str(path) for path in paths]})


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the strongest baseline on validation and run the prespecified paired test grid.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    config = load_context(args.config)
    artifacts = path_from(config, "artifacts_dir")
    output = path_from(config, "reports_dir") / "statistics"
    output.mkdir(parents=True, exist_ok=True)
    validation_rows = []
    for path in sorted(artifacts.glob("models/classical/*/*/overall_metrics.csv")):
        frame = pd.read_csv(path)
        validation = frame[frame["split"].eq("validation")]
        if len(validation):
            validation_rows.append(
                {
                    "encoder": path.parents[1].name,
                    "classifier": path.parent.name,
                    "validation_auc_macro": float(validation.iloc[0]["auc_macro"]),
                    "test_predictions": str(path.parent / "test_case_predictions.csv"),
                }
            )
    candidates = pd.DataFrame(validation_rows).sort_values("validation_auc_macro", ascending=False)
    if candidates.empty:
        raise FileNotFoundError("No classical validation metrics were found")
    write_csv(output / "baseline_selection_by_validation.csv", candidates)
    strongest = candidates.iloc[0]
    baseline_path = Path(strongest["test_predictions"])
    baseline_name = f"{strongest['encoder']}+{strongest['classifier']}:case_mean"
    ensemble_paths: dict[str, Path] = {}
    for encoder in ("siglip2_so400m", "derm_foundation"):
        seed_paths = sorted((artifacts / "models" / "mil" / encoder).glob("seed_*/test_predictions.csv"))
        if seed_paths:
            ensemble_paths[encoder] = ensemble_predictions(
                seed_paths,
                config["labels"],
                f"{encoder}+mil:seed_mean",
                output / f"{encoder}_mil_test_ensemble_predictions.csv",
            )
    mode = config["study"]["run_mode"]
    replicates = args.replicates or int(config[mode]["bootstrap_replicates"])
    comparisons = []
    for encoder, mil_path in ensemble_paths.items():
        name = f"{encoder}_mil_vs_strongest_baseline"
        compare_prediction_files(
            mil_path,
            baseline_path,
            config["labels"],
            output / name,
            replicates=replicates,
            seed=args.seed,
            first_name=f"{encoder}+MIL",
            second_name=baseline_name,
        )
        comparisons.append(name)
    if {"siglip2_so400m", "derm_foundation"}.issubset(ensemble_paths):
        name = "siglip2_mil_vs_derm_foundation_mil"
        compare_prediction_files(
            ensemble_paths["siglip2_so400m"],
            ensemble_paths["derm_foundation"],
            config["labels"],
            output / name,
            replicates=replicates,
            seed=args.seed + 1,
            first_name="SigLIP2+MIL",
            second_name="Derm Foundation+MIL",
        )
        comparisons.append(name)
    write_json(
        output / "comparison_grid_manifest.json",
        {
            "baseline_selected_on": "validation_auc_macro",
            "selected_baseline": baseline_name,
            "comparisons": comparisons,
            "replicates": replicates,
            "complete": True,
        },
    )


if __name__ == "__main__":
    main()
