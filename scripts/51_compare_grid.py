#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import load_context, mil_run_root

from hawk_derm.config import path_from
from hawk_derm.evaluation.metrics import multilabel_metrics
from hawk_derm.evaluation.predictions import arrays_from_prediction_frame, prediction_frame, save_predictions
from hawk_derm.io import read_json, sha256_file, write_csv, write_json
from hawk_derm.runtime import resolve_cpu_workers
from hawk_derm.statistics.bootstrap import case_bootstrap
from hawk_derm.statistics.comparison import compare_prediction_files


def assert_identical_cases(paths: dict[str, Path], labels: list[str]) -> None:
    reference_name = next(iter(paths))
    reference = pd.read_csv(paths[reference_name]).sort_values("case_id").reset_index(drop=True)
    reference_ids = reference["case_id"].astype(str)
    reference_truth, _ = arrays_from_prediction_frame(reference, labels)
    for name, path in paths.items():
        frame = pd.read_csv(path).sort_values("case_id").reset_index(drop=True)
        truth, _ = arrays_from_prediction_frame(frame, labels)
        if not reference_ids.equals(frame["case_id"].astype(str)):
            raise ValueError(f"{name} does not contain the identical locked case IDs as {reference_name}")
        if not np.array_equal(reference_truth, truth):
            raise ValueError(f"{name} ground truth differs from {reference_name}")


def match_prediction_files(
    paths: dict[str, Path],
    reference_path: Path,
    labels: list[str],
    output: Path,
) -> tuple[dict[str, Path], list[dict[str, str]], int]:
    """Create comparison-only prediction files on one explicit reference cohort.

    Source predictions remain untouched. A model may contain additional cases, but it
    must contain every reference case exactly once and carry identical ground truth.
    """
    reference = pd.read_csv(reference_path).copy()
    if reference["case_id"].duplicated().any():
        raise ValueError(f"Comparison cohort contains duplicate case IDs: {reference_path}")
    reference["case_id"] = reference["case_id"].astype(str)
    reference = reference.sort_values("case_id").reset_index(drop=True)
    reference_ids = set(reference["case_id"])
    reference_truth, _ = arrays_from_prediction_frame(reference, labels)

    cohort = reference[["case_id"]].copy()
    cohort.insert(0, "comparison_row", np.arange(1, len(cohort) + 1))
    write_csv(output / "comparison_cohort.csv", cohort)

    matched: dict[str, Path] = {}
    excluded: list[dict[str, str]] = []
    paired_dir = output / "paired_predictions"
    for name, path in paths.items():
        frame = pd.read_csv(path).copy()
        if frame["case_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate case IDs: {path}")
        frame["case_id"] = frame["case_id"].astype(str)
        source_ids = set(frame["case_id"])
        missing = sorted(reference_ids - source_ids)
        if missing:
            raise ValueError(
                f"{name} is missing {len(missing)} required comparison cases; "
                f"first missing IDs: {missing[:5]}"
            )
        extras = sorted(source_ids - reference_ids)
        excluded.extend({"model": name, "case_id": case_id, "reason": "outside_common_750_cohort"} for case_id in extras)
        frame = frame[frame["case_id"].isin(reference_ids)].sort_values("case_id").reset_index(drop=True)
        truth, _ = arrays_from_prediction_frame(frame, labels)
        if not frame["case_id"].equals(reference["case_id"]) or not np.array_equal(truth, reference_truth):
            raise ValueError(f"{name} does not match the reference cohort ground truth")
        destination = paired_dir / f"{name}.csv"
        write_csv(destination, frame)
        write_json(
            destination.with_suffix(".metadata.json"),
            {
                "complete": True,
                "rows": len(frame),
                "source": str(path),
                "source_sha256": sha256_file(path),
                "comparison_cohort": str(output / "comparison_cohort.csv"),
                "excluded_case_count": len(extras),
                "excluded_case_ids": extras,
            },
        )
        matched[name] = destination
    write_csv(
        output / "excluded_cases.csv",
        pd.DataFrame(excluded, columns=["model", "case_id", "reason"]),
    )
    return matched, excluded, len(reference)


def probability_ensemble(paths: list[Path], labels: list[str], output: Path, model_name: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"No seed predictions found for {model_name}")
    frames = [pd.read_csv(path).sort_values("case_id").reset_index(drop=True) for path in paths]
    truth, _ = arrays_from_prediction_frame(frames[0], labels)
    case_ids = frames[0]["case_id"].astype(str)
    probabilities = []
    for path, frame in zip(paths, frames, strict=True):
        candidate_truth, candidate_probability = arrays_from_prediction_frame(frame, labels)
        if not case_ids.equals(frame["case_id"].astype(str)) or not np.array_equal(truth, candidate_truth):
            raise ValueError(f"Unpaired seed predictions: {path}")
        probabilities.append(candidate_probability)
    result = prediction_frame(
        case_ids,
        truth,
        np.mean(probabilities, axis=0),
        labels,
        split="test",
        model=model_name,
    )
    return save_predictions(
        output,
        result,
        labels,
        {"aggregation": "mean_probability_across_seeds", "seed_files": [str(path) for path in paths]},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the SigLIP2 MIL ensemble with matched MIL and RF models.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--encoder", default="siglip2_so400m")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--comparison-cohort-predictions",
        default=None,
        help="Prediction CSV defining the exact paired test cohort. Defaults to the archived 750-case Derm Foundation MIL ensemble.",
    )
    args = parser.parse_args()

    config = load_context(args.config)
    workers = resolve_cpu_workers(args.workers)
    artifacts = path_from(config, "artifacts_dir")
    tag = args.run_tag or "canonical"
    output = path_from(config, "reports_dir") / "reanalysis" / f"{args.encoder}_mil_{tag}" / "statistics"
    output.mkdir(parents=True, exist_ok=True)

    siglip_root = mil_run_root(config, args.encoder, args.run_tag)
    siglip_mil = siglip_root / "ensemble" / "test_predictions.csv"
    derm_root = artifacts / "models" / "mil" / "derm_foundation"
    derm_ensemble = path_from(config, "reports_dir") / "statistics" / "derm_foundation_mil_test_ensemble_predictions.csv"
    if not derm_ensemble.is_file():
        derm_ensemble = derm_root / "ensemble" / "test_predictions.csv"
        if not derm_ensemble.is_file():
            derm_ensemble = probability_ensemble(
                sorted(derm_root.glob("seed_*/test_predictions.csv")),
                config["labels"],
                output / "derm_foundation_mil_test_ensemble_predictions.csv",
                "derm_foundation+mil:probability_ensemble",
            )

    models = {
        "siglip2_mil": siglip_mil,
        "siglip2_random_forest": artifacts
        / "models"
        / "classical"
        / "siglip2_so400m"
        / "random_forest"
        / "test_case_predictions.csv",
        "derm_foundation_mil": derm_ensemble,
        "derm_foundation_random_forest": artifacts
        / "models"
        / "classical"
        / "derm_foundation"
        / "random_forest"
        / "test_case_predictions.csv",
    }
    missing = {name: str(path) for name, path in models.items() if not path.is_file()}
    if missing:
        raise FileNotFoundError(f"Required paired comparison predictions are missing: {missing}")
    reference_path = (
        Path(args.comparison_cohort_predictions)
        if args.comparison_cohort_predictions
        else path_from(config, "reports_dir") / "statistics" / "derm_foundation_mil_test_ensemble_predictions.csv"
    )
    if not reference_path.is_file():
        raise FileNotFoundError(f"Comparison-cohort prediction file is missing: {reference_path}")
    matched_models, excluded, comparison_case_count = match_prediction_files(
        models,
        reference_path,
        config["labels"],
        output,
    )
    assert_identical_cases(matched_models, config["labels"])
    matched_metric_rows = []
    for name, path in matched_models.items():
        truth, probability = arrays_from_prediction_frame(pd.read_csv(path), config["labels"])
        matched_metric_rows.append(
            {
                "model": name,
                "cases": len(truth),
                **multilabel_metrics(truth, probability),
            }
        )
    write_csv(output / "matched_model_metrics.csv", pd.DataFrame(matched_metric_rows))

    mode = config["study"]["run_mode"]
    replicates = args.replicates or int(config[mode]["bootstrap_replicates"])
    comparisons = []
    primary_name = "siglip2_mil"
    primary_path = matched_models[primary_name]
    primary_frame = pd.read_csv(primary_path)
    primary_truth, primary_probability = arrays_from_prediction_frame(primary_frame, config["labels"])
    confidence_rows = []
    for offset, (metric, averaging) in enumerate(
        (("roc_auc", "macro"), ("roc_auc_micro", "micro"), ("pr_auc", "macro"), ("pr_auc_micro", "micro"))
    ):
        result = case_bootstrap(
            primary_truth,
            primary_probability,
            metric=metric,
            replicates=replicates,
            seed=args.seed + 100 + offset,
            workers=workers,
        )
        confidence_rows.append(
            {
                "model": primary_name,
                "metric": "roc_auc" if metric.startswith("roc_auc") else "pr_auc",
                "averaging": averaging,
                "estimate": result.estimate,
                "ci_95_lower": result.lower,
                "ci_95_upper": result.upper,
                "cases": len(primary_truth),
                "bootstrap_replicates": replicates,
            }
        )
    write_csv(output / "siglip2_mil_auc_confidence_intervals.csv", pd.DataFrame(confidence_rows))
    comparison_pairs = [
        ((primary_name, primary_path), (name, path))
        for name, path in matched_models.items()
        if name != primary_name
    ]
    for index, ((first_name, first_path), (second_name, second_path)) in enumerate(comparison_pairs):
        name = f"{first_name}_vs_{second_name}"
        summary_path = output / name / "comparison_summary.json"
        if summary_path.is_file() and int(read_json(summary_path).get("replicates", 0)) == replicates:
            summary = read_json(summary_path)
        else:
            summary = compare_prediction_files(
                first_path,
                second_path,
                config["labels"],
                output / name,
                replicates=replicates,
                seed=args.seed + index,
                first_name=first_name,
                second_name=second_name,
                workers=workers,
            )
        comparisons.append({"name": name, **summary})
    write_json(
        output / "comparison_grid_manifest.json",
        {
            "source_models": {name: str(path) for name, path in models.items()},
            "paired_models": {name: str(path) for name, path in matched_models.items()},
            "comparison_cohort_source": str(reference_path),
            "comparison_cohort_source_sha256": sha256_file(reference_path),
            "comparison_case_count": comparison_case_count,
            "excluded_rows": excluded,
            "case_membership": "explicit_common_750_test_cohort",
            "comparison_scope": "siglip2_mil_against_each_prespecified_comparator",
            "comparisons": comparisons,
            "replicates": replicates,
            "complete": True,
        },
    )


if __name__ == "__main__":
    main()
