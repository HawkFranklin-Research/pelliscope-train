#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import load_context, mil_run_root

from hawk_derm.config import path_from
from hawk_derm.features.registry import load_encoder_registry
from hawk_derm.data.splits import assert_existing_full_split
from hawk_derm.io import read_json, sha256_file, write_csv, write_json


def verify_tagged_grid(config: dict, encoders: list[str], run_tag: str) -> None:
    split_path = path_from(config, "split_manifest")
    splits = pd.read_csv(split_path)
    split_hash = sha256_file(split_path)
    expected = {name: set(splits.loc[splits["split"].eq(name), "case_id"].astype(str)) for name in ("train", "validation", "test")}
    artifacts = path_from(config, "artifacts_dir")
    seeds = [int(seed) for seed in config[config["study"]["run_mode"]]["seeds"]]
    failures = []
    for encoder in encoders:
        root = mil_run_root(config, encoder, run_tag)
        selected = root / "tuning" / "best_mil_config.yaml"
        feature_bank = artifacts / "features" / encoder / "feature_bank.npz"
        required = [feature_bank, selected, root / "cross_validation" / "cross_validation_metrics.csv", root / "repeated_metrics_long.csv", root / "ensemble" / "test_predictions.csv", root / "thresholds" / "validation_selected_thresholds.csv"]
        for path in required:
            if not path.is_file() or path.stat().st_size == 0:
                failures.append(f"missing {path}")
        selected_hash = sha256_file(selected) if selected.is_file() else None
        feature_hash = sha256_file(feature_bank) if feature_bank.is_file() else None
        for seed in seeds:
            seed_root = root / f"seed_{seed}"
            summary = seed_root / "summary.json"
            if not summary.is_file():
                failures.append(f"missing {summary}")
            else:
                provenance = read_json(summary).get("provenance", {})
                if provenance.get("split_manifest_sha256") != split_hash:
                    failures.append(f"stale split in {summary}")
                if selected_hash and provenance.get("selected_mil_config_sha256") != selected_hash:
                    failures.append(f"stale selected configuration in {summary}")
                if feature_hash and provenance.get("feature_bank_sha256") != feature_hash:
                    failures.append(f"stale feature bank in {summary}")
            for split in expected:
                path = seed_root / f"{split}_predictions.csv"
                if not path.is_file():
                    failures.append(f"missing {path}")
                    continue
                frame = pd.read_csv(path, usecols=["case_id"])
                if len(frame) != len(expected[split]) or set(frame["case_id"].astype(str)) != expected[split]:
                    failures.append(f"wrong {split} membership in {path}")
        for split in ("validation", "test"):
            path = root / "ensemble" / f"{split}_predictions.csv"
            if not path.is_file():
                failures.append(f"missing {path}")
                continue
            frame = pd.read_csv(path, usecols=["case_id"])
            if len(frame) != len(expected[split]) or set(frame["case_id"].astype(str)) != expected[split]:
                failures.append(f"wrong {split} membership in {path}")
        for classifier in ("gradient_boosting", "knn", "logistic", "random_forest", "svm_linear"):
            classical_root = artifacts / "models" / "classical" / encoder / classifier
            path = classical_root / "test_case_predictions.csv"
            if not path.is_file():
                failures.append(f"missing {path}")
                continue
            frame = pd.read_csv(path, usecols=["case_id"])
            if len(frame) != len(expected["test"]) or set(frame["case_id"].astype(str)) != expected["test"]:
                failures.append(f"wrong locked test membership in {path}")
            run_manifest = classical_root / "run_manifest.json"
            if not run_manifest.is_file():
                failures.append(f"missing {run_manifest}")
            else:
                inputs = read_json(run_manifest).get("inputs", [])
                if not any(item.get("sha256") == split_hash and Path(item.get("path", "")).name == split_path.name for item in inputs):
                    failures.append(f"classical model was not fitted on the locked split: {run_manifest}")
    if failures:
        raise RuntimeError("Tagged model grid is incomplete: " + "; ".join(failures[:20]))


def verify_tagged_mil_release(config: dict, encoder: str, run_tag: str) -> None:
    artifacts = path_from(config, "artifacts_dir")
    reports = path_from(config, "reports_dir")
    root = mil_run_root(config, encoder, run_tag)
    case_path = path_from(config, "case_manifest")
    split_path = path_from(config, "split_manifest")
    cases = pd.read_csv(case_path)
    splits = pd.read_csv(split_path)
    expected_case_count = int(config["study"]["canonical_case_count"])
    expected_split_counts = splits.groupby("split")["case_id"].nunique().to_dict()
    expected_seeds = [int(seed) for seed in config[config["study"]["run_mode"]]["seeds"]]
    selected_config = root / "tuning" / "best_mil_config.yaml"
    feature_bank = artifacts / "features" / encoder / "feature_bank.npz"
    statistics_root = reports / "reanalysis" / f"{encoder}_mil_{run_tag}" / "statistics"
    canonical_test_predictions = statistics_root / "paired_predictions" / f"{encoder.removesuffix('_so400m')}_mil.csv"
    expected_test_case_count = int(config["study"]["canonical_test_case_count"])
    required = [
        feature_bank,
        selected_config,
        root / "tuning" / "selection_manifest.json",
        root / "cross_validation" / "cross_validation_metrics.csv",
        root / "repeated_metrics_long.csv",
        root / "repeated_metrics_summary.csv",
        root / "ensemble" / "validation_predictions.csv",
        root / "ensemble" / "test_predictions.csv",
        root / "ensemble" / "overall_metrics.csv",
        root / "ensemble" / "per_class_metrics.csv",
        root / "ensemble" / "ensemble_manifest.json",
        root / "thresholds" / "validation_selected_thresholds.csv",
        root / "thresholds" / "test_operating_points.csv",
        root / "thresholds" / "threshold_manifest.json",
        root / "final_model" / "model.pt",
        root / "final_model" / "train_predictions.csv",
        root / "final_model" / "validation_predictions.csv",
        root / "final_model" / "test_predictions.csv",
        root / "final_model" / "final_model_manifest.json",
        statistics_root / "comparison_grid_manifest.json",
        canonical_test_predictions,
    ]
    failures: list[str] = []
    if len(cases) != expected_case_count or cases["case_id"].nunique() != expected_case_count:
        failures.append(f"case manifest must contain {expected_case_count} unique rows; found {len(cases)}")
    if splits["case_id"].nunique() != expected_case_count or len(splits) != expected_case_count:
        failures.append("split manifest does not contain every canonical case exactly once")
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        failures.append(f"missing or empty required artifacts: {missing}")

    if canonical_test_predictions.is_file():
        canonical_test = pd.read_csv(canonical_test_predictions)
        canonical_ids = canonical_test["case_id"].astype(str)
        if len(canonical_test) != expected_test_case_count or canonical_ids.nunique() != expected_test_case_count:
            failures.append(
                f"canonical test predictions must contain {expected_test_case_count} unique cases: "
                f"{canonical_test_predictions}"
            )
        for paired_path in sorted((statistics_root / "paired_predictions").glob("*.csv")):
            paired = pd.read_csv(paired_path)
            paired_ids = paired["case_id"].astype(str)
            if len(paired) != expected_test_case_count or set(paired_ids) != set(canonical_ids):
                failures.append(f"paired prediction membership differs from the canonical test cohort: {paired_path}")

    manifests = (
        list(root.rglob("*manifest.json"))
        + list(root.rglob("run_manifest.json"))
        + [case_path.with_suffix(".metadata.json"), split_path.with_suffix(".metadata.json")]
    )
    for manifest_path in sorted(set(manifests)):
        if not manifest_path.is_file():
            continue
        payload = read_json(manifest_path)
        if payload.get("run_mode") == "smoke":
            failures.append(f"production artifact is marked run_mode=smoke: {manifest_path}")

    seed_dirs = [root / f"seed_{seed}" for seed in expected_seeds]
    if sum(path.is_dir() for path in seed_dirs) != len(expected_seeds):
        failures.append(f"expected {len(expected_seeds)} seed directories")
    provenance_values: list[tuple[str, str, str]] = []
    expected_ids = {
        split: set(splits.loc[splits["split"].eq(split), "case_id"].astype(str))
        for split in ("train", "validation", "test")
    }
    for seed_dir in seed_dirs:
        summary_path = seed_dir / "summary.json"
        if not summary_path.is_file():
            failures.append(f"missing seed summary: {summary_path}")
            continue
        summary = read_json(summary_path)
        provenance = summary.get("provenance", {})
        provenance_values.append(
            (
                str(provenance.get("feature_bank_sha256", "")),
                str(provenance.get("selected_mil_config_sha256", "")),
                str(provenance.get("split_manifest_sha256", "")),
            )
        )
        for split, expected in expected_ids.items():
            path = seed_dir / f"{split}_predictions.csv"
            if not path.is_file():
                failures.append(f"missing raw predictions: {path}")
                continue
            frame = pd.read_csv(path)
            actual = set(frame["case_id"].astype(str))
            if len(frame) != expected_split_counts.get(split, 0) or actual != expected:
                failures.append(f"unexpected {split} membership or row count: {path}")
    if provenance_values and len(set(provenance_values)) != 1:
        failures.append("feature/config/split hashes differ across repeated seeds")
    if provenance_values and any(not value for value in provenance_values[0]):
        failures.append("one or more repeated-seed provenance hashes are missing")

    for path, split in (
        (root / "ensemble" / "validation_predictions.csv", "validation"),
        (root / "ensemble" / "test_predictions.csv", "test"),
        (root / "final_model" / "train_predictions.csv", "train"),
        (root / "final_model" / "validation_predictions.csv", "validation"),
        (root / "final_model" / "test_predictions.csv", "test"),
    ):
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if len(frame) != expected_split_counts.get(split, 0):
            failures.append(f"unexpected row count in {path}")
        if set(frame["case_id"].astype(str)) != expected_ids[split]:
            failures.append(f"unexpected locked membership in {path}")

    if (root / "ensemble" / "ensemble_manifest.json").is_file():
        ensemble_manifest = read_json(root / "ensemble" / "ensemble_manifest.json")
        if ensemble_manifest.get("aggregation") != "mean_probability_across_seeds":
            failures.append("ensemble manifest does not identify probability averaging")
        if ensemble_manifest.get("mean_per_seed_auc_is_not_ensemble_auc") is not True:
            failures.append("mean per-seed AUC could be mislabeled as ensemble AUC")
    if feature_bank.is_file() and selected_config.is_file() and provenance_values:
        if provenance_values[0][0] != sha256_file(feature_bank):
            failures.append("seed feature-bank hash differs from the current feature bank")
        if provenance_values[0][1] != sha256_file(selected_config):
            failures.append("seed selected-config hash differs from best_mil_config.yaml")
        if provenance_values[0][2] != sha256_file(split_path):
            failures.append("seed split hash differs from the locked split manifest")
    final_manifest = root / "final_model" / "final_model_manifest.json"
    if final_manifest.is_file() and read_json(final_manifest).get("split_manifest_sha256") != sha256_file(split_path):
        failures.append("final model was fitted on a different split")

    records = pd.DataFrame(
        [
            {
                "path": str(path),
                "exists": path.is_file(),
                "size": path.stat().st_size if path.is_file() else 0,
                "sha256": sha256_file(path) if path.is_file() else "",
            }
            for path in required
        ]
    )
    output = reports / "reanalysis" / f"{encoder}_mil_{run_tag}" / "verification"
    write_csv(output / "release_verification.csv", records)
    payload = {
        "complete": not failures,
        "encoder": encoder,
        "run_tag": run_tag,
        "case_count": len(cases),
        "split_counts": expected_split_counts,
        "seed_count": len(seed_dirs),
        "checked": len(required),
        "failures": failures,
    }
    write_json(output / "release_verification.json", payload)
    if failures:
        raise RuntimeError("Tagged MIL release is incomplete: " + "; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify release completeness without recomputing experiments.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--encoders", default=None, help="Comma-separated encoder keys expected in this release.")
    parser.add_argument("--primary-encoder", default="siglip2_so400m")
    parser.add_argument("--mil-run-tag", default=None, help="Verify an isolated MIL run instead of the legacy release.")
    args = parser.parse_args()
    config = load_context(args.config)
    if args.mil_run_tag:
        if config["study"]["run_mode"] == "full":
            assert_existing_full_split(config)
        if args.encoders:
            verify_tagged_grid(config, [item.strip() for item in args.encoders.split(",") if item.strip()], args.mil_run_tag)
        verify_tagged_mil_release(config, args.primary_encoder, args.mil_run_tag)
        return
    root = Path(config["repository_root"])
    artifacts = path_from(config, "artifacts_dir")
    reports = path_from(config, "reports_dir")
    required = [
        path_from(config, "case_manifest"),
        path_from(config, "image_manifest"),
        path_from(config, "split_manifest"),
        reports / "tables" / "headline_study_numbers.csv",
        reports / "tables" / "cohort_sizes_by_split.csv",
        reports / "figures" / "figure1_cohort_preprocessing.png",
        reports / "figures" / "graphical_abstract.png",
        reports / "figures" / "figure2_encoder_classifier_heatmap.png",
        # Figure families are emitted as independent panels/pages for the
        # 25-class study. The old 10-class composite filenames are no longer
        # release artifacts; final manuscript assembly remains separate.
        reports / "figures" / "figure3a_learning_curves.png",
        reports / "figures" / "figure3c_roc_pr_curves.png",
        reports / "figures" / "figure4_threshold_curves_page_01.png",
        reports / "figures" / "figure6a_operating_points_page_01.png",
        reports / "figures" / "figure6b_split_metrics_page_01.png",
        reports / "figures" / "figure5b_accuracy_comparison.png",
        reports / "figures" / "figure_generation_manifest.json",
        reports / "statistics" / "comparison_grid_manifest.json",
        reports / "evaluations" / f"{args.primary_encoder}_final" / "target_positive_only_overall_metrics.csv",
        artifacts / "models" / "mil" / args.primary_encoder / "final_model" / "model.pt",
    ]
    full_registry = load_encoder_registry()
    selected = [item.strip() for item in args.encoders.split(",") if item.strip()] if args.encoders else list(full_registry)
    unknown = set(selected).difference(full_registry)
    if unknown:
        raise ValueError(f"Unknown encoder keys in --encoders: {sorted(unknown)}")
    if args.primary_encoder not in selected:
        raise ValueError("--primary-encoder must be included in --encoders")
    registry = {encoder: full_registry[encoder] for encoder in selected}
    for encoder in registry:
        required.extend(
            [
                artifacts / "features" / encoder / "feature_bank.metadata.json",
                artifacts / "models" / "mil" / encoder / "repeated_metrics_long.csv",
            ]
        )
    records = []
    for path in required:
        exists = path.is_file()
        metadata_complete = None
        if exists and path.suffix == ".json":
            metadata_complete = bool(read_json(path).get("complete", False))
        records.append(
            {
                "path": str(path.relative_to(root) if root in path.parents else path),
                "exists": exists,
                "size": path.stat().st_size if exists else 0,
                "sha256": sha256_file(path) if exists else "",
                "metadata_complete": metadata_complete,
            }
        )
    frame = pd.DataFrame(records)
    output = reports / "run_manifests" / "release_verification.csv"
    write_csv(output, frame)
    feature_gate = []
    for encoder, spec in registry.items():
        metadata_path = artifacts / "features" / encoder / "feature_bank.metadata.json"
        if metadata_path.is_file():
            metadata = read_json(metadata_path)
            feature_gate.append(
                {
                    "encoder": encoder,
                    "dimension": metadata.get("dimension"),
                    "expected_dimension": spec.dimension,
                    "row_count": metadata.get("row_count"),
                    "dimension_matches": metadata.get("dimension") == spec.dimension,
                }
            )
    feature_gate_frame = pd.DataFrame(feature_gate)
    write_csv(reports / "run_manifests" / "feature_gate.csv", feature_gate_frame)
    feature_dimensions_match = bool(len(feature_gate_frame) == len(registry) and feature_gate_frame["dimension_matches"].all())
    feature_rows_match = bool(
        len(feature_gate_frame) == len(registry) and feature_gate_frame["row_count"].nunique() == 1
    )
    floating_revisions = {
        encoder: spec.revision
        for encoder, spec in registry.items()
        if str(spec.revision).lower() in {"main", "master", "latest", "release-provided"}
    }
    revisions_are_immutable = not floating_revisions
    complete = bool(
        frame["exists"].all()
        and not frame.loc[frame["metadata_complete"].notna(), "metadata_complete"].eq(False).any()
        and feature_dimensions_match
        and feature_rows_match
        and revisions_are_immutable
    )
    write_json(
        reports / "run_manifests" / "release_verification.json",
        {
            "complete": complete,
            "checked": len(frame),
            "feature_dimensions_match": feature_dimensions_match,
            "feature_row_counts_match": feature_rows_match,
            "revisions_are_immutable": revisions_are_immutable,
            "floating_revisions": floating_revisions,
        },
    )
    if not complete:
        missing = frame.loc[~frame["exists"], "path"].tolist()
        failures = []
        if missing:
            failures.append(f"missing {len(missing)} required artifacts")
        if not feature_dimensions_match:
            failures.append("feature dimensions do not match the encoder registry")
        if not feature_rows_match:
            failures.append("feature banks do not contain identical row counts")
        if not revisions_are_immutable:
            failures.append(f"floating model revisions remain: {floating_revisions}")
        raise RuntimeError("Release is incomplete: " + "; ".join(failures))


if __name__ == "__main__":
    main()
