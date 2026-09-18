#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import load_context

from hawk_derm.config import path_from
from hawk_derm.features.registry import load_encoder_registry
from hawk_derm.io import read_json, sha256_file, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify release completeness without recomputing experiments.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--encoders", default=None, help="Comma-separated encoder keys expected in this release.")
    parser.add_argument("--primary-encoder", default="siglip2_so400m")
    args = parser.parse_args()
    config = load_context(args.config)
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
