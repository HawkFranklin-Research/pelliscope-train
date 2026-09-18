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
        reports / "figures" / "figure3_mil_diagnostics.png",
        reports / "figures" / "figure4_threshold_curves.png",
        reports / "figures" / "figure6_operating_points.png",
        reports / "statistics" / "comparison_grid_manifest.json",
        reports / "evaluations" / "siglip2_so400m_final" / "target_positive_only_overall_metrics.csv",
        artifacts / "models" / "mil" / "siglip2_so400m" / "final_model" / "model.pt",
    ]
    registry = load_encoder_registry()
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
        raise RuntimeError(f"Release is incomplete; missing {len(missing)} required artifacts")


if __name__ == "__main__":
    main()
