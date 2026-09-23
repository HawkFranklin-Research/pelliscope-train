#!/usr/bin/env python3
"""Audit, normalize, and preserve the raw predictions used in this analysis."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_lib import (
    CLASSIFIERS,
    CONFIG,
    ENCODERS,
    HERE,
    PROB_COLUMNS,
    REPO_ROOT,
    TRUE_COLUMNS,
    canonical_case_ids,
    sha256,
    validate_prediction_frame,
    write_json,
)


INPUT = HERE / "inputs" / "hf_production"
OUTPUT = HERE / "outputs"
RAW = OUTPUT / "raw_predictions"
MANIFESTS = OUTPUT / "manifests"


def inventory_row(path: Path, role: str, source: str, expected_ids: list[str]) -> dict:
    frame = pd.read_csv(path)
    ids = frame["case_id"].astype(str) if "case_id" in frame else pd.Series(dtype=str)
    return {
        "role": role,
        "source": source,
        "path": str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path),
        "rows": len(frame),
        "unique_case_ids": ids.nunique(),
        "canonical_case_overlap": len(set(ids) & set(expected_ids)),
        "extra_case_ids": len(set(ids) - set(expected_ids)),
        "sha256": sha256(path),
    }


def validate_image_predictions(path: Path, expected_ids: list[str]) -> None:
    frame = pd.read_csv(path)
    required = {"case_id", *TRUE_COLUMNS, *PROB_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing image-level columns {sorted(missing)}")
    ids = set(frame["case_id"].astype(str))
    if set(expected_ids) - ids:
        raise ValueError(f"{path}: image-level predictions do not cover all canonical cases")
    probability = frame[PROB_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError(f"{path}: invalid image-level probabilities")


def write_normalized(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def same_truth(reference: pd.DataFrame, candidate: pd.DataFrame, source: Path) -> None:
    if not np.array_equal(reference[TRUE_COLUMNS].to_numpy(), candidate[TRUE_COLUMNS].to_numpy()):
        raise ValueError(f"Ground-truth labels disagree in {source}")


def mean_seed_predictions(seed_frames: list[pd.DataFrame]) -> pd.DataFrame:
    ensemble = seed_frames[0][["case_id", *TRUE_COLUMNS]].copy()
    probability = np.mean([frame[PROB_COLUMNS].to_numpy(dtype=float) for frame in seed_frames], axis=0)
    ensemble[PROB_COLUMNS] = probability
    return ensemble[["case_id", *TRUE_COLUMNS, *PROB_COLUMNS]]


def main() -> None:
    expected_ids = canonical_case_ids()
    expected_count = int(CONFIG["canonical"]["expected_cases"])
    if not INPUT.exists():
        raise FileNotFoundError("Run 00_fetch_prediction_sources.py before the audit")

    if RAW.exists():
        shutil.rmtree(RAW)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    inventory: list[dict] = []
    registry: list[dict] = []
    truth_reference: pd.DataFrame | None = None

    # Preserve and validate both image-level and case-aggregated classical outputs.
    for encoder in ENCODERS:
        for classifier in [name for name in CLASSIFIERS if name != "mil"]:
            model_id = f"{encoder}__{classifier}"
            model_dir = INPUT / "classical" / encoder / classifier
            case_path = model_dir / "test_case_predictions.csv"
            image_path = model_dir / "test_image_predictions.csv"
            if not case_path.exists() or not image_path.exists():
                raise FileNotFoundError(f"Missing production predictions for {model_id}")
            case_frame = validate_prediction_frame(pd.read_csv(case_path), case_path, expected_ids)
            validate_image_predictions(image_path, expected_ids)
            if truth_reference is None:
                truth_reference = case_frame[["case_id", *TRUE_COLUMNS]].copy()
            else:
                same_truth(truth_reference, case_frame, case_path)
            normalized = RAW / "classical_case_aggregated" / encoder / f"{classifier}.csv"
            write_normalized(case_frame, normalized)
            inventory.append(inventory_row(case_path, "classical_case_aggregated", "huggingface_pinned", expected_ids))
            inventory.append(inventory_row(image_path, "classical_image_level", "huggingface_pinned", expected_ids))
            registry.append(
                {
                    "model_id": model_id,
                    "encoder": encoder,
                    "classifier": classifier,
                    "model_family": "classical_case_aggregated",
                    "prediction_path": str(normalized.relative_to(HERE)),
                    "cases": expected_count,
                    "source_revision": CONFIG["hub"]["revision"],
                }
            )

    # Average ten production MIL seeds for the six unchanged encoders.
    for encoder in [name for name in ENCODERS if name != "siglip2_so400m"]:
        seed_frames: list[pd.DataFrame] = []
        for seed in range(42, 52):
            path = INPUT / "mil" / encoder / f"seed_{seed}" / "test_predictions.csv"
            frame = validate_prediction_frame(pd.read_csv(path), path, expected_ids)
            same_truth(truth_reference, frame, path)  # type: ignore[arg-type]
            normalized_seed = RAW / "mil_individual_seeds" / encoder / f"seed_{seed}.csv"
            write_normalized(frame, normalized_seed)
            seed_frames.append(frame)
            inventory.append(inventory_row(path, "mil_individual_seed", "huggingface_pinned", expected_ids))
        ensemble = mean_seed_predictions(seed_frames)
        ensemble_path = RAW / "mil_probability_ensembles" / f"{encoder}.csv"
        write_normalized(ensemble, ensemble_path)
        registry.append(
            {
                "model_id": f"{encoder}__mil",
                "encoder": encoder,
                "classifier": "mil",
                "model_family": "mil_probability_ensemble",
                "prediction_path": str(ensemble_path.relative_to(HERE)),
                "cases": expected_count,
                "source_revision": CONFIG["hub"]["revision"],
            }
        )

    # The revised SigLIP2 MIL run supersedes the older Hub seed files.
    canonical_root = REPO_ROOT / "artifacts/models/mil/siglip2_so400m/canonical"
    canonical_seed_frames: list[pd.DataFrame] = []
    for seed in range(42, 52):
        path = canonical_root / f"seed_{seed}" / "test_predictions.csv"
        frame = validate_prediction_frame(pd.read_csv(path), path, expected_ids)
        same_truth(truth_reference, frame, path)  # type: ignore[arg-type]
        normalized_seed = RAW / "mil_individual_seeds" / "siglip2_so400m" / f"seed_{seed}.csv"
        write_normalized(frame, normalized_seed)
        canonical_seed_frames.append(frame)
        inventory.append(inventory_row(path, "mil_individual_seed", "canonical_siglip2_local", expected_ids))
    seed_mean = mean_seed_predictions(canonical_seed_frames)
    ensemble_source = canonical_root / "ensemble" / "test_predictions.csv"
    ensemble = validate_prediction_frame(pd.read_csv(ensemble_source), ensemble_source, expected_ids)
    same_truth(truth_reference, ensemble, ensemble_source)  # type: ignore[arg-type]
    max_difference = float(np.abs(seed_mean[PROB_COLUMNS].to_numpy() - ensemble[PROB_COLUMNS].to_numpy()).max())
    if max_difference > 1e-10:
        raise ValueError(f"Canonical SigLIP2 ensemble is not the ten-seed probability mean: {max_difference}")
    ensemble_path = RAW / "mil_probability_ensembles" / "siglip2_so400m.csv"
    write_normalized(ensemble, ensemble_path)
    inventory.append(inventory_row(ensemble_source, "mil_probability_ensemble", "canonical_siglip2_local", expected_ids))
    registry.append(
        {
            "model_id": "siglip2_so400m__mil",
            "encoder": "siglip2_so400m",
            "classifier": "mil",
            "model_family": "mil_probability_ensemble",
            "prediction_path": str(ensemble_path.relative_to(HERE)),
            "cases": expected_count,
            "source_revision": "canonical_local_ten_seed_ensemble",
        }
    )

    # Explicitly record the older Hub SigLIP2 seed files as superseded inputs.
    for seed in range(42, 52):
        path = INPUT / "mil" / "siglip2_so400m" / f"seed_{seed}" / "test_predictions.csv"
        row = inventory_row(path, "mil_individual_seed_superseded", "huggingface_pinned", expected_ids)
        inventory.append(row)

    registry_frame = pd.DataFrame(registry).sort_values(["encoder", "classifier"])
    if len(registry_frame) != 42 or registry_frame["model_id"].nunique() != 42:
        raise ValueError("The normalized model grid must contain exactly 42 unique cells")
    inventory_frame = pd.DataFrame(inventory)
    inventory_frame.to_csv(MANIFESTS / "prediction_inventory.csv", index=False)
    registry_frame.to_csv(MANIFESTS / "model_registry.csv", index=False)
    pd.DataFrame({"comparison_row": range(expected_count), "case_id": expected_ids}).to_csv(
        MANIFESTS / "canonical_case_ids.csv", index=False
    )
    truth_reference.to_csv(MANIFESTS / "canonical_ground_truth.csv", index=False)  # type: ignore[union-attr]
    write_json(
        MANIFESTS / "audit_summary.json",
        {
            "complete": True,
            "canonical_cases": expected_count,
            "labels": len(TRUE_COLUMNS),
            "classical_case_models": 35,
            "classical_image_files_audited": 35,
            "mil_encoders": 7,
            "mil_seeds_per_encoder": 10,
            "normalized_model_grid_cells": 42,
            "hub_repo": CONFIG["hub"]["repo_id"],
            "hub_revision": CONFIG["hub"]["revision"],
            "siglip2_ensemble_max_probability_difference_from_seed_mean": max_difference,
        },
    )
    print(f"Audit passed: {expected_count} cases, 25 labels, 42 normalized model outputs")


if __name__ == "__main__":
    main()

