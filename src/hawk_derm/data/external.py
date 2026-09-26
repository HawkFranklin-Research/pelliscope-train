"""Inference-only external cohorts (e.g. SD-198).

An external cohort is prepared once into adapter manifests that follow this pipeline's own
schema (`case_id`, `is_<label>` columns, `resolved_image_path`, `bag_slot`, `retained_for_bag`),
so the unchanged feature extractor and model code can score it. External data never enter
training, tuning, calibration or threshold selection.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hawk_derm.config import REPOSITORY_ROOT, load_yaml, resolve_path
from hawk_derm.constants import slugify
from hawk_derm.features.bank import FeatureBank, load_feature_bank
from hawk_derm.io import sha256_file


@dataclass(frozen=True)
class ExternalCohort:
    name: str
    display_name: str
    config_path: Path
    dataset_root: Path
    output_root: Path
    settings: dict[str, Any]

    @property
    def manifests_dir(self) -> Path:
        return self.output_root / "manifests"

    @property
    def case_manifest_path(self) -> Path:
        return self.manifests_dir / "case_manifest.csv"

    @property
    def image_manifest_path(self) -> Path:
        return self.manifests_dir / "image_manifest.csv"

    @property
    def prepared(self) -> bool:
        return self.case_manifest_path.is_file() and self.image_manifest_path.is_file()

    def source_path(self, key: str) -> Path:
        return self.dataset_root / self.settings["paths"][key]

    def feature_bank_path(self, encoder: str) -> Path:
        return self.output_root / "features" / encoder / "feature_bank.npz"

    def prediction_name(self) -> str:
        """Prefix used for this cohort's prediction files, e.g. `external_sd198`."""
        return f"external_{self.name}"


@dataclass(frozen=True)
class ExternalInput:
    """One prepared cohort with a feature bank for a specific encoder."""

    name: str
    cases: pd.DataFrame
    bank: FeatureBank
    bank_sha256: str
    case_manifest_sha256: str

    def identity(self) -> dict[str, str]:
        return {"name": self.name, "bank_sha256": self.bank_sha256, "case_manifest_sha256": self.case_manifest_sha256}


def load_external_cohort(path: str | Path, root: str | Path | None = None) -> ExternalCohort:
    config_path = resolve_path(path, root)
    settings = load_yaml(config_path)
    name = str(settings["cohort"]["name"])
    dataset_value = os.getenv(f"HAWK_DERM_EXTERNAL_{name.upper()}_ROOT") or settings["paths"]["dataset_root"]
    return ExternalCohort(
        name=name,
        display_name=str(settings["cohort"].get("display_name", name)),
        config_path=config_path,
        dataset_root=resolve_path(dataset_value, root),
        output_root=resolve_path(settings["paths"]["output_root"], root),
        settings=settings,
    )


def load_external_cohorts(config: dict[str, Any]) -> list[ExternalCohort]:
    root = config.get("repository_root", REPOSITORY_ROOT)
    return [load_external_cohort(path, root) for path in config.get("external_cohorts", []) or []]


def parse_label_schema(payload: Any) -> list[str]:
    """Return labels in output order from a schema JSON that is a list or a dict with a `labels` list."""
    entries = payload.get("labels", payload.get("classes")) if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError("Label schema must contain a non-empty list of labels")
    if all(isinstance(entry, str) for entry in entries):
        return list(entries)
    rows = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Unsupported label schema entry: {entry!r}")
        label = entry.get("label") or entry.get("name") or entry.get("label_25")
        index = entry.get("index", entry.get("label_index", entry.get("label_index_25", position)))
        rows.append((int(index), str(label)))
    rows.sort()
    if [index for index, _ in rows] != list(range(len(rows))):
        raise ValueError("Label schema indices must be 0..N-1 without gaps")
    return [label for _, label in rows]


def build_external_manifests(cohort: ExternalCohort, labels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate the downloaded cohort and return (cases, images, summary) in pipeline schema."""
    schema = parse_label_schema(json.loads(cohort.source_path("label_schema").read_text()))
    if [slugify(label) for label in schema] != [slugify(label) for label in labels]:
        raise ValueError(
            "External label schema order differs from the study labels: "
            f"{[slugify(label) for label in schema]} != {[slugify(label) for label in labels]}"
        )
    source = pd.read_csv(cohort.source_path("image_manifest"))
    required = {"image_id", "pseudo_case_id", "image_path", "label_index_25"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"External image manifest lacks columns: {sorted(missing)}")
    if source["image_id"].duplicated().any():
        raise ValueError("External image IDs must be unique")
    indices = source["label_index_25"].astype(int)
    if indices.min() < 0 or indices.max() >= len(labels):
        raise ValueError("External label_index_25 values fall outside the 25-label schema")
    resolved = [cohort.dataset_root / str(path) for path in source["image_path"]]
    absent = [str(path) for path in resolved if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} external images are missing, e.g. {absent[:3]}")
    hashes = [sha256_file(path) for path in resolved]
    if "sha256" in source:
        mismatched = [
            image_id
            for image_id, recorded, digest in zip(source["image_id"], source["sha256"].astype(str), hashes, strict=True)
            if recorded and recorded != "nan" and recorded != digest
        ]
        if mismatched:
            raise ValueError(f"External image hashes differ from the manifest for {len(mismatched)} images, e.g. {mismatched[:3]}")
    duplicate_group = (
        source["duplicate_group_id"].fillna("").astype(str)
        if "duplicate_group_id" in source
        else pd.Series([""] * len(source))
    )
    # Images without a duplicate group are their own resampling unit.
    resampling_group = [group if group else f"unique::{image_id}" for group, image_id in zip(duplicate_group, source["image_id"], strict=True)]
    images = pd.DataFrame(
        {
            "case_id": source["pseudo_case_id"].astype(str),
            "image_id": source["image_id"].astype(str),
            "image_order": 0,
            "bag_slot": 0,
            "retained_for_bag": True,
            "resolved_image_path": [str(path) for path in resolved],
            "sha256": hashes,
            "duplicate_group_id": duplicate_group.to_numpy(),
            "resampling_group": resampling_group,
        }
    )
    if images["case_id"].duplicated().any():
        raise ValueError("Each external pseudo-case must contain exactly one image")
    one_hot = np.zeros((len(source), len(labels)), dtype=np.uint8)
    one_hot[np.arange(len(source)), indices.to_numpy()] = 1
    cases = pd.DataFrame({"case_id": images["case_id"], "label_index_25": indices.to_numpy(), "label_25": [labels[i] for i in indices]})
    for position, label in enumerate(labels):
        cases[f"is_{slugify(label)}"] = one_hot[:, position]
    cases["resampling_group"] = resampling_group
    cases["bag_length"] = 1
    represented = [labels[i] for i in sorted(set(indices))]
    summary = {
        "cohort": cohort.name,
        "images": int(len(images)),
        "unique_image_contents": int(pd.Series(hashes).nunique()),
        "represented_labels": represented,
        "represented_label_count": len(represented),
        "images_per_label": {labels[i]: int(count) for i, count in indices.value_counts().sort_index().items()},
        "source_image_manifest_sha256": sha256_file(cohort.source_path("image_manifest")),
        "label_schema_sha256": sha256_file(cohort.source_path("label_schema")),
    }
    expected = cohort.settings.get("expected", {})
    checks = {
        "images": summary["images"],
        "unique_images": summary["unique_image_contents"],
        "represented_labels": summary["represented_label_count"],
    }
    for key, actual in checks.items():
        if key in expected and int(expected[key]) != actual:
            raise ValueError(f"External cohort {cohort.name}: expected {key}={expected[key]}, found {actual}")
    return cases, images, summary


def represented_labels(cases: pd.DataFrame, labels: list[str]) -> list[str]:
    return [label for label in labels if cases[f"is_{slugify(label)}"].sum() > 0]


def external_inputs(config: dict[str, Any], encoder: str) -> list[ExternalInput]:
    """Prepared cohorts that have a feature bank for `encoder`; silently empty otherwise."""
    inputs = []
    for cohort in load_external_cohorts(config):
        bank_path = cohort.feature_bank_path(encoder)
        if not cohort.prepared or not bank_path.is_file():
            continue
        inputs.append(
            ExternalInput(
                name=cohort.prediction_name(),
                cases=pd.read_csv(cohort.case_manifest_path),
                bank=load_feature_bank(bank_path),
                bank_sha256=sha256_file(bank_path),
                case_manifest_sha256=sha256_file(cohort.case_manifest_path),
            )
        )
    return inputs


def external_paths(config: dict[str, Any], encoder: str) -> list[Path]:
    """Input files that external scoring depends on, for run-manifest hashing."""
    paths: list[Path] = []
    for cohort in load_external_cohorts(config):
        bank_path = cohort.feature_bank_path(encoder)
        if cohort.prepared and bank_path.is_file():
            paths.extend([bank_path, cohort.case_manifest_path])
    return paths
