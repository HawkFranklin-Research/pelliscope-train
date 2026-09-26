from __future__ import annotations

from typing import Any
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from hawk_derm.config import path_from
from hawk_derm.io import sha256_file, write_csv, write_json


def assert_complete_image_audit(images: pd.DataFrame, audit: pd.DataFrame, *, run_mode: str) -> None:
    columns = ["case_id", "image_id", "image_order"]
    if "resolved_image_path" in images:
        columns.append("resolved_image_path")
    if any(column not in audit for column in [*columns, "sha256", "decode_ok", "run_mode"]):
        raise ValueError("Image audit lacks required identity, hash, or run-mode columns")
    if len(audit) != len(images) or Counter(map(tuple, audit[columns].astype(str).to_numpy())) != Counter(
        map(tuple, images[columns].astype(str).to_numpy())
    ):
        raise ValueError("Image audit does not cover every image-manifest row")
    if not audit["run_mode"].eq(run_mode).all():
        raise ValueError(f"Image audit contains rows outside the {run_mode} run")


def split_membership_sha256(manifest: pd.DataFrame) -> str:
    if manifest["case_id"].duplicated().any():
        raise ValueError("Split manifest contains duplicate case IDs")
    rows = sorted(zip(manifest["case_id"].astype(str), manifest["split"].astype(str), strict=True))
    return hashlib.sha256("".join(f"{case_id},{split}\n" for case_id, split in rows).encode()).hexdigest()


def assert_locked_split(config: dict[str, Any], manifest: pd.DataFrame) -> None:
    if config["study"]["run_mode"] != "full":
        return
    expected_counts = config["study"]["locked_split_counts"]
    actual_counts = manifest["split"].value_counts().to_dict()
    if actual_counts != expected_counts:
        raise ValueError(f"Full split counts differ from the study lock: {actual_counts} != {expected_counts}")
    actual_hash = split_membership_sha256(manifest)
    if actual_hash != config["study"]["locked_split_membership_sha256"]:
        raise ValueError("Full split case IDs differ from the locked production cohort")


def assert_existing_full_split(config: dict[str, Any]) -> None:
    if config["study"]["run_mode"] != "full":
        return
    image_path = path_from(config, "image_manifest")
    audit_path = path_from(config, "audit_dir") / "image_audit.csv"
    split_path = path_from(config, "split_manifest")
    metadata_path = split_path.with_suffix(".metadata.json")
    for path in (image_path, audit_path, split_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Full split prerequisite is missing: {path}")
    assert_complete_image_audit(pd.read_csv(image_path), pd.read_csv(audit_path), run_mode="full")
    assert_locked_split(config, pd.read_csv(split_path))
    import json

    metadata = json.loads(metadata_path.read_text())
    for name, path in (("image_manifest_sha256", image_path), ("image_audit_sha256", audit_path), ("split_manifest_sha256", split_path)):
        if metadata.get(name) != sha256_file(path):
            raise ValueError(f"Full split metadata has a stale or missing {name}")


def assert_classical_locked(config: dict[str, Any], encoders: list[str], classifiers: list[str]) -> None:
    if config["study"]["run_mode"] != "full":
        return
    import json

    split_path = path_from(config, "split_manifest")
    split = pd.read_csv(split_path)
    test_ids = set(split.loc[split["split"].eq("test"), "case_id"].astype(str))
    split_hash = sha256_file(split_path)
    for encoder in encoders:
        for classifier in classifiers:
            root = path_from(config, "artifacts_dir") / "models" / "classical" / encoder / classifier
            prediction_path = root / "test_case_predictions.csv"
            run_path = root / "run_manifest.json"
            if not prediction_path.is_file() or not run_path.is_file():
                raise FileNotFoundError(f"Locked classical result is missing: {root}")
            predictions = pd.read_csv(prediction_path, usecols=["case_id"])
            if len(predictions) != len(test_ids) or set(predictions["case_id"].astype(str)) != test_ids:
                raise ValueError(f"Classical test predictions differ from the locked test cases: {prediction_path}")
            run = json.loads(run_path.read_text())
            if not run.get("complete") or run.get("run_mode") != "full" or not any(
                item.get("sha256") == split_hash and Path(item.get("path", "")).name == split_path.name
                for item in run.get("inputs", [])
            ):
                raise ValueError(f"Classical model was not fitted with the locked split: {run_path}")


def freeze_splits(config: dict[str, Any], cases: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    split_config = config["split"]
    train_fraction = float(split_config["train_fraction"])
    validation_fraction = float(split_config["validation_fraction"])
    test_fraction = float(split_config["test_fraction"])
    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("Split fractions must sum to one")
    seed = int(split_config["seed"])
    label_columns = [column for column in cases.columns if column.startswith("is_")]
    labels = cases[label_columns].to_numpy(dtype=np.uint8)
    indices = np.arange(len(cases))

    outer = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1.0 - train_fraction, random_state=seed)
    train_indices, remainder_indices = next(outer.split(indices, labels))
    validation_share = validation_fraction / (validation_fraction + test_fraction)
    inner = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1.0 - validation_share, random_state=seed + 1)
    validation_local, test_local = next(inner.split(remainder_indices, labels[remainder_indices]))
    validation_indices = remainder_indices[validation_local]
    test_indices = remainder_indices[test_local]

    split = np.empty(len(cases), dtype=object)
    split[train_indices] = "train"
    split[validation_indices] = "validation"
    split[test_indices] = "test"
    manifest = cases[["case_id"]].copy()
    manifest["split"] = split
    manifest["split_seed"] = seed
    manifest["split_version"] = "v1"

    # Reconcile cross-case duplicate images so identical image hashes do not cross split boundaries
    image_audit_path = path_from(config, "audit_dir") / "image_audit.csv"
    if config["study"]["run_mode"] == "full" and not image_audit_path.is_file():
        raise FileNotFoundError(f"Full split requires a complete image audit: {image_audit_path}")
    if image_audit_path.is_file():
        image_audit = pd.read_csv(image_audit_path)
        if config["study"]["run_mode"] == "full":
            assert_complete_image_audit(images, image_audit, run_mode="full")
        valid_hashes = image_audit[image_audit["sha256"].ne("") & image_audit["decode_ok"]]
        hash_to_cases = valid_hashes.groupby("sha256")["case_id"].apply(lambda s: sorted(set(s.astype(str)))).to_dict()
        case_to_split = dict(zip(manifest["case_id"].astype(str), manifest["split"]))
        for h, c_list in hash_to_cases.items():
            if len(c_list) > 1:
                splits_in_group = [case_to_split[c] for c in c_list if c in case_to_split]
                target_split = "train" if "train" in splits_in_group else splits_in_group[0]
                for c in c_list:
                    if c in case_to_split:
                        case_to_split[c] = target_split
        manifest["split"] = manifest["case_id"].astype(str).map(case_to_split)

    assert_locked_split(config, manifest)
    output = path_from(config, "split_manifest")
    write_csv(output, manifest)

    joined = cases.merge(manifest, on="case_id", validate="one_to_one").merge(
        images.groupby("case_id").size().rename("image_count"), left_on="case_id", right_index=True, how="left"
    )
    rows = []
    for split_name, frame in joined.groupby("split", sort=False):
        row: dict[str, Any] = {
            "split": split_name,
            "cases": frame["case_id"].nunique(),
            "images": int(frame["image_count"].fillna(0).sum()),
            "mean_label_cardinality": frame[label_columns].sum(axis=1).mean(),
            "all_zero_cases": frame[label_columns].sum(axis=1).eq(0).sum(),
        }
        row.update({f"positive__{column.removeprefix('is_')}": int(frame[column].sum()) for column in label_columns})
        rows.append(row)
    audit = pd.DataFrame(rows)
    audit_path = path_from(config, "audit_dir") / "split_audit.csv"
    write_csv(audit_path, audit)
    demographic_columns = [
        column
        for column in joined.columns
        if any(token in column.lower() for token in ("age", "sex", "fitzpatrick", "monk", "skin_tone", "gradab"))
        and column not in label_columns
    ]
    demographic_rows = []
    for column in demographic_columns:
        counts = joined.groupby(["split", joined[column].fillna("Missing").astype(str)]).size()
        for (split_name, category), count in counts.items():
            demographic_rows.append({"field": column, "category": category, "split": split_name, "cases": int(count)})
    write_csv(path_from(config, "audit_dir") / "split_demographic_audit.csv", pd.DataFrame(demographic_rows))
    write_json(
        output.with_suffix(".metadata.json"),
        {
            "run_mode": config["study"]["run_mode"],
            "split_version": "v1",
            "seed": seed,
            "fractions": {"train": train_fraction, "validation": validation_fraction, "test": test_fraction},
            "counts": manifest["split"].value_counts().to_dict(),
            "image_manifest_sha256": sha256_file(path_from(config, "image_manifest")),
            "image_audit_sha256": sha256_file(image_audit_path) if image_audit_path.is_file() else None,
            "split_manifest_sha256": sha256_file(output),
            "complete": True,
        },
    )
    return manifest


def assert_no_split_leakage(split_manifest: pd.DataFrame, image_audit: pd.DataFrame) -> None:
    joined = image_audit.merge(split_manifest[["case_id", "split"]], on="case_id", how="left", validate="many_to_one")
    leaking = joined[joined["sha256"].ne("")].groupby("sha256")["split"].nunique()
    leaking = leaking[leaking > 1]
    if not leaking.empty:
        raise ValueError(f"{len(leaking)} image hashes cross split boundaries")
