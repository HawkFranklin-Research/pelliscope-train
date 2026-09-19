from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from hawk_derm.config import path_from
from hawk_derm.io import write_csv, write_json


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
    if image_audit_path.is_file():
        image_audit = pd.read_csv(image_audit_path)
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
