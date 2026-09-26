import json

import pandas as pd
import pytest

from hawk_derm.data.splits import assert_classical_locked, assert_complete_image_audit, assert_locked_split, assert_no_split_leakage, split_membership_sha256
from hawk_derm.io import sha256_file


def test_hash_leakage_is_rejected() -> None:
    splits = pd.DataFrame({"case_id": ["a", "b"], "split": ["train", "test"]})
    audit = pd.DataFrame({"case_id": ["a", "b"], "sha256": ["same", "same"]})
    try:
        assert_no_split_leakage(splits, audit)
    except ValueError:
        return
    raise AssertionError("Expected cross-split duplicate hash to be rejected")


def test_full_audit_rejects_smoke_subset() -> None:
    images = pd.DataFrame({"case_id": ["a", "b"], "image_id": ["x", "y"], "image_order": [1, 1]})
    audit = images.iloc[:1].assign(sha256="hash", decode_ok=True, run_mode="smoke")
    with pytest.raises(ValueError, match="cover every"):
        assert_complete_image_audit(images, audit, run_mode="full")


def test_locked_membership_checks_ids_as_well_as_counts() -> None:
    split = pd.DataFrame({"case_id": ["a", "b", "c"], "split": ["train", "validation", "test"]})
    config = {"study": {"run_mode": "full", "locked_split_counts": {"train": 1, "validation": 1, "test": 1}, "locked_split_membership_sha256": split_membership_sha256(split)}}
    assert_locked_split(config, split)
    changed = split.copy()
    changed.loc[2, "case_id"] = "different"
    with pytest.raises(ValueError, match="case IDs"):
        assert_locked_split(config, changed)


def test_classical_gate_rejects_smoke_prediction_copy(tmp_path) -> None:
    split_path = tmp_path / "split.csv"
    pd.DataFrame({"case_id": ["a", "b"], "split": ["train", "test"]}).to_csv(split_path, index=False)
    root = tmp_path / "artifacts" / "models" / "classical" / "encoder" / "random_forest"
    root.mkdir(parents=True)
    prediction = root / "test_case_predictions.csv"
    pd.DataFrame({"case_id": ["b"]}).to_csv(prediction, index=False)
    (root / "run_manifest.json").write_text(json.dumps({
        "complete": True,
        "run_mode": "full",
        "inputs": [{"path": str(split_path), "sha256": sha256_file(split_path)}],
    }))
    config = {"repository_root": str(tmp_path), "paths": {"split_manifest": str(split_path), "artifacts_dir": str(tmp_path / "artifacts")}, "study": {"run_mode": "full"}}
    assert_classical_locked(config, ["encoder"], ["random_forest"])
    pd.DataFrame({"case_id": ["a"]}).to_csv(prediction, index=False)
    with pytest.raises(ValueError, match="locked test cases"):
        assert_classical_locked(config, ["encoder"], ["random_forest"])
