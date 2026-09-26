from copy import deepcopy
import importlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from hawk_derm.io import sha256_file, write_json
from hawk_derm.models import experiments


def test_mil_stage_resume_requires_matching_command_options(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    pipeline = importlib.import_module("run_mil_pipeline")
    recorded = {"config": "configs/study_25class.yaml", "trials": 12, "strict_locked_split": True}
    assert pipeline.requested_options_match(
        ["--config", "configs/study_25class.yaml", "--trials", "12", "--strict-locked-split"], recorded
    )
    assert not pipeline.requested_options_match(["--trials", "20"], recorded)
    assert not pipeline.requested_options_match(["--run-tag", "different"], recorded)


def test_old_mil_summary_cannot_skip_training(monkeypatch, tmp_path) -> None:
    write_json(tmp_path / "summary.json", {"complete": True, "best_score": 1.0})

    def attempted_training(*_args, **_kwargs):
        raise RuntimeError("training was reached")

    monkeypatch.setattr(experiments, "prepare_bag_data", attempted_training)
    with pytest.raises(RuntimeError, match="training was reached"):
        experiments.run_mil_experiment(
            SimpleNamespace(encoder="test", model_id="test", revision="test", dimension=2),
            pd.DataFrame(),
            pd.DataFrame({"case_id": ["a", "b"], "split": ["train", "validation"]}),
            ["eczema"],
            {"architecture": {}, "training": {}},
            tmp_path,
            seed=42,
            provenance={"feature_bank_sha256": "bank", "case_manifest_sha256": "cases"},
        )


def test_mil_cache_requires_matching_inputs_and_complete_artifacts(tmp_path) -> None:
    bank = SimpleNamespace(encoder="test", model_id="test", revision="test", dimension=2)
    splits = pd.DataFrame({"case_id": ["a", "b"], "split": ["train", "validation"]})
    config = {"architecture": {"dropout": 0.1}, "training": {"epochs": 3}}
    provenance = {"feature_bank_sha256": "bank", "case_manifest_sha256": "cases"}
    kwargs = {"seed": 42, "max_images": 3, "evaluation_splits": ("train", "validation"), "provenance": provenance}
    fingerprint = experiments.mil_experiment_fingerprint(bank, splits, ["eczema"], config, **kwargs)
    assert fingerprint is not None
    paths = experiments.mil_result_artifacts(tmp_path, kwargs["evaluation_splits"])
    for path in paths:
        path.write_text(path.name)
    summary = {
        "complete": True,
        "experiment_fingerprint": fingerprint,
        "artifact_sha256": {path.name: sha256_file(path) for path in paths},
    }
    assert experiments.reusable_mil_summary(summary, tmp_path, kwargs["evaluation_splits"], fingerprint)

    changed_splits = splits.copy()
    changed_splits.loc[1, "split"] = "test"
    changed_config = deepcopy(config)
    changed_config["training"]["epochs"] = 4
    assert experiments.mil_experiment_fingerprint(bank, changed_splits, ["eczema"], config, **kwargs) != fingerprint
    assert experiments.mil_experiment_fingerprint(bank, splits, ["eczema"], changed_config, **kwargs) != fingerprint
    assert experiments.mil_experiment_fingerprint(bank, splits, ["eczema"], config, **{**kwargs, "seed": 43}) != fingerprint
    assert experiments.mil_experiment_fingerprint(bank, splits, ["eczema"], config, **{**kwargs, "provenance": None}) is None
    assert not experiments.reusable_mil_summary(summary, tmp_path, kwargs["evaluation_splits"], "different")

    paths[0].write_text("changed model")
    assert not experiments.reusable_mil_summary(summary, tmp_path, kwargs["evaluation_splits"], fingerprint)
    paths[0].unlink()
    assert not experiments.reusable_mil_summary(summary, tmp_path, kwargs["evaluation_splits"], fingerprint)
