import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("first", "last", "expected_calls", "expected_ranges"),
    [
        ("tuning", "thresholds", 2, [("tuning", "thresholds"), ("tuning", "thresholds")]),
        ("final", "verify", 1, [("final", "verification")]),
    ],
)
def test_full_pipeline_delegates_mil_stages(monkeypatch, tmp_path: Path, first: str, last: str, expected_calls: int, expected_ranges: list[tuple[str, str]]) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    pipeline = importlib.import_module("run_pipeline")
    config = {
        "repository_root": str(tmp_path),
        "paths": {"artifacts_dir": "artifacts", "reports_dir": "reports"},
        "study": {"run_mode": "full"},
    }
    monkeypatch.setattr(pipeline, "load_context", lambda *_: config)
    monkeypatch.setattr(pipeline, "assert_existing_full_split", lambda *_: None)
    monkeypatch.setattr(pipeline, "assert_classical_locked", lambda *_: None)
    monkeypatch.setattr(pipeline, "load_encoder_registry", lambda: {
        "siglip2_so400m": SimpleNamespace(import_bank=None),
        "derm_foundation": SimpleNamespace(import_bank=None),
    })
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda script, *args, **kwargs: calls.append((script, list(args))))
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--run-mode", "full", "--from-stage", first, "--to-stage", last])
    pipeline.main()
    assert len(calls) == expected_calls
    assert all(script == "run_mil_pipeline.py" for script, _ in calls)
    assert [(args[args.index("--from-stage") + 1], args[args.index("--to-stage") + 1]) for _, args in calls] == expected_ranges
    assert all("--run-tag" in args and "canonical" in args for _, args in calls)


def test_smoke_outputs_are_isolated(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    common = importlib.import_module("_common")
    smoke = common.load_context("configs/study_25class.yaml", "smoke")
    full = common.load_context("configs/study_25class.yaml", "full")
    for key in ("case_manifest", "image_manifest", "split_manifest", "audit_dir", "artifacts_dir", "reports_dir"):
        assert "smoke_runs" in smoke["paths"][key]
        assert "smoke_runs" not in full["paths"][key]


def test_resume_rejects_overwritten_predictions(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    pipeline = importlib.import_module("run_pipeline")
    output = tmp_path / "test_case_predictions.csv"
    manifest = tmp_path / "run_manifest.json"
    output.write_text("complete prediction\n")
    from hawk_derm.io import sha256_file

    manifest.write_text(json.dumps({
        "complete": True,
        "run_mode": "full",
        "inputs": [],
        "outputs": [{"path": str(output), "sha256": sha256_file(output)}],
    }))
    assert pipeline.completed_artifact([output], manifest, "full")
    output.write_text("six-row smoke copy\n")
    assert not pipeline.completed_artifact([output], manifest, "full")


def test_full_resume_rebuilds_stale_split(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    pipeline = importlib.import_module("run_pipeline")
    config = {
        "repository_root": str(tmp_path),
        "paths": {"split_manifest": str(tmp_path / "split.csv"), "artifacts_dir": str(tmp_path / "artifacts")},
        "study": {"run_mode": "full"},
    }
    monkeypatch.setattr(pipeline, "load_context", lambda *_: config)
    monkeypatch.setattr(pipeline, "load_encoder_registry", lambda: {"siglip2_so400m": SimpleNamespace(import_bank=None)})
    rebuilt = False
    calls = []

    def check_split(_config):
        if not rebuilt:
            raise ValueError("old split")

    def record_run(script, *_args, **kwargs):
        nonlocal rebuilt
        calls.append((script, kwargs["resume"]))
        rebuilt = True

    monkeypatch.setattr(pipeline, "assert_existing_full_split", check_split)
    monkeypatch.setattr(pipeline, "run", record_run)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--run-mode", "full", "--resume", "--from-stage", "split", "--to-stage", "split"])
    pipeline.main()
    assert calls == [("03_freeze_splits.py", False)]
