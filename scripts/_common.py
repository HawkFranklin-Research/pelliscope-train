from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hawk_derm.config import load_study_config, load_yaml, path_from  # noqa: E402
from hawk_derm.runtime import configure_process  # noqa: E402


def load_context(config_path: str, run_mode: str | None = None) -> dict[str, Any]:
    configure_process()
    config = load_study_config(config_path)
    selected_mode = run_mode or os.getenv("HAWK_DERM_RUN_MODE")
    if selected_mode:
        config["study"]["run_mode"] = selected_mode
    if selected_mode == "smoke":
        # Smoke output must never replace production manifests or predictions.
        smoke_root = REPOSITORY_ROOT / "smoke_runs"
        for key in ("case_manifest", "image_manifest", "split_manifest"):
            config["paths"][key] = str(smoke_root / "data" / Path(config["paths"][key]).name)
        for key in ("audit_dir", "artifacts_dir", "reports_dir"):
            config["paths"][key] = str(smoke_root / key.removesuffix("_dir"))
    return config


def load_manifests(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(path_from(config, "case_manifest")),
        pd.read_csv(path_from(config, "image_manifest")),
        pd.read_csv(path_from(config, "split_manifest")),
    )


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def mil_run_root(config: dict[str, Any], encoder: str, run_tag: str | None = None) -> Path:
    root = path_from(config, "artifacts_dir") / "models" / "mil" / encoder
    return root / run_tag if run_tag else root


def selected_mil_config_path(config: dict[str, Any], encoder: str, run_tag: str | None = None) -> Path:
    return mil_run_root(config, encoder, run_tag) / "tuning" / "best_mil_config.yaml"


def load_selected_mil_config(
    config: dict[str, Any],
    encoder: str,
    run_tag: str | None,
    requested: str | Path | None,
) -> tuple[dict[str, Any], Path]:
    path = Path(requested) if requested else selected_mil_config_path(config, encoder, run_tag)
    if path.is_file():
        return load_yaml(path), path
    fallback = load_yaml("configs/mil.yaml")
    if config["study"]["run_mode"] == "full" and fallback.get("selected_config_required_for_full_run", False):
        raise FileNotFoundError(f"A selected MIL configuration is required for a full run: {path}")
    return fallback, Path("configs/mil.yaml")
