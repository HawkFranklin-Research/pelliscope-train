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

from hawk_derm.config import load_study_config, path_from  # noqa: E402


def load_context(config_path: str, run_mode: str | None = None) -> dict[str, Any]:
    config = load_study_config(config_path)
    selected_mode = run_mode or os.getenv("HAWK_DERM_RUN_MODE")
    if selected_mode:
        config["study"]["run_mode"] = selected_mode
    return config


def load_manifests(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(path_from(config, "case_manifest")),
        pd.read_csv(path_from(config, "image_manifest")),
        pd.read_csv(path_from(config, "split_manifest")),
    )


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]
