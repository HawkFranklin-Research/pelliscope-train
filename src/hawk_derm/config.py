from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_study_config(path: str | Path = "configs/study_25class.yaml") -> dict[str, Any]:
    config = load_yaml(path)
    config["repository_root"] = str(REPOSITORY_ROOT)
    paths = config.setdefault("paths", {})
    raw_root_override = os.getenv("HAWK_DERM_RAW_DATA_ROOT")
    if raw_root_override:
        raw_root = Path(raw_root_override).expanduser()
        paths["raw_dataset_root"] = str(raw_root)
        paths["raw_images_dir"] = str(raw_root / "images")
        paths["raw_metadata_csv"] = str(raw_root / "metadata.csv")
    for key in list(paths):
        override = os.getenv(f"HAWK_DERM_PATH_{key.upper()}")
        if override:
            paths[key] = override
    return config


def resolve_path(value: str | Path, root: str | Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(root or REPOSITORY_ROOT) / path


def path_from(config: Mapping[str, Any], key: str) -> Path:
    return resolve_path(config["paths"][key], config.get("repository_root", REPOSITORY_ROOT))
