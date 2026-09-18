from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def write_json(path: str | Path, value: Any) -> Path:
    path = ensure_parent(path)
    payload = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    _atomic_text(path, payload)
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    path = ensure_parent(path)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)
    return path


def append_csv_row(path: str | Path, columns: Iterable[str], row: dict[str, Any]) -> None:
    path = ensure_parent(path)
    columns = list(columns)
    frame = pd.DataFrame([{column: row.get(column, "") for column in columns}])
    frame.to_csv(path, mode="a", index=False, header=not path.exists() or path.stat().st_size == 0)


def _atomic_text(path: Path, payload: str) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    os.replace(temporary, path)
