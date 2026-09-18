from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from hawk_derm.constants import slugify
from hawk_derm.io import write_csv, write_json


def prediction_frame(
    case_ids: Iterable[str],
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    *,
    thresholds: np.ndarray | None = None,
    split: str | None = None,
    seed: int | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    if y_true.shape != probabilities.shape or y_true.shape[1] != len(labels):
        raise ValueError("Prediction and target shapes must match the label vocabulary")
    thresholds = np.full(len(labels), 0.5) if thresholds is None else np.asarray(thresholds)
    frame = pd.DataFrame({"case_id": list(case_ids)})
    if split is not None:
        frame["split"] = split
    if seed is not None:
        frame["seed"] = seed
    if model is not None:
        frame["model"] = model
    for index, label in enumerate(labels):
        slug = slugify(label)
        frame[f"true__{slug}"] = y_true[:, index].astype(np.uint8)
        frame[f"prob__{slug}"] = probabilities[:, index].astype(float)
        frame[f"pred__{slug}"] = (probabilities[:, index] >= thresholds[index]).astype(np.uint8)
    return frame


def save_predictions(path: str | Path, frame: pd.DataFrame, labels: list[str], metadata: dict | None = None) -> Path:
    path = write_csv(path, frame)
    write_json(
        path.with_suffix(".metadata.json"),
        {"labels": labels, "rows": len(frame), "complete": True, **(metadata or {})},
    )
    return path


def arrays_from_prediction_frame(frame: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    slugs = [slugify(label) for label in labels]
    y_true = frame[[f"true__{slug}" for slug in slugs]].to_numpy(dtype=np.uint8)
    probabilities = frame[[f"prob__{slug}" for slug in slugs]].to_numpy(dtype=float)
    return y_true, probabilities


def align_prediction_frames(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if left["case_id"].duplicated().any() or right["case_id"].duplicated().any():
        raise ValueError("Paired prediction files must contain one row per case ID")
    shared = sorted(set(left["case_id"].astype(str)) & set(right["case_id"].astype(str)))
    left_indexed = left.assign(case_id=left["case_id"].astype(str)).set_index("case_id").loc[shared].reset_index()
    right_indexed = right.assign(case_id=right["case_id"].astype(str)).set_index("case_id").loc[shared].reset_index()
    if not left_indexed["case_id"].equals(right_indexed["case_id"]):
        raise RuntimeError("Prediction alignment failed")
    return left_indexed, right_indexed
