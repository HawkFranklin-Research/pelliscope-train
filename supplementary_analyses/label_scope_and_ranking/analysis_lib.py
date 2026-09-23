"""Shared utilities for the isolated label-scope analysis."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CONFIG = yaml.safe_load((HERE / "analysis_config.yaml").read_text())
STUDY = yaml.safe_load((REPO_ROOT / "configs" / "study_25class.yaml").read_text())
LABELS = list(STUDY["labels"])
ENCODERS = ["resnet50", "inception_v3", "bit50", "vit_base", "clip_vitb32", "derm_foundation", "siglip2_so400m"]
CLASSIFIERS = ["gradient_boosting", "svm_linear", "logistic", "mil", "random_forest", "knn"]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


LABEL_SLUGS = [slug(label) for label in LABELS]
TRUE_COLUMNS = [f"true__{name}" for name in LABEL_SLUGS]
PROB_COLUMNS = [f"prob__{name}" for name in LABEL_SLUGS]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_case_ids() -> list[str]:
    path = REPO_ROOT / "reports/reanalysis/siglip2_so400m_mil_canonical/statistics/comparison_cohort.csv"
    frame = pd.read_csv(path).sort_values("comparison_row")
    ids = frame["case_id"].astype(str).tolist()
    expected = int(CONFIG["canonical"]["expected_cases"])
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError(f"Canonical comparison cohort must contain {expected} unique cases")
    return ids


def validate_prediction_frame(frame: pd.DataFrame, path: Path, expected_ids: Iterable[str]) -> pd.DataFrame:
    expected_ids = list(expected_ids)
    required = {"case_id", *TRUE_COLUMNS, *PROB_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    frame = frame.copy()
    frame["case_id"] = frame["case_id"].astype(str)
    if frame["case_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate case IDs")
    available = set(frame["case_id"])
    absent = sorted(set(expected_ids) - available)
    if absent:
        raise ValueError(f"{path}: missing {len(absent)} canonical cases")
    frame = frame.set_index("case_id").loc[expected_ids].reset_index()
    truth = frame[TRUE_COLUMNS].to_numpy(dtype=float)
    probability = frame[PROB_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(truth).all() or not np.isin(truth, [0, 1]).all():
        raise ValueError(f"{path}: labels must be finite binary values")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError(f"{path}: probabilities must be finite and in [0, 1]")
    return frame[["case_id", *TRUE_COLUMNS, *PROB_COLUMNS]]


def arrays(frame: pd.DataFrame, label_slugs: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    names = label_slugs or LABEL_SLUGS
    return (
        frame[[f"true__{name}" for name in names]].to_numpy(dtype=int),
        frame[[f"prob__{name}" for name in names]].to_numpy(dtype=float),
    )


def safe_metrics(truth: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    valid = [index for index in range(truth.shape[1]) if np.unique(truth[:, index]).size == 2]
    if not valid:
        return {name: float("nan") for name in ("roc_auc_macro", "roc_auc_micro", "pr_auc_macro", "pr_auc_micro")}
    per_roc = [roc_auc_score(truth[:, index], probability[:, index]) for index in valid]
    per_pr = [average_precision_score(truth[:, index], probability[:, index]) for index in valid]
    return {
        "roc_auc_macro": float(np.mean(per_roc)),
        "roc_auc_micro": float(roc_auc_score(truth.ravel(), probability.ravel())),
        "pr_auc_macro": float(np.mean(per_pr)),
        "pr_auc_micro": float(average_precision_score(truth.ravel(), probability.ravel())),
    }


def curve_values(truth: np.ndarray, probability: np.ndarray, curve: str, averaging: str, grid: np.ndarray) -> np.ndarray:
    if averaging == "micro":
        columns = [(truth.ravel(), probability.ravel())]
    else:
        columns = [
            (truth[:, index], probability[:, index])
            for index in range(truth.shape[1])
            if np.unique(truth[:, index]).size == 2
        ]
    interpolated: list[np.ndarray] = []
    for target, score in columns:
        if curve == "roc":
            x, y, _ = roc_curve(target, score)
            interpolated.append(np.interp(grid, x, y))
        else:
            precision, recall, _ = precision_recall_curve(target, score)
            x = recall[::-1]
            y = precision[::-1]
            unique_x, inverse = np.unique(x, return_inverse=True)
            best_y = np.zeros_like(unique_x)
            for position in range(len(unique_x)):
                best_y[position] = y[inverse == position].max()
            interpolated.append(np.interp(grid, unique_x, best_y))
    return np.mean(interpolated, axis=0)


def bootstrap_interval(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(finite, [0.025, 0.975]))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

