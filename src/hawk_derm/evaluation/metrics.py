from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, score)) if np.unique(y_true).size == 2 else float("nan")


def safe_average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y_true, score)) if np.any(y_true == 1) else float("nan")


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    memberships = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        mask = memberships == index
        if mask.any():
            error += mask.mean() * abs(float(probability[mask].mean()) - float(y_true[mask].mean()))
    return float(error)


def multilabel_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.uint8)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = probabilities >= thresholds
    per_class_auc = [safe_auc(y_true[:, index], probabilities[:, index]) for index in range(y_true.shape[1])]
    per_class_ap = [safe_average_precision(y_true[:, index], probabilities[:, index]) for index in range(y_true.shape[1])]
    return {
        "subset_accuracy": float(accuracy_score(y_true, predictions)),
        "precision_micro": float(precision_score(y_true, predictions, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, predictions, average="micro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, predictions, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "auc_micro": safe_auc(y_true.ravel(), probabilities.ravel()),
        "auc_macro": float(np.nanmean(per_class_auc)),
        "pr_auc_micro": safe_average_precision(y_true.ravel(), probabilities.ravel()),
        "pr_auc_macro": float(np.nanmean(per_class_ap)),
        "brier_macro": float(np.mean([brier_score_loss(y_true[:, i], probabilities[:, i]) for i in range(y_true.shape[1])])),
        "ece_macro": float(np.mean([expected_calibration_error(y_true[:, i], probabilities[:, i]) for i in range(y_true.shape[1])])),
    }


def per_class_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    thresholds: np.ndarray | float = 0.5,
) -> pd.DataFrame:
    threshold_array = np.full(len(labels), float(thresholds)) if np.isscalar(thresholds) else np.asarray(thresholds)
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        truth = y_true[:, index].astype(np.uint8)
        probability = probabilities[:, index]
        predicted = probability >= threshold_array[index]
        tp = int(np.sum(predicted & (truth == 1)))
        fp = int(np.sum(predicted & (truth == 0)))
        tn = int(np.sum(~predicted & (truth == 0)))
        fn = int(np.sum(~predicted & (truth == 1)))
        sensitivity = tp / (tp + fn) if tp + fn else float("nan")
        specificity = tn / (tn + fp) if tn + fp else float("nan")
        rows.append(
            {
                "label": label,
                "threshold": float(threshold_array[index]),
                "support_positive": int(truth.sum()),
                "support_negative": int((1 - truth).sum()),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
                "precision": float(precision_score(truth, predicted, zero_division=0)),
                "recall": float(recall_score(truth, predicted, zero_division=0)),
                "f1": float(f1_score(truth, predicted, zero_division=0)),
                "auc": safe_auc(truth, probability),
                "pr_auc": safe_average_precision(truth, probability),
                "brier": float(brier_score_loss(truth, probability)),
                "ece": expected_calibration_error(truth, probability),
            }
        )
    return pd.DataFrame(rows)
