from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def threshold_sweep(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    grid: np.ndarray | None = None,
) -> pd.DataFrame:
    grid = np.arange(0.0, 1.0001, 0.01) if grid is None else np.asarray(grid)
    rows: list[dict[str, Any]] = []
    for class_index, label in enumerate(labels):
        truth = y_true[:, class_index].astype(bool)
        for threshold in grid:
            predicted = probabilities[:, class_index] >= threshold
            tp = int(np.sum(predicted & truth))
            fp = int(np.sum(predicted & ~truth))
            tn = int(np.sum(~predicted & ~truth))
            fn = int(np.sum(~predicted & truth))
            sensitivity = tp / (tp + fn) if tp + fn else float("nan")
            specificity = tn / (tn + fp) if tn + fp else float("nan")
            precision = tp / (tp + fp) if tp + fp else 0.0
            f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
            rows.append(
                {
                    "label": label,
                    "threshold": float(threshold),
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
                    "youden_j": sensitivity + specificity - 1,
                    "f1": f1,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                }
            )
    return pd.DataFrame(rows)


def select_thresholds(sweep: pd.DataFrame, rule: str = "balance") -> pd.DataFrame:
    selected = []
    for _, group in sweep.groupby("label", sort=False):
        group = group.copy()
        if rule == "balance":
            group["objective"] = (group["sensitivity"] - group["specificity"]).abs()
            ordered = group.sort_values(["objective", "threshold"], ascending=[True, True])
        elif rule == "youden":
            ordered = group.sort_values(["youden_j", "threshold"], ascending=[False, True])
        elif rule == "f1":
            ordered = group.sort_values(["f1", "threshold"], ascending=[False, True])
        else:
            raise ValueError(f"Unknown threshold selection rule: {rule}")
        selected.append(ordered.iloc[0])
    return pd.DataFrame(selected).drop(columns=["objective"], errors="ignore").reset_index(drop=True)
