"""Metrics for single-image, single-label external cohorts such as SD-198.

Each external image carries exactly one of the 25 labels. Within the represented labels, the
other represented labels' images serve as negatives (one-vs-rest); this assumes one diagnosis per
image. Unrepresented labels have no positives, so only their false-alarm rate is reported.
Resampling is by duplicate-content group so identical images never split across a resample.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from hawk_derm.statistics.inference import bootstrap_two_sided_p_value, wilson_interval

HEADLINE_METRICS = ("macro_auc_represented", "macro_ap_represented", "top1_accuracy", "top3_accuracy")


def represented_columns(truth: np.ndarray) -> np.ndarray:
    return np.flatnonzero(truth.sum(axis=0) > 0)


def _topk_accuracy(truth: np.ndarray, probability: np.ndarray, k: int, columns: np.ndarray | None = None) -> float:
    scores = probability if columns is None else probability[:, columns]
    target = truth if columns is None else truth[:, columns]
    top = np.argsort(-scores, axis=1)[:, :k]
    return float(np.mean(np.take_along_axis(target, top, axis=1).any(axis=1)))


def _per_label(truth: np.ndarray, probability: np.ndarray, columns: np.ndarray, metric: Callable) -> np.ndarray:
    values = []
    for column in columns:
        y = truth[:, column]
        values.append(metric(y, probability[:, column]) if 0 < y.sum() < len(y) else np.nan)
    return np.asarray(values, dtype=float)


def external_metrics(
    truth: np.ndarray,
    probability: np.ndarray,
    thresholds: np.ndarray | None = None,
    top_k: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.uint8)
    probability = np.asarray(probability, dtype=float)
    columns = represented_columns(truth)
    auc = _per_label(truth, probability, columns, roc_auc_score)
    ap = _per_label(truth, probability, columns, average_precision_score)
    sub_truth, sub_probability = truth[:, columns].ravel(), probability[:, columns].ravel()
    metrics: dict[str, float] = {
        "images": float(len(truth)),
        "represented_labels": float(len(columns)),
        "macro_auc_represented": float(np.nanmean(auc)) if np.isfinite(auc).any() else float("nan"),
        "micro_auc_represented": float(roc_auc_score(sub_truth, sub_probability)) if 0 < sub_truth.sum() < len(sub_truth) else float("nan"),
        "macro_ap_represented": float(np.nanmean(ap)) if np.isfinite(ap).any() else float("nan"),
        "micro_ap_represented": float(average_precision_score(sub_truth, sub_probability)) if sub_truth.sum() else float("nan"),
        "chance_macro_ap_represented": float(truth[:, columns].mean(axis=0).mean()) if len(columns) else float("nan"),
        "top1_accuracy_represented_only": _topk_accuracy(truth, probability, 1, columns),
    }
    for k in top_k:
        metrics[f"top{k}_accuracy"] = _topk_accuracy(truth, probability, k)
    if thresholds is not None:
        predicted = probability >= np.asarray(thresholds)[None, :]
        sensitivity = [predicted[truth[:, c] == 1, c].mean() for c in columns]
        metrics["macro_sensitivity_represented"] = float(np.mean(sensitivity)) if sensitivity else float("nan")
        unrepresented = np.setdiff1d(np.arange(truth.shape[1]), columns)
        metrics["off_target_alarm_rate"] = float(predicted[:, unrepresented].mean()) if len(unrepresented) else float("nan")
        metrics["mean_positive_calls_per_image"] = float(predicted.sum(axis=1).mean())
    return metrics


def per_label_table(
    truth: np.ndarray, probability: np.ndarray, labels: list[str], thresholds: np.ndarray | None = None
) -> pd.DataFrame:
    truth = np.asarray(truth, dtype=np.uint8)
    rows = []
    for column, label in enumerate(labels):
        y, p = truth[:, column], probability[:, column]
        positives = int(y.sum())
        row = {
            "label": label,
            "represented": positives > 0,
            "positives": positives,
            "prevalence": positives / len(y),
            "roc_auc": roc_auc_score(y, p) if 0 < positives < len(y) else np.nan,
            "average_precision": average_precision_score(y, p) if positives else np.nan,
        }
        row["ap_lift_over_chance"] = row["average_precision"] / row["prevalence"] if positives else np.nan
        if thresholds is not None:
            called = p >= thresholds[column]
            if positives:
                tp = int(called[y == 1].sum())
                lower, upper = wilson_interval(tp, positives)
                row.update({"threshold": thresholds[column], "sensitivity": tp / positives, "sensitivity_ci_lower": lower, "sensitivity_ci_upper": upper})
            else:
                row.update({"threshold": thresholds[column], "off_target_alarm_rate": float(called.mean())})
        rows.append(row)
    return pd.DataFrame(rows)


def confusion_among_represented(truth: np.ndarray, probability: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Rows: true label; columns: highest-scoring label among the represented labels."""
    columns = represented_columns(truth)
    names = [labels[c] for c in columns]
    true_index = truth[:, columns].argmax(axis=1)
    predicted_index = probability[:, columns].argmax(axis=1)
    table = pd.crosstab(
        pd.Categorical([names[i] for i in true_index], categories=names),
        pd.Categorical([names[i] for i in predicted_index], categories=names),
        dropna=False,
    )
    table.index.name, table.columns.name = "true_label", "top_scoring_represented_label"
    return table


def group_resample_indices(groups: np.ndarray, replicates: int, seed: int) -> list[np.ndarray]:
    """Resample whole groups with replacement; duplicate images always move together."""
    groups = np.asarray(groups).astype(str)
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))]) for _ in range(replicates)]


def bootstrap_intervals(
    truth: np.ndarray, probability: np.ndarray, samples: list[np.ndarray], metrics: tuple[str, ...] = HEADLINE_METRICS
) -> dict[str, tuple[float, float]]:
    draws = {name: [] for name in metrics}
    for index in samples:
        values = external_metrics(truth[index], probability[index])
        for name in metrics:
            draws[name].append(values[name])
    return {name: tuple(np.nanpercentile(values, [2.5, 97.5]).tolist()) for name, values in draws.items()}


def paired_difference(
    truth: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    samples: list[np.ndarray],
    metrics: tuple[str, ...] = HEADLINE_METRICS,
) -> pd.DataFrame:
    estimate_first, estimate_second = external_metrics(truth, first), external_metrics(truth, second)
    draws = {name: [] for name in metrics}
    for index in samples:
        a, b = external_metrics(truth[index], first[index]), external_metrics(truth[index], second[index])
        for name in metrics:
            draws[name].append(a[name] - b[name])
    rows = []
    for name in metrics:
        distribution = np.asarray(draws[name], dtype=float)
        lower, upper = np.nanpercentile(distribution, [2.5, 97.5])
        rows.append(
            {
                "metric": name,
                "first": estimate_first[name],
                "second": estimate_second[name],
                "difference": estimate_first[name] - estimate_second[name],
                "ci_lower": lower,
                "ci_upper": upper,
                "bootstrap_p": bootstrap_two_sided_p_value(distribution),
            }
        )
    return pd.DataFrame(rows)


def first_of_each_group(groups: np.ndarray) -> np.ndarray:
    """Indices keeping one image per duplicate-content group (unique-image sensitivity analysis)."""
    _, first = np.unique(np.asarray(groups).astype(str), return_index=True)
    return np.sort(first)
