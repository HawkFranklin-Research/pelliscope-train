from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from hawk_derm.evaluation.metrics import safe_auc, safe_average_precision


def macro_metric(y_true: np.ndarray, probabilities: np.ndarray, metric: str) -> float:
    function: Callable[[np.ndarray, np.ndarray], float]
    if metric == "roc_auc":
        function = safe_auc
    elif metric == "pr_auc":
        function = safe_average_precision
    else:
        raise ValueError(metric)
    values = [function(y_true[:, index], probabilities[:, index]) for index in range(y_true.shape[1])]
    return float(np.nanmean(values))


@dataclass
class BootstrapResult:
    estimate: float
    lower: float
    upper: float
    distribution: np.ndarray


def percentile_interval(distribution: np.ndarray, estimate: float, alpha: float = 0.05) -> BootstrapResult:
    finite = distribution[np.isfinite(distribution)]
    if not len(finite):
        return BootstrapResult(estimate, float("nan"), float("nan"), distribution)
    lower, upper = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(estimate, float(lower), float(upper), distribution)


def case_bootstrap(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    metric: str,
    replicates: int,
    seed: int,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    count = len(y_true)
    distribution = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        indices = rng.integers(0, count, count)
        distribution[replicate] = macro_metric(y_true[indices], probabilities[indices], metric)
    return percentile_interval(distribution, macro_metric(y_true, probabilities, metric))


def paired_case_bootstrap(
    y_true: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    metric: str,
    replicates: int,
    seed: int,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    count = len(y_true)
    distribution = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        indices = rng.integers(0, count, count)
        distribution[replicate] = macro_metric(y_true[indices], first[indices], metric) - macro_metric(
            y_true[indices], second[indices], metric
        )
    estimate = macro_metric(y_true, first, metric) - macro_metric(y_true, second, metric)
    return percentile_interval(distribution, estimate)


def paired_case_permutation(
    y_true: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    metric: str,
    replicates: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    """Swap complete case-level prediction vectors between models under the paired null."""
    rng = np.random.default_rng(seed)
    observed = macro_metric(y_true, first, metric) - macro_metric(y_true, second, metric)
    distribution = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        swap = rng.random(len(y_true)) < 0.5
        permuted_first = first.copy()
        permuted_second = second.copy()
        permuted_first[swap] = second[swap]
        permuted_second[swap] = first[swap]
        distribution[replicate] = macro_metric(y_true, permuted_first, metric) - macro_metric(y_true, permuted_second, metric)
    finite = distribution[np.isfinite(distribution)]
    p_value = (np.sum(np.abs(finite) >= abs(observed)) + 1) / (len(finite) + 1) if len(finite) else float("nan")
    return float(p_value), distribution
