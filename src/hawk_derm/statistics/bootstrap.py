from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from joblib import Parallel, delayed, parallel_config

from hawk_derm.evaluation.metrics import safe_auc, safe_average_precision


def macro_metric(y_true: np.ndarray, probabilities: np.ndarray, metric: str) -> float:
    if metric == "roc_auc_micro":
        return safe_auc(y_true.ravel(), probabilities.ravel())
    if metric == "pr_auc_micro":
        return safe_average_precision(y_true.ravel(), probabilities.ravel())
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


def _ranges(length: int, workers: int) -> list[tuple[int, int]]:
    chunks = min(length, max(1, workers * 2))
    boundaries = np.linspace(0, length, chunks + 1, dtype=int)
    return [(int(start), int(stop)) for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True) if stop > start]


def _case_bootstrap_chunk(
    indices: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metric: str,
) -> np.ndarray:
    return np.asarray([macro_metric(y_true[index], probabilities[index], metric) for index in indices], dtype=float)


def _paired_bootstrap_chunk(
    indices: np.ndarray,
    y_true: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    metric: str,
) -> np.ndarray:
    return np.asarray(
        [
            macro_metric(y_true[index], first[index], metric) - macro_metric(y_true[index], second[index], metric)
            for index in indices
        ],
        dtype=float,
    )


def _paired_permutation_chunk(
    swaps: np.ndarray,
    y_true: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    metric: str,
) -> np.ndarray:
    values = np.empty(len(swaps), dtype=float)
    for position, swap in enumerate(swaps):
        permuted_first = np.where(swap[:, None], second, first)
        permuted_second = np.where(swap[:, None], first, second)
        values[position] = macro_metric(y_true, permuted_first, metric) - macro_metric(y_true, permuted_second, metric)
    return values


def _parallel_chunks(function: Callable[..., np.ndarray], values: np.ndarray, workers: int, *args: object) -> np.ndarray:
    ranges = _ranges(len(values), workers)
    if workers == 1 or len(ranges) == 1:
        return function(values, *args)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        chunks = Parallel(n_jobs=min(workers, len(ranges)), max_nbytes="10M", mmap_mode="r")(
            delayed(function)(values[start:stop], *args) for start, stop in ranges
        )
    return np.concatenate(chunks)


def case_bootstrap(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    metric: str,
    replicates: int,
    seed: int,
    workers: int = 1,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    count = len(y_true)
    indices = rng.integers(0, count, size=(replicates, count))
    distribution = _parallel_chunks(_case_bootstrap_chunk, indices, workers, y_true, probabilities, metric)
    return percentile_interval(distribution, macro_metric(y_true, probabilities, metric))


def paired_case_bootstrap(
    y_true: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    metric: str,
    replicates: int,
    seed: int,
    workers: int = 1,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    count = len(y_true)
    indices = rng.integers(0, count, size=(replicates, count))
    distribution = _parallel_chunks(_paired_bootstrap_chunk, indices, workers, y_true, first, second, metric)
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
    workers: int = 1,
) -> tuple[float, np.ndarray]:
    """Swap complete case-level prediction vectors between models under the paired null."""
    rng = np.random.default_rng(seed)
    observed = macro_metric(y_true, first, metric) - macro_metric(y_true, second, metric)
    swaps = rng.random((replicates, len(y_true))) < 0.5
    distribution = _parallel_chunks(_paired_permutation_chunk, swaps, workers, y_true, first, second, metric)
    finite = distribution[np.isfinite(distribution)]
    p_value = (np.sum(np.abs(finite) >= abs(observed)) + 1) / (len(finite) + 1) if len(finite) else float("nan")
    return float(p_value), distribution
