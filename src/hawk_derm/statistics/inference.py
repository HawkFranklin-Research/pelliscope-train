from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def wilson_interval(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = norm.ppf(1 - alpha / 2)
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half_width = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    return center - half_width, center + half_width


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def bootstrap_two_sided_p_value(distribution: np.ndarray) -> float:
    distribution = distribution[np.isfinite(distribution)]
    if not len(distribution):
        return float("nan")
    lower = (np.sum(distribution <= 0) + 1) / (len(distribution) + 1)
    upper = (np.sum(distribution >= 0) + 1) / (len(distribution) + 1)
    return float(min(1.0, 2 * min(lower, upper)))
