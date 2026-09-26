from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _logit(probability: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), eps, 1 - eps)
    return np.log(clipped / (1 - clipped))


def fit_platt(
    truth: np.ndarray,
    probability: np.ndarray,
    labels: list[str],
    *,
    l2: float = 1e-3,
) -> pd.DataFrame:
    """Fit sigmoid(slope * logit(p) + offset) separately for each label.

    A small L2 penalty pulls each fit toward the identity (slope 1, offset 0), which keeps
    labels with few positives stable. Slopes are kept positive, so each label's case ranking,
    and therefore its AP and ROC-AUC, is unchanged. Labels without both classes stay identity.
    """
    truth = np.asarray(truth, dtype=float)
    logits = _logit(probability)
    rows = []
    for index, label in enumerate(labels):
        y, z = truth[:, index], logits[:, index]
        if y.min() == y.max():
            rows.append({"label": label, "slope": 1.0, "offset": 0.0, "fitted": False})
            continue

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            log_slope, offset = params
            slope = np.exp(log_slope)
            score = slope * z + offset
            probability_hat = 1 / (1 + np.exp(-score))
            loss = np.mean(np.logaddexp(0, score) - y * score)
            residual = probability_hat - y
            gradient_slope = np.mean(residual * z) * slope + 2 * l2 * (slope - 1) * slope
            gradient_offset = np.mean(residual) + 2 * l2 * offset
            return loss + l2 * ((slope - 1) ** 2 + offset**2), np.array([gradient_slope, gradient_offset])

        result = minimize(objective, np.zeros(2), jac=True, method="L-BFGS-B")
        rows.append({"label": label, "slope": float(np.exp(result.x[0])), "offset": float(result.x[1]), "fitted": True})
    return pd.DataFrame(rows)


def apply_platt(probability: np.ndarray, parameters: pd.DataFrame, labels: list[str]) -> np.ndarray:
    table = parameters.set_index("label").loc[labels]
    score = table["slope"].to_numpy() * _logit(probability) + table["offset"].to_numpy()
    return 1 / (1 + np.exp(-score))
