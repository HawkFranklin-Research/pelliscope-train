"""Fit regularized label offsets and a shared slope on validation only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold

from common import HERE, LABELS, MODELS, ROOT, SCORES, TRUTH, load, locked_ids


OUT = HERE / "outputs/calibration"
VALIDATION = {
    "siglip2_mil": ROOT / "artifacts/models/mil/siglip2_so400m/canonical/ensemble/validation_predictions.csv",
    "siglip2_rf": ROOT / "artifacts/models/classical/siglip2_so400m/random_forest/validation_case_predictions.csv",
}
PENALTY = 0.01  # Fixed before inspecting test performance; no test tuning.


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "macro_ap": float(average_precision_score(y, p, average="macro")),
        "micro_ap": float(average_precision_score(y, p, average="micro")),
        "macro_roc_auc": float(roc_auc_score(y, p, average="macro")),
        "micro_roc_auc": float(roc_auc_score(y, p, average="micro")),
    }


def fit(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    x = logit(np.clip(p, 1e-12, 1 - 1e-12))
    count = x.shape[1]

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        slope, offsets = theta[0], theta[1:]
        z = slope * x + offsets
        residual = expit(z) - y
        loss = np.mean(np.logaddexp(0, z) - y * z)
        loss += PENALTY * ((slope - 1) ** 2 + np.mean(offsets**2))
        grad = np.r_[
            np.mean(residual * x) + 2 * PENALTY * (slope - 1),
            np.mean(residual, axis=0) / count + 2 * PENALTY * offsets / count,
        ]
        return float(loss), grad

    result = minimize(
        objective,
        np.r_[1.0, np.zeros(count)],
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.25, 4.0), *[(-8.0, 8.0)] * count],
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def transform(p: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return expit(theta[0] * logit(np.clip(p, 1e-12, 1 - 1e-12)) + theta[1:])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    validation = {name: load(path) for name, path in VALIDATION.items()}
    shared = sorted(set.intersection(*(set(frame["case_id"]) for frame in validation.values())))
    if len(shared) < 700:
        raise ValueError("Unexpectedly small validation overlap")
    validation = {name: load(VALIDATION[name], shared) for name in validation}
    y_val = validation["siglip2_mil"][TRUTH].to_numpy(int)
    if not np.array_equal(y_val, validation["siglip2_rf"][TRUTH].to_numpy(int)):
        raise ValueError("Validation labels disagree between models")
    test = {name: load(path, locked_ids()) for name, path in MODELS.items() if name in VALIDATION}
    y_test = test["siglip2_mil"][TRUTH].to_numpy(int)
    if not np.array_equal(y_test, test["siglip2_rf"][TRUTH].to_numpy(int)):
        raise ValueError("Locked test labels disagree between models")

    rows, parameters = [], []
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    for name in VALIDATION:
        p_val = validation[name][SCORES].to_numpy(float)
        p_test = test[name][SCORES].to_numpy(float)
        oof = np.empty_like(p_val)
        for train_idx, held_idx in folds.split(p_val):
            theta = fit(y_val[train_idx], p_val[train_idx])
            oof[held_idx] = transform(p_val[held_idx], theta)
        theta = fit(y_val, p_val)
        calibrated = transform(p_test, theta)
        for split, truth, original, updated in (
            ("validation_oof", y_val, p_val, oof),
            ("locked_test", y_test, p_test, calibrated),
        ):
            for variant, values in (("original", original), ("calibrated", updated)):
                rows.append(
                    {"model": name, "split": split, "variant": variant, "cases": len(truth), **metrics(truth, values)}
                )
        # A single fitted monotone map per label preserves every test per-label rank.
        for col in range(len(LABELS)):
            before = average_precision_score(y_test[:, col], p_test[:, col])
            after = average_precision_score(y_test[:, col], calibrated[:, col])
            if not np.isclose(before, after, atol=1e-12):
                raise AssertionError(f"Per-condition AP changed: {name}, {LABELS[col]}")
            parameters.append(
                {"model": name, "condition": LABELS[col], "shared_slope": theta[0], "offset": theta[col + 1]}
            )
        pd.DataFrame(
            {"case_id": test[name]["case_id"], **{col: calibrated[:, idx] for idx, col in enumerate(SCORES)}}
        ).to_csv(OUT / f"{name}_locked_test_calibrated_scores.csv", index=False)

    pd.DataFrame(rows).to_csv(OUT / "metric_comparison.csv", index=False)
    pd.DataFrame(parameters).to_csv(OUT / "parameters.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"Validation overlap: {len(shared)} cases; locked test: {len(y_test)} cases")


if __name__ == "__main__":
    main()
