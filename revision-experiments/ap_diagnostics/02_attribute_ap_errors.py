"""Attribute each condition's AP deficit to high-ranked negative case pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import HERE, LABELS, MODELS, SCORES, TRUTH, load, locked_ids


OUT = HERE / "outputs/error_attribution"


def negative_burdens(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, float]:
    """Split 1-AP over negatives at or above each positive's tied score group."""
    if not 0 < y.sum() < len(y):
        raise ValueError("Each condition needs positives and negatives")
    unique, ascending_group = np.unique(score, return_inverse=True)
    group = len(unique) - 1 - ascending_group  # Descending score; ties stay grouped.
    positives = np.bincount(group, weights=y, minlength=len(unique))
    negatives = np.bincount(group, weights=1 - y, minlength=len(unique))
    total = np.cumsum(positives + negatives)
    precision = np.cumsum(positives) / total
    ap = float(np.sum(positives * precision) / y.sum())
    group_burden = np.cumsum((positives / (y.sum() * total))[::-1])[::-1]
    burden = np.where(y == 0, group_burden[group], 0.0)
    if not np.isclose(ap, average_precision_score(y, score), atol=1e-12):
        raise AssertionError("Tie-aware AP does not match scikit-learn")
    if not np.isclose(burden.sum(), 1 - ap, atol=1e-12):
        raise AssertionError("Negative burdens do not sum to the AP deficit")
    return burden, ap


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ids = locked_ids()
    frames = {name: load(path, ids) for name, path in MODELS.items()}
    truth = frames["siglip2_mil"][TRUTH].to_numpy(int)
    for name, frame in frames.items():
        if not np.array_equal(truth, frame[TRUTH].to_numpy(int)):
            raise ValueError(f"Ground-truth labels differ: {name}")
    all_zero = truth.sum(axis=1) == 0
    detail, condition_rows = [], []
    for name, frame in frames.items():
        scores = frame[SCORES].to_numpy(float)
        for col, label in enumerate(LABELS):
            burdens, ap = negative_burdens(truth[:, col], scores[:, col])
            condition_rows.append(
                {
                    "model": name,
                    "condition": label,
                    "positives": int(truth[:, col].sum()),
                    "ap": ap,
                    "ap_deficit": 1 - ap,
                }
            )
            for idx in np.flatnonzero(truth[:, col] == 0):
                detail.append(
                    {
                        "model": name,
                        "case_id": ids[idx],
                        "condition": label,
                        "negative_group": "no_target" if all_zero[idx] else "other_target",
                        "score": scores[idx, col],
                        "ap_deficit_burden": burdens[idx],
                    }
                )
    pairs = pd.DataFrame(detail)
    pairs.to_csv(OUT / "negative_case_condition_burdens.csv", index=False)
    pd.DataFrame(condition_rows).to_csv(OUT / "per_condition_ap.csv", index=False)
    group = (
        pairs.groupby(["model", "negative_group"])
        .agg(
            negative_pairs=("ap_deficit_burden", "size"),
            total_ap_deficit_burden=("ap_deficit_burden", "sum"),
            mean_burden_per_negative_pair=("ap_deficit_burden", "mean"),
        )
        .reset_index()
    )
    group["share_of_model_ap_deficit"] = group["total_ap_deficit_burden"] / group.groupby("model")[
        "total_ap_deficit_burden"
    ].transform("sum")
    group.to_csv(OUT / "negative_group_summary.csv", index=False)
    cases = pairs.groupby(["model", "case_id", "negative_group"], as_index=False).agg(
        conditions_negative=("condition", "size"), total_burden=("ap_deficit_burden", "sum")
    )
    cases.sort_values(["model", "total_burden"], ascending=[True, False]).groupby("model", group_keys=False).head(
        25
    ).to_csv(OUT / "top_25_cases_per_model.csv", index=False)
    print(group.round(5).to_string(index=False))
    print(f"Locked test: {len(ids)} cases; {len(pairs)} negative case-condition pairs across {len(MODELS)} models")


if __name__ == "__main__":
    main()
