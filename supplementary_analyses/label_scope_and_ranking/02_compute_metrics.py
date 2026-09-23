#!/usr/bin/env python3
"""Calculate Top-k and nested prevalence-scope metrics from audited predictions."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score

from analysis_lib import (
    CONFIG,
    HERE,
    LABELS,
    LABEL_SLUGS,
    REPO_ROOT,
    arrays,
    bootstrap_interval,
    curve_values,
    safe_metrics,
)


OUTPUT = HERE / "outputs"
MANIFESTS = OUTPUT / "manifests"
TABLES = OUTPUT / "tables"
PLOT_DATA = OUTPUT / "plot_data"


def load_registry() -> pd.DataFrame:
    summary = MANIFESTS / "audit_summary.json"
    if not summary.exists() or '"complete": true' not in summary.read_text().lower():
        raise RuntimeError("The prediction audit has not completed successfully")
    return pd.read_csv(MANIFESTS / "model_registry.csv")


def load_prediction(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(HERE / relative_path)


def training_prevalence() -> pd.DataFrame:
    cases = pd.read_csv(REPO_ROOT / "data/manifests/case_manifest.csv")
    splits = pd.read_csv(REPO_ROOT / "data/splits/split_manifest_v1.csv")
    cases["case_id"] = cases["case_id"].astype(str)
    splits["case_id"] = splits["case_id"].astype(str)
    train_ids = set(splits.loc[splits["split"] == "train", "case_id"])
    train = cases[cases["case_id"].isin(train_ids)]
    rows = []
    for order, (label, name) in enumerate(zip(LABELS, LABEL_SLUGS, strict=True)):
        column = f"is_{name}"
        if column not in train.columns:
            raise KeyError(f"Training manifest is missing {column}")
        positives = int(train[column].sum())
        rows.append(
            {
                "label": label,
                "label_slug": name,
                "configured_order": order,
                "training_positives": positives,
                "training_prevalence": positives / len(train),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["training_positives", "configured_order"], ascending=[False, True]
    )
    frame.insert(0, "prevalence_rank", range(1, len(frame) + 1))
    return frame


def topk_values(truth: np.ndarray, probability: np.ndarray, k_values: list[int]) -> dict[tuple[int, str], float]:
    positive = truth.sum(axis=1) > 0
    truth = truth[positive]
    probability = probability[positive]
    order = np.argsort(-probability, axis=1)
    output: dict[tuple[int, str], float] = {}
    for k in k_values:
        top = order[:, :k]
        retrieved = np.take_along_axis(truth, top, axis=1).sum(axis=1)
        output[(k, "hit_rate")] = float(np.mean(retrieved > 0))
        output[(k, "recall_at_k")] = float(np.mean(retrieved / truth.sum(axis=1)))
        output[(k, "precision_at_k")] = float(np.mean(retrieved / k))
    return output


def bootstrap_topk(
    model_id: str,
    truth: np.ndarray,
    probability: np.ndarray,
    k_values: list[int],
    replicates: int,
    seed: int,
) -> list[dict]:
    positive = truth.sum(axis=1) > 0
    truth = truth[positive]
    probability = probability[positive]
    rng = np.random.default_rng(seed)
    distributions = {(k, metric): [] for k in k_values for metric in ("hit_rate", "recall_at_k", "precision_at_k")}
    for _ in range(replicates):
        sample = rng.integers(0, len(truth), len(truth))
        values = topk_values(truth[sample], probability[sample], k_values)
        for key, value in values.items():
            distributions[key].append(value)
    rows = []
    for (k, metric), values in distributions.items():
        lower, upper = bootstrap_interval(np.asarray(values))
        rows.append({"model_id": model_id, "k": k, "metric": metric, "ci_95_lower": lower, "ci_95_upper": upper})
    return rows


def bootstrap_scope(
    model_id: str,
    scope_n: int,
    truth: np.ndarray,
    probability: np.ndarray,
    replicates: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    names = ("roc_auc_macro", "roc_auc_micro", "pr_auc_macro", "pr_auc_micro")
    distributions = {name: [] for name in names}
    for _ in range(replicates):
        sample = rng.integers(0, len(truth), len(truth))
        values = safe_metrics(truth[sample], probability[sample])
        for name in names:
            distributions[name].append(values[name])
    rows = []
    for name, values in distributions.items():
        lower, upper = bootstrap_interval(np.asarray(values))
        rows.append(
            {
                "model_id": model_id,
                "scope_n": scope_n,
                "metric": name,
                "ci_95_lower": lower,
                "ci_95_upper": upper,
            }
        )
    return rows


def bootstrap_per_condition(
    model_id: str,
    truth: np.ndarray,
    probability: np.ndarray,
    replicates: int,
    seed: int,
) -> list[dict]:
    """Case-bootstrap per-condition ROC-AUC intervals for one model."""
    rng = np.random.default_rng(seed)
    distributions = np.full((replicates, truth.shape[1]), np.nan, dtype=float)
    for replicate in range(replicates):
        sample = rng.integers(0, len(truth), len(truth))
        sampled_truth = truth[sample]
        sampled_probability = probability[sample]
        for index in range(truth.shape[1]):
            if np.unique(sampled_truth[:, index]).size == 2:
                distributions[replicate, index] = roc_auc_score(
                    sampled_truth[:, index], sampled_probability[:, index]
                )
    rows = []
    for index, name in enumerate(LABEL_SLUGS):
        lower, upper = bootstrap_interval(distributions[:, index])
        rows.append(
            {
                "model_id": model_id,
                "label_slug": name,
                "roc_auc_ci_95_lower": lower,
                "roc_auc_ci_95_upper": upper,
            }
        )
    return rows


def bootstrap_curve(
    scope_n: int,
    curve: str,
    averaging: str,
    truth: np.ndarray,
    probability: np.ndarray,
    grid: np.ndarray,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    point = curve_values(truth, probability, curve, averaging, grid)
    samples = []
    for _ in range(replicates):
        selection = rng.integers(0, len(truth), len(truth))
        samples.append(curve_values(truth[selection], probability[selection], curve, averaging, grid))
    distribution = np.asarray(samples)
    return pd.DataFrame(
        {
            "scope_n": scope_n,
            "curve": curve,
            "averaging": averaging,
            "x": grid,
            "value": point,
            "ci_95_lower": np.nanquantile(distribution, 0.025, axis=0),
            "ci_95_upper": np.nanquantile(distribution, 0.975, axis=0),
        }
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    PLOT_DATA.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    prevalence = training_prevalence()
    prevalence.to_csv(MANIFESTS / "training_label_prevalence.csv", index=False)
    scopes = [int(value) for value in CONFIG["canonical"]["label_scopes"]]
    k_values = [int(value) for value in CONFIG["canonical"]["top_k"]]
    ordered_slugs = prevalence["label_slug"].tolist()
    scope_rows = []
    for scope_n in scopes:
        for position, name in enumerate(ordered_slugs[:scope_n], start=1):
            scope_rows.append({"scope_n": scope_n, "within_scope_rank": position, "label_slug": name, "label": LABELS[LABEL_SLUGS.index(name)]})
    pd.DataFrame(scope_rows).to_csv(MANIFESTS / "nested_label_scopes.csv", index=False)

    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    topk_rows: list[dict] = []
    scope_metric_rows: list[dict] = []
    per_condition_rows: list[dict] = []
    for item in registry.itertuples(index=False):
        frame = load_prediction(item.prediction_path)
        truth, probability = arrays(frame)
        predictions[item.model_id] = (truth, probability)
        for (k, metric), value in topk_values(truth, probability, k_values).items():
            topk_rows.append(
                {
                    "model_id": item.model_id,
                    "encoder": item.encoder,
                    "classifier": item.classifier,
                    "target_positive_cases": int((truth.sum(axis=1) > 0).sum()),
                    "k": k,
                    "metric": metric,
                    "estimate": value,
                }
            )
        for index, name in enumerate(LABEL_SLUGS):
            target = truth[:, index]
            per_condition_rows.append(
                {
                    "model_id": item.model_id,
                    "encoder": item.encoder,
                    "classifier": item.classifier,
                    "label": LABELS[index],
                    "label_slug": name,
                    "test_positives": int(target.sum()),
                    "test_prevalence": float(target.mean()),
                    "roc_auc": float(roc_auc_score(target, probability[:, index])) if np.unique(target).size == 2 else np.nan,
                    "pr_auc": float(average_precision_score(target, probability[:, index])) if target.sum() else np.nan,
                }
            )
        for scope_n in scopes:
            indices = [LABEL_SLUGS.index(name) for name in ordered_slugs[:scope_n]]
            scoped_truth = truth[:, indices]
            scoped_probability = probability[:, indices]
            values = safe_metrics(scoped_truth, scoped_probability)
            baseline = float(scoped_truth.mean())
            row = {
                "model_id": item.model_id,
                "encoder": item.encoder,
                "classifier": item.classifier,
                "scope_n": scope_n,
                "selected_positive_labels": int(scoped_truth.sum()),
                "scope_positive_cases": int((scoped_truth.sum(axis=1) > 0).sum()),
                "scope_all_zero_cases": int((scoped_truth.sum(axis=1) == 0).sum()),
                "pr_no_skill_baseline": baseline,
                **values,
            }
            row["pr_macro_lift_ratio"] = values["pr_auc_macro"] / baseline
            row["pr_micro_lift_ratio"] = values["pr_auc_micro"] / baseline
            row["pr_macro_normalized_gain"] = (values["pr_auc_macro"] - baseline) / (1 - baseline)
            row["pr_micro_normalized_gain"] = (values["pr_auc_micro"] - baseline) / (1 - baseline)
            scope_metric_rows.append(row)

    topk_frame = pd.DataFrame(topk_rows)
    scope_frame = pd.DataFrame(scope_metric_rows)
    per_condition_frame = pd.DataFrame(per_condition_rows)

    # Case-bootstrap uncertainty is calculated for the four principal comparisons.
    principal = list(CONFIG["principal_models"])
    replicates = int(CONFIG["statistics"]["metric_bootstrap_replicates"])
    base_seed = int(CONFIG["statistics"]["bootstrap_seed"])
    workers = min(8, os.cpu_count() or 1)
    topk_jobs = Parallel(n_jobs=workers, prefer="processes")
    topk_ci_nested = topk_jobs(
        delayed(bootstrap_topk)(model_id, *predictions[model_id], k_values, replicates, base_seed + index)
        for index, model_id in enumerate(principal)
    )
    topk_ci = pd.DataFrame([row for group in topk_ci_nested for row in group])
    topk_frame = topk_frame.merge(topk_ci, on=["model_id", "k", "metric"], how="left")

    scope_tasks = []
    for model_index, model_id in enumerate(principal):
        truth, probability = predictions[model_id]
        for scope_n in scopes:
            indices = [LABEL_SLUGS.index(name) for name in ordered_slugs[:scope_n]]
            scope_tasks.append(
                (model_id, scope_n, truth[:, indices], probability[:, indices], base_seed + 100 + model_index * 10 + scope_n)
            )
    scope_ci_nested = Parallel(n_jobs=workers, prefer="processes")(
        delayed(bootstrap_scope)(model_id, scope_n, truth, probability, replicates, seed)
        for model_id, scope_n, truth, probability, seed in scope_tasks
    )
    scope_ci = pd.DataFrame([row for group in scope_ci_nested for row in group])
    scope_ci_wide = scope_ci.pivot(index=["model_id", "scope_n"], columns="metric", values=["ci_95_lower", "ci_95_upper"])
    scope_ci_wide.columns = [f"{metric}_{bound}" for bound, metric in scope_ci_wide.columns]
    scope_ci_wide = scope_ci_wide.reset_index()
    scope_frame = scope_frame.merge(scope_ci_wide, on=["model_id", "scope_n"], how="left")

    condition_ci_nested = Parallel(n_jobs=workers, prefer="processes")(
        delayed(bootstrap_per_condition)(
            model_id,
            *predictions[model_id],
            replicates,
            base_seed + 500 + index,
        )
        for index, model_id in enumerate(principal)
    )
    condition_ci = pd.DataFrame([row for group in condition_ci_nested for row in group])
    per_condition_frame = per_condition_frame.merge(
        condition_ci, on=["model_id", "label_slug"], how="left"
    )

    topk_frame.to_csv(TABLES / "topk_ranking_metrics.csv", index=False)
    scope_frame.to_csv(TABLES / "label_scope_metrics.csv", index=False)
    per_condition_frame.to_csv(TABLES / "per_condition_metrics.csv", index=False)
    scope_frame.to_csv(PLOT_DATA / "label_scope_metric_trajectories.csv", index=False)

    # Curves and case-bootstrap ribbons for the primary model.
    primary_model = str(CONFIG["primary_model"])
    primary_truth, primary_probability = predictions[primary_model]
    grid = np.linspace(0, 1, 201)
    curve_replicates = int(CONFIG["statistics"]["curve_bootstrap_replicates"])
    curve_tasks = []
    for scope_n in scopes:
        indices = [LABEL_SLUGS.index(name) for name in ordered_slugs[:scope_n]]
        for curve in ("roc", "pr"):
            for averaging in ("macro", "micro"):
                curve_tasks.append(
                    (
                        scope_n,
                        curve,
                        averaging,
                        primary_truth[:, indices],
                        primary_probability[:, indices],
                        base_seed + 1000 + scope_n * 10 + (0 if curve == "roc" else 2) + (0 if averaging == "macro" else 1),
                    )
                )
    curve_frames = Parallel(n_jobs=workers, prefer="processes")(
        delayed(bootstrap_curve)(scope_n, curve, averaging, truth, probability, grid, curve_replicates, seed)
        for scope_n, curve, averaging, truth, probability, seed in curve_tasks
    )
    curves = pd.concat(curve_frames, ignore_index=True)
    baselines = scope_frame.loc[scope_frame["model_id"] == primary_model, ["scope_n", "pr_no_skill_baseline"]]
    curves = curves.merge(baselines, on="scope_n", how="left")
    curves.to_csv(PLOT_DATA / "cumulative_roc_pr_curves.csv", index=False)

    print(
        f"Calculated Top-k metrics and {len(scopes)} nested label scopes for {len(registry)} models; "
        f"bootstrap CIs completed for {len(principal)} principal models"
    )


if __name__ == "__main__":
    main()
