#!/usr/bin/env python3
"""Figures pairing the internal locked test set with each external cohort.

Reads the tables written by 82_evaluate_external.py plus the internal prediction files.
Only figure types that are meaningful for one-image pseudo-cases are produced; attention,
photo-count and learning-curve figures stay internal-only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from _common import load_context, mil_run_root  # noqa: E402

from hawk_derm.config import load_yaml, path_from  # noqa: E402
from hawk_derm.data.external import load_external_cohorts  # noqa: E402
from hawk_derm.evaluation.predictions import arrays_from_prediction_frame  # noqa: E402
from hawk_derm.figures.style import manuscript_style  # noqa: E402
from hawk_derm.io import write_json  # noqa: E402

CLASSIFIER_ORDER = ["mil_ensemble", "random_forest", "gradient_boosting", "logistic", "svm_linear", "knn"]


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def grid_heatmaps(grid: pd.DataFrame, output: Path, cohort_name: str) -> Path:
    panels = [
        ("internal_macro_auc_same_labels", "Internal test (same labels)"),
        ("macro_auc_represented", f"External: {cohort_name}"),
    ]
    data = grid[grid["classifier"].isin(CLASSIFIER_ORDER)]
    encoders = sorted(data["encoder"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.55 * len(encoders) + 2.5), constrained_layout=True)
    for axis, (column, title) in zip(axes, panels, strict=True):
        table = data.pivot(index="encoder", columns="classifier", values=column).reindex(index=encoders, columns=CLASSIFIER_ORDER)
        image = axis.imshow(table.to_numpy(dtype=float), vmin=0.5, vmax=1.0, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(CLASSIFIER_ORDER)), CLASSIFIER_ORDER, rotation=35, ha="right")
        axis.set_yticks(range(len(encoders)), encoders)
        for (row, col), value in np.ndenumerate(table.to_numpy(dtype=float)):
            if np.isfinite(value):
                axis.text(col, row, f"{value:.3f}", ha="center", va="center", color="white" if value < 0.8 else "black", fontsize=8)
        axis.set_title(f"{title}\nmacro ROC-AUC")
    fig.colorbar(image, ax=axes, shrink=0.8)
    return save(fig, output / "external_grid_heatmap.png")


def per_label_auc(internal_path: Path, per_label: pd.DataFrame, labels: list[str], model: str, output: Path) -> Path | None:
    external = per_label[(per_label["model"] == model) & per_label["represented"]].set_index("label")
    if external.empty or not internal_path.is_file():
        return None
    truth, probability = arrays_from_prediction_frame(pd.read_csv(internal_path), labels)
    names = external.index.tolist()
    internal = [
        roc_auc_score(truth[:, labels.index(name)], probability[:, labels.index(name)])
        if 0 < truth[:, labels.index(name)].sum() < len(truth) else np.nan
        for name in names
    ]
    positions = np.arange(len(names))
    fig, axis = plt.subplots(figsize=(max(8, 0.8 * len(names)), 4.5), constrained_layout=True)
    axis.bar(positions - 0.2, internal, 0.4, label="Internal test")
    axis.bar(positions + 0.2, external["roc_auc"], 0.4, label="External")
    axis.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    axis.set_xticks(positions, names, rotation=35, ha="right")
    axis.set_ylim(0.3, 1.0)
    axis.set_ylabel("ROC-AUC")
    axis.set_title(f"{model}: per-condition ROC-AUC")
    axis.legend(frameon=False)
    return save(fig, output / f"external_per_label_auc_{model}.png")


def sensitivity(operating_points: Path, per_label: pd.DataFrame, model: str, output: Path) -> Path | None:
    external = per_label[(per_label["model"] == model) & per_label["represented"]].set_index("label")
    if external.empty or "sensitivity" not in external or not operating_points.is_file():
        return None
    internal = pd.read_csv(operating_points).set_index("label").reindex(external.index)
    positions = np.arange(len(external))
    fig, axis = plt.subplots(figsize=(max(8, 0.8 * len(external)), 4.5), constrained_layout=True)
    for offset, frame, name in ((-0.2, internal, "Internal test"), (0.2, external, "External")):
        values = np.nan_to_num(frame["sensitivity"].to_numpy(dtype=float), nan=0.0)
        lower = frame["sensitivity_ci_lower"].to_numpy(dtype=float) if "sensitivity_ci_lower" in frame else values
        upper = frame["sensitivity_ci_upper"].to_numpy(dtype=float) if "sensitivity_ci_upper" in frame else values
        lower_err = np.maximum(0.0, np.nan_to_num(values - lower, nan=0.0))
        upper_err = np.maximum(0.0, np.nan_to_num(upper - values, nan=0.0))
        errors = np.vstack([lower_err, upper_err])
        axis.bar(positions + offset, values, 0.4, yerr=errors, capsize=3, label=name)
    axis.set_xticks(positions, external.index, rotation=35, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Sensitivity at locked threshold (95% Wilson CI)")
    axis.set_title(f"{model}: sensitivity at validation-selected thresholds")
    axis.legend(frameon=False)
    return save(fig, output / f"external_sensitivity_{model}.png")


def topk(grid: pd.DataFrame, output: Path) -> Path:
    data = grid[grid["classifier"].isin(["mil_ensemble", "random_forest"])].sort_values(["encoder", "classifier"])
    names = (data["encoder"] + "\n" + data["classifier"]).tolist()
    positions = np.arange(len(names))
    fig, axis = plt.subplots(figsize=(max(9, 0.7 * len(names)), 4.8), constrained_layout=True)
    axis.bar(positions - 0.3, data["internal_top3_hit"], 0.2, label="Internal top-3")
    axis.bar(positions - 0.1, data["top3_accuracy"], 0.2, label="External top-3")
    axis.bar(positions + 0.1, data["internal_top1_hit"], 0.2, label="Internal top-1")
    axis.bar(positions + 0.3, data["top1_accuracy"], 0.2, label="External top-1")
    axis.set_xticks(positions, names, rotation=45, ha="right", fontsize=8)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Correct diagnosis in top-k")
    axis.legend(frameon=False, ncol=4)
    return save(fig, output / "external_topk.png")


def confusion(path: Path, model: str, output: Path) -> Path | None:
    if not path.is_file():
        return None
    table = pd.read_csv(path).set_index("true_label")
    counts = table.to_numpy(dtype=float)
    shares = counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None)
    fig, axis = plt.subplots(figsize=(0.6 * len(table) + 3, 0.6 * len(table) + 2.5), constrained_layout=True)
    axis.imshow(shares, vmin=0, vmax=1, cmap="Blues")
    axis.set_xticks(range(len(table.columns)), table.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(table.index)), table.index)
    for (row, col), value in np.ndenumerate(counts):
        axis.text(col, row, f"{int(value)}", ha="center", va="center", fontsize=8, color="white" if shares[row, col] > 0.5 else "black")
    axis.set_xlabel("Top-scoring represented label")
    axis.set_ylabel("True label")
    axis.set_title(f"{model}: confusion among represented labels")
    return save(fig, output / f"external_confusion_{model}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--run-tag", default="canonical")
    parser.add_argument("--cohort", default=None)
    args = parser.parse_args()
    config = load_context(args.config)
    labels = config["labels"]
    style = load_yaml("configs/figures.yaml").get("style", {})
    production = load_yaml("configs/production.yaml").get("production_models", {})
    encoders = [entry["encoder"] for entry in production.values() if entry.get("encoder")] or ["siglip2_so400m"]
    for cohort in load_external_cohorts(config):
        if args.cohort not in (None, cohort.name):
            continue
        tables = path_from(config, "reports_dir") / "external" / cohort.name
        if not (tables / "model_grid.csv").is_file():
            print(f"[external] {cohort.name}: run 82_evaluate_external.py first")
            continue
        grid, per_label = pd.read_csv(tables / "model_grid.csv"), pd.read_csv(tables / "per_label.csv")
        output = tables / "figures"
        written = []
        with manuscript_style(style):
            written.append(grid_heatmaps(grid, output, cohort.display_name))
            written.append(topk(grid, output))
            for encoder in encoders:
                model = f"{encoder}+mil_ensemble"
                root = mil_run_root(config, encoder, args.run_tag)
                written.append(per_label_auc(root / "ensemble" / "test_predictions.csv", per_label, labels, model, output))
                written.append(sensitivity(root / "thresholds" / "test_operating_points.csv", per_label, model, output))
                written.append(confusion(tables / "confusion" / f"{model}.csv", model, output))
        written = [str(path) for path in written if path is not None]
        write_json(output / "external_figure_manifest.json", {"cohort": cohort.name, "figures": written, "complete": True})
        print(f"[external] {cohort.name}: {len(written)} figures -> {output}")


if __name__ == "__main__":
    main()
