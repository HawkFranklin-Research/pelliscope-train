#!/usr/bin/env python3
"""Generate isolated supplementary figures from the audited prediction tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from analysis_lib import CLASSIFIERS, CONFIG, ENCODERS, HERE, LABELS, LABEL_SLUGS


OUTPUT = HERE / "outputs"
TABLES = OUTPUT / "tables"
PLOT_DATA = OUTPUT / "plot_data"
FIGURES = OUTPUT / "figures"

SCOPE_COLORS = {
    5: "#19323C",
    10: "#2E7D6E",
    15: "#7A9E9F",
    20: "#C06C4D",
    25: "#E2B33C",
}
MODEL_COLORS = {
    "siglip2_so400m__mil": "#19323C",
    "siglip2_so400m__random_forest": "#C06C4D",
    "derm_foundation__mil": "#2E7D6E",
    "derm_foundation__random_forest": "#7A4EAB",
}
MODEL_NAMES = {
    "siglip2_so400m__mil": "SigLIP2 + MIL ensemble",
    "siglip2_so400m__random_forest": "SigLIP2 + random forest",
    "derm_foundation__mil": "Derm Foundation + MIL ensemble",
    "derm_foundation__random_forest": "Derm Foundation + random forest",
}
ENCODER_NAMES = {
    "resnet50": "ResNet-50",
    "inception_v3": "Inception-v3",
    "bit50": "BiT-50",
    "vit_base": "ViT-Base",
    "clip_vitb32": "CLIP ViT-B/32",
    "derm_foundation": "Derm Foundation",
    "siglip2_so400m": "SigLIP2 SO400M",
}
CLASSIFIER_NAMES = {
    "gradient_boosting": "Gradient\nboosting",
    "svm_linear": "Linear SVM",
    "logistic": "Logistic",
    "mil": "MIL",
    "random_forest": "Random\nforest",
    "knn": "k-NN",
}
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "indigo_violet_coral_gold",
    ["#24124D", "#5F328B", "#A43A78", "#E66B5B", "#F6C85F"],
)


def style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 450,
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, filename: str) -> Path:
    path = FIGURES / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def topk_figure(topk: pd.DataFrame) -> Path:
    metrics = [
        ("hit_rate", "Hit@k", "Cases with ≥1 true diagnosis in Top-k"),
        ("recall_at_k", "Recall@k", "Fraction of true diagnoses recovered"),
        ("precision_at_k", "Precision@k", "Fraction of Top-k predictions that are true"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True, sharey=True, constrained_layout=True)
    principal = list(CONFIG["principal_models"])
    for axis, (metric, title, ylabel) in zip(axes, metrics, strict=True):
        for model_id in principal:
            rows = topk[(topk["model_id"] == model_id) & (topk["metric"] == metric)].sort_values("k")
            axis.errorbar(
                rows["k"],
                rows["estimate"],
                yerr=np.vstack(
                    [rows["estimate"] - rows["ci_95_lower"], rows["ci_95_upper"] - rows["estimate"]]
                ),
                marker="o",
                markersize=6,
                linewidth=2.2,
                capsize=3,
                color=MODEL_COLORS[model_id],
                label=MODEL_NAMES[model_id],
            )
        axis.set(title=title, xlabel="Number of ranked predictions (k)", ylabel=ylabel)
        axis.set_xticks([1, 3, 5])
        axis.set_ylim(0, 1.03)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.09))
    fig.suptitle("ImageNet-style ranking performance on target-positive test cases", y=1.18, fontsize=16)
    fig.text(
        0.5,
        -0.04,
        "Bars show 95% case-bootstrap confidence intervals. All-zero target cases are excluded because Hit@k is undefined.",
        ha="center",
        fontsize=9,
        color="#4C5860",
    )
    return save(fig, "figure_s1_topk_ranking_performance.png")


def cumulative_curves_figure(curves: pd.DataFrame, scope_metrics: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.2), constrained_layout=True)
    panels = [
        ("roc", "macro", "Macro-average ROC", "False-positive rate", "True-positive rate"),
        ("roc", "micro", "Micro-average ROC", "False-positive rate", "True-positive rate"),
        ("pr", "macro", "Macro-average precision–recall", "Recall", "Precision"),
        ("pr", "micro", "Micro-average precision–recall", "Recall", "Precision"),
    ]
    primary = scope_metrics[scope_metrics["model_id"] == CONFIG["primary_model"]].set_index("scope_n")
    for axis, (curve, averaging, title, xlabel, ylabel) in zip(axes.flat, panels, strict=True):
        for scope_n, color in SCOPE_COLORS.items():
            rows = curves[
                (curves["scope_n"] == scope_n)
                & (curves["curve"] == curve)
                & (curves["averaging"] == averaging)
            ].sort_values("x")
            metric = primary.loc[scope_n, f"{curve}_auc_{averaging}" if curve == "pr" else f"roc_auc_{averaging}"]
            label = f"Top {scope_n}: AUC {metric:.3f}"
            if curve == "pr":
                baseline = float(primary.loc[scope_n, "pr_no_skill_baseline"])
                label += f"; prevalence {baseline:.3f}"
                axis.axhline(baseline, color=color, linestyle=":", linewidth=1.0, alpha=0.65)
            axis.plot(rows["x"], rows["value"], color=color, linewidth=2.2, label=label)
            axis.fill_between(
                rows["x"].to_numpy(),
                rows["ci_95_lower"].to_numpy(),
                rows["ci_95_upper"].to_numpy(),
                color=color,
                alpha=0.11,
                linewidth=0,
            )
        if curve == "roc":
            axis.plot([0, 1], [0, 1], linestyle="--", color="#9A9A9A", linewidth=1)
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.01))
        axis.legend(loc="best", frameon=True, framealpha=0.9, fontsize=8)
    fig.suptitle(
        "SigLIP2 MIL performance as progressively rarer conditions are added",
        fontsize=16,
    )
    fig.text(
        0.5,
        -0.015,
        "Top-N labels are selected once by training prevalence. Curves use the same 25-class model and 750-case test cohort.",
        ha="center",
        fontsize=9,
        color="#4C5860",
    )
    return save(fig, "figure_s2_cumulative_roc_pr_curves.png")


def trajectory_figure(scope_metrics: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), sharex=True, constrained_layout=True)
    panels = [
        ("roc_auc_macro", "Macro ROC-AUC"),
        ("roc_auc_micro", "Micro ROC-AUC"),
        ("pr_auc_macro", "Macro PR-AUC"),
        ("pr_auc_micro", "Micro PR-AUC"),
    ]
    for axis, (metric, title) in zip(axes.flat, panels, strict=True):
        for model_id in CONFIG["principal_models"]:
            rows = scope_metrics[scope_metrics["model_id"] == model_id].sort_values("scope_n")
            lower = rows[f"{metric}_ci_95_lower"]
            upper = rows[f"{metric}_ci_95_upper"]
            axis.errorbar(
                rows["scope_n"],
                rows[metric],
                yerr=np.vstack([rows[metric] - lower, upper - rows[metric]]),
                marker="o",
                markersize=5,
                linewidth=2,
                capsize=3,
                color=MODEL_COLORS[model_id],
                label=MODEL_NAMES[model_id],
            )
        if metric.startswith("pr_"):
            baseline = scope_metrics[
                scope_metrics["model_id"] == CONFIG["primary_model"]
            ].sort_values("scope_n")
            axis.plot(
                baseline["scope_n"],
                baseline["pr_no_skill_baseline"],
                linestyle="--",
                marker="x",
                color="#717171",
                linewidth=1.5,
                label="Positive-prevalence reference",
            )
        axis.set(title=title, xlabel="Number of conditions included", ylabel=title)
        axis.set_xticks([5, 10, 15, 20, 25])
        axis.set_ylim(0, 1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles_pr, labels_pr = axes[1, 0].get_legend_handles_labels()
    if "Positive-prevalence reference" in labels_pr:
        index = labels_pr.index("Positive-prevalence reference")
        handles.append(handles_pr[index])
        labels.append(labels_pr[index])
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Performance trajectories across nested prevalence-defined label sets", y=1.14, fontsize=16)
    return save(fig, "figure_s3_label_scope_metric_trajectories.png")


def heatmap_figure(scope_metrics: pd.DataFrame, scope_n: int) -> Path:
    subset = scope_metrics[scope_metrics["scope_n"] == scope_n]
    panels = [
        ("roc_auc_macro", "Macro ROC-AUC", 0.50, 0.95),
        ("roc_auc_micro", "Micro ROC-AUC", 0.50, 0.95),
        ("pr_auc_macro", "Macro PR-AUC", 0.00, 0.55),
        ("pr_auc_micro", "Micro PR-AUC", 0.00, 0.55),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.2), constrained_layout=True)
    for axis, (metric, title, vmin, vmax) in zip(axes.flat, panels, strict=True):
        matrix = subset.pivot(index="encoder", columns="classifier", values=metric).reindex(
            index=ENCODERS, columns=CLASSIFIERS
        )
        matrix.index = [ENCODER_NAMES[value] for value in matrix.index]
        matrix.columns = [CLASSIFIER_NAMES[value] for value in matrix.columns]
        sns.heatmap(
            matrix,
            ax=axis,
            cmap=HEAT_CMAP,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".3f",
            linewidths=0.6,
            linecolor="white",
            cbar_kws={"label": title, "shrink": 0.82},
            annot_kws={"fontsize": 8},
        )
        axis.set(title=title, xlabel="Classifier", ylabel="Frozen image encoder")
        axis.tick_params(axis="x", rotation=0)
        axis.tick_params(axis="y", rotation=0)
    fig.suptitle(
        f"Case-level test performance: Top {scope_n} conditions by training prevalence",
        fontsize=16,
    )
    fig.text(
        0.5,
        -0.01,
        "All panels use identical metric-specific color limits across Top 5/10/15/20/25 figures.",
        ha="center",
        fontsize=9,
        color="#4C5860",
    )
    return save(fig, f"figure_s4_heatmaps_top{scope_n:02d}.png")


def radar_figure(per_condition: pd.DataFrame, prevalence: pd.DataFrame, scope_n: int) -> Path:
    top = prevalence.sort_values("prevalence_rank").head(scope_n)
    slugs = top["label_slug"].tolist()
    labels = [LABELS[LABEL_SLUGS.index(value)] for value in slugs]
    abbreviations = {
        "Allergic Contact Dermatitis": "Allergic contact\ndermatitis",
        "Irritant Contact Dermatitis": "Irritant contact\ndermatitis",
        "Herpes Zoster": "Herpes\nzoster",
        "Insect Bite": "Insect\nbite",
        "Drug Rash": "Drug\nrash",
    }
    labels = [abbreviations.get(value, value) for value in labels]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    numbered_labels = scope_n >= 15
    if numbered_labels:
        axis_labels = [str(index) for index in range(1, scope_n + 1)]
    else:
        axis_labels = labels
    fig = plt.figure(figsize=(11.2, 10.5))
    if numbered_labels:
        axis = fig.add_axes([0.15, 0.32, 0.70, 0.52], polar=True)
    else:
        axis = fig.add_axes([0.10, 0.08, 0.80, 0.74], polar=True)
    axis.set_theta_offset(np.pi / 2)
    axis.set_theta_direction(-1)
    for model_id in CONFIG["principal_models"]:
        rows = per_condition[
            (per_condition["model_id"] == model_id) & per_condition["label_slug"].isin(slugs)
        ].set_index("label_slug").loc[slugs]
        values = rows["roc_auc"].to_numpy()
        lower = rows["roc_auc_ci_95_lower"].to_numpy()
        upper = rows["roc_auc_ci_95_upper"].to_numpy()
        closed_values = np.r_[values, values[0]]
        closed_lower = np.r_[lower, lower[0]]
        closed_upper = np.r_[upper, upper[0]]
        axis.plot(closed_angles, closed_values, linewidth=2.2, color=MODEL_COLORS[model_id], label=MODEL_NAMES[model_id])
        axis.fill_between(
            closed_angles,
            closed_lower,
            closed_upper,
            color=MODEL_COLORS[model_id],
            alpha=0.055,
            linewidth=0,
        )
    axis.set_xticks(angles)
    axis.set_xticklabels(axis_labels, fontsize=8 if numbered_labels else (7 if scope_n > 10 else 9))
    axis.set_ylim(0.5, 1.0)
    axis.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    axis.set_yticklabels(["0.6", "0.7", "0.8", "0.9", "1.0"], color="#58636B", fontsize=8)
    axis.set_rlabel_position(102)
    axis.grid(color="#CBD3D2", linewidth=0.8)
    axis.spines["polar"].set_color("#AEB9B6")
    handles, legend_labels = axis.get_legend_handles_labels()
    fig.suptitle(
        f"Per-condition ROC-AUC for the {scope_n} most prevalent training conditions",
        y=0.985,
        fontsize=15,
    )
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.865,
        "Shaded ribbons show 95% case-bootstrap confidence intervals.",
        ha="center",
        fontsize=9,
        color="#4C5860",
    )
    if numbered_labels:
        key_lines = [f"{index}. {label.replace(chr(10), ' ')}" for index, label in enumerate(labels, start=1)]
        midpoint = (len(key_lines) + 1) // 2
        fig.text(0.05, 0.025, "\n".join(key_lines[:midpoint]), va="bottom", fontsize=7.2, linespacing=1.25)
        fig.text(0.53, 0.025, "\n".join(key_lines[midpoint:]), va="bottom", fontsize=7.2, linespacing=1.25)
    return save(fig, f"figure_s5_principal_models_top{scope_n:02d}_radar.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    style()
    topk = pd.read_csv(TABLES / "topk_ranking_metrics.csv")
    scope_metrics = pd.read_csv(TABLES / "label_scope_metrics.csv")
    per_condition = pd.read_csv(TABLES / "per_condition_metrics.csv")
    prevalence = pd.read_csv(OUTPUT / "manifests" / "training_label_prevalence.csv")
    curves = pd.read_csv(PLOT_DATA / "cumulative_roc_pr_curves.csv")

    outputs = [
        topk_figure(topk),
        cumulative_curves_figure(curves, scope_metrics),
        trajectory_figure(scope_metrics),
    ]
    outputs.extend(heatmap_figure(scope_metrics, scope_n) for scope_n in CONFIG["canonical"]["label_scopes"])
    outputs.extend(
        radar_figure(per_condition, prevalence, int(scope_n))
        for scope_n in CONFIG["canonical"]["label_scopes"]
    )
    manifest = {
        "complete": True,
        "figures": [str(path.relative_to(HERE)) for path in outputs],
        "figure_count": len(outputs),
        "canonical_cases": int(CONFIG["canonical"]["expected_cases"]),
        "note": "All figures use saved probabilities; no model was retrained.",
    }
    (FIGURES / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {len(outputs)} supplementary figures in {FIGURES}")


if __name__ == "__main__":
    main()
