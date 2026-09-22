from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from hawk_derm.figures.style import manuscript_style
from hawk_derm.io import write_json


def _save(fig: plt.Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def _page_path(base: Path, page: int, page_count: int) -> Path:
    return base if page_count == 1 else base.with_name(f"{base.stem}_page_{page + 1:02d}{base.suffix}")


def _label_pages(data: pd.DataFrame, labels: list[str], per_page: int) -> list[pd.DataFrame]:
    pages = []
    for start in range(0, len(labels), per_page):
        selected = labels[start : start + per_page]
        page = data[data["label"].isin(selected)].copy()
        page["label"] = pd.Categorical(page["label"], selected, ordered=True)
        pages.append(page.sort_values("label"))
    return pages


def split_donut(split_data: pd.DataFrame, output: Path, style: dict[str, Any]) -> Path:
    ordered = split_data.set_index("split").reindex(["train", "validation", "test"])
    case_colors = [style["split_cases_train"], style["split_cases_validation"], style["split_cases_test"]]
    image_colors = [style["split_images_train"], style["split_images_validation"], style["split_images_test"]]
    fig, axis = plt.subplots(figsize=(8, 8))
    axis.pie(
        ordered["images"], radius=1.0, labels=[f"{int(value):,}" for value in ordered["images"]],
        colors=image_colors, startangle=90, wedgeprops={"width": 0.28, "edgecolor": "white"},
        textprops={"color": style["split_label_text"], "weight": "bold"},
    )
    axis.pie(
        ordered["cases"], radius=0.68, labels=[f"{int(value):,}" for value in ordered["cases"]],
        colors=case_colors, startangle=90, wedgeprops={"width": 0.28, "edgecolor": "white"},
        textprops={"color": style["split_label_text"], "weight": "bold"},
    )
    axis.text(0, 0, "Cases", ha="center", va="center", fontsize=16, weight="bold", color=style["text"])
    axis.text(0, -1.2, "Images", ha="center", va="center", fontsize=12, color=style["text"])
    axis.legend(["Train", "Validation", "Test"], loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    axis.set_title("Data Overview", fontsize=22)
    return _save(fig, output, int(style["dpi"]))


def cohort_overview(
    split_data: pd.DataFrame,
    condition_data: pd.DataFrame,
    cases: pd.DataFrame,
    images: pd.DataFrame,
    output: Path,
    style: dict[str, Any],
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes[0, 0].axis("off")
    axes[0, 0].text(
        0.5, 0.5,
        f"Canonical analytic cohort\n\n{cases['case_id'].nunique():,} cases\n{len(images):,} manifest image rows\n"
        f"{len(condition_data):,} target conditions\n{images['resolved_image_path'].nunique():,} unique resolved paths",
        ha="center", va="center", fontsize=16,
        bbox={"boxstyle": "round,pad=1", "facecolor": style["mil_panel"], "edgecolor": style["text"]},
    )
    ordered = split_data.set_index("split").reindex(["train", "validation", "test"])
    axes[0, 1].pie(ordered["images"], radius=1, colors=[style["split_images_train"], style["split_images_validation"], style["split_images_test"]], startangle=90, wedgeprops={"width": 0.28, "edgecolor": "white"})
    axes[0, 1].pie(ordered["cases"], radius=0.68, colors=[style["split_cases_train"], style["split_cases_validation"], style["split_cases_test"]], startangle=90, wedgeprops={"width": 0.28, "edgecolor": "white"})
    axes[0, 1].legend([f"{name.title()}: {int(row.cases):,} cases / {int(row.images):,} images" for name, row in ordered.iterrows()], loc="lower center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    axes[0, 1].set_title("Split composition", weight="bold")
    top = condition_data.sort_values("train_images").tail(25)
    axes[1, 0].barh(top["label"], top["train_images"], color=sns.color_palette(style["cohort_condition_palette"], len(top)))
    axes[1, 0].set_xlabel("Training image rows")
    axes[1, 0].set_title("Training-set condition image counts", weight="bold")
    anatomical_column = next((column for column in images if "body_part" in column.lower() or "anatom" in column.lower()), None)
    if anatomical_column:
        anatomy = images[anatomical_column].fillna("Other").value_counts().head(12).sort_values()
        axes[1, 1].barh(anatomy.index.astype(str), anatomy.values, color=sns.color_palette(style["anatomy_palette"], len(anatomy)))
        axes[1, 1].set_xlabel("Images")
        axes[1, 1].set_title("Anatomical-site distribution", weight="bold")
    else:
        bag_counts = images.groupby("case_id").size().clip(upper=3).value_counts().sort_index()
        axes[1, 1].bar(bag_counts.index.astype(str), bag_counts.values, color=style["image_count"])
        axes[1, 1].set_xlabel("Images retained per case")
        axes[1, 1].set_ylabel("Cases")
        axes[1, 1].set_title("Case bag lengths", weight="bold")
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def performance_heatmap(
    data: pd.DataFrame,
    output: Path,
    style: dict[str, Any],
    *,
    metric: str = "auc_macro",
) -> Path:
    averaging = metric.removeprefix("auc_")
    matrix = data.pivot(index="encoder", columns="classifier", values=metric)
    fig, axis = plt.subplots(figsize=(12, max(5, 0.7 * len(matrix))))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        vmin=0.5,
        vmax=0.95,
        cmap=style["performance_heatmap"],
        ax=axis,
        cbar_kws={"label": f"Test {averaging} ROC-AUC"},
    )
    axis.set(xlabel="Classifier", ylabel="Frozen encoder", title=f"Case-level test {averaging} ROC-AUC")
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def learning_curves(data: pd.DataFrame, output: Path, style: dict[str, Any]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    specifications = [
        (axes[0], "train_loss", "validation_loss", style["train_loss"], style["validation_loss"], "MIL loss across runs", "Binary cross-entropy loss"),
        (axes[1], "train_accuracy", "validation_accuracy", style["train_accuracy"], style["validation_accuracy"], "MIL accuracy across runs", "Binary accuracy"),
    ]
    for axis, train_column, validation_column, train_color, validation_color, title, ylabel in specifications:
        if train_column not in data or validation_column not in data:
            axis.text(0.5, 0.5, "Per-epoch accuracy was not recorded\nin this earlier smoke artifact.", ha="center", va="center")
            axis.set_axis_off()
            continue
        for _, run in data.groupby("seed"):
            axis.plot(run["epoch"], run[train_column], color=train_color, alpha=float(style["individual_alpha"]))
            axis.plot(run["epoch"], run[validation_column], color=validation_color, alpha=float(style["individual_alpha"]))
        summary = data.groupby("epoch")[[train_column, validation_column]].agg(["mean", "std"])
        for column, color, label in ((train_column, train_color, "Train"), (validation_column, validation_color, "Validation")):
            mean = summary[(column, "mean")]
            sd = summary[(column, "std")].fillna(0)
            axis.plot(mean.index, mean, color=color, lw=float(style["mean_linewidth"]), label=f"{label} (mean)")
            axis.fill_between(mean.index, mean - sd, mean + sd, color=color, alpha=float(style["uncertainty_alpha"]))
        axis.set(xlabel="Epoch", ylabel=ylabel, title=title)
        axis.legend(frameon=True)
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def per_class_auc_pages(data: pd.DataFrame, labels: list[str], output: Path, style: dict[str, Any], per_page: int) -> list[Path]:
    pages = _label_pages(data, labels, per_page)
    outputs = []
    split_order = ["train", "validation", "test"]
    palette = {"train": style["train"], "validation": style["validation"], "test": style["test"]}
    for page_number, page in enumerate(pages):
        summary = page.groupby(["label", "split"], observed=True)["auc"].agg(["mean", "std"]).reset_index()
        fig, axis = plt.subplots(figsize=(max(12, 1.25 * page["label"].nunique()), 6))
        sns.barplot(data=summary, x="label", y="mean", hue="split", hue_order=split_order, palette=palette, errorbar=None, ax=axis)
        category_positions = {str(label): index for index, label in enumerate(page["label"].cat.categories)}
        width = 0.8 / len(split_order)
        for _, row in summary.iterrows():
            x = category_positions[str(row["label"])] + (split_order.index(row["split"]) - 1) * width
            axis.errorbar(x, row["mean"], yerr=0 if pd.isna(row["std"]) else row["std"], color="#5B3954", capsize=2, lw=1, fmt="none")
        axis.set_ylim(0, 1)
        axis.set(xlabel="Diagnostic condition", ylabel="ROC-AUC", title="Diagnostic-condition MIL ROC-AUC (mean ± SD across runs)")
        axis.tick_params(axis="x", rotation=28)
        axis.legend(title="Split", frameon=True)
        fig.tight_layout()
        outputs.append(_save(fig, _page_path(output, page_number, len(pages)), int(style["dpi"])))
    return outputs


def _plot_curve_summary(axis: plt.Axes, data: pd.DataFrame, curve: str, style: dict[str, Any]) -> None:
    for averaging in ("micro", "macro"):
        group = data[data["curve"].eq(curve) & data["averaging"].eq(averaging)]
        if group.empty:
            continue
        for _, run in group.groupby("seed"):
            axis.plot(run["x"], run["y"], color=style[f"{curve}_{averaging}"], alpha=float(style["uncertainty_alpha"]), lw=1)
        summary = group.groupby("x")["y"].agg(["mean", "std"])
        score = group.groupby("seed")["score"].first().dropna()
        score_text = f" ({score.mean():.3f}±{score.std(ddof=1):.3f})" if len(score) > 1 else ""
        axis.plot(summary.index, summary["mean"], color=style[f"{curve}_{averaging}"], lw=float(style["mean_linewidth"]), label=f"{averaging.title()}{score_text}")
        sd = summary["std"].fillna(0)
        axis.fill_between(summary.index, summary["mean"] - sd, summary["mean"] + sd, color=style[f"{curve}_{averaging}"], alpha=float(style["uncertainty_alpha"]))


def roc_pr_curves(data: pd.DataFrame, output: Path, style: dict[str, Any]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _plot_curve_summary(axes[0], data, "roc", style)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="MIL ROC-AUC (test set, repeated runs)", xlim=(0, 1), ylim=(0, 1))
    _plot_curve_summary(axes[1], data, "pr", style)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="MIL precision-recall (test set, repeated runs)", xlim=(0, 1), ylim=(0, 1))
    for axis in axes:
        axis.legend(frameon=True)
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def roc_with_sensitivity_specificity(
    curves: pd.DataFrame,
    sens_spec: pd.DataFrame,
    output: Path,
    style: dict[str, Any],
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    _plot_curve_summary(axes[0], curves, "roc", style)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="MIL ROC-AUC (test set, repeated runs)", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(frameon=True)
    configured = style.get("class_curve_colors", {})
    fallback = sns.color_palette("tab20", sens_spec["label"].nunique())
    for index, (label, group) in enumerate(sens_spec.groupby("label", sort=False)):
        group = group.sort_values("threshold")
        color = configured.get(str(label), fallback[index])
        axes[1].plot(group["specificity_mean"], group["sensitivity_mean"], color=color, lw=1.8, label=str(label))
        chosen = group[group["selected"]]
        if len(chosen):
            row = chosen.iloc[0]
            axes[1].scatter(row["specificity_mean"], row["sensitivity_mean"], color=color, s=25, zorder=3)
    axes[1].set(xlabel="Specificity", ylabel="Sensitivity", title="MIL sensitivity vs. specificity (test set, repeated runs)", xlim=(0, 1), ylim=(0, 1))
    axes[1].legend(loc="lower left", fontsize=6, frameon=False, ncol=2 if sens_spec["label"].nunique() > 10 else 1)
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def threshold_curve_pages(data: pd.DataFrame, labels: list[str], output: Path, style: dict[str, Any], per_page: int, columns: int) -> list[Path]:
    pages = _label_pages(data, labels, per_page)
    outputs = []
    for page_number, page in enumerate(pages):
        page_labels = list(page["label"].cat.categories)
        rows = math.ceil(len(page_labels) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(7.2 * columns, 3.8 * rows), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).reshape(-1)
        for axis, label in zip(axes, page_labels, strict=False):
            group = page[page["label"].eq(label)].sort_values("threshold")
            for metric, color in (("sensitivity", style["sensitivity"]), ("specificity", style["specificity"])):
                mean = group[f"{metric}_mean"].to_numpy(float)
                sd = group[f"{metric}_sd"].to_numpy(float)
                x = group["threshold"].to_numpy(float)
                axis.plot(x, mean, color=color, lw=2, label=metric.title())
                axis.fill_between(x, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), color=color, alpha=float(style["threshold_band_alpha"]))
            selected = group[group["selected"]]
            if len(selected):
                row = selected.iloc[0]
                axis.axvline(row["threshold"], color="black", linestyle="--", lw=1)
                axis.text(0.02, 0.82, f"Se={row['sensitivity_mean']:.3f}\nSp={row['specificity_mean']:.3f}", transform=axis.transAxes, bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"})
                axis.text(row["threshold"] + 0.01, 0.04, f"t={row['threshold']:.3f}", rotation=90, fontsize=7)
            axis.set(title=str(label), xlim=(0, 1), ylim=(0, 1))
        for axis in axes[len(page_labels) :]:
            axis.remove()
        axes[0].legend(frameon=True, ncol=2)
        fig.supxlabel("Decision threshold")
        fig.supylabel("Metric value")
        fig.tight_layout()
        outputs.append(_save(fig, _page_path(output, page_number, len(pages)), int(style["dpi"])))
    return outputs


def youden_pages(data: pd.DataFrame, labels: list[str], output: Path, style: dict[str, Any], per_page: int, columns: int) -> list[Path]:
    pages = _label_pages(data, labels, per_page)
    outputs = []
    for page_number, page in enumerate(pages):
        page_labels = list(page["label"].cat.categories)
        rows = math.ceil(len(page_labels) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(7.2 * columns, 3.8 * rows), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).reshape(-1)
        for axis, label in zip(axes, page_labels, strict=False):
            group = page[page["label"].eq(label)].sort_values("threshold")
            axis.plot(group["threshold"], group["youden_j_mean"], color=style["youden_curve"], lw=2)
            mean = group["youden_j_mean"].to_numpy(float)
            sd = group["youden_j_sd"].to_numpy(float)
            axis.fill_between(group["threshold"], mean - sd, mean + sd, color=style["youden_curve"], alpha=float(style["threshold_band_alpha"]))
            selected = group[group["selected"]]
            if len(selected):
                row = selected.iloc[0]
                axis.scatter(row["threshold"], row["youden_j_mean"], color=style["youden_point"], zorder=3)
                axis.axvline(row["threshold"], color="black", linestyle="--", lw=1)
            axis.set(title=str(label), xlim=(0, 1))
        for axis in axes[len(page_labels) :]:
            axis.remove()
        fig.supxlabel("Decision threshold")
        fig.supylabel("Youden J")
        fig.tight_layout()
        outputs.append(_save(fig, _page_path(output, page_number, len(pages)), int(style["dpi"])))
    return outputs


def operating_point_pages(data: pd.DataFrame, labels: list[str], output: Path, style: dict[str, Any], per_page: int, columns: int) -> list[Path]:
    pages = _label_pages(data, labels, per_page)
    outputs = []
    for page_number, page in enumerate(pages):
        page_labels = list(page["label"].cat.categories)
        rows = math.ceil(len(page_labels) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(7.2 * columns, 3.8 * rows), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).reshape(-1)
        for axis, label in zip(axes, page_labels, strict=False):
            group = page[page["label"].eq(label)].sort_values("threshold")
            axis.plot(group["specificity_mean"], group["sensitivity_mean"], color=style["operating_curve"], lw=2)
            default = group.iloc[(group["threshold"] - 0.5).abs().argsort()[:1]]
            balanced = group[group["selected"]]
            if len(default):
                row = default.iloc[0]
                axis.scatter(row["specificity_mean"], row["sensitivity_mean"], color=style["threshold_default"], s=45, label="t=0.50")
            if len(balanced):
                row = balanced.iloc[0]
                axis.scatter(row["specificity_mean"], row["sensitivity_mean"], color=style["threshold_balanced"], s=45, label="Balanced")
            axis.set(title=str(label), xlim=(0, 1), ylim=(0, 1))
        for axis in axes[len(page_labels) :]:
            axis.remove()
        axes[0].legend(frameon=True)
        fig.supxlabel("Specificity")
        fig.supylabel("Sensitivity")
        fig.tight_layout()
        outputs.append(_save(fig, _page_path(output, page_number, len(pages)), int(style["dpi"])))
    return outputs


def split_metric_pages(data: pd.DataFrame, labels: list[str], output: Path, style: dict[str, Any], per_page: int) -> list[Path]:
    pages = _label_pages(data, labels, per_page)
    outputs = []
    palette = {"train": style["split_metric_train"], "validation": style["split_metric_validation"], "test": style["split_metric_test"]}
    for page_number, page in enumerate(pages):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
        for axis, metric in zip(axes.ravel(), ("accuracy", "precision", "recall", "f1"), strict=True):
            sns.barplot(data=page, x="label", y=metric, hue="split", hue_order=["train", "validation", "test"], palette=palette, alpha=0.8, ax=axis)
            axis.set_ylim(0, 1)
            axis.set_title(metric.replace("_", " ").title())
            axis.tick_params(axis="x", rotation=32)
            if metric != "accuracy" and axis.legend_:
                axis.legend_.remove()
        axes[0, 0].legend(title="Split", frameon=True)
        fig.tight_layout()
        outputs.append(_save(fig, _page_path(output, page_number, len(pages)), int(style["dpi"])))
    return outputs


def condition_distributions(data: pd.DataFrame, output: Path, style: dict[str, Any]) -> Path:
    ordered = data.sort_values("total_cases")
    fig, axes = plt.subplots(1, 2, figsize=(18, max(7, len(ordered) * 0.32)), sharey=True)
    axes[0].barh(ordered["label"], ordered["total_cases"], color=sns.color_palette(style["distribution_palette"], len(ordered)))
    axes[0].set(xlabel="Cases with positive target label", title="Conditions – selected vocabulary")
    weighted = ordered["total_cases"] * ordered.get("effective_number_weight", 1.0)
    axes[1].barh(ordered["label"], weighted, color=sns.color_palette(style["weighted_distribution_palette"], len(ordered)))
    axes[1].set(xlabel="Cases × normalized effective-number weight", title="Conditions (weighted labels)")
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def graphical_abstract(condition_count: int, heatmap: pd.DataFrame | None, output: Path, style: dict[str, Any]) -> Path:
    fig, axis = plt.subplots(figsize=(16, 6))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    boxes = [(0.03, "1–3 patient\nphotographs", style["baseline_panel"]), (0.25, "Frozen image\nencoder", "#D9F0E3"), (0.47, "Masked gated\nattention MIL", style["mil_panel"]), (0.69, f"{condition_count}-condition\ncase probabilities", "#DDEBFA"), (0.86, "Validation-selected\noperating points", "#E1F2E8")]
    for x, text, color in boxes:
        axis.text(x + 0.07, 0.55, text, ha="center", va="center", fontsize=13, weight="bold", bbox={"boxstyle": "round,pad=1", "facecolor": color, "edgecolor": style["text"]})
    for left, right in zip(boxes[:-1], boxes[1:], strict=True):
        axis.annotate("", xy=(right[0], 0.55), xytext=(left[0] + 0.14, 0.55), arrowprops={"arrowstyle": "->", "lw": 2, "color": style["text"]})
    subtitle = "Shared locked test cases; raw case-level probabilities retained"
    if heatmap is not None and len(heatmap):
        best = heatmap.loc[heatmap["auc_macro"].idxmax()]
        subtitle += f"\nBest completed cell: {best['encoder']} + {best['classifier']} ({best['auc_macro']:.3f} macro ROC-AUC)"
    axis.text(0.5, 0.12, subtitle, ha="center", va="center", fontsize=11)
    axis.set_title("Case-level multi-image teledermatology workflow", fontsize=19, weight="bold", pad=20)
    return _save(fig, output, int(style["dpi"]))


def closed_model_heatmap(data: pd.DataFrame, output: Path, style: dict[str, Any]) -> Path:
    matrix = data.pivot(index="model", columns="label", values="auc")
    fig, axis = plt.subplots(figsize=(max(14, 0.52 * len(matrix.columns)), max(6, 0.55 * len(matrix))))
    sns.heatmap(matrix, annot=len(matrix.columns) <= 12, fmt=".3f", vmin=0.5, vmax=1.0, cmap=style["reviewer_heatmap"], ax=axis, cbar_kws={"label": "Test ROC-AUC"})
    axis.set(title="Closed multimodal and case-level model comparison", xlabel="Condition", ylabel="Model")
    axis.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def threshold_accuracy_comparison(data: pd.DataFrame, labels: list[str], output: Path, style: dict[str, Any]) -> Path:
    ordered = data.set_index("label").reindex(labels).reset_index()
    x = np.arange(len(ordered))
    fig, axis = plt.subplots(figsize=(max(12, len(ordered) * 0.65), 6))
    axis.bar(x - 0.2, ordered["accuracy_default"], width=0.4, color="#9ECAE1", label="t=0.5")
    axis.bar(x + 0.2, ordered["accuracy_balanced"], width=0.4, color="#2171B5", label="Balanced t")
    axis.axhline(ordered["accuracy_default"].mean(), color="#6BAED6", linestyle="--")
    axis.axhline(ordered["accuracy_balanced"].mean(), color="#08306B", linestyle="--")
    axis.set_xticks(x, ordered["label"], rotation=35, ha="right")
    axis.set(ylim=(0, 1), ylabel="Accuracy", title="Test accuracy comparison")
    axis.legend(frameon=True)
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def closed_model_overall(data: pd.DataFrame, output: Path, style: dict[str, Any]) -> Path:
    metrics = [("subset_accuracy", "Subset Accuracy"), ("precision_micro", "Precision (micro)"), ("recall_micro", "Recall (micro)"), ("f1_micro", "F1 (micro)"), ("auc_micro", "ROC AUC (micro)"), ("pr_auc_micro", "PR AUC (micro)")]
    metrics = [(column, title) for column, title in metrics if column in data]
    rows, columns = 2, 3
    fig, axes = plt.subplots(rows, columns, figsize=(20, 10), sharey=True)
    colors = [style.get("closed_model_colors", {}).get(str(model), sns.color_palette("crest", len(data))[index]) for index, model in enumerate(data["model"])]
    for axis, (column, title) in zip(axes.ravel(), metrics, strict=False):
        bars = axis.bar(data["model"], data[column], color=colors)
        axis.bar_label(bars, fmt="%.3f", fontsize=8)
        axis.set(title=title, ylim=(0, 1.05))
        axis.tick_params(axis="x", rotation=35)
    for axis in axes.ravel()[len(metrics) :]:
        axis.remove()
    fig.suptitle("Overall Test Performance by Model", fontsize=16)
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def auc_radar(data: pd.DataFrame, output: Path, style: dict[str, Any]) -> Path:
    labels = list(dict.fromkeys(data["label"]))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    fig, axis = plt.subplots(figsize=(12, 12), subplot_kw={"projection": "polar"})
    colors = style.get("radar_colors", {})
    for model, group in data.groupby("model", sort=False):
        values = group.set_index("label").reindex(labels)["auc"].to_numpy(dtype=float)
        axis.plot(closed_angles, np.r_[values, values[0]], color=colors.get(str(model)), linestyle=":" if str(model).startswith("DermAssist") else "-", lw=2 if str(model).startswith("DermAssist") else 1.4, label=model)
    axis.set_xticks(angles, labels, fontsize=7)
    axis.set_ylim(0, 1)
    axis.set_yticks(np.arange(0.2, 1.01, 0.2))
    axis.legend(bbox_to_anchor=(1.3, 1.05), loc="upper left", fontsize=7, frameon=False)
    axis.set_title("Per-condition test ROC-AUC", weight="bold", pad=25)
    return _save(fig, output, int(style["dpi"]))


def metadata_figure(cases: pd.DataFrame, output: Path, style: dict[str, Any], kind: str) -> Path | None:
    if kind == "confidence":
        x_column = next((column for column in cases if "self_reported" in column.lower() and "num" in column.lower()), None)
        y_column = next((column for column in cases if "confidence" in column.lower()), None)
        if not x_column or not y_column:
            return None
        frame = cases[[x_column, y_column]].dropna().copy()
        bins = [-np.inf, 3, 6, 9, np.inf]
        frame["self_report_bin"] = pd.cut(frame[x_column], bins=bins, labels=["0–3", "4–6", "7–9", "10+"])
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.regplot(data=frame, x=x_column, y=y_column, scatter_kws={"alpha": 0.25, "color": "#4C78A8"}, line_kws={"color": "#C53030"}, ax=axes[0])
        sns.boxplot(data=frame, x="self_report_bin", y=y_column, color="#6B8299", ax=axes[1])
        axes[0].set_title("Confidence vs. self-reported variables")
        axes[1].set_title("Confidence by self-report count")
    elif kind == "gradability_mst":
        columns = [column for column in cases if any(token in column.lower() for token in ("gradable", "monk_skin_tone"))][:3]
        if not columns:
            return None
        fig, axes = plt.subplots(1, len(columns), figsize=(6 * len(columns), 5))
        axes = np.atleast_1d(axes)
        for axis, column in zip(axes, columns, strict=True):
            counts = cases[column].fillna("Missing").astype(str).value_counts().sort_index()
            color = style["gradability"] if "gradable" in column.lower() else style["monk_skin_tone"]
            axis.bar(counts.index, counts.values, color=color)
            axis.set_title(column.replace("_", " ").title(), fontsize=10)
            axis.tick_params(axis="x", rotation=45)
    else:
        raise ValueError(kind)
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def image_counts_shot_types(images: pd.DataFrame, output: Path, style: dict[str, Any]) -> Path:
    counts = images.groupby("case_id").size().value_counts().sort_index()
    shot_column = next((column for column in images if "shot" in column.lower() or "view" in column.lower()), None)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(counts.index.astype(str), counts.values, color=style["image_count"])
    axes[0].set_title("Images per case", weight="bold")
    shot_colors = [style["shot_close_up"], style["shot_angle"], style["shot_distance"]]
    if shot_column:
        shot_counts = images[shot_column].fillna("Missing").value_counts()
        axes[1].bar(shot_counts.index.astype(str), shot_counts.values, color=[shot_colors[index % 3] for index in range(len(shot_counts))])
        axes[1].tick_params(axis="x", rotation=35)
        axes[1].set_title("Shot types", weight="bold")
    else:
        order_counts = images["bag_slot"].value_counts().sort_index()
        axes[1].bar([f"View {value + 1}" for value in order_counts.index], order_counts.values, color=shot_colors[: len(order_counts)])
        axes[1].set_title("Retained view positions", weight="bold")
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def generate_figures(
    cases: pd.DataFrame,
    images: pd.DataFrame,
    reports_dir: str | Path,
    artifacts_dir: str | Path,
    figure_config: dict[str, Any],
    *,
    only: set[str] | None = None,
) -> list[Path]:
    del artifacts_dir  # Rendering is intentionally isolated from model artifacts.
    reports_dir = Path(reports_dir)
    plot_dir, figure_dir = reports_dir / "plot_data", reports_dir / "figures"
    style = figure_config["style"]
    pagination = figure_config.get("pagination", {})
    per_page = int(pagination.get("conditions_per_page", 10))
    columns = int(pagination.get("columns", 2))
    labels = pd.read_csv(plot_dir / "condition_support.csv")["label"].tolist()
    selected = only or {"cohort", "performance", "mil", "thresholds", "operating", "supplementary", "closed"}
    outputs: list[Path] = []

    def include(group: str) -> bool:
        return group in selected

    with manuscript_style(style):
        split_data = pd.read_csv(plot_dir / "split_composition.csv")
        condition_data = pd.read_csv(plot_dir / "condition_support.csv")
        heatmap_path = plot_dir / "encoder_classifier_heatmap.csv"
        heatmap_data = pd.read_csv(heatmap_path) if heatmap_path.exists() else None

        if include("cohort"):
            outputs.append(cohort_overview(split_data, condition_data, cases, images, figure_dir / "figure1_cohort_preprocessing.png", style))
            outputs.append(graphical_abstract(len(condition_data), heatmap_data, figure_dir / "graphical_abstract.png", style))
        if include("performance") and heatmap_data is not None:
            outputs.append(performance_heatmap(heatmap_data, figure_dir / "figure2_encoder_classifier_heatmap.png", style))
            if "auc_micro" in heatmap_data:
                outputs.append(
                    performance_heatmap(
                        heatmap_data,
                        figure_dir / "figure2_encoder_classifier_heatmap_micro.png",
                        style,
                        metric="auc_micro",
                    )
                )

        history_path = plot_dir / "mil_history_long.csv"
        per_class_path = plot_dir / "mil_per_class_metrics_long.csv"
        curves_path = plot_dir / "mil_roc_pr_curves.csv"
        test_curves_path = plot_dir / "test_sensitivity_specificity_curves.csv"
        if include("mil"):
            if history_path.exists():
                outputs.append(learning_curves(pd.read_csv(history_path), figure_dir / "figure3a_learning_curves.png", style))
            if per_class_path.exists():
                outputs.extend(per_class_auc_pages(pd.read_csv(per_class_path), labels, figure_dir / "figure3b_per_class_auc.png", style, per_page))
            if curves_path.exists():
                curves = pd.read_csv(curves_path)
                outputs.append(roc_pr_curves(curves, figure_dir / "figure3c_roc_pr_curves.png", style))
                outputs.append(roc_pr_curves(curves, figure_dir / "mil_roc_pr_curves.png", style))
                if test_curves_path.exists():
                    test_curves = pd.read_csv(test_curves_path)
                    outputs.append(roc_with_sensitivity_specificity(curves, test_curves, figure_dir / "figure3c_roc_with_sens_spec.png", style))
                    outputs.append(roc_with_sensitivity_specificity(curves, test_curves, figure_dir / "mil_roc_with_sens_spec.png", style))

        validation_curves_path = plot_dir / "validation_threshold_curves.csv"
        if include("thresholds") and validation_curves_path.exists():
            threshold_data = pd.read_csv(validation_curves_path)
            outputs.extend(threshold_curve_pages(threshold_data, labels, figure_dir / "figure4_threshold_curves.png", style, per_page, columns))
            outputs.extend(threshold_curve_pages(threshold_data, labels, figure_dir / "mil_sensitivity_specificity_curves_with_thresholds.png", style, per_page, columns))
            outputs.extend(youden_pages(threshold_data, labels, figure_dir / "mil_youden_j_curves.png", style, per_page, columns))

        split_metrics_path = plot_dir / "mil_split_per_class_metrics.csv"
        if include("operating") and test_curves_path.exists():
            outputs.extend(operating_point_pages(pd.read_csv(test_curves_path), labels, figure_dir / "figure6a_operating_points.png", style, per_page, columns))
            if split_metrics_path.exists():
                outputs.extend(split_metric_pages(pd.read_csv(split_metrics_path), labels, figure_dir / "figure6b_split_metrics.png", style, per_page))
            accuracy_path = plot_dir / "threshold_accuracy_comparison.csv"
            if accuracy_path.exists():
                outputs.append(threshold_accuracy_comparison(pd.read_csv(accuracy_path), labels, figure_dir / "figure5b_accuracy_comparison.png", style))

        if include("supplementary"):
            outputs.append(split_donut(split_data, figure_dir / "mil_split_soft.png", style))
            outputs.append(condition_distributions(condition_data, figure_dir / "condition_distributions.png", style))
            outputs.append(image_counts_shot_types(images, figure_dir / "images_counts_shot_types.png", style))
            for kind, filename in (("confidence", "confidence_vs_selfreported.png"), ("gradability_mst", "gradability_mst.png")):
                generated = metadata_figure(cases, figure_dir / filename, style, kind)
                if generated:
                    outputs.append(generated)

        closed_per_class = plot_dir / "closed_models_per_class.csv"
        closed_overall = plot_dir / "closed_models_overall.csv"
        if include("closed") and closed_per_class.exists():
            closed_data = pd.read_csv(closed_per_class)
            outputs.append(closed_model_heatmap(closed_data, figure_dir / "figure5a_closed_model_heatmap.png", style))
            outputs.append(auc_radar(closed_data, figure_dir / "all_models_auc_radar_plotly.png", style))
        if include("closed") and closed_overall.exists():
            closed_data = pd.read_csv(closed_overall)
            outputs.append(closed_model_overall(closed_data, figure_dir / "overall_test_metrics_closed_models.png", style))

    write_json(
        figure_dir / "figure_generation_manifest.json",
        {"outputs": [str(path) for path in outputs], "groups": sorted(selected), "complete": True},
    )
    return outputs
