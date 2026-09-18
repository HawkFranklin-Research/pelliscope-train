from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from hawk_derm.figures.style import manuscript_style
from hawk_derm.io import write_json


def _save(fig: plt.Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def split_donut(split_data: pd.DataFrame, output: Path, style: dict) -> Path:
    ordered = split_data.set_index("split").reindex(["train", "validation", "test"])
    colors = [style["split_train"], style["split_validation"], style["split_test"]]
    fig, axis = plt.subplots(figsize=(8, 8))
    axis.pie(ordered["images"], radius=1.0, labels=[f"{name.title()} images\n{int(value):,}" for name, value in ordered["images"].items()], colors=colors, startangle=90, wedgeprops={"width": 0.28, "edgecolor": "white"})
    axis.pie(ordered["cases"], radius=0.68, labels=[f"{int(value):,}" for value in ordered["cases"]], colors=colors, startangle=90, wedgeprops={"width": 0.28, "edgecolor": "white"})
    axis.text(0, 0, "Cases\n(inner)\nImages\n(outer)", ha="center", va="center", weight="bold")
    axis.set_title("Shared train/validation/test composition", weight="bold")
    return _save(fig, output, int(style["dpi"]))


def cohort_overview(split_data: pd.DataFrame, condition_data: pd.DataFrame, cases: pd.DataFrame, images: pd.DataFrame, output: Path, style: dict) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes[0, 0].axis("off")
    source_text = (
        f"Canonical analytic cohort\n\n{cases['case_id'].nunique():,} cases\n"
        f"{len(images):,} manifest image rows\n{len(condition_data):,} target conditions\n"
        f"{images['resolved_image_path'].nunique():,} unique resolved paths"
    )
    axes[0, 0].text(0.5, 0.5, source_text, ha="center", va="center", fontsize=16, bbox={"boxstyle": "round,pad=1", "facecolor": style["mil_panel"], "edgecolor": style["text"]})
    ordered = split_data.set_index("split").reindex(["train", "validation", "test"])
    colors = [style["split_train"], style["split_validation"], style["split_test"]]
    axes[0, 1].pie(ordered["images"], radius=1, colors=colors, startangle=90, wedgeprops={"width": 0.28, "edgecolor": "white"})
    axes[0, 1].pie(ordered["cases"], radius=0.68, colors=colors, startangle=90, wedgeprops={"width": 0.28, "edgecolor": "white"})
    axes[0, 1].legend([f"{name.title()}: {int(row.cases):,} cases / {int(row.images):,} images" for name, row in ordered.iterrows()], loc="lower center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    axes[0, 1].set_title("Split composition", weight="bold")
    top = condition_data.sort_values("train_images").tail(25)
    axes[1, 0].barh(top["label"], top["train_images"], color=sns.color_palette("viridis", len(top)))
    axes[1, 0].set_xlabel("Training image rows")
    axes[1, 0].set_title("Training-set condition image counts", weight="bold")
    anatomical_column = next((column for column in images if "body_part" in column.lower() or "anatom" in column.lower()), None)
    if anatomical_column:
        anatomy = images[anatomical_column].fillna("Other").value_counts().head(12).sort_values()
        axes[1, 1].barh(anatomy.index.astype(str), anatomy.values, color=sns.color_palette("Spectral", len(anatomy)))
        axes[1, 1].set_xlabel("Images")
        axes[1, 1].set_title("Anatomical-site distribution", weight="bold")
    else:
        bag_counts = images.groupby("case_id").size().clip(upper=3).value_counts().sort_index()
        axes[1, 1].bar(bag_counts.index.astype(str), bag_counts.values, color=style["train"])
        axes[1, 1].set_xlabel("Images retained per case")
        axes[1, 1].set_ylabel("Cases")
        axes[1, 1].set_title("Case bag lengths", weight="bold")
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def performance_heatmap(data: pd.DataFrame, output: Path, style: dict) -> Path:
    matrix = data.pivot(index="encoder", columns="classifier", values="auc_macro")
    fig, axis = plt.subplots(figsize=(12, max(5, 0.7 * len(matrix))))
    sns.heatmap(matrix, annot=True, fmt=".3f", vmin=0.5, vmax=0.95, cmap=style["performance_heatmap"], ax=axis, cbar_kws={"label": "Test macro ROC-AUC"})
    axis.set_xlabel("Classifier")
    axis.set_ylabel("Frozen encoder")
    axis.set_title("Case-level test macro ROC-AUC", weight="bold")
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def mil_diagnostics(artifact_root: Path, output: Path, style: dict) -> Path:
    histories = []
    metric_frames = []
    primary_root = artifact_root / "models" / "mil" / "siglip2_so400m"
    history_pattern = primary_root.glob("seed_*/history.json") if primary_root.exists() else artifact_root.glob("models/mil/*/seed_*/history.json")
    for history_path in sorted(history_pattern):
        import json

        payload = json.loads(history_path.read_text())
        frame = pd.DataFrame(payload["history"])
        frame["run"] = history_path.parent.name
        frame["encoder"] = history_path.parents[1].name
        histories.append(frame)
        per_class_path = history_path.parent / "test_per_class_metrics.csv"
        if per_class_path.exists():
            metrics = pd.read_csv(per_class_path)
            metrics["run"] = history_path.parent.name
            metric_frames.append(metrics)
    if not histories or not metric_frames:
        raise FileNotFoundError("MIL histories and per-class metrics are required for diagnostics")
    history = pd.concat(histories, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    prediction_pattern = primary_root.glob("seed_*/test_predictions.csv") if primary_root.exists() else artifact_root.glob("models/mil/*/seed_*/test_predictions.csv")
    prediction_frames = [pd.read_csv(path) for path in sorted(prediction_pattern)]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    pr_axis = axes[2].inset_axes([0.54, 0.08, 0.42, 0.40])
    for _, run in history.groupby(["encoder", "run"]):
        axes[0].plot(run["epoch"], run["train_loss"], color=style["train"], alpha=0.15)
        axes[0].plot(run["epoch"], run["validation_loss"], color=style["validation"], alpha=0.15)
    means = history.groupby("epoch")[["train_loss", "validation_loss"]].mean()
    axes[0].plot(means.index, means["train_loss"], color=style["train"], lw=3, label="Train")
    axes[0].plot(means.index, means["validation_loss"], color=style["validation"], lw=3, label="Validation")
    axes[0].legend(frameon=False)
    axes[0].set_title("MIL learning curves across runs", weight="bold")
    sns.boxplot(data=metrics, x="label", y="auc", color=style["test"], ax=axes[1])
    sns.stripplot(data=metrics, x="label", y="auc", color=style["text"], size=2.5, alpha=0.5, ax=axes[1])
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Condition-level test ROC-AUC", weight="bold")
    for frame in prediction_frames:
        true_columns = [column for column in frame if column.startswith("true__")]
        probability_columns = [column.replace("true__", "prob__", 1) for column in true_columns]
        truth = frame[true_columns].to_numpy().ravel()
        probability = frame[probability_columns].to_numpy().ravel()
        fpr, tpr, _ = roc_curve(truth, probability)
        axes[2].plot(fpr, tpr, color="#7B2CBF", alpha=0.15)
        precision, recall, _ = precision_recall_curve(truth, probability)
        pr_axis.plot(recall, precision, color="#D45087", alpha=0.15)
    if prediction_frames:
        combined_truth = prediction_frames[0][[column for column in prediction_frames[0] if column.startswith("true__")]].to_numpy().ravel()
        mean_probability = np.mean(
            [frame[[column for column in frame if column.startswith("prob__")]].to_numpy().ravel() for frame in prediction_frames],
            axis=0,
        )
        fpr, tpr, _ = roc_curve(combined_truth, mean_probability)
        axes[2].plot(fpr, tpr, color="#7B2CBF", lw=3, label="Mean micro ROC")
        precision, recall, _ = precision_recall_curve(combined_truth, mean_probability)
        pr_axis.plot(recall, precision, color="#D45087", lw=2, label="Mean micro PR")
    axes[2].plot([0, 1], [0, 1], color="#7A869A", linestyle="--", lw=1)
    axes[2].set_xlabel("False-positive rate")
    axes[2].set_ylabel("True-positive rate")
    axes[2].set_title("Aggregated test ROC across runs", weight="bold")
    axes[2].legend(frameon=False)
    pr_axis.set_xlim(0, 1)
    pr_axis.set_ylim(0, 1)
    pr_axis.set_xlabel("Recall", fontsize=7)
    pr_axis.set_ylabel("Precision", fontsize=7)
    pr_axis.tick_params(labelsize=6)
    pr_axis.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def threshold_curves(data: pd.DataFrame, output: Path, style: dict) -> Path:
    labels = list(dict.fromkeys(data["label"]))
    columns = 5
    rows = math.ceil(len(labels) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(20, 3.6 * rows), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    for axis, label in zip(axes, labels, strict=False):
        group = data[data["label"].eq(label)].sort_values("threshold")
        axis.plot(group["threshold"], group["sensitivity"], color=style["sensitivity"], label="Sensitivity")
        axis.plot(group["threshold"], group["specificity"], color=style["specificity"], label="Specificity")
        if "selected" in group:
            selected = group[group["selected"].astype(bool)]
            if len(selected):
                axis.axvline(selected.iloc[0]["threshold"], color="black", linestyle="--", lw=1)
        axis.set_title(label, fontsize=9)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    for axis in axes[len(labels) :]:
        axis.remove()
    axes[0].legend(frameon=False, fontsize=8)
    fig.supxlabel("Decision threshold")
    fig.supylabel("Metric")
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def operating_points(data: pd.DataFrame, output: Path, style: dict, split_metrics: pd.DataFrame | None = None) -> Path:
    primary_encoder = "siglip2_so400m" if "siglip2_so400m" in set(data["encoder"]) else data["encoder"].iloc[0]
    frame = data[data["encoder"].eq(primary_encoder)].copy().sort_values("balanced_accuracy")
    positions = np.arange(len(frame))
    fig, axes = plt.subplots(2, 2, figsize=(16, max(11, len(frame) * 0.52)), gridspec_kw={"height_ratios": [2, 1]})
    axes[0, 0].barh(positions - 0.18, frame["sensitivity"], height=0.36, color=style["sensitivity"], label="Sensitivity")
    axes[0, 0].barh(positions + 0.18, frame["specificity"], height=0.36, color=style["specificity"], label="Specificity")
    if {"default_sensitivity", "default_specificity"}.issubset(frame.columns):
        axes[0, 0].scatter(frame["default_sensitivity"], positions, color=style["threshold_default"], marker="x", label="Sensitivity at 0.5")
        axes[0, 0].scatter(frame["default_specificity"], positions, color=style["threshold_default"], marker="+", label="Specificity at 0.5")
    axes[0, 0].set_yticks(positions, frame["label"])
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_title(f"{primary_encoder}: locked-test operating points", weight="bold")
    axes[0, 1].scatter(frame["threshold"], positions, color=style["threshold_balanced"], s=55)
    axes[0, 1].set_yticks(positions, [])
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_xlabel("Validation-selected threshold")
    axes[0, 1].set_title("Balanced thresholds", weight="bold")
    if split_metrics is not None and len(split_metrics):
        model_rows = split_metrics[split_metrics["encoder"].eq(primary_encoder)]
        if "classifier" in model_rows:
            model_rows = model_rows[model_rows["classifier"].eq("mil")]
        summary = model_rows.groupby("split")[[column for column in ("subset_accuracy", "precision_micro", "recall_micro", "f1_micro") if column in model_rows]].mean()
        summary.T.plot(kind="bar", ax=axes[1, 0], color=[style["train"], style["validation"], style["test"]][: len(summary)])
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_title("Split-wise case metrics", weight="bold")
        axes[1, 0].tick_params(axis="x", rotation=0)
        axes[1, 0].legend(frameon=False)
    else:
        axes[1, 0].axis("off")
    axes[1, 1].axis("off")
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def condition_distributions(data: pd.DataFrame, output: Path, style: dict) -> Path:
    ordered = data.sort_values("total_cases")
    fig, axes = plt.subplots(1, 2, figsize=(18, max(7, len(ordered) * 0.32)), sharey=True)
    axes[0].barh(ordered["label"], ordered["total_cases"], color=sns.color_palette("viridis", len(ordered)))
    axes[0].set_xlabel("Cases with positive target label")
    axes[0].set_title("Raw condition distribution", weight="bold")
    weighted = ordered["total_cases"] * ordered.get("effective_number_weight", 1.0)
    axes[1].barh(ordered["label"], weighted, color=sns.color_palette("magma", len(ordered)))
    axes[1].set_xlabel("Cases × normalized effective-number weight")
    axes[1].set_title("Class-balanced weighted distribution", weight="bold")
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def graphical_abstract(condition_count: int, heatmap: pd.DataFrame | None, output: Path, style: dict) -> Path:
    fig, axis = plt.subplots(figsize=(16, 6))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    boxes = [
        (0.03, "1–3 patient\nphotographs", style["baseline_panel"]),
        (0.25, "Frozen image\nencoder", "#D9F0E3"),
        (0.47, "Masked gated\nattention MIL", style["mil_panel"]),
        (0.69, f"{condition_count}-condition\ncase probabilities", "#DDEBFA"),
        (0.86, "Validation-selected\noperating points", "#E1F2E8"),
    ]
    for x, text, color in boxes:
        axis.text(x + 0.07, 0.55, text, ha="center", va="center", fontsize=13, weight="bold", bbox={"boxstyle": "round,pad=1", "facecolor": color, "edgecolor": style["text"]})
    for left, right in zip(boxes[:-1], boxes[1:], strict=True):
        axis.annotate("", xy=(right[0], 0.55), xytext=(left[0] + 0.14, 0.55), arrowprops={"arrowstyle": "->", "lw": 2, "color": style["text"]})
    subtitle = "Shared locked test cases; raw case-level probabilities retained"
    if heatmap is not None and len(heatmap):
        best = heatmap.loc[heatmap["auc_macro"].idxmax()]
        subtitle += f"\nBest completed smoke/full cell: {best['encoder']} + {best['classifier']} ({best['auc_macro']:.3f} macro ROC-AUC)"
    axis.text(0.5, 0.12, subtitle, ha="center", va="center", fontsize=11)
    axis.set_title("Case-level multi-image teledermatology workflow", fontsize=19, weight="bold", pad=20)
    return _save(fig, output, int(style["dpi"]))


def closed_model_heatmap(data: pd.DataFrame, output: Path, style: dict) -> Path:
    matrix = data.pivot(index="model", columns="label", values="auc")
    fig, axis = plt.subplots(figsize=(max(14, 0.52 * len(matrix.columns)), max(6, 0.55 * len(matrix))))
    sns.heatmap(
        matrix,
        annot=len(matrix.columns) <= 12,
        fmt=".3f",
        vmin=0.5,
        vmax=1.0,
        cmap=style["reviewer_heatmap"],
        ax=axis,
        cbar_kws={"label": "Test ROC-AUC"},
    )
    axis.set_title("Closed multimodal and case-level model comparison", weight="bold")
    axis.set_xlabel("Condition")
    axis.set_ylabel("Model")
    axis.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def closed_model_overall(data: pd.DataFrame, output: Path, style: dict) -> Path:
    metrics = [column for column in ("subset_accuracy", "precision_micro", "recall_micro", "f1_micro", "auc_micro", "pr_auc_micro") if column in data]
    melted = data.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="value")
    fig, axis = plt.subplots(figsize=(15, 7))
    sns.barplot(data=melted, x="model", y="value", hue="metric", ax=axis, palette="viridis")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=35)
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    axis.set_title("Closed multimodal model test metrics", weight="bold")
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def auc_radar(data: pd.DataFrame, output: Path, style: dict) -> Path:
    labels = list(dict.fromkeys(data["label"]))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    fig, axis = plt.subplots(figsize=(12, 12), subplot_kw={"projection": "polar"})
    for model, group in data.groupby("model", sort=False):
        values = group.set_index("label").reindex(labels)["auc"].to_numpy(dtype=float)
        axis.plot(closed_angles, np.r_[values, values[0]], lw=1.4, label=model)
    axis.set_xticks(angles, labels, fontsize=7)
    axis.set_ylim(0, 1)
    axis.set_yticks(np.arange(0.2, 1.01, 0.2))
    axis.legend(bbox_to_anchor=(1.3, 1.05), loc="upper left", fontsize=7, frameon=False)
    axis.set_title("Per-condition test ROC-AUC", weight="bold", pad=25)
    return _save(fig, output, int(style["dpi"]))


def metadata_figure(cases: pd.DataFrame, output: Path, style: dict, kind: str) -> Path | None:
    if kind == "confidence":
        x_column = next((column for column in cases if "self_reported" in column.lower() and "num" in column.lower()), None)
        y_column = next((column for column in cases if "confidence" in column.lower()), None)
        if not x_column or not y_column:
            return None
        fig, axis = plt.subplots(figsize=(8, 6))
        sns.regplot(data=cases, x=x_column, y=y_column, scatter_kws={"alpha": 0.25}, line_kws={"color": "#C53030"}, ax=axis)
        axis.set_title("Dermatologist confidence vs. self-reported variables", weight="bold")
    elif kind == "gradability_mst":
        columns = [column for column in cases if any(token in column.lower() for token in ("gradable", "monk_skin_tone"))][:3]
        if not columns:
            return None
        fig, axes = plt.subplots(1, len(columns), figsize=(6 * len(columns), 5))
        axes = np.atleast_1d(axes)
        for axis, column in zip(axes, columns, strict=True):
            counts = cases[column].fillna("Missing").astype(str).value_counts().sort_index()
            axis.bar(counts.index, counts.values, color="#6BA6A6")
            axis.set_title(column.replace("_", " ").title(), fontsize=10)
            axis.tick_params(axis="x", rotation=45)
    else:
        raise ValueError(kind)
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def image_counts_shot_types(images: pd.DataFrame, output: Path, style: dict) -> Path:
    counts = images.groupby("case_id").size().value_counts().sort_index()
    shot_column = next((column for column in images if "shot" in column.lower() or "view" in column.lower()), None)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(counts.index.astype(str), counts.values, color=style["train"])
    axes[0].set_title("Images per case", weight="bold")
    if shot_column:
        shot_counts = images[shot_column].fillna("Missing").value_counts()
        axes[1].bar(shot_counts.index.astype(str), shot_counts.values, color=["#4DB6AC", "#F3D36A", "#B39DDB"][: len(shot_counts)])
        axes[1].tick_params(axis="x", rotation=35)
        axes[1].set_title("Shot types", weight="bold")
    else:
        order_counts = images["bag_slot"].value_counts().sort_index()
        axes[1].bar([f"View {value + 1}" for value in order_counts.index], order_counts.values, color="#4DB6AC")
        axes[1].set_title("Retained view positions", weight="bold")
    fig.tight_layout()
    return _save(fig, output, int(style["dpi"]))


def generate_figures(
    cases: pd.DataFrame,
    images: pd.DataFrame,
    reports_dir: str | Path,
    artifacts_dir: str | Path,
    figure_config: dict[str, Any],
) -> list[Path]:
    reports_dir, artifacts_dir = Path(reports_dir), Path(artifacts_dir)
    plot_dir, figure_dir = reports_dir / "plot_data", reports_dir / "figures"
    style = figure_config["style"]
    outputs: list[Path] = []
    with manuscript_style(style):
        split_data = pd.read_csv(plot_dir / "split_composition.csv")
        condition_data = pd.read_csv(plot_dir / "condition_support.csv")
        outputs.append(cohort_overview(split_data, condition_data, cases, images, figure_dir / "figure1_cohort_preprocessing.png", style))
        outputs.append(split_donut(split_data, figure_dir / "mil_split_soft.png", style))
        outputs.append(condition_distributions(condition_data, figure_dir / "condition_distributions.png", style))
        outputs.append(image_counts_shot_types(images, figure_dir / "images_counts_shot_types.png", style))
        for kind, filename in (("confidence", "confidence_vs_selfreported.png"), ("gradability_mst", "gradability_mst.png")):
            generated = metadata_figure(cases, figure_dir / filename, style, kind)
            if generated:
                outputs.append(generated)
        heatmap_path = plot_dir / "encoder_classifier_heatmap.csv"
        heatmap_data = pd.read_csv(heatmap_path) if heatmap_path.exists() else None
        outputs.append(graphical_abstract(len(condition_data), heatmap_data, figure_dir / "graphical_abstract.png", style))
        if heatmap_path.exists():
            outputs.append(performance_heatmap(heatmap_data, figure_dir / "figure2_encoder_classifier_heatmap.png", style))
        outputs.append(mil_diagnostics(artifacts_dir, figure_dir / "figure3_mil_diagnostics.png", style))
        sweep_files = sorted(artifacts_dir.glob("models/mil/*/thresholds/validation_threshold_sweep.csv"))
        if sweep_files:
            sweep = pd.read_csv(sweep_files[0])
            outputs.append(threshold_curves(sweep, figure_dir / "figure4_threshold_curves.png", style))
            outputs.append(threshold_curves(sweep, figure_dir / "mil_sensitivity_specificity_curves_with_thresholds.png", style))
            youden = sweep.copy()
            fig, axis = plt.subplots(figsize=(12, 8))
            for label, group in youden.groupby("label", sort=False):
                axis.plot(group["threshold"], group["youden_j"], alpha=0.65, label=label)
            axis.set_xlabel("Decision threshold")
            axis.set_ylabel("Youden J")
            axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, frameon=False)
            outputs.append(_save(fig, figure_dir / "mil_youden_j_curves.png", int(style["dpi"])))
        operating_path = plot_dir / "operating_points.csv"
        if operating_path.exists():
            split_metrics_path = reports_dir / "tables" / "all_model_metrics.csv"
            split_metrics = pd.read_csv(split_metrics_path) if split_metrics_path.exists() else None
            outputs.append(operating_points(pd.read_csv(operating_path), figure_dir / "figure6_operating_points.png", style, split_metrics))
        closed_per_class = plot_dir / "closed_models_per_class.csv"
        closed_overall = plot_dir / "closed_models_overall.csv"
        if closed_per_class.exists():
            closed_data = pd.read_csv(closed_per_class)
            outputs.append(closed_model_heatmap(closed_data, figure_dir / "figure5_closed_model_comparison.png", style))
            outputs.append(auc_radar(closed_data, figure_dir / "all_models_auc_radar_plotly.png", style))
        if closed_overall.exists():
            outputs.append(closed_model_overall(pd.read_csv(closed_overall), figure_dir / "overall_test_metrics_closed_models.png", style))
    write_json(figure_dir / "figure_generation_manifest.json", {"outputs": [str(path) for path in outputs], "complete": True})
    return outputs
