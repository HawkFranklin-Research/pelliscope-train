from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hawk_derm.constants import slugify
from hawk_derm.features.registry import load_encoder_registry
from hawk_derm.config import load_yaml
from hawk_derm.figures.plot_data import generate_mil_plot_data
from hawk_derm.io import write_csv, write_json


def generate_tables(
    cases: pd.DataFrame,
    images: pd.DataFrame,
    splits: pd.DataFrame,
    labels: list[str],
    artifacts_dir: str | Path,
    reports_dir: str | Path,
    primary_mil_root: str | Path | None = None,
) -> list[Path]:
    artifacts_dir, reports_dir = Path(artifacts_dir), Path(reports_dir)
    table_dir = reports_dir / "tables"
    plot_dir = reports_dir / "plot_data"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    label_columns = [f"is_{slugify(label)}" for label in labels]
    cohort = pd.DataFrame(
        [
            {
                "cases": cases["case_id"].nunique(),
                "image_rows": len(images),
                "unique_resolved_paths": images["resolved_image_path"].nunique(),
                "target_conditions": len(labels),
                "all_zero_target_cases": int(cases[label_columns].sum(axis=1).eq(0).sum()),
                "single_label_cases": int(cases[label_columns].sum(axis=1).eq(1).sum()),
                "multi_label_cases": int(cases[label_columns].sum(axis=1).gt(1).sum()),
                "median_images_per_case": float(images.groupby("case_id").size().median()),
            }
        ]
    )
    outputs.append(write_csv(table_dir / "headline_study_numbers.csv", cohort))

    image_counts = images.groupby("case_id").size().rename("images")
    split_table = splits.merge(image_counts, left_on="case_id", right_index=True, how="left").groupby("split").agg(
        cases=("case_id", "nunique"), images=("images", "sum")
    ).reset_index()
    split_table["split"] = pd.Categorical(split_table["split"], ["train", "validation", "test"], ordered=True)
    split_table = split_table.sort_values("split")
    outputs.append(write_csv(table_dir / "cohort_sizes_by_split.csv", split_table))
    outputs.append(write_csv(plot_dir / "split_composition.csv", split_table))

    label_rows = []
    joined = cases.merge(splits[["case_id", "split"]], on="case_id", validate="one_to_one")
    for label, column in zip(labels, label_columns, strict=True):
        row: dict[str, Any] = {"label": label, "total_cases": int(cases[column].sum())}
        for split_name in ("train", "validation", "test"):
            row[f"{split_name}_cases"] = int(joined.loc[joined["split"].eq(split_name), column].sum())
        row["train_images"] = int(images.merge(joined[["case_id", column]], on="case_id")[column].sum())
        label_rows.append(row)
    label_support = pd.DataFrame(label_rows)
    beta = float(load_yaml("configs/mil.yaml")["training"]["class_balanced_beta"])
    effective = (1 - beta) / (1 - np.power(beta, label_support["train_cases"].clip(lower=1)))
    label_support["effective_number_weight"] = effective / effective.mean()
    outputs.append(write_csv(table_dir / "class_imbalance_diagnostics.csv", label_support))
    outputs.append(write_csv(plot_dir / "condition_support.csv", label_support))

    registry = load_encoder_registry()
    encoder_table = pd.DataFrame(
        [
            {
                "encoder": spec.display_name,
                "key": spec.key,
                "model_id": spec.model_id,
                "revision": spec.revision,
                "embedding_dimension": spec.dimension,
                "backend": spec.backend,
            }
            for spec in registry.values()
        ]
    )
    outputs.append(write_csv(table_dir / "encoder_registry.csv", encoder_table))

    metric_files = sorted(artifacts_dir.glob("models/classical/*/*/overall_metrics.csv"))
    mil_metric_files = sorted(artifacts_dir.glob("models/mil/*/repeated_metrics_long.csv"))
    if primary_mil_root:
        primary_mil_root = Path(primary_mil_root)
        primary_encoder = primary_mil_root.parent.name
        mil_metric_files = [path for path in mil_metric_files if path.parent.name != primary_encoder]
        mil_metric_files.append(primary_mil_root / "repeated_metrics_long.csv")
    metric_files += mil_metric_files
    metric_files = [path for path in metric_files if path.is_file()]
    metric_frames = []
    for path in metric_files:
        frame = pd.read_csv(path)
        if "classifier" not in frame:
            frame["classifier"] = "mil"
        metric_frames.append(frame)
    if metric_frames:
        model_metrics = pd.concat(metric_frames, ignore_index=True)
        outputs.append(write_csv(table_dir / "all_model_metrics.csv", model_metrics))
        test = model_metrics[model_metrics["split"].eq("test")]
        heatmap = test.groupby(["encoder", "classifier"], as_index=False)[["auc_macro", "auc_micro"]].mean()
        outputs.append(write_csv(plot_dir / "encoder_classifier_heatmap.csv", heatmap))

    threshold_files = (
        [Path(primary_mil_root) / "thresholds" / "test_operating_points.csv"]
        if primary_mil_root
        else sorted(artifacts_dir.glob("models/mil/*/thresholds/test_operating_points.csv"))
    )
    threshold_files = [path for path in threshold_files if path.is_file()]
    if threshold_files:
        threshold_tables = []
        for path in threshold_files:
            frame = pd.read_csv(path)
            encoder = Path(primary_mil_root).parents[0].name if primary_mil_root else path.parents[1].name
            frame.insert(0, "encoder", encoder)
            threshold_tables.append(frame)
        operating = pd.concat(threshold_tables, ignore_index=True)
        outputs.append(write_csv(table_dir / "operating_points.csv", operating))
        outputs.append(write_csv(plot_dir / "operating_points.csv", operating))

    tuning_files = (
        [Path(primary_mil_root) / "tuning" / "trials.csv"]
        if primary_mil_root
        else sorted(artifacts_dir.glob("models/mil/*/tuning/trials.csv"))
    )
    tuning_files = [path for path in tuning_files if path.is_file()]
    if tuning_files:
        trials = []
        for path in tuning_files:
            frame = pd.read_csv(path)
            encoder = Path(primary_mil_root).parents[0].name if primary_mil_root else path.parents[1].name
            frame.insert(0, "encoder", encoder)
            trials.append(frame)
        outputs.append(write_csv(table_dir / "hyperparameter_trials.csv", pd.concat(trials, ignore_index=True)))

    latency_files = sorted(table_dir.glob("inference_latency_*.csv"))
    if latency_files:
        outputs.append(write_csv(table_dir / "inference_latency_all_devices.csv", pd.concat([pd.read_csv(path) for path in latency_files], ignore_index=True)))

    mil_config = load_yaml("configs/mil.yaml")
    threshold_config = mil_config["thresholds"]
    threshold_grid = np.arange(
        float(threshold_config["grid_start"]),
        float(threshold_config["grid_stop"]) + float(threshold_config["grid_step"]) / 2,
        float(threshold_config["grid_step"]),
    )
    figure_config = load_yaml("configs/figures.yaml")
    outputs.extend(
        generate_mil_plot_data(
            artifacts_dir,
            plot_dir,
            labels,
            primary_encoder=str(figure_config.get("primary_encoder", "siglip2_so400m")),
            threshold_grid=threshold_grid,
            primary_root=primary_mil_root,
        )
    )

    write_json(table_dir / "table_generation_manifest.json", {"outputs": [str(path) for path in outputs], "complete": True})
    return outputs
