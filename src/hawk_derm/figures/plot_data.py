from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from hawk_derm.evaluation.metrics import per_class_metrics
from hawk_derm.evaluation.predictions import arrays_from_prediction_frame
from hawk_derm.evaluation.thresholds import threshold_sweep
from hawk_derm.io import write_csv


def _seed(path: Path) -> int:
    return int(path.parent.name.removeprefix("seed_"))


def _prediction_arrays(paths: list[Path], labels: list[str]) -> list[tuple[int, pd.DataFrame, np.ndarray, np.ndarray]]:
    runs: list[tuple[int, pd.DataFrame, np.ndarray, np.ndarray]] = []
    reference_ids: pd.Series | None = None
    reference_truth: np.ndarray | None = None
    for path in paths:
        frame = pd.read_csv(path).sort_values("case_id").reset_index(drop=True)
        truth, probability = arrays_from_prediction_frame(frame, labels)
        case_ids = frame["case_id"].astype(str)
        if reference_ids is None:
            reference_ids = case_ids
            reference_truth = truth
        elif not reference_ids.equals(case_ids) or not np.array_equal(reference_truth, truth):
            raise ValueError(f"Repeated predictions are not paired by case ID and truth: {path}")
        runs.append((_seed(path), frame, truth, probability))
    return runs


def _safe_macro_auc(truth: np.ndarray, probability: np.ndarray) -> float:
    values = [roc_auc_score(truth[:, index], probability[:, index]) for index in range(truth.shape[1]) if np.unique(truth[:, index]).size == 2]
    return float(np.mean(values)) if values else float("nan")


def _curve_rows(seed: int, truth: np.ndarray, probability: np.ndarray, points: int = 201) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    grid = np.linspace(0.0, 1.0, points)

    micro_fpr, micro_tpr, _ = roc_curve(truth.ravel(), probability.ravel())
    roc_series = {"micro": np.interp(grid, micro_fpr, micro_tpr)}
    class_rocs = []
    for index in range(truth.shape[1]):
        if np.unique(truth[:, index]).size != 2:
            continue
        fpr, tpr, _ = roc_curve(truth[:, index], probability[:, index])
        class_rocs.append(np.interp(grid, fpr, tpr))
    if class_rocs:
        roc_series["macro"] = np.mean(class_rocs, axis=0)
    for averaging, values in roc_series.items():
        auc = roc_auc_score(truth.ravel(), probability.ravel()) if averaging == "micro" else _safe_macro_auc(truth, probability)
        rows.extend({"seed": seed, "curve": "roc", "averaging": averaging, "x": x, "y": y, "score": auc} for x, y in zip(grid, values, strict=True))

    micro_precision, micro_recall, _ = precision_recall_curve(truth.ravel(), probability.ravel())
    order = np.argsort(micro_recall)
    pr_series = {"micro": np.interp(grid, micro_recall[order], micro_precision[order])}
    class_prs = []
    for index in range(truth.shape[1]):
        if truth[:, index].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(truth[:, index], probability[:, index])
        order = np.argsort(recall)
        class_prs.append(np.interp(grid, recall[order], precision[order]))
    if class_prs:
        pr_series["macro"] = np.mean(class_prs, axis=0)
    per_class_ap = [
        average_precision_score(truth[:, index], probability[:, index])
        for index in range(truth.shape[1])
        if truth[:, index].sum() > 0
    ]
    pr_scores = {
        "micro": float(average_precision_score(truth.ravel(), probability.ravel())),
        "macro": float(np.mean(per_class_ap)) if per_class_ap else float("nan"),
    }
    for averaging, values in pr_series.items():
        rows.extend(
            {"seed": seed, "curve": "pr", "averaging": averaging, "x": x, "y": y, "score": pr_scores[averaging]}
            for x, y in zip(grid, values, strict=True)
        )
    return rows


def _history_data(primary_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(primary_root.glob("seed_*/history.json")):
        payload = json.loads(path.read_text())
        frame = pd.DataFrame(payload.get("history", []))
        if frame.empty:
            continue
        frame.insert(0, "seed", _seed(path))
        frame.insert(0, "encoder", primary_root.name)
        frame["best_epoch"] = payload.get("best_epoch")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _per_class_data(primary_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(primary_root.glob("seed_*/*_per_class_metrics.csv")):
        split = path.name.removesuffix("_per_class_metrics.csv")
        frame = pd.read_csv(path)
        frame.insert(0, "split", split)
        frame.insert(0, "seed", _seed(path))
        frame.insert(0, "encoder", primary_root.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _threshold_curve_data(
    runs: list[tuple[int, pd.DataFrame, np.ndarray, np.ndarray]],
    labels: list[str],
    grid: np.ndarray,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for seed, _, truth, probability in runs:
        frame = threshold_sweep(truth, probability, labels, grid)
        frame.insert(0, "seed", seed)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    long = pd.concat(frames, ignore_index=True)
    aggregated = long.groupby(["label", "threshold"], as_index=False).agg(
        sensitivity_mean=("sensitivity", "mean"),
        sensitivity_sd=("sensitivity", "std"),
        specificity_mean=("specificity", "mean"),
        specificity_sd=("specificity", "std"),
        youden_j_mean=("youden_j", "mean"),
        youden_j_sd=("youden_j", "std"),
    )
    aggregated[["sensitivity_sd", "specificity_sd", "youden_j_sd"]] = aggregated[["sensitivity_sd", "specificity_sd", "youden_j_sd"]].fillna(0.0)
    selected_map = selected.set_index("label")["threshold"]
    aggregated["selected_threshold"] = aggregated["label"].map(selected_map)
    aggregated["selected"] = np.isclose(aggregated["threshold"], aggregated["selected_threshold"])
    return aggregated


def _split_metrics(
    primary_root: Path,
    labels: list[str],
    selected: pd.DataFrame,
) -> pd.DataFrame:
    thresholds = selected.set_index("label").loc[labels, "threshold"].to_numpy(dtype=float)
    frames = []
    for split in ("train", "validation", "test"):
        runs = _prediction_arrays(sorted(primary_root.glob(f"seed_*/{split}_predictions.csv")), labels)
        if not runs:
            continue
        truth = runs[0][2]
        probability = np.mean([run[3] for run in runs], axis=0)
        metrics = per_class_metrics(truth, probability, labels, thresholds)
        metrics["accuracy"] = (metrics["tp"] + metrics["tn"]) / (metrics["support_positive"] + metrics["support_negative"])
        metrics.insert(0, "split", split)
        metrics.insert(0, "encoder", primary_root.name)
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _threshold_accuracy_comparison(
    runs: list[tuple[int, pd.DataFrame, np.ndarray, np.ndarray]],
    labels: list[str],
    selected: pd.DataFrame,
) -> pd.DataFrame:
    if not runs:
        return pd.DataFrame()
    truth = runs[0][2]
    probability = np.mean([run[3] for run in runs], axis=0)
    selected_thresholds = selected.set_index("label").loc[labels, "threshold"].to_numpy(dtype=float)
    balanced = per_class_metrics(truth, probability, labels, selected_thresholds)
    default = per_class_metrics(truth, probability, labels, 0.5)
    return pd.DataFrame(
        {
            "label": labels,
            "accuracy_default": (default["tp"] + default["tn"]) / (default["support_positive"] + default["support_negative"]),
            "accuracy_balanced": (balanced["tp"] + balanced["tn"]) / (balanced["support_positive"] + balanced["support_negative"]),
            "balanced_threshold": selected_thresholds,
        }
    )


def generate_mil_plot_data(
    artifacts_dir: str | Path,
    plot_dir: str | Path,
    labels: list[str],
    *,
    primary_encoder: str,
    threshold_grid: np.ndarray,
    primary_root: str | Path | None = None,
) -> list[Path]:
    artifacts_dir, plot_dir = Path(artifacts_dir), Path(plot_dir)
    primary_root = Path(primary_root) if primary_root else artifacts_dir / "models" / "mil" / primary_encoder
    if not primary_root.exists():
        candidates = sorted((artifacts_dir / "models" / "mil").glob("*"))
        if not candidates:
            return []
        primary_root = candidates[0]
    outputs: list[Path] = []

    history = _history_data(primary_root)
    if not history.empty:
        outputs.append(write_csv(plot_dir / "mil_history_long.csv", history))

    per_class = _per_class_data(primary_root)
    if not per_class.empty:
        outputs.append(write_csv(plot_dir / "mil_per_class_metrics_long.csv", per_class))

    selected_path = primary_root / "thresholds" / "validation_selected_thresholds.csv"
    if not selected_path.exists():
        return outputs
    selected = pd.read_csv(selected_path)

    curve_rows = []
    test_runs = _prediction_arrays(sorted(primary_root.glob("seed_*/test_predictions.csv")), labels)
    for seed, _, truth, probability in test_runs:
        curve_rows.extend(_curve_rows(seed, truth, probability))
    if curve_rows:
        outputs.append(write_csv(plot_dir / "mil_roc_pr_curves.csv", pd.DataFrame(curve_rows)))

    validation_runs = _prediction_arrays(sorted(primary_root.glob("seed_*/validation_predictions.csv")), labels)
    validation_curves = _threshold_curve_data(validation_runs, labels, threshold_grid, selected)
    if not validation_curves.empty:
        validation_curves.insert(0, "split", "validation")
        outputs.append(write_csv(plot_dir / "validation_threshold_curves.csv", validation_curves))

    test_curves = _threshold_curve_data(test_runs, labels, threshold_grid, selected)
    if not test_curves.empty:
        test_curves.insert(0, "split", "test")
        outputs.append(write_csv(plot_dir / "test_sensitivity_specificity_curves.csv", test_curves))

    threshold_accuracy = _threshold_accuracy_comparison(test_runs, labels, selected)
    if not threshold_accuracy.empty:
        outputs.append(write_csv(plot_dir / "threshold_accuracy_comparison.csv", threshold_accuracy))

    split_metrics = _split_metrics(primary_root, labels, selected)
    if not split_metrics.empty:
        outputs.append(write_csv(plot_dir / "mil_split_per_class_metrics.csv", split_metrics))
    return outputs
