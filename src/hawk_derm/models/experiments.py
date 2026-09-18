from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from hawk_derm.constants import slugify
from hawk_derm.evaluation.metrics import multilabel_metrics, per_class_metrics
from hawk_derm.evaluation.predictions import prediction_frame, save_predictions
from hawk_derm.features.bank import FeatureBank, build_case_bags
from hawk_derm.io import write_csv, write_json
from hawk_derm.models.mil import BagDataset, predict_mil, train_mil


def choose_device(choice: str = "auto") -> str:
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def prepare_bag_data(
    bank: FeatureBank,
    cases: pd.DataFrame,
    splits: pd.DataFrame,
    labels: list[str],
    max_images: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    label_slugs = [slugify(label) for label in labels]
    ordered = cases.merge(splits[["case_id", "split"]], on="case_id", validate="one_to_one")
    arrays = build_case_bags(bank, ordered, label_slugs, max_images=max_images)
    if np.any(~arrays["masks"].any(axis=1)):
        missing = arrays["case_ids"][~arrays["masks"].any(axis=1)]
        raise ValueError(f"Cases without embeddings cannot enter MIL: {missing[:10].tolist()}")
    return arrays, ordered["split"].to_numpy()


def subset_dataset(arrays: dict[str, np.ndarray], mask: np.ndarray) -> BagDataset:
    return BagDataset(arrays["bags"][mask], arrays["masks"][mask], arrays["labels"][mask], arrays["case_ids"][mask])


def run_mil_experiment(
    bank: FeatureBank,
    cases: pd.DataFrame,
    splits: pd.DataFrame,
    labels: list[str],
    mil_config: dict[str, Any],
    output_dir: str | Path,
    *,
    seed: int,
    device: str = "auto",
    max_images: int = 3,
    evaluation_splits: tuple[str, ...] = ("train", "validation", "test"),
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays, split_values = prepare_bag_data(bank, cases, splits, labels, max_images)
    datasets = {name: subset_dataset(arrays, split_values == name) for name in ("train", "validation", "test")}
    target_device = choose_device(device)
    result = train_mil(
        datasets["train"],
        datasets["validation"],
        bank.dimension,
        len(labels),
        mil_config["architecture"],
        mil_config["training"],
        device=target_device,
        seed=seed,
    )
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "architecture": mil_config["architecture"],
            "input_dim": bank.dimension,
            "class_count": len(labels),
            "labels": labels,
            "seed": seed,
            "encoder": bank.encoder,
        },
        output_dir / "model.pt",
    )
    write_json(output_dir / "history.json", {"history": result.history, "best_epoch": result.best_epoch, "complete": True})
    metric_rows = []
    for split_name in evaluation_splits:
        dataset = datasets[split_name]
        probabilities, attention = predict_mil(
            result.model, dataset, device=target_device, batch_size=int(mil_config["training"]["batch_size"])
        )
        frame = prediction_frame(
            dataset.case_ids,
            dataset.labels.numpy(),
            probabilities,
            labels,
            split=split_name,
            seed=seed,
            model=f"{bank.encoder}+mil",
        )
        for slot in range(attention.shape[1]):
            frame[f"attention__slot_{slot + 1}"] = attention[:, slot]
        save_predictions(
            output_dir / f"{split_name}_predictions.csv",
            frame,
            labels,
            {"evaluation_unit": "case", "attention_is_model_internal": True},
        )
        overall = multilabel_metrics(dataset.labels.numpy(), probabilities)
        metric_rows.append({"encoder": bank.encoder, "seed": seed, "split": split_name, **overall})
        write_csv(output_dir / f"{split_name}_per_class_metrics.csv", per_class_metrics(dataset.labels.numpy(), probabilities, labels))
    write_csv(output_dir / "overall_metrics.csv", pd.DataFrame(metric_rows))
    summary = {
        "encoder": bank.encoder,
        "seed": seed,
        "device": target_device,
        "best_epoch": result.best_epoch,
        "complete": True,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def sample_search_configs(base: dict[str, Any], search: dict[str, Any], trials: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(trials):
        trial = deepcopy(base)
        trial["training"]["learning_rate"] = float(np.exp(rng.uniform(np.log(search["learning_rate"][0]), np.log(search["learning_rate"][1]))))
        trial["training"]["weight_decay"] = float(np.exp(rng.uniform(np.log(search["weight_decay"][0]), np.log(search["weight_decay"][1]))))
        for key in ("dropout", "instance_dim", "attention_dim", "shared_dim"):
            trial["architecture"][key] = rng.choice(search[key]).item()
        trial["training"]["loss"] = str(rng.choice(search["loss"]))
        configs.append(trial)
    return configs


def run_mil_search(
    bank: FeatureBank,
    cases: pd.DataFrame,
    splits: pd.DataFrame,
    labels: list[str],
    mil_config: dict[str, Any],
    output_dir: str | Path,
    *,
    trials: int,
    seed: int,
    device: str,
    max_images: int,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    rows = []
    for trial_index, config in enumerate(sample_search_configs(mil_config, mil_config["search"], trials, seed)):
        trial_dir = output_dir / f"trial_{trial_index:03d}"
        run_mil_experiment(
            bank,
            cases,
            splits,
            labels,
            config,
            trial_dir,
            seed=seed + trial_index,
            device=device,
            max_images=max_images,
            evaluation_splits=("train", "validation"),
        )
        metrics = pd.read_csv(trial_dir / "overall_metrics.csv")
        validation = metrics.loc[metrics["split"].eq("validation")].iloc[0]
        rows.append(
            {
                "trial": trial_index,
                "validation_auc_macro": validation["auc_macro"],
                "validation_pr_auc_macro": validation["pr_auc_macro"],
                **{f"architecture__{key}": value for key, value in config["architecture"].items()},
                **{f"training__{key}": value for key, value in config["training"].items() if np.isscalar(value)},
            }
        )
    results = pd.DataFrame(rows).sort_values(["validation_auc_macro", "validation_pr_auc_macro"], ascending=False)
    write_csv(output_dir / "trials.csv", results)
    return results


def run_mil_cross_validation(
    bank: FeatureBank,
    cases: pd.DataFrame,
    splits: pd.DataFrame,
    labels: list[str],
    mil_config: dict[str, Any],
    output_dir: str | Path,
    *,
    folds: int,
    seed: int,
    device: str,
    max_images: int,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    development = splits[~splits["split"].eq("test")][["case_id"]].merge(cases, on="case_id", validate="one_to_one")
    label_columns = [f"is_{slugify(label)}" for label in labels]
    splitter = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for fold, (train_index, validation_index) in enumerate(splitter.split(development, development[label_columns]), start=1):
        fold_splits = splits.copy()
        fold_splits["split"] = "test"
        fold_splits.loc[fold_splits["case_id"].isin(development.iloc[train_index]["case_id"]), "split"] = "train"
        fold_splits.loc[fold_splits["case_id"].isin(development.iloc[validation_index]["case_id"]), "split"] = "validation"
        fold_dir = output_dir / f"fold_{fold:02d}"
        run_mil_experiment(
            bank,
            cases,
            fold_splits,
            labels,
            mil_config,
            fold_dir,
            seed=seed + fold,
            device=device,
            max_images=max_images,
            evaluation_splits=("train", "validation"),
        )
        metrics = pd.read_csv(fold_dir / "overall_metrics.csv")
        validation = metrics.loc[metrics["split"].eq("validation")].iloc[0].to_dict()
        rows.append({"fold": fold, **validation})
    frame = pd.DataFrame(rows)
    write_csv(output_dir / "cross_validation_metrics.csv", frame)
    return frame


def run_repeated_mil(
    bank: FeatureBank,
    cases: pd.DataFrame,
    splits: pd.DataFrame,
    labels: list[str],
    mil_config: dict[str, Any],
    output_dir: str | Path,
    *,
    seeds: list[int],
    device: str,
    max_images: int,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    metrics = []
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        run_mil_experiment(bank, cases, splits, labels, mil_config, seed_dir, seed=seed, device=device, max_images=max_images)
        frame = pd.read_csv(seed_dir / "overall_metrics.csv")
        metrics.append(frame)
    combined = pd.concat(metrics, ignore_index=True)
    write_csv(output_dir / "repeated_metrics_long.csv", combined)
    numeric = combined.select_dtypes(include=[np.number]).columns.drop("seed", errors="ignore")
    summary = combined.groupby(["encoder", "split"])[list(numeric)].agg(["mean", "std"]).reset_index()
    summary.columns = ["__".join(filter(None, map(str, column))) if isinstance(column, tuple) else str(column) for column in summary.columns]
    write_csv(output_dir / "repeated_metrics_summary.csv", summary)
    return combined
