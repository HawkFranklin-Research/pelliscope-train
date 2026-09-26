from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from hawk_derm.constants import slugify
from hawk_derm.data.external import ExternalInput
from hawk_derm.evaluation.metrics import multilabel_metrics, per_class_metrics
from hawk_derm.evaluation.predictions import prediction_frame, save_predictions
from hawk_derm.features.bank import FeatureBank
from hawk_derm.io import write_csv, write_json


@dataclass
class ConstantProbability:
    probability: float

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = np.full(len(features), self.probability, dtype=float)
        return np.column_stack([1 - positive, positive])


def make_estimator(name: str, config: dict[str, Any], seed: int, workers: int = 1) -> BaseEstimator:
    params = dict(config.get("params", {}))
    if name == "logistic":
        estimator: BaseEstimator = LogisticRegression(random_state=seed, **params)
    elif name == "svm_linear":
        folds = int(params.pop("calibration_folds", 3))
        estimator = CalibratedClassifierCV(
            LinearSVC(random_state=seed, **params), cv=folds, method="sigmoid", n_jobs=workers
        )
    elif name == "random_forest":
        if int(params.get("n_jobs", -1)) == -1:
            params["n_jobs"] = workers
        estimator = RandomForestClassifier(random_state=seed, **params)
    elif name == "gradient_boosting":
        estimator = GradientBoostingClassifier(random_state=seed, **params)
    elif name == "knn":
        params.setdefault("n_jobs", workers)
        estimator = KNeighborsClassifier(**params)
    else:
        raise ValueError(f"Unknown classifier: {name}")
    if config.get("scale", False):
        return Pipeline([("scale", StandardScaler()), ("model", estimator)])
    return estimator


def _fit_label_model(
    index: int,
    classifier_name: str,
    classifier_config: dict[str, Any],
    embeddings: np.ndarray,
    image_truth: np.ndarray,
    train_mask: np.ndarray,
    seed: int,
    estimator_workers: int,
) -> Any:
    target = image_truth[train_mask, index]
    if np.unique(target).size < 2:
        return ConstantProbability(float(target.mean()))
    effective_config = dict(classifier_config)
    effective_config["params"] = dict(classifier_config.get("params", {}))
    if classifier_name == "svm_linear" and np.bincount(target).min() < int(
        effective_config["params"].get("calibration_folds", 3)
    ):
        effective_config["params"]["calibration_folds"] = int(np.bincount(target).min())
        if effective_config["params"]["calibration_folds"] < 2:
            model: Any = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed + index)
        else:
            model = make_estimator(classifier_name, effective_config, seed + index, estimator_workers)
    else:
        model = make_estimator(classifier_name, effective_config, seed + index, estimator_workers)
    model.fit(embeddings[train_mask], target)
    return model


def _positive_probability(model: Any, features: np.ndarray) -> np.ndarray:
    values = model.predict_proba(features)
    return values[:, -1] if values.ndim == 2 else values


def aggregate_case_probabilities(
    case_ids: np.ndarray,
    probabilities: np.ndarray,
    labels_by_case: pd.DataFrame,
    labels: list[str],
    rule: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.DataFrame(probabilities, columns=[slugify(label) for label in labels])
    frame.insert(0, "case_id", case_ids.astype(str))
    if rule == "mean":
        aggregated = frame.groupby("case_id", sort=False).mean()
    elif rule == "max":
        aggregated = frame.groupby("case_id", sort=False).max()
    else:
        raise ValueError(f"Unsupported case aggregation: {rule}")
    labels_indexed = labels_by_case.assign(case_id=labels_by_case["case_id"].astype(str)).set_index("case_id")
    truth = labels_indexed.loc[aggregated.index, [f"is_{slugify(label)}" for label in labels]].to_numpy(dtype=np.uint8)
    return aggregated.index.to_numpy(), truth, aggregated.to_numpy(dtype=float)


def run_classical_experiment(
    bank: FeatureBank,
    cases: pd.DataFrame,
    splits: pd.DataFrame,
    labels: list[str],
    classifier_name: str,
    classifier_config: dict[str, Any],
    output_dir: str | Path,
    *,
    seed: int = 42,
    aggregation: str = "mean",
    workers: int = 1,
    external: list[ExternalInput] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_table = cases.merge(splits[["case_id", "split"]], on="case_id", validate="one_to_one")
    case_lookup = case_table.set_index(case_table["case_id"].astype(str))
    image_splits = np.asarray([case_lookup.loc[str(case_id), "split"] for case_id in bank.case_ids])
    image_truth = np.asarray(
        [case_lookup.loc[str(case_id), [f"is_{slugify(label)}" for label in labels]].to_numpy(dtype=np.uint8) for case_id in bank.case_ids]
    )
    train_mask = image_splits == "train"
    # Forests already parallelize each tree ensemble internally. k-NN fitting is
    # only data storage and uses its worker budget during prediction. The three
    # otherwise serial estimator families parallelize the 25 one-vs-rest label
    # fits within this single top-level model configuration.
    parallel_labels = classifier_name in {"logistic", "svm_linear", "gradient_boosting"} and workers > 1
    if parallel_labels:
        label_workers = min(workers, len(labels))
        with parallel_config(backend="loky", inner_max_num_threads=1):
            models = Parallel(n_jobs=label_workers, max_nbytes="10M", mmap_mode="r")(
                delayed(_fit_label_model)(
                    index,
                    classifier_name,
                    classifier_config,
                    bank.embeddings,
                    image_truth,
                    train_mask,
                    seed,
                    1,
                )
                for index in range(len(labels))
            )
    else:
        models = [
            _fit_label_model(
                index,
                classifier_name,
                classifier_config,
                bank.embeddings,
                image_truth,
                train_mask,
                seed,
                workers,
            )
            for index in range(len(labels))
        ]
    joblib.dump(models, output_dir / "model.joblib")

    all_probabilities = np.column_stack([_positive_probability(model, bank.embeddings) for model in models])
    metric_rows = []
    outputs = []
    for split_name in ("train", "validation", "test"):
        mask = image_splits == split_name
        image_frame = prediction_frame(
            [str(value) for value in bank.image_ids[mask]],
            image_truth[mask],
            all_probabilities[mask],
            labels,
            split=split_name,
            seed=seed,
            model=f"{bank.encoder}+{classifier_name}:image_level",
        ).rename(columns={"case_id": "image_id"})
        image_frame.insert(0, "case_id", bank.case_ids[mask].astype(str))
        image_path = output_dir / f"{split_name}_image_predictions.csv"
        save_predictions(image_path, image_frame, labels, {"evaluation_unit": "image"})
        outputs.append(image_path)

        case_ids, truth, probabilities = aggregate_case_probabilities(
            bank.case_ids[mask], all_probabilities[mask], cases, labels, aggregation
        )
        case_frame = prediction_frame(
            case_ids,
            truth,
            probabilities,
            labels,
            split=split_name,
            seed=seed,
            model=f"{bank.encoder}+{classifier_name}:case_{aggregation}",
        )
        case_path = output_dir / f"{split_name}_case_predictions.csv"
        save_predictions(case_path, case_frame, labels, {"evaluation_unit": "case", "aggregation": aggregation})
        outputs.append(case_path)
        overall = multilabel_metrics(truth, probabilities)
        metric_rows.append({"encoder": bank.encoder, "classifier": classifier_name, "split": split_name, **overall})
        write_csv(output_dir / f"{split_name}_per_class_metrics.csv", per_class_metrics(truth, probabilities, labels))
    for item in external or []:
        # Same fitted label models, same image-to-case aggregation as the study splits.
        external_probabilities = np.column_stack([_positive_probability(model, item.bank.embeddings) for model in models])
        external_truth = item.cases.set_index(item.cases["case_id"].astype(str)).loc[
            item.bank.case_ids.astype(str), [f"is_{slugify(label)}" for label in labels]
        ].to_numpy(dtype=np.uint8)
        image_frame = prediction_frame(
            [str(value) for value in item.bank.image_ids],
            external_truth,
            external_probabilities,
            labels,
            split=item.name,
            seed=seed,
            model=f"{bank.encoder}+{classifier_name}:image_level",
        ).rename(columns={"case_id": "image_id"})
        image_frame.insert(0, "case_id", item.bank.case_ids.astype(str))
        image_path = output_dir / f"{item.name}_image_predictions.csv"
        save_predictions(image_path, image_frame, labels, {"evaluation_unit": "image", "external_bank_sha256": item.bank_sha256})
        case_ids, truth, probabilities = aggregate_case_probabilities(
            item.bank.case_ids, external_probabilities, item.cases, labels, aggregation
        )
        case_path = output_dir / f"{item.name}_case_predictions.csv"
        save_predictions(
            case_path,
            prediction_frame(case_ids, truth, probabilities, labels, split=item.name, seed=seed, model=f"{bank.encoder}+{classifier_name}:case_{aggregation}"),
            labels,
            {"evaluation_unit": "external_pseudo_case", "aggregation": aggregation, "external_bank_sha256": item.bank_sha256},
        )
        outputs.extend([image_path, case_path])
        metric_rows.append({"encoder": bank.encoder, "classifier": classifier_name, "split": item.name, **multilabel_metrics(truth, probabilities)})
    write_csv(output_dir / "overall_metrics.csv", pd.DataFrame(metric_rows))
    summary = {
        "encoder": bank.encoder,
        "classifier": classifier_name,
        "seed": seed,
        "aggregation": aggregation,
        "model_count": len(models),
        "cpu_workers": workers,
        "parallel_strategy": "one_vs_rest_labels" if parallel_labels else "estimator_internal",
        "complete": True,
    }
    write_json(output_dir / "summary.json", summary)
    return summary
