#!/usr/bin/env python3
"""Evaluate every trained model on each external cohort, next to its internal-test result.

Reads predictions already written during training (no model is loaded here): all MIL
ensembles and final models, and all classical encoder x classifier models. Thresholds are
the MIL validation-selected ones, applied unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score

from _common import load_context, mil_run_root

from hawk_derm.config import load_yaml, path_from
from hawk_derm.constants import slugify
from hawk_derm.data.external import load_external_cohorts
from hawk_derm.evaluation.external import (
    bootstrap_intervals,
    confusion_among_represented,
    external_metrics,
    first_of_each_group,
    group_resample_indices,
    paired_difference,
    per_label_table,
)
from hawk_derm.evaluation.metrics import multilabel_metrics
from hawk_derm.evaluation.predictions import arrays_from_prediction_frame
from hawk_derm.features.registry import load_encoder_registry
from hawk_derm.io import sha256_file, write_csv, write_json
from hawk_derm.runtime import resolve_cpu_workers
from hawk_derm.statistics.inference import holm_adjust


def model_sources(config: dict, cohort_prefix: str, run_tag: str) -> list[dict]:
    artifacts = path_from(config, "artifacts_dir")
    classifiers = list(load_yaml("configs/classifiers.yaml")["classifiers"])
    sources = []
    for encoder in load_encoder_registry():
        root = mil_run_root(config, encoder, run_tag)
        thresholds = root / "thresholds" / "validation_selected_thresholds.csv"
        sources.append(
            {
                "model": f"{encoder}+mil_ensemble", "encoder": encoder, "classifier": "mil_ensemble",
                "external": root / "ensemble" / f"{cohort_prefix}_predictions.csv",
                "internal": root / "ensemble" / "test_predictions.csv",
                "thresholds": thresholds if thresholds.is_file() else None,
            }
        )
        sources.append(
            {
                "model": f"{encoder}+mil_final_model", "encoder": encoder, "classifier": "mil_final_model",
                "external": root / "final_model" / f"{cohort_prefix}_predictions.csv",
                "internal": root / "final_model" / "test_predictions.csv",
                "thresholds": None,
            }
        )
        for classifier in classifiers:
            model_root = artifacts / "models" / "classical" / encoder / classifier
            sources.append(
                {
                    "model": f"{encoder}+{classifier}", "encoder": encoder, "classifier": classifier,
                    "external": model_root / f"{cohort_prefix}_case_predictions.csv",
                    "internal": model_root / "test_case_predictions.csv",
                    "thresholds": None,
                }
            )
    return [source for source in sources if source["external"].is_file()]


def load_scores(path: Path, labels: list[str], case_order: list[str] | None = None) -> tuple[list[str], np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    frame["case_id"] = frame["case_id"].astype(str)
    if case_order is not None:
        frame = frame.set_index("case_id").loc[case_order].reset_index()
    truth, probability = arrays_from_prediction_frame(frame, labels)
    return frame["case_id"].tolist(), truth, probability


def internal_restricted(path: Path, labels: list[str], columns: np.ndarray) -> dict[str, float]:
    """Internal-test metrics on the same labels the external cohort represents, for side-by-side reporting."""
    if not path.is_file():
        return {}
    _, truth, probability = load_scores(path, labels)
    auc = [roc_auc_score(truth[:, c], probability[:, c]) for c in columns if 0 < truth[:, c].sum() < len(truth)]
    ap = [average_precision_score(truth[:, c], probability[:, c]) for c in columns if truth[:, c].sum()]
    overall = multilabel_metrics(truth, probability)
    return {
        "internal_macro_auc_same_labels": float(np.mean(auc)) if auc else np.nan,
        "internal_macro_ap_same_labels": float(np.mean(ap)) if ap else np.nan,
        "internal_macro_auc_all_labels": overall["auc_macro"],
        "internal_macro_ap_all_labels": overall["pr_auc_macro"],
        "internal_top1_hit": overall["top1_hit"],
        "internal_top3_hit": overall["top3_hit"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--run-tag", default="canonical")
    parser.add_argument("--cohort", default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    config = load_context(args.config)
    labels = config["labels"]
    workers = resolve_cpu_workers(args.workers)
    production = load_yaml("configs/production.yaml").get("production_models", {})
    production_encoders = {entry["encoder"] for entry in production.values() if entry.get("encoder")}

    for cohort in load_external_cohorts(config):
        if args.cohort not in (None, cohort.name) or not cohort.prepared:
            continue
        settings = cohort.settings.get("evaluation", {})
        replicates = args.replicates or int(settings.get("bootstrap_replicates", 2000))
        seed = int(settings.get("seed", 20260926))
        top_k = tuple(int(k) for k in settings.get("top_k", [1, 3, 5]))
        output = path_from(config, "reports_dir") / "external" / cohort.name
        sources = model_sources(config, cohort.prediction_name(), args.run_tag)
        if not sources:
            print(f"[external] {cohort.name}: no external predictions found; train models after extracting external features")
            continue
        cases = pd.read_csv(cohort.case_manifest_path)
        cases["case_id"] = cases["case_id"].astype(str)
        case_order = cases["case_id"].tolist()
        expected_truth = cases[[f"is_{slugify(label)}" for label in labels]].to_numpy(dtype=np.uint8)
        groups = cases["resampling_group"].to_numpy()
        samples = group_resample_indices(groups, replicates, seed)
        unique = first_of_each_group(groups)

        loaded = {}
        for source in sources:
            case_ids, truth, probability = load_scores(source["external"], labels, case_order)
            if case_ids != case_order or not np.array_equal(truth, expected_truth):
                raise ValueError(f"Ground truth in {source['external']} differs from the external case manifest")
            thresholds = None
            if source["thresholds"] is not None:
                thresholds = pd.read_csv(source["thresholds"]).set_index("label").loc[labels, "threshold"].to_numpy(dtype=float)
            loaded[source["model"]] = (source, truth, probability, thresholds)
        truth = next(iter(loaded.values()))[1]
        columns = np.flatnonzero(truth.sum(axis=0) > 0)

        rows, per_label = [], []
        for name, (source, _, probability, thresholds) in loaded.items():
            row = {
                "model": name, "encoder": source["encoder"], "classifier": source["classifier"],
                "production_encoder": source["encoder"] in production_encoders,
                **external_metrics(truth, probability, thresholds, top_k),
                **{f"unique_images_{key}": value for key, value in external_metrics(truth[unique], probability[unique], None, top_k).items() if key in ("macro_auc_represented", "macro_ap_represented", "top1_accuracy", "top3_accuracy")},
                **internal_restricted(source["internal"], labels, columns),
                "external_predictions_sha256": sha256_file(source["external"]),
            }
            rows.append(row)
            table = per_label_table(truth, probability, labels, thresholds)
            table.insert(0, "model", name)
            per_label.append(table)
            if source["classifier"] == "mil_ensemble":
                write_csv(output / "confusion" / f"{name}.csv", confusion_among_represented(truth, probability, labels).reset_index())
        grid = pd.DataFrame(rows)

        intervals = Parallel(n_jobs=workers)(
            delayed(bootstrap_intervals)(truth, probability, samples) for _, (_, _, probability, _) in loaded.items()
        )
        for name, interval in zip(loaded, intervals, strict=True):
            for metric, (lower, upper) in interval.items():
                grid.loc[grid["model"].eq(name), f"{metric}_ci_lower"] = lower
                grid.loc[grid["model"].eq(name), f"{metric}_ci_upper"] = upper
        write_csv(output / "model_grid.csv", grid.sort_values(["encoder", "classifier"]))
        write_csv(output / "per_label.csv", pd.concat(per_label, ignore_index=True))

        # Paired comparisons: MIL ensemble vs random forest for every encoder (Holm across encoders),
        # plus each production encoder's MIL ensemble vs every other encoder's random forest.
        pairs = []
        for encoder in sorted({source["encoder"] for source, *_ in loaded.values()}):
            first, second = f"{encoder}+mil_ensemble", f"{encoder}+random_forest"
            if first in loaded and second in loaded:
                pairs.append(("same_encoder_mil_vs_rf", first, second))
        for encoder in sorted(production_encoders):
            first = f"{encoder}+mil_ensemble"
            for other in sorted({source["encoder"] for source, *_ in loaded.values()} - {encoder}):
                second = f"{other}+random_forest"
                if first in loaded and second in loaded:
                    pairs.append(("production_vs_other_encoder_rf", first, second))
        comparisons = Parallel(n_jobs=workers)(
            delayed(paired_difference)(truth, loaded[first][2], loaded[second][2], samples) for _, first, second in pairs
        )
        frames = []
        for (family, first, second), frame in zip(pairs, comparisons, strict=True):
            frames.append(frame.assign(family=family, first_model=first, second_model=second))
        if frames:
            paired = pd.concat(frames, ignore_index=True)
            paired["holm_p_within_family_metric"] = np.nan
            for (_, _), block in paired.groupby(["family", "metric"]):
                paired.loc[block.index, "holm_p_within_family_metric"] = holm_adjust(block["bootstrap_p"].to_numpy())
            write_csv(output / "paired_comparisons.csv", paired)

        write_json(
            output / "external_evaluation_manifest.json",
            {
                "cohort": cohort.name,
                "run_tag": args.run_tag,
                "models": len(loaded),
                "replicates": replicates,
                "resampling_unit": "duplicate_content_group",
                "represented_labels": [labels[c] for c in columns],
                "thresholds": "MIL validation-selected thresholds applied unchanged; classical models threshold-free",
                "external_case_manifest_sha256": sha256_file(cohort.case_manifest_path),
                "complete": True,
            },
        )
        print(f"[external] {cohort.name}: evaluated {len(loaded)} models -> {output}")


if __name__ == "__main__":
    main()
