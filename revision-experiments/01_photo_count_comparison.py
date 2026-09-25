"""Paired MIL-versus-RF analysis by the number of retained photos per case."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "outputs/photo_count"
LOCK = ROOT / "reports/reanalysis/siglip2_so400m_mil_canonical/statistics/comparison_cohort.csv"
IMAGES = ROOT / "data/manifests/image_manifest.csv"
RAW = ROOT / "supplementary_analyses/label_scope_and_ranking/outputs/raw_predictions"
MODELS = {
    "siglip2_mil_ensemble": RAW / "mil_probability_ensembles/siglip2_so400m.csv",
    "siglip2_random_forest": RAW / "classical_case_aggregated/siglip2_so400m/random_forest.csv",
}
LABELS = yaml.safe_load((ROOT / "configs/study_25class.yaml").read_text())["labels"]
SLUGS = [re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") for label in LABELS]
TRUE_COLS = [f"true__{slug}" for slug in SLUGS]
PROB_COLS = [f"prob__{slug}" for slug in SLUGS]
METRICS = ("macro_roc_auc", "micro_roc_auc", "macro_ap", "micro_ap")
REPLICATES = 2000
SEED = 20260926


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_metrics(truth: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """ROC-AUC and tie-aware non-interpolated AP, matching scikit-learn."""
    positives = int(truth.sum())
    negatives = len(truth) - positives
    if positives == 0 or negatives == 0:
        return float("nan"), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_truth = truth[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
    positive_groups = np.diff(np.r_[0, np.cumsum(sorted_truth)[ends]])
    group_sizes = np.diff(np.r_[0, ends + 1])
    negative_groups = group_sizes - positive_groups
    cumulative_positive = np.cumsum(positive_groups)
    cumulative_negative = np.cumsum(negative_groups)
    ap = np.sum(positive_groups * cumulative_positive / (ends + 1)) / positives
    auc = np.sum(positive_groups * (negatives - cumulative_negative + negative_groups / 2)) / (positives * negatives)
    return float(auc), float(ap)


def metrics(truth: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    per_label = np.asarray([binary_metrics(truth[:, col], scores[:, col]) for col in range(truth.shape[1])])
    valid = np.isfinite(per_label[:, 0])
    micro_roc, micro_ap = binary_metrics(truth.ravel(), scores.ravel())
    return {
        "macro_roc_auc": float(np.mean(per_label[valid, 0])) if valid.any() else float("nan"),
        "micro_roc_auc": micro_roc,
        "macro_ap": float(np.mean(per_label[valid, 1])) if valid.any() else float("nan"),
        "micro_ap": micro_ap,
        "evaluable_labels": int(valid.sum()),
    }


def read_inputs() -> tuple[list[str], np.ndarray, dict[str, np.ndarray], np.ndarray]:
    lock = pd.read_csv(LOCK, dtype={"case_id": str}).sort_values("comparison_row")
    ids = lock["case_id"].tolist()
    if len(ids) != 750 or len(set(ids)) != 750:
        raise ValueError("The locked comparison cohort must have 750 unique IDs")
    images = pd.read_csv(IMAGES, dtype={"case_id": str, "image_id": str})
    kept = images.loc[images["retained_for_bag"].astype(bool)]
    if kept.duplicated(["case_id", "image_id"]).any():
        raise ValueError("Repeated retained image within a case")
    if not kept.loc[kept["case_id"].isin(ids), "file_exists"].astype(bool).all():
        raise ValueError("Locked test case has an unavailable retained image")
    counts = kept.groupby("case_id").size().reindex(ids)
    if counts.isna().any() or not counts.isin([1, 2, 3]).all():
        raise ValueError("Missing or invalid photo count for a locked test case")

    predictions: dict[str, np.ndarray] = {}
    truth: np.ndarray | None = None
    for name, path in MODELS.items():
        frame = pd.read_csv(path, dtype={"case_id": str})
        if frame["case_id"].duplicated().any() or set(frame["case_id"]) != set(ids):
            raise ValueError(f"Prediction case IDs differ from the lock: {path}")
        frame = frame.set_index("case_id").loc[ids]
        labels = frame[TRUE_COLS].to_numpy(int)
        scores = frame[PROB_COLS].to_numpy(float)
        if not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all():
            raise ValueError(f"Invalid labels or scores: {path}")
        if ((scores < 0) | (scores > 1)).any():
            raise ValueError(f"Out-of-range probabilities: {path}")
        if truth is None:
            truth = labels
        elif not np.array_equal(truth, labels):
            raise ValueError("MIL and random forest ground-truth labels disagree")
        predictions[name] = scores
    assert truth is not None
    return ids, truth, predictions, counts.to_numpy(int)


def percentile_interval(draws: np.ndarray) -> tuple[float, float]:
    finite = draws[np.isfinite(draws)]
    if len(finite) < REPLICATES * 0.9:
        raise ValueError("Too many undefined bootstrap draws")
    return tuple(float(value) for value in np.percentile(finite, [2.5, 97.5]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ids, truth, predictions, photo_counts = read_inputs()
    model_names = list(MODELS)
    names = {1: "one_photo", 2: "two_photos", 3: "three_photos"}
    rng = np.random.default_rng(SEED)
    counts = pd.DataFrame({"case_id": ids, "photos": photo_counts})
    counts.to_csv(OUT / "locked_case_photo_counts.csv", index=False)
    support, estimates, draws = [], [], []
    delta_draws: dict[int, dict[str, np.ndarray]] = {}
    one_support = truth[photo_counts == 1].sum(axis=0)
    three_support = truth[photo_counts == 3].sum(axis=0)
    shared_macro_labels = np.flatnonzero(
        (one_support > 0)
        & (one_support < (photo_counts == 1).sum())
        & (three_support > 0)
        & (three_support < (photo_counts == 3).sum())
    )
    shared_macro_points: dict[int, dict[str, float]] = {}
    shared_macro_draws: dict[int, dict[str, np.ndarray]] = {}

    for photos in (1, 2, 3):
        members = np.flatnonzero(photo_counts == photos)
        group_truth = truth[members]
        group_scores = {name: scores[members] for name, scores in predictions.items()}
        label_positive = group_truth.sum(axis=0)
        support.extend(
            {
                "photos": photos,
                "condition": label,
                "positive_cases": int(label_positive[col]),
                "negative_cases": int(len(members) - label_positive[col]),
            }
            for col, label in enumerate(LABELS)
        )
        point = {name: metrics(group_truth, scores) for name, scores in group_scores.items()}
        if photos in (1, 3):
            restricted = {
                name: metrics(group_truth[:, shared_macro_labels], scores[:, shared_macro_labels])
                for name, scores in group_scores.items()
            }
            shared_macro_points[photos] = {
                metric: restricted[model_names[0]][metric] - restricted[model_names[1]][metric]
                for metric in ("macro_roc_auc", "macro_ap")
            }
        # Verify the fast tie-aware implementation against scikit-learn on every observed subgroup.
        for name, scores in group_scores.items():
            for col in range(len(LABELS)):
                if 0 < label_positive[col] < len(members):
                    auc, ap = binary_metrics(group_truth[:, col], scores[:, col])
                    if not np.isclose(auc, roc_auc_score(group_truth[:, col], scores[:, col]), atol=1e-12):
                        raise AssertionError("ROC-AUC mismatch with scikit-learn")
                    if not np.isclose(ap, average_precision_score(group_truth[:, col], scores[:, col]), atol=1e-12):
                        raise AssertionError("AP mismatch with scikit-learn")

        boot = {name: {metric: np.empty(REPLICATES) for metric in METRICS} for name in model_names}
        restricted_boot = {
            name: {metric: np.empty(REPLICATES) for metric in ("macro_roc_auc", "macro_ap")} for name in model_names
        }
        for replicate in range(REPLICATES):
            draw = rng.integers(len(members), size=len(members))
            for name in model_names:
                result = metrics(group_truth[draw], group_scores[name][draw])
                for metric in METRICS:
                    boot[name][metric][replicate] = result[metric]
                if photos in (1, 3):
                    restricted_result = metrics(
                        group_truth[draw][:, shared_macro_labels],
                        group_scores[name][draw][:, shared_macro_labels],
                    )
                    for metric in ("macro_roc_auc", "macro_ap"):
                        restricted_boot[name][metric][replicate] = restricted_result[metric]
        delta_draws[photos] = {}
        for metric in METRICS:
            delta_draws[photos][metric] = boot[model_names[0]][metric] - boot[model_names[1]][metric]
        if photos in (1, 3):
            shared_macro_draws[photos] = {
                metric: restricted_boot[model_names[0]][metric] - restricted_boot[model_names[1]][metric]
                for metric in ("macro_roc_auc", "macro_ap")
            }

        common = {
            "photos": photos,
            "group": names[photos],
            "cases": len(members),
            "target_positive_cases": int((group_truth.sum(axis=1) > 0).sum()),
            "positive_case_label_pairs": int(group_truth.sum()),
            "pooled_prevalence": float(group_truth.mean()),
        }
        for name in model_names:
            for metric in METRICS:
                low, high = percentile_interval(boot[name][metric])
                estimates.append(
                    {
                        **common,
                        "comparison": name,
                        "metric": metric,
                        "estimate": point[name][metric],
                        "ci_95_lower": low,
                        "ci_95_upper": high,
                        "evaluable_labels": point[name]["evaluable_labels"],
                    }
                )
        for metric in METRICS:
            low, high = percentile_interval(delta_draws[photos][metric])
            estimates.append(
                {
                    **common,
                    "comparison": "mil_minus_rf",
                    "metric": metric,
                    "estimate": point[model_names[0]][metric] - point[model_names[1]][metric],
                    "ci_95_lower": low,
                    "ci_95_upper": high,
                    "evaluable_labels": point[model_names[0]]["evaluable_labels"],
                }
            )
            draws.extend(
                {"photos": photos, "replicate": index, "metric": metric, "mil_minus_rf": value}
                for index, value in enumerate(delta_draws[photos][metric])
            )

    interaction = []
    table = pd.DataFrame(estimates)
    for metric in METRICS:
        if metric.startswith("macro"):
            difference = shared_macro_draws[3][metric] - shared_macro_draws[1][metric]
            point_difference = shared_macro_points[3][metric] - shared_macro_points[1][metric]
            labels_compared = len(shared_macro_labels)
        else:
            difference = delta_draws[3][metric] - delta_draws[1][metric]
            three = table[(table.photos == 3) & (table.comparison == "mil_minus_rf") & (table.metric == metric)].iloc[0]
            one = table[(table.photos == 1) & (table.comparison == "mil_minus_rf") & (table.metric == metric)].iloc[0]
            point_difference = three.estimate - one.estimate
            labels_compared = len(LABELS)
        low, high = percentile_interval(difference)
        interaction.append(
            {
                "metric": metric,
                "comparison": "three_photo_advantage_minus_one_photo_advantage",
                "estimate": point_difference,
                "ci_95_lower": low,
                "ci_95_upper": high,
                "labels_compared": labels_compared,
            }
        )

    table.to_csv(OUT / "photo_count_metrics_and_paired_intervals.csv", index=False)
    pd.DataFrame(interaction).to_csv(OUT / "three_vs_one_interaction.csv", index=False)
    pd.DataFrame(support).to_csv(OUT / "condition_support_by_photo_count.csv", index=False)
    pd.DataFrame({"condition": [LABELS[index] for index in shared_macro_labels]}).to_csv(
        OUT / "shared_macro_interaction_labels.csv", index=False
    )
    pd.DataFrame(draws).to_csv(OUT / "paired_bootstrap_deltas.csv", index=False)
    sources = {str(path.relative_to(ROOT)): file_hash(path) for path in [LOCK, IMAGES, *MODELS.values()]}
    (OUT / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "test_cases": 750,
                "photo_groups": counts.photos.value_counts().sort_index().to_dict(),
                "bootstrap_replicates": REPLICATES,
                "bootstrap_seed": SEED,
                "shared_macro_interaction_labels": [LABELS[index] for index in shared_macro_labels],
                "sources_sha256": sources,
                "interpretation": "exploratory paired subgroup analysis; report every group",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        table[table.comparison.eq("mil_minus_rf")][
            ["photos", "cases", "metric", "estimate", "ci_95_lower", "ci_95_upper", "evaluable_labels"]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("\nThree-photo minus one-photo difference in MIL advantage:")
    print(pd.DataFrame(interaction).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
