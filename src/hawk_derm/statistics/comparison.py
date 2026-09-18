from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hawk_derm.constants import slugify
from hawk_derm.evaluation.metrics import safe_auc
from hawk_derm.evaluation.predictions import align_prediction_frames, arrays_from_prediction_frame
from hawk_derm.io import write_csv, write_json
from hawk_derm.statistics.bootstrap import paired_case_bootstrap, paired_case_permutation
from hawk_derm.statistics.inference import bootstrap_two_sided_p_value, holm_adjust


def compare_prediction_files(
    first_path: str | Path,
    second_path: str | Path,
    labels: list[str],
    output_dir: str | Path,
    *,
    replicates: int,
    seed: int,
    first_name: str,
    second_name: str,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    first, second = align_prediction_frames(pd.read_csv(first_path), pd.read_csv(second_path))
    first_truth, first_probability = arrays_from_prediction_frame(first, labels)
    second_truth, second_probability = arrays_from_prediction_frame(second, labels)
    if not np.array_equal(first_truth, second_truth):
        raise ValueError("Paired prediction files disagree on ground truth")

    rows = []
    distributions: dict[str, np.ndarray] = {}
    for metric in ("roc_auc", "pr_auc"):
        result = paired_case_bootstrap(
            first_truth,
            first_probability,
            second_probability,
            metric=metric,
            replicates=replicates,
            seed=seed + (0 if metric == "roc_auc" else 1),
        )
        distributions[metric] = result.distribution
        permutation_p, permutation_distribution = paired_case_permutation(
            first_truth,
            first_probability,
            second_probability,
            metric=metric,
            replicates=replicates,
            seed=seed + (10 if metric == "roc_auc" else 11),
        )
        distributions[f"{metric}_permutation_null"] = permutation_distribution
        rows.append(
            {
                "comparison": f"{first_name} - {second_name}",
                "metric": f"macro_{metric}",
                "difference": result.estimate,
                "ci_lower": result.lower,
                "ci_upper": result.upper,
                "bootstrap_p": bootstrap_two_sided_p_value(result.distribution),
                "paired_permutation_p": permutation_p,
                "replicates": replicates,
            }
        )

    per_class = []
    rng = np.random.default_rng(seed + 100)
    for class_index, label in enumerate(labels):
        truth = first_truth[:, class_index]
        estimate = safe_auc(truth, first_probability[:, class_index]) - safe_auc(truth, second_probability[:, class_index])
        distribution = np.empty(replicates)
        for replicate in range(replicates):
            indices = rng.integers(0, len(truth), len(truth))
            distribution[replicate] = safe_auc(truth[indices], first_probability[indices, class_index]) - safe_auc(
                truth[indices], second_probability[indices, class_index]
            )
        finite = distribution[np.isfinite(distribution)]
        lower, upper = (np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan))
        permutation_distribution = np.empty(replicates)
        for replicate in range(replicates):
            swap = rng.random(len(truth)) < 0.5
            permuted_first = first_probability[:, class_index].copy()
            permuted_second = second_probability[:, class_index].copy()
            permuted_first[swap] = second_probability[swap, class_index]
            permuted_second[swap] = first_probability[swap, class_index]
            permutation_distribution[replicate] = safe_auc(truth, permuted_first) - safe_auc(truth, permuted_second)
        permutation_finite = permutation_distribution[np.isfinite(permutation_distribution)]
        permutation_p = (
            (np.sum(np.abs(permutation_finite) >= abs(estimate)) + 1) / (len(permutation_finite) + 1)
            if len(permutation_finite)
            else np.nan
        )
        per_class.append(
            {
                "label": label,
                "auc_difference": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "bootstrap_p": bootstrap_two_sided_p_value(distribution),
                "paired_permutation_p": permutation_p,
            }
        )
    per_class_frame = pd.DataFrame(per_class)
    estimable = per_class_frame["paired_permutation_p"].notna()
    per_class_frame["holm_adjusted_p"] = np.nan
    per_class_frame.loc[estimable, "holm_adjusted_p"] = holm_adjust(
        per_class_frame.loc[estimable, "paired_permutation_p"].to_numpy()
    )
    write_csv(output_dir / "paired_macro_comparison.csv", pd.DataFrame(rows))
    write_csv(output_dir / "paired_per_class_auc_comparison.csv", per_class_frame)
    np.savez_compressed(output_dir / "paired_bootstrap_distributions.npz", **distributions)
    summary = {
        "first_model": first_name,
        "second_model": second_name,
        "paired_case_count": len(first),
        "replicates": replicates,
        "seed": seed,
        "complete": True,
    }
    write_json(output_dir / "comparison_summary.json", summary)
    return summary
