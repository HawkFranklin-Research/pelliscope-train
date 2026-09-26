"""Two pre-declared checks for the revised MIL training setup. Neither uses test labels.

1. Base-rate check: per label, mean predicted probability divided by prevalence, on MIL
   validation predictions. Ratios far above 1 across labels confirm that the original
   focal-plus-positive-weight loss inflated scores for rare labels.
2. Rank gate for the manifold-residual projection: effective rank of the L2-normalised
   training-split photo embeddings. The projection is enabled only if the ratio
   effective_rank / dimension is at most RANK_GATE (declared here before running).

Usage, from the repository root:
    python revision-experiments/02_loss_and_rank_diagnostics.py \
        [--predictions PATH] [--encoder siglip2_so400m]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hawk_derm.features.bank import load_feature_bank  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs/loss_and_rank"
DEFAULT_PREDICTIONS = ROOT / "artifacts/models/mil/siglip2_so400m/canonical/ensemble/validation_predictions.csv"
RANK_GATE = 0.25
MAX_ROWS = 5000


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def base_rate_report(predictions: Path, labels: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(predictions)
    rows = []
    for label in labels:
        truth = frame[f"true__{slug(label)}"].to_numpy(dtype=float)
        probability = frame[f"prob__{slug(label)}"].to_numpy(dtype=float)
        prevalence = truth.mean()
        rows.append(
            {
                "label": label,
                "positives": int(truth.sum()),
                "prevalence": prevalence,
                "mean_predicted_probability": probability.mean(),
                "ratio": probability.mean() / prevalence if prevalence > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def effective_rank(embeddings: np.ndarray, seed: int = 0) -> float:
    """exp(entropy of the normalised squared singular values) of row-L2-normalised embeddings."""
    rng = np.random.default_rng(seed)
    if len(embeddings) > MAX_ROWS:
        embeddings = embeddings[rng.choice(len(embeddings), MAX_ROWS, replace=False)]
    normalised = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    energy = np.linalg.svd(normalised, compute_uv=False) ** 2
    share = energy[energy > 0] / energy.sum()
    return float(np.exp(-(share * np.log(share)).sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--encoder", default="siglip2_so400m")
    args = parser.parse_args()
    study = yaml.safe_load((ROOT / "configs/study_25class.yaml").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"rank_gate": RANK_GATE}

    if args.predictions.is_file():
        report = base_rate_report(args.predictions, study["labels"])
        report.to_csv(OUT / "base_rate_report.csv", index=False)
        summary["base_rate"] = {
            "predictions": str(args.predictions),
            "median_ratio": float(report["ratio"].median()),
            "labels_with_ratio_above_2": int((report["ratio"] > 2).sum()),
            "labels": len(report),
        }
    else:
        summary["base_rate"] = f"skipped: {args.predictions} not found"

    bank_path = ROOT / "artifacts/features" / args.encoder / "feature_bank.npz"
    if bank_path.is_file():
        bank = load_feature_bank(bank_path)
        splits = pd.read_csv(ROOT / study["paths"]["split_manifest"])
        train_ids = set(splits.loc[splits["split"].eq("train"), "case_id"].astype(str))
        embeddings = bank.embeddings[np.isin(bank.case_ids.astype(str), list(train_ids))]
        rank = effective_rank(embeddings)
        summary["effective_rank"] = {
            "encoder": args.encoder,
            "training_photos": int(len(embeddings)),
            "dimension": bank.dimension,
            "effective_rank": rank,
            "ratio": rank / bank.dimension,
            "enable_manifold_residual": rank / bank.dimension <= RANK_GATE,
        }
    else:
        summary["effective_rank"] = f"skipped: {bank_path} not found"

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
