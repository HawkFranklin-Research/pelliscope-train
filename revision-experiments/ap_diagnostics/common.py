"""Small, read-only input helpers for the isolated AP diagnostics."""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LABELS = yaml.safe_load((ROOT / "configs/study_25class.yaml").read_text())["labels"]
SLUGS = [re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") for label in LABELS]
TRUTH = [f"true__{name}" for name in SLUGS]
SCORES = [f"prob__{name}" for name in SLUGS]
RAW = ROOT / "supplementary_analyses/label_scope_and_ranking/outputs/raw_predictions"
LOCK = ROOT / "reports/reanalysis/siglip2_so400m_mil_canonical/statistics/comparison_cohort.csv"

MODELS = {
    "siglip2_mil": RAW / "mil_probability_ensembles/siglip2_so400m.csv",
    "siglip2_rf": RAW / "classical_case_aggregated/siglip2_so400m/random_forest.csv",
    "derm_mil": RAW / "mil_probability_ensembles/derm_foundation.csv",
    "derm_rf": RAW / "classical_case_aggregated/derm_foundation/random_forest.csv",
}


def load(path: Path, case_ids: list[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"case_id": str})
    if frame["case_id"].duplicated().any():
        raise ValueError(f"Duplicate case IDs: {path}")
    if case_ids is not None:
        if not set(case_ids).issubset(set(frame["case_id"])):
            raise ValueError(f"Missing locked cases: {path}")
        frame = frame.set_index("case_id").loc[case_ids].reset_index()
    values = frame[SCORES].to_numpy(float)
    truth = frame[TRUTH].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"Invalid probabilities: {path}")
    if not np.isin(truth, (0, 1)).all():
        raise ValueError(f"Invalid labels: {path}")
    return frame[["case_id", *TRUTH, *SCORES]]


def locked_ids() -> list[str]:
    ids = pd.read_csv(LOCK, dtype={"case_id": str}).sort_values("comparison_row")["case_id"].tolist()
    if len(ids) != 750 or len(set(ids)) != 750:
        raise ValueError("Expected exactly 750 unique locked test cases")
    return ids
