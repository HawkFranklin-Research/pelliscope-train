from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hawk_derm.config import load_yaml, path_from
from hawk_derm.features.bank import load_feature_bank
from hawk_derm.io import sha256_file, write_json

# Pre-registered in revision-experiments/PREREGISTRATION.md: the manifold-residual projection is
# used only when effective rank / embedding dimension <= RANK_GATE on training-split SigLIP2 photos.
RANK_GATE = 0.25
RANK_GATE_ENCODER = "siglip2_so400m"
MAX_RANK_ROWS = 5000


def effective_rank(embeddings: np.ndarray, *, max_rows: int = MAX_RANK_ROWS, seed: int = 0) -> float:
    """exp(entropy of the normalised squared singular values) of row-L2-normalised embeddings."""
    rng = np.random.default_rng(seed)
    if len(embeddings) > max_rows:
        embeddings = embeddings[rng.choice(len(embeddings), max_rows, replace=False)]
    normalised = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    energy = np.linalg.svd(normalised.astype(np.float64), compute_uv=False) ** 2
    share = energy[energy > 0] / energy.sum()
    return float(np.exp(-(share * np.log(share)).sum()))


def rank_gate(config: dict[str, Any], encoder: str = RANK_GATE_ENCODER) -> dict[str, Any] | None:
    """Evaluate the gate on training-split photos, or return None when the feature bank is absent."""
    bank_path = path_from(config, "artifacts_dir") / "features" / encoder / "feature_bank.npz"
    if not bank_path.is_file():
        return None
    bank = load_feature_bank(bank_path)
    split_path = path_from(config, "split_manifest")
    splits = pd.read_csv(split_path)
    train_ids = splits.loc[splits["split"].eq("train"), "case_id"].astype(str).to_numpy()
    embeddings = bank.embeddings[np.isin(bank.case_ids.astype(str), train_ids)]
    rank = effective_rank(embeddings)
    return {
        "encoder": encoder,
        "training_photos": int(len(embeddings)),
        "dimension": int(bank.dimension),
        "effective_rank": rank,
        "ratio": rank / bank.dimension,
        "threshold": RANK_GATE,
        "enable_manifold_residual": bool(rank / bank.dimension <= RANK_GATE),
        "feature_bank_sha256": sha256_file(bank_path),
        "split_manifest_sha256": sha256_file(split_path),
    }


def assert_rank_gate(config: dict[str, Any], mil_config_path: str | Path = "configs/mil.yaml") -> dict[str, Any]:
    """Stop a full run when configs/mil.yaml disagrees with the pre-registered rank gate."""
    record = rank_gate(config)
    output = path_from(config, "reports_dir") / "revision" / "rank_gate.json"
    if record is None:
        payload = {"evaluated": False, "reason": f"{RANK_GATE_ENCODER} feature bank is not on disk"}
        write_json(output, payload)
        print(f"[rank-gate] skipped: {payload['reason']}", flush=True)
        return payload
    configured = load_yaml(mil_config_path)["architecture"].get("instance_projection", "dense")
    required = "manifold_residual" if record["enable_manifold_residual"] else "dense"
    payload = {"evaluated": True, **record, "configured_instance_projection": configured, "required": required}
    write_json(output, payload)
    print(
        f"[rank-gate] effective rank {record['effective_rank']:.1f} / {record['dimension']} "
        f"= {record['ratio']:.3f} -> {required} (configured: {configured})",
        flush=True,
    )
    if configured != required:
        raise ValueError(
            f"Pre-registered rank gate requires instance_projection: {required} in {mil_config_path}, "
            f"but it is {configured}. Update the config, commit it, then resume from the tuning stage."
        )
    return payload
