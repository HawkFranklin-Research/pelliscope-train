from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hawk_derm.io import ensure_parent, write_json


@dataclass
class FeatureBank:
    embeddings: np.ndarray
    case_ids: np.ndarray
    image_ids: np.ndarray
    image_paths: np.ndarray
    image_sha256: np.ndarray
    encoder: str
    model_id: str
    revision: str
    dimension: int


def save_feature_bank(path: str | Path, bank: FeatureBank, metadata: dict[str, Any]) -> Path:
    path = ensure_parent(path)
    np.savez_compressed(
        path,
        embeddings=bank.embeddings.astype(np.float32),
        case_ids=bank.case_ids.astype(str),
        image_ids=bank.image_ids.astype(str),
        image_paths=bank.image_paths.astype(str),
        image_sha256=bank.image_sha256.astype(str),
        encoder=bank.encoder,
        model_id=bank.model_id,
        revision=bank.revision,
        dimension=np.int32(bank.dimension),
    )
    write_json(path.with_suffix(".metadata.json"), {**metadata, "complete": True, "row_count": len(bank.embeddings)})
    return path


def load_feature_bank(path: str | Path) -> FeatureBank:
    values = np.load(path, allow_pickle=False)
    embeddings = values["embeddings"].astype(np.float32)
    return FeatureBank(
        embeddings=embeddings,
        case_ids=values["case_ids"].astype(str),
        image_ids=values["image_ids"].astype(str),
        image_paths=values["image_paths"].astype(str),
        image_sha256=values["image_sha256"].astype(str),
        encoder=str(values["encoder"]),
        model_id=str(values["model_id"]),
        revision=str(values["revision"]),
        dimension=int(values["dimension"]),
    )


def import_legacy_casebag(
    legacy_path: str | Path,
    case_manifest: pd.DataFrame,
    image_manifest: pd.DataFrame,
    *,
    encoder: str,
    model_id: str,
    revision: str,
) -> FeatureBank:
    values = np.load(legacy_path, allow_pickle=True)
    feature_key = next(key for key in ("features", "X", "embeddings", "bags") if key in values.files)
    features = values[feature_key].astype(np.float32)
    case_key = next((key for key in ("case_ids", "case_id", "ids") if key in values.files), None)
    case_ids = values[case_key].astype(str) if case_key else case_manifest["case_id"].astype(str).to_numpy()
    mask_key = next((key for key in ("masks", "mask", "attention_mask") if key in values.files), None)
    masks = values[mask_key].astype(bool) if mask_key else np.linalg.norm(features, axis=-1) > 0
    legacy_paths = values["image_paths"].astype(str) if "image_paths" in values.files else None
    if features.ndim != 3 or masks.shape != features.shape[:2]:
        raise ValueError(f"Legacy casebag must have [cases, slots, dimension] features and matching masks; got {features.shape}")
    canonical_ids = set(case_manifest["case_id"].astype(str))
    if set(case_ids.astype(str)) != canonical_ids:
        missing = canonical_ids - set(case_ids.astype(str))
        extra = set(case_ids.astype(str)) - canonical_ids
        raise ValueError(f"Legacy case IDs do not match the canonical manifest: missing={len(missing)}, extra={len(extra)}")
    rows = []
    for case_index, case_id in enumerate(case_ids):
        manifest_rows = image_manifest[image_manifest["case_id"].astype(str).eq(str(case_id))].sort_values("bag_slot")
        for slot in range(features.shape[1]):
            if not masks[case_index, slot]:
                continue
            metadata = manifest_rows.iloc[slot] if slot < len(manifest_rows) else None
            if legacy_paths is not None and metadata is not None:
                legacy_name = Path(str(legacy_paths[case_index, slot])).name
                canonical_name = Path(str(metadata["resolved_image_path"])).name
                if legacy_name and legacy_name != canonical_name:
                    raise ValueError(
                        f"Legacy image order/path mismatch for case {case_id}, slot {slot}: {legacy_name} != {canonical_name}"
                    )
            rows.append(
                (
                    features[case_index, slot],
                    str(case_id),
                    str(metadata["image_id"]) if metadata is not None else f"{case_id}:{slot}",
                    str(metadata["resolved_image_path"]) if metadata is not None else "",
                    str(metadata.get("sha256", "")) if metadata is not None else "",
                )
            )
    return FeatureBank(
        embeddings=np.stack([row[0] for row in rows]),
        case_ids=np.asarray([row[1] for row in rows]),
        image_ids=np.asarray([row[2] for row in rows]),
        image_paths=np.asarray([row[3] for row in rows]),
        image_sha256=np.asarray([row[4] for row in rows]),
        encoder=encoder,
        model_id=model_id,
        revision=revision,
        dimension=features.shape[-1],
    )


def build_case_bags(
    bank: FeatureBank,
    cases: pd.DataFrame,
    labels: list[str],
    max_images: int = 3,
) -> dict[str, np.ndarray]:
    label_columns = [f"is_{label}" if not label.startswith("is_") else label for label in labels]
    by_case: dict[str, list[int]] = {}
    for index, case_id in enumerate(bank.case_ids.astype(str)):
        by_case.setdefault(case_id, []).append(index)
    bags = np.zeros((len(cases), max_images, bank.dimension), dtype=np.float32)
    masks = np.zeros((len(cases), max_images), dtype=bool)
    image_ids = np.full((len(cases), max_images), "", dtype="U256")
    for case_index, case_id in enumerate(cases["case_id"].astype(str)):
        indices = by_case.get(case_id, [])[:max_images]
        if indices:
            bags[case_index, : len(indices)] = bank.embeddings[indices]
            masks[case_index, : len(indices)] = True
            image_ids[case_index, : len(indices)] = bank.image_ids[indices]
    return {
        "bags": bags,
        "masks": masks,
        "labels": cases[label_columns].to_numpy(dtype=np.float32),
        "case_ids": cases["case_id"].astype(str).to_numpy(),
        "image_ids": image_ids,
    }
