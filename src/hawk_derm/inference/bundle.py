"""Production bundles: the calibrated 10-seed MIL ensemble packaged for inference.

One bundle = every seed checkpoint + the selected MIL config + per-label Platt calibration +
validation-selected thresholds + label order + encoder identity, with a manifest of hashes.
The same `ProductionBundle.predict` is used by the self-check, by external evaluation code
that wants to re-score, and by the released model, so evaluated and published behaviour match.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from hawk_derm.config import load_yaml, path_from
from hawk_derm.evaluation.calibration import apply_platt
from hawk_derm.features.registry import load_encoder_registry
from hawk_derm.io import read_json, sha256_file, write_json
from hawk_derm.models.mil import GatedAttentionMIL
from hawk_derm.provenance.run import git_commit

MANIFEST = "bundle_manifest.json"


def mil_root(config: dict[str, Any], encoder: str, run_tag: str) -> Path:
    return path_from(config, "artifacts_dir") / "models" / "mil" / encoder / run_tag


def production_entries(config_path: str | Path = "configs/production.yaml") -> dict[str, dict[str, Any]]:
    """Production models with an encoder assigned; entries left null are skipped on purpose."""
    entries = load_yaml(config_path).get("production_models", {}) or {}
    return {name: entry for name, entry in entries.items() if entry.get("encoder")}


def build_bundle(config: dict[str, Any], name: str, entry: dict[str, Any], output: Path) -> Path:
    if entry.get("model") != "mil_ensemble":
        raise ValueError(f"{name}: only mil_ensemble production models are supported")
    encoder, run_tag = entry["encoder"], entry.get("run_tag", "canonical")
    root = mil_root(config, encoder, run_tag)
    expected_seeds = [int(seed) for seed in config[config["study"]["run_mode"]]["seeds"]]
    seed_paths = [root / f"seed_{seed}" / "model.pt" for seed in expected_seeds]
    required = [
        *seed_paths,
        root / "tuning" / "best_mil_config.yaml",
        root / "ensemble" / "calibration_parameters.csv",
        root / "ensemble" / "ensemble_manifest.json",
        root / "thresholds" / "validation_selected_thresholds.csv",
        root / "thresholds" / "test_operating_points.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{name}: bundle inputs are missing: {missing[:5]}")
    if output.exists():
        shutil.rmtree(output)
    (output / "models").mkdir(parents=True)
    for seed, path in zip(expected_seeds, seed_paths, strict=True):
        shutil.copy2(path, output / "models" / f"seed_{seed}.pt")
    for path in required[len(seed_paths):]:
        shutil.copy2(path, output / path.name)
    for path in sorted((root / "thresholds").glob("external_*_operating_points.csv")):
        shutil.copy2(path, output / path.name)
    spec = load_encoder_registry()[encoder]
    files = sorted(path for path in output.rglob("*") if path.is_file())
    write_json(
        output / MANIFEST,
        {
            "name": name,
            "display_name": entry.get("display_name", name),
            "model": "mil_ensemble",
            "aggregation": "mean_probability_across_seeds_then_per_label_platt",
            "encoder": {"key": encoder, "model_id": spec.model_id, "revision": spec.revision, "dimension": spec.dimension},
            "labels": config["labels"],
            "seeds": expected_seeds,
            "max_images_per_case": int(config["study"]["max_images_per_case"]),
            "run_tag": run_tag,
            "split_manifest_sha256": sha256_file(path_from(config, "split_manifest")),
            "source_ensemble_manifest": read_json(root / "ensemble" / "ensemble_manifest.json"),
            "files": {str(path.relative_to(output)): sha256_file(path) for path in files},
            "git_commit": git_commit(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "self_check": {"passed": False},
        },
    )
    return output


@dataclass
class ProductionBundle:
    root: Path
    manifest: dict[str, Any]
    models: list[GatedAttentionMIL]
    calibration: pd.DataFrame
    thresholds: np.ndarray

    @classmethod
    def load(cls, root: str | Path, *, verify_hashes: bool = True, device: str = "cpu") -> "ProductionBundle":
        root = Path(root)
        manifest = read_json(root / MANIFEST)
        if verify_hashes:
            changed = [name for name, digest in manifest["files"].items() if sha256_file(root / name) != digest]
            if changed:
                raise ValueError(f"Bundle files changed since packaging: {changed}")
        models = []
        for seed in manifest["seeds"]:
            payload = torch.load(root / "models" / f"seed_{seed}.pt", map_location=device, weights_only=False)
            model = GatedAttentionMIL(input_dim=int(payload["input_dim"]), class_count=int(payload["class_count"]), **payload["architecture"])
            model.load_state_dict(payload["state_dict"])
            models.append(model.to(device).eval())
        labels = manifest["labels"]
        thresholds = pd.read_csv(root / "validation_selected_thresholds.csv").set_index("label").loc[labels, "threshold"]
        return cls(root, manifest, models, pd.read_csv(root / "calibration_parameters.csv"), thresholds.to_numpy(dtype=float))

    @property
    def labels(self) -> list[str]:
        return list(self.manifest["labels"])

    def predict(self, bags: np.ndarray, masks: np.ndarray, batch_size: int = 256) -> dict[str, np.ndarray]:
        """bags: [cases, max_images, dimension] embeddings; masks: [cases, max_images] True for real photos."""
        bags_tensor = torch.as_tensor(np.asarray(bags), dtype=torch.float32)
        masks_tensor = torch.as_tensor(np.asarray(masks), dtype=torch.bool)
        if not masks_tensor.any(dim=1).all():
            raise ValueError("Every case needs at least one photo")
        seed_probabilities = []
        with torch.inference_mode():
            for model in self.models:
                device = next(model.parameters()).device
                parts = [
                    torch.sigmoid(model(bags_tensor[i : i + batch_size].to(device), masks_tensor[i : i + batch_size].to(device))[0]).cpu().numpy()
                    for i in range(0, len(bags_tensor), batch_size)
                ]
                seed_probabilities.append(np.concatenate(parts))
        uncalibrated = np.mean(seed_probabilities, axis=0)
        probability = apply_platt(uncalibrated, self.calibration, self.labels)
        return {"probability": probability, "uncalibrated_probability": uncalibrated, "decision": probability >= self.thresholds[None, :]}
