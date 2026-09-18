#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import load_context

from hawk_derm.config import path_from
from hawk_derm.features.bank import import_legacy_casebag, save_feature_bank
from hawk_derm.features.extractors import extract_feature_bank
from hawk_derm.features.registry import load_encoder_registry
from hawk_derm.io import read_json, sha256_file
from hawk_derm.provenance import RunRecorder


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract or import a standardized frozen feature bank.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--import-bank", type=Path, default=None)
    parser.add_argument("--import-metadata", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_context(args.config)
    registry = load_encoder_registry()
    if args.encoder not in registry:
        raise ValueError(f"Unknown encoder {args.encoder!r}; choose from {sorted(registry)}")
    spec = registry[args.encoder]
    cases = pd.read_csv(path_from(config, "case_manifest"))
    images = pd.read_csv(path_from(config, "image_manifest"))
    if args.max_images:
        images = images.head(args.max_images).copy()
    output_dir = path_from(config, "artifacts_dir") / "features" / args.encoder
    output = output_dir / "feature_bank.npz"
    with RunRecorder("extract_features", vars(args), output_dir, inputs=[path_from(config, "image_manifest")]) as run:
        if args.import_bank:
            metadata_path = args.import_metadata or args.import_bank.with_suffix(".metadata.json")
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    "A legacy feature bank requires an adjacent or explicit metadata JSON containing model_id, revision, dimension, and optional sha256"
                )
            import_metadata = read_json(metadata_path)
            if int(import_metadata["dimension"]) != spec.dimension:
                raise ValueError("Imported feature dimension does not match the encoder registry")
            if import_metadata.get("model_id") != spec.model_id:
                raise ValueError("Imported feature model_id does not match the encoder registry")
            if import_metadata.get("revision") != spec.revision:
                raise ValueError("Imported feature revision does not match the encoder registry")
            if import_metadata.get("sha256") and import_metadata["sha256"] != sha256_file(args.import_bank):
                raise ValueError("Imported feature bank checksum does not match its metadata")
            bank = import_legacy_casebag(
                args.import_bank,
                cases,
                images,
                encoder=spec.key,
                model_id=spec.model_id,
                revision=spec.revision,
            )
            save_feature_bank(
                output,
                bank,
                {
                    "source": str(args.import_bank),
                    "source_sha256": sha256_file(args.import_bank),
                    "source_metadata": str(metadata_path),
                    "preprocessing": import_metadata.get("preprocessing"),
                    "imported": True,
                    "run_mode": config["study"]["run_mode"],
                    "encoder": bank.encoder,
                    "model_id": bank.model_id,
                    "revision": bank.revision,
                    "dimension": bank.dimension,
                    "dtype": "float32",
                },
            )
        else:
            extract_feature_bank(spec, images, output, device=args.device, batch_size=args.batch_size, force=args.force)
        run.complete([output, output.with_suffix(".metadata.json")], run_mode=config["study"]["run_mode"])


if __name__ == "__main__":
    main()
