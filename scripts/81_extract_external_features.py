#!/usr/bin/env python3
"""Extract external-cohort feature banks with the same code path as the study banks.

Run this before the model stages. Models trained later score these banks while they are in
memory, so external predictions are produced alongside the internal test predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _common import load_context

from hawk_derm.data.external import load_external_cohorts
from hawk_derm.features.extractors import extract_feature_bank
from hawk_derm.features.registry import load_encoder_registry
from hawk_derm.io import sha256_file
from hawk_derm.provenance import RunRecorder
from hawk_derm.runtime import resolve_cpu_workers


def complete(output_dir) -> bool:
    manifest = output_dir / "run_manifest.json"
    bank = output_dir / "feature_bank.npz"
    if not manifest.is_file() or not bank.is_file():
        return False
    payload = json.loads(manifest.read_text())
    if not payload.get("complete"):
        return False
    items = [*payload.get("inputs", []), *payload.get("outputs", [])]
    return bool(items) and all(Path(item["path"]).is_file() and sha256_file(item["path"]) == item.get("sha256") for item in items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--cohort", default=None)
    parser.add_argument("--encoders", default=None, help="Comma-separated encoder keys (default: all registered).")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Re-extract even when a matching bank exists.")
    args = parser.parse_args()
    config = load_context(args.config)
    workers = resolve_cpu_workers(args.workers)
    registry = load_encoder_registry()
    encoders = [item.strip() for item in args.encoders.split(",")] if args.encoders else list(registry)
    unknown = set(encoders).difference(registry)
    if unknown:
        raise ValueError(f"Unknown encoders: {sorted(unknown)}")
    for cohort in load_external_cohorts(config):
        if args.cohort not in (None, cohort.name):
            continue
        if not cohort.prepared:
            raise FileNotFoundError(f"Run scripts/80_prepare_external.py first: {cohort.manifests_dir}")
        images = pd.read_csv(cohort.image_manifest_path)
        for encoder in encoders:
            output_dir = cohort.feature_bank_path(encoder).parent
            if not args.force and complete(output_dir):
                print(f"[external] {cohort.name}/{encoder}: verified bank already present", flush=True)
                continue
            with RunRecorder(
                "extract_external_features",
                {**vars(args), "cohort": cohort.name, "encoder": encoder},
                output_dir,
                inputs=[cohort.image_manifest_path],
            ) as run:
                output = cohort.feature_bank_path(encoder)
                bank = extract_feature_bank(
                    registry[encoder],
                    images,
                    output,
                    device=args.device,
                    batch_size=args.batch_size,
                    workers=workers,
                    force=args.force,
                )
                # Unlike the study banks, every external image must be scored: a silent gap would bias results.
                if len(bank.image_ids) != len(images):
                    raise RuntimeError(
                        f"{cohort.name}/{encoder}: {len(images) - len(bank.image_ids)} images failed; see the extraction_failures CSV"
                    )
                run.complete([output, output.with_suffix(".metadata.json")], cohort=cohort.name)
            print(f"[external] {cohort.name}/{encoder}: extracted {len(images)} images", flush=True)


if __name__ == "__main__":
    main()
