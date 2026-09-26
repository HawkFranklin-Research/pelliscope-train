#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _common import load_context, load_manifests

from hawk_derm.config import load_yaml, path_from
from hawk_derm.data.external import external_inputs, external_paths
from hawk_derm.features.bank import load_feature_bank
from hawk_derm.models.classical import run_classical_experiment
from hawk_derm.provenance import RunRecorder
from hawk_derm.runtime import resolve_cpu_workers


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one classical classifier over one frozen encoder.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aggregation", choices=["mean", "max"], default="mean")
    parser.add_argument("--workers", type=int, default=None, help="CPU budget for this classifier configuration.")
    args = parser.parse_args()
    config = load_context(args.config)
    workers = resolve_cpu_workers(args.workers)
    classifiers = load_yaml("configs/classifiers.yaml")["classifiers"]
    if args.classifier not in classifiers:
        raise ValueError(f"Unknown classifier: {args.classifier}")
    cases, _, splits = load_manifests(config)
    bank_path = path_from(config, "artifacts_dir") / "features" / args.encoder / "feature_bank.npz"
    bank = load_feature_bank(bank_path)
    output_dir = path_from(config, "artifacts_dir") / "models" / "classical" / args.encoder / args.classifier
    external = external_inputs(config, args.encoder)
    with RunRecorder(
        "train_classical",
        vars(args),
        output_dir,
        inputs=[bank_path, path_from(config, "split_manifest"), *external_paths(config, args.encoder)],
    ) as run:
        run_classical_experiment(
            bank,
            cases,
            splits,
            config["labels"],
            args.classifier,
            classifiers[args.classifier],
            output_dir,
            seed=args.seed,
            aggregation=args.aggregation,
            workers=workers,
            external=external,
        )
        run.complete([output_dir / "overall_metrics.csv", output_dir / "test_case_predictions.csv"])


if __name__ == "__main__":
    main()
