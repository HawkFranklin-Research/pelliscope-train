#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _common import load_context, load_manifests

from hawk_derm.config import load_yaml, path_from
from hawk_derm.features.bank import load_feature_bank
from hawk_derm.models.experiments import run_mil_search
from hawk_derm.provenance import RunRecorder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation-only MIL hyperparameter search.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_context(args.config)
    mil = load_yaml("configs/mil.yaml")
    if args.epochs:
        mil["training"]["epochs"] = args.epochs
    trials = args.trials or int(mil["search"]["trials"])
    cases, _, splits = load_manifests(config)
    bank_path = path_from(config, "artifacts_dir") / "features" / args.encoder / "feature_bank.npz"
    bank = load_feature_bank(bank_path)
    output_dir = path_from(config, "artifacts_dir") / "models" / "mil" / args.encoder / "tuning"
    with RunRecorder("tune_mil", vars(args), output_dir, inputs=[bank_path, path_from(config, "split_manifest")]) as run:
        run_mil_search(
            bank,
            cases,
            splits,
            config["labels"],
            mil,
            output_dir,
            trials=trials,
            seed=args.seed,
            device=args.device,
            max_images=int(config["study"]["max_images_per_case"]),
        )
        run.complete([output_dir / "trials.csv"])


if __name__ == "__main__":
    main()
