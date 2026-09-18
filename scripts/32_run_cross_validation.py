#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _common import load_context, load_manifests

from hawk_derm.config import load_yaml, path_from
from hawk_derm.features.bank import load_feature_bank
from hawk_derm.models.experiments import run_mil_cross_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multilabel-stratified development cross-validation.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_context(args.config)
    mil = load_yaml("configs/mil.yaml")
    if args.epochs:
        mil["training"]["epochs"] = args.epochs
    cases, _, splits = load_manifests(config)
    bank = load_feature_bank(path_from(config, "artifacts_dir") / "features" / args.encoder / "feature_bank.npz")
    run_mil_cross_validation(
        bank,
        cases,
        splits,
        config["labels"],
        mil,
        path_from(config, "artifacts_dir") / "models" / "mil" / args.encoder / "cross_validation",
        folds=args.folds,
        seed=args.seed,
        device=args.device,
        max_images=int(config["study"]["max_images_per_case"]),
    )


if __name__ == "__main__":
    main()
