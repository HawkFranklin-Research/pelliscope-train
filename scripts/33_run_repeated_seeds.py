#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _common import load_context, load_manifests, parse_seeds

from hawk_derm.config import load_yaml, path_from
from hawk_derm.features.bank import load_feature_bank
from hawk_derm.models.experiments import run_repeated_mil


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated MIL initializations on the shared locked split.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_context(args.config)
    mil = load_yaml("configs/mil.yaml")
    if args.epochs:
        mil["training"]["epochs"] = args.epochs
    mode = config["study"]["run_mode"]
    seeds = parse_seeds(args.seeds) if args.seeds else list(config[mode]["seeds"])
    cases, _, splits = load_manifests(config)
    bank = load_feature_bank(path_from(config, "artifacts_dir") / "features" / args.encoder / "feature_bank.npz")
    run_repeated_mil(
        bank,
        cases,
        splits,
        config["labels"],
        mil,
        path_from(config, "artifacts_dir") / "models" / "mil" / args.encoder,
        seeds=seeds,
        device=args.device,
        max_images=int(config["study"]["max_images_per_case"]),
    )


if __name__ == "__main__":
    main()
