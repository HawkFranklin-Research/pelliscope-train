#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _common import load_context, load_manifests, load_selected_mil_config, mil_run_root, parse_seeds

from hawk_derm.config import path_from
from hawk_derm.features.bank import load_feature_bank
from hawk_derm.io import sha256_file
from hawk_derm.models.experiments import run_repeated_mil
from hawk_derm.provenance import RunRecorder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated MIL initializations on the shared locked split.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mil-config", default=None, help="Selected best_mil_config.yaml from tuning.")
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()
    config = load_context(args.config)
    mil, mil_config_path = load_selected_mil_config(config, args.encoder, args.run_tag, args.mil_config)
    if args.epochs:
        mil["training"]["epochs"] = args.epochs
    mode = config["study"]["run_mode"]
    seeds = parse_seeds(args.seeds) if args.seeds else list(config[mode]["seeds"])
    cases, _, splits = load_manifests(config)
    bank_path = path_from(config, "artifacts_dir") / "features" / args.encoder / "feature_bank.npz"
    bank = load_feature_bank(bank_path)
    output_dir = mil_run_root(config, args.encoder, args.run_tag)
    split_path = path_from(config, "split_manifest")
    provenance = {
        "feature_bank_sha256": sha256_file(bank_path),
        "case_manifest_sha256": sha256_file(path_from(config, "case_manifest")),
        "split_manifest_sha256": sha256_file(split_path),
        "selected_mil_config": str(mil_config_path),
        "selected_mil_config_sha256": sha256_file(mil_config_path),
    }
    with RunRecorder(
        "repeat_mil",
        vars(args),
        output_dir,
        inputs=[bank_path, path_from(config, "case_manifest"), split_path, mil_config_path],
    ) as run:
        run_repeated_mil(
            bank,
            cases,
            splits,
            config["labels"],
            mil,
            output_dir,
            seeds=seeds,
            device=args.device,
            max_images=int(config["study"]["max_images_per_case"]),
            provenance=provenance,
        )
        run.complete([output_dir / "repeated_metrics_long.csv", output_dir / "repeated_metrics_summary.csv"])


if __name__ == "__main__":
    main()
