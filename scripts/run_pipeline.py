#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from _common import REPOSITORY_ROOT, load_context

from hawk_derm.config import load_yaml
from hawk_derm.features.registry import load_encoder_registry


def run(script: str, *arguments: str, environment: dict[str, str]) -> None:
    command = [sys.executable, str(REPOSITORY_ROOT / "scripts" / script), *arguments]
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)


def comma_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the staged smoke or full reproduction workflow.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--run-mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--publish-features", action="store_true")
    parser.add_argument("--download-data", action="store_true", help="Download the raw SCIN dataset before building manifests.")
    parser.add_argument("--data-revision", default="main", help="Hugging Face dataset revision used with --download-data.")
    parser.add_argument("--skip-encoders", default="", help="Comma-separated encoder keys to omit from this run.")
    parser.add_argument("--primary-encoder", default=None, help="Encoder used for tuning, cross-validation, and final fitting.")
    args = parser.parse_args()
    config = load_context(args.config, args.run_mode)
    environment = dict(os.environ, HAWK_DERM_RUN_MODE=args.run_mode)
    config_args = ["--config", args.config]
    if args.download_data:
        run("00_download_data.py", *config_args, "--revision", args.data_revision, environment=environment)
    run("01_build_manifest.py", *config_args, "--run-mode", args.run_mode, environment=environment)
    audit_arguments = [*config_args]
    if args.run_mode == "full":
        audit_arguments.append("--strict")
    run("02_audit_data.py", *audit_arguments, environment=environment)
    run("03_freeze_splits.py", *config_args, environment=environment)
    registry = load_encoder_registry()
    skipped = set(comma_values(args.skip_encoders))
    unknown = skipped.difference(registry)
    if unknown:
        raise ValueError(f"Unknown encoder keys in --skip-encoders: {sorted(unknown)}")
    encoders = [encoder for encoder in registry if encoder not in skipped]
    if not encoders:
        raise ValueError("At least one encoder must remain after --skip-encoders")
    if args.primary_encoder:
        primary = args.primary_encoder
        if primary not in encoders:
            raise ValueError("--primary-encoder must be one of the encoders included in this run")
    elif "siglip2_so400m" in encoders:
        primary = "siglip2_so400m"
    elif "derm_foundation" in encoders:
        primary = "derm_foundation"
    else:
        primary = encoders[0]
    classifiers = list(load_yaml("configs/classifiers.yaml")["classifiers"])
    uploads: list[subprocess.Popen] = []
    for encoder in encoders:
        extraction_arguments = [*config_args, "--encoder", encoder, "--device", args.device]
        import_key = f"HAWK_DERM_IMPORT_BANK_{encoder.upper()}"
        import_bank = environment.get(import_key) or registry[encoder].import_bank
        if import_bank:
            extraction_arguments += ["--import-bank", import_bank]
        run("10_extract_features.py", *extraction_arguments, environment=environment)
        if args.publish_features:
            uploads.append(
                subprocess.Popen(
                    [sys.executable, str(REPOSITORY_ROOT / "scripts" / "11_publish_feature_bank.py"), *config_args, "--encoder", encoder],
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                )
            )
        for classifier in classifiers:
            run("20_train_classical.py", *config_args, "--encoder", encoder, "--classifier", classifier, environment=environment)
        repeated_arguments = ["--encoder", encoder, "--device", args.device]
        if args.run_mode == "smoke":
            repeated_arguments += ["--epochs", str(config["smoke"]["epochs"]), "--seeds", ",".join(map(str, config["smoke"]["seeds"]))]
        run("33_run_repeated_seeds.py", *config_args, *repeated_arguments, environment=environment)
        run("40_select_thresholds.py", *config_args, "--encoder", encoder, environment=environment)
    tuning_arguments = ["--encoder", primary, "--device", args.device]
    cv_arguments = ["--encoder", primary, "--device", args.device]
    if args.run_mode == "smoke":
        tuning_arguments += ["--trials", str(config["smoke"]["trials"]), "--epochs", str(config["smoke"]["epochs"])]
        cv_arguments += ["--folds", str(config["smoke"]["folds"]), "--epochs", str(config["smoke"]["epochs"])]
    run("31_tune_mil.py", *config_args, *tuning_arguments, environment=environment)
    run("32_run_cross_validation.py", *config_args, *cv_arguments, environment=environment)
    final_epochs = str(config["smoke"]["epochs"] if args.run_mode == "smoke" else load_yaml("configs/mil.yaml")["training"]["epochs"])
    run(
        "34_fit_final_model.py",
        *config_args,
        "--encoder",
        primary,
        "--device",
        args.device,
        "--epochs",
        final_epochs,
        environment=environment,
    )
    run(
        "50_evaluate.py",
        *config_args,
        "--predictions",
        f"artifacts/models/mil/{primary}/final_model/test_predictions.csv",
        "--thresholds",
        f"artifacts/models/mil/{primary}/thresholds/validation_selected_thresholds.csv",
        "--output-dir",
        f"reports/evaluations/{primary}_final",
        environment=environment,
    )
    comparison_encoders = [encoder for encoder in ("siglip2_so400m", "derm_foundation") if encoder in encoders]
    if not comparison_encoders:
        comparison_encoders = [primary]
    run("51_compare_grid.py", *config_args, "--mil-encoders", ",".join(comparison_encoders), environment=environment)
    run("60_generate_tables.py", *config_args, environment=environment)
    run("61_generate_figures.py", *config_args, environment=environment)
    for process in uploads:
        if process.wait() != 0:
            raise RuntimeError("A background Hugging Face upload failed")
    run(
        "70_verify_release.py",
        *config_args,
        "--encoders",
        ",".join(encoders),
        "--primary-encoder",
        primary,
        environment=environment,
    )


if __name__ == "__main__":
    main()
