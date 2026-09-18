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


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the staged smoke or full reproduction workflow.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--run-mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--publish-features", action="store_true")
    args = parser.parse_args()
    config = load_context(args.config, args.run_mode)
    environment = dict(os.environ, HAWK_DERM_RUN_MODE=args.run_mode)
    config_args = ["--config", args.config]
    run("01_build_manifest.py", *config_args, "--run-mode", args.run_mode, environment=environment)
    audit_arguments = [*config_args]
    if args.run_mode == "full":
        audit_arguments.append("--strict")
    run("02_audit_data.py", *audit_arguments, environment=environment)
    run("03_freeze_splits.py", *config_args, environment=environment)
    encoders = list(load_encoder_registry())
    classifiers = list(load_yaml("configs/classifiers.yaml")["classifiers"])
    uploads: list[subprocess.Popen] = []
    registry = load_encoder_registry()
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
    primary = "siglip2_so400m"
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
        "artifacts/models/mil/siglip2_so400m/final_model/test_predictions.csv",
        "--thresholds",
        "artifacts/models/mil/siglip2_so400m/thresholds/validation_selected_thresholds.csv",
        "--output-dir",
        "reports/evaluations/siglip2_so400m_final",
        environment=environment,
    )
    run("51_compare_grid.py", *config_args, environment=environment)
    run("60_generate_tables.py", *config_args, environment=environment)
    run("61_generate_figures.py", *config_args, environment=environment)
    for process in uploads:
        if process.wait() != 0:
            raise RuntimeError("A background Hugging Face upload failed")
    run("70_verify_release.py", *config_args, environment=environment)


if __name__ == "__main__":
    main()
