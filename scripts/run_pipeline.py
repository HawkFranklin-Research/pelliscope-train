#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from _common import REPOSITORY_ROOT, load_context

from hawk_derm.config import load_yaml, path_from
from hawk_derm.features.registry import load_encoder_registry
from hawk_derm.runtime import cpu_environment, resolve_cpu_workers


STAGES = (
    "download",
    "manifest",
    "audit",
    "split",
    "features",
    "classical",
    "mil",
    "thresholds",
    "tuning",
    "cross-validation",
    "final",
    "evaluation",
    "statistics",
    "tables",
    "figures",
    "verify",
)


def completed_artifact(outputs: list[Path], manifest: Path | None, run_mode: str) -> bool:
    if not outputs or any(not path.is_file() or path.stat().st_size == 0 for path in outputs):
        return False
    if manifest is None or not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return bool(payload.get("complete")) and payload.get("run_mode") == run_mode


def run(
    script: str,
    *arguments: str,
    environment: dict[str, str],
    resume: bool = False,
    outputs: list[Path] | None = None,
    manifest: Path | None = None,
) -> None:
    if resume and completed_artifact(outputs or [], manifest, environment["HAWK_DERM_RUN_MODE"]):
        print(f"[resume] {script}: verified outputs already complete", flush=True)
        return
    command = [sys.executable, str(REPOSITORY_ROOT / "scripts" / script), *arguments]
    print(f"[run] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)


def comma_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def stage_selected(stage: str, first: str, last: str) -> bool:
    return STAGES.index(first) <= STAGES.index(stage) <= STAGES.index(last)


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
    parser.add_argument("--cpu-workers", type=int, default=None, help="CPU budget assigned to each top-level pipeline job.")
    parser.add_argument("--download-workers", type=int, default=None, help="Concurrent Hugging Face file downloads (maximum 32).")
    parser.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--to-stage", choices=STAGES, default=STAGES[-1])
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip expensive stages only when their non-empty outputs and matching smoke/full run manifest are complete.",
    )
    args = parser.parse_args()
    if STAGES.index(args.from_stage) > STAGES.index(args.to_stage):
        parser.error("--from-stage must not come after --to-stage")
    config = load_context(args.config, args.run_mode)
    workers = resolve_cpu_workers(args.cpu_workers)
    environment = cpu_environment(workers, dict(os.environ, HAWK_DERM_RUN_MODE=args.run_mode))
    config_args = ["--config", args.config]
    artifacts = path_from(config, "artifacts_dir")
    if args.download_data and stage_selected("download", args.from_stage, args.to_stage):
        download_workers = min(32, resolve_cpu_workers(args.download_workers or workers))
        run(
            "00_download_data.py",
            *config_args,
            "--revision",
            args.data_revision,
            "--max-workers",
            str(download_workers),
            environment=environment,
        )
    if stage_selected("manifest", args.from_stage, args.to_stage):
        case_manifest = path_from(config, "case_manifest")
        image_manifest = path_from(config, "image_manifest")
        run(
            "01_build_manifest.py",
            *config_args,
            "--run-mode",
            args.run_mode,
            environment=environment,
            resume=args.resume,
            outputs=[case_manifest, image_manifest],
            manifest=case_manifest.with_suffix(".metadata.json"),
        )
    audit_arguments = [*config_args]
    if args.run_mode == "full":
        audit_arguments.append("--strict")
    if stage_selected("audit", args.from_stage, args.to_stage):
        audit_summary = path_from(config, "audit_dir") / "data_audit_summary.json"
        run(
            "02_audit_data.py",
            *audit_arguments,
            "--workers",
            str(workers),
            environment=environment,
            resume=args.resume,
            outputs=[audit_summary, path_from(config, "audit_dir") / "image_audit.csv"],
            manifest=audit_summary,
        )
    if stage_selected("split", args.from_stage, args.to_stage):
        split_manifest = path_from(config, "split_manifest")
        run(
            "03_freeze_splits.py",
            *config_args,
            environment=environment,
            resume=args.resume,
            outputs=[split_manifest],
            manifest=split_manifest.with_suffix(".metadata.json"),
        )
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
        extraction_arguments = [
            *config_args,
            "--encoder",
            encoder,
            "--device",
            args.device,
            "--workers",
            str(workers),
        ]
        import_key = f"HAWK_DERM_IMPORT_BANK_{encoder.upper()}"
        import_bank = environment.get(import_key) or registry[encoder].import_bank
        if import_bank:
            extraction_arguments += ["--import-bank", import_bank]
        feature_dir = artifacts / "features" / encoder
        if stage_selected("features", args.from_stage, args.to_stage):
            run(
                "10_extract_features.py",
                *extraction_arguments,
                environment=environment,
                resume=args.resume,
                outputs=[feature_dir / "feature_bank.npz", feature_dir / "feature_bank.metadata.json"],
                manifest=feature_dir / "run_manifest.json",
            )
        if args.publish_features and stage_selected("features", args.from_stage, args.to_stage):
            uploads.append(
                subprocess.Popen(
                    [sys.executable, str(REPOSITORY_ROOT / "scripts" / "11_publish_feature_bank.py"), *config_args, "--encoder", encoder],
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                )
            )
        if stage_selected("classical", args.from_stage, args.to_stage):
            for classifier in classifiers:
                model_dir = artifacts / "models" / "classical" / encoder / classifier
                run(
                    "20_train_classical.py",
                    *config_args,
                    "--encoder",
                    encoder,
                    "--classifier",
                    classifier,
                    "--workers",
                    str(workers),
                    environment=environment,
                    resume=args.resume,
                    outputs=[model_dir / "overall_metrics.csv", model_dir / "test_case_predictions.csv"],
                    manifest=model_dir / "run_manifest.json",
                )
        repeated_arguments = ["--encoder", encoder, "--device", args.device]
        if args.run_mode == "smoke":
            repeated_arguments += ["--epochs", str(config["smoke"]["epochs"]), "--seeds", ",".join(map(str, config["smoke"]["seeds"]))]
        mil_dir = artifacts / "models" / "mil" / encoder
        if stage_selected("mil", args.from_stage, args.to_stage):
            run(
                "33_run_repeated_seeds.py",
                *config_args,
                *repeated_arguments,
                environment=environment,
                resume=args.resume,
                outputs=[mil_dir / "repeated_metrics_long.csv", mil_dir / "repeated_metrics_summary.csv"],
                manifest=mil_dir / "run_manifest.json",
            )
        if stage_selected("thresholds", args.from_stage, args.to_stage):
            run("40_select_thresholds.py", *config_args, "--encoder", encoder, environment=environment)
    tuning_arguments = ["--encoder", primary, "--device", args.device]
    cv_arguments = ["--encoder", primary, "--device", args.device]
    if args.run_mode == "smoke":
        tuning_arguments += ["--trials", str(config["smoke"]["trials"]), "--epochs", str(config["smoke"]["epochs"])]
        cv_arguments += ["--folds", str(config["smoke"]["folds"]), "--epochs", str(config["smoke"]["epochs"])]
    primary_mil_dir = artifacts / "models" / "mil" / primary
    if stage_selected("tuning", args.from_stage, args.to_stage):
        run(
            "31_tune_mil.py",
            *config_args,
            *tuning_arguments,
            environment=environment,
            resume=args.resume,
            outputs=[primary_mil_dir / "tuning" / "trials.csv"],
            manifest=primary_mil_dir / "tuning" / "run_manifest.json",
        )
    if stage_selected("cross-validation", args.from_stage, args.to_stage):
        run(
            "32_run_cross_validation.py",
            *config_args,
            *cv_arguments,
            environment=environment,
            resume=args.resume,
            outputs=[primary_mil_dir / "cross_validation" / "cross_validation_metrics.csv"],
            manifest=primary_mil_dir / "cross_validation" / "run_manifest.json",
        )
    final_epochs = str(config["smoke"]["epochs"] if args.run_mode == "smoke" else load_yaml("configs/mil.yaml")["training"]["epochs"])
    if stage_selected("final", args.from_stage, args.to_stage):
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
    if stage_selected("evaluation", args.from_stage, args.to_stage):
        evaluation_dir = path_from(config, "reports_dir") / "evaluations" / f"{primary}_final"
        run(
            "50_evaluate.py",
            *config_args,
            "--predictions",
            str(primary_mil_dir / "final_model" / "test_predictions.csv"),
            "--thresholds",
            str(primary_mil_dir / "thresholds" / "validation_selected_thresholds.csv"),
            "--output-dir",
            str(evaluation_dir),
            environment=environment,
        )
    comparison_encoders = [encoder for encoder in ("siglip2_so400m", "derm_foundation") if encoder in encoders]
    if not comparison_encoders:
        comparison_encoders = [primary]
    if stage_selected("statistics", args.from_stage, args.to_stage):
        run(
            "51_compare_grid.py",
            *config_args,
            "--mil-encoders",
            ",".join(comparison_encoders),
            "--workers",
            str(workers),
            environment=environment,
        )
    if stage_selected("tables", args.from_stage, args.to_stage):
        run("60_generate_tables.py", *config_args, environment=environment)
    if stage_selected("figures", args.from_stage, args.to_stage):
        run("61_generate_figures.py", *config_args, environment=environment)
    for process in uploads:
        if process.wait() != 0:
            raise RuntimeError("A background Hugging Face upload failed")
    if stage_selected("verify", args.from_stage, args.to_stage):
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
