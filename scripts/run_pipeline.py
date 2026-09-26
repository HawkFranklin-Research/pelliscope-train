#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from _common import REPOSITORY_ROOT, load_context

from hawk_derm.config import load_yaml, path_from
from hawk_derm.data.external import load_external_cohorts
from hawk_derm.features.registry import load_encoder_registry
from hawk_derm.data.splits import assert_classical_locked, assert_complete_image_audit, assert_existing_full_split
from hawk_derm.io import sha256_file
from hawk_derm.runtime import cpu_environment, resolve_cpu_workers


STAGES = (
    "download",
    "manifest",
    "audit",
    "split",
    "external-prepare",
    "external-features",
    "features",
    "classical",
    "tuning",
    "cross-validation",
    "mil",
    "ensemble",
    "thresholds",
    "final",
    "evaluation",
    "statistics",
    "tables",
    "figures",
    "verify",
    "external-evaluate",
    "external-figures",
    "bundle",
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
    if not (payload.get("complete") and payload.get("run_mode") == run_mode):
        return False
    if payload.get("split_version"):
        if not all(payload.get(key) for key in ("image_manifest_sha256", "image_audit_sha256", "split_manifest_sha256")):
            return False
        if sha256_file(outputs[0]) != payload["split_manifest_sha256"]:
            return False
    for item in payload.get("inputs", []):
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            return False
    for item in payload.get("outputs", []):
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            return False
    return True


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
    parser.add_argument("--mil-run-tag", default="canonical", help="Shared MIL output namespace for the full run.")
    parser.add_argument(
        "--archive-to-hub",
        action="store_true",
        help="Upload every trained model and report to the private model repo in the background (scripts/86_archive_run_to_hub.py).",
    )
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
        audit_path = path_from(config, "audit_dir") / "image_audit.csv"
        audit_resume = args.resume
        if audit_resume and args.run_mode == "full" and audit_path.is_file():
            try:
                assert_complete_image_audit(
                    pd.read_csv(path_from(config, "image_manifest")),
                    pd.read_csv(audit_path),
                    run_mode="full",
                )
            except (ValueError, FileNotFoundError):
                audit_resume = False
        run(
            "02_audit_data.py",
            *audit_arguments,
            "--workers",
            str(workers),
            environment=environment,
            resume=audit_resume,
            outputs=[audit_summary, audit_path],
            manifest=audit_summary,
        )
    if stage_selected("split", args.from_stage, args.to_stage):
        split_manifest = path_from(config, "split_manifest")
        split_resume = args.resume
        if split_resume and args.run_mode == "full":
            try:
                assert_existing_full_split(config)
            except (ValueError, FileNotFoundError):
                split_resume = False
        run(
            "03_freeze_splits.py",
            *config_args,
            environment=environment,
            resume=split_resume,
            outputs=[split_manifest],
            manifest=split_manifest.with_suffix(".metadata.json"),
        )
    if args.run_mode == "full" and STAGES.index(args.to_stage) >= STAGES.index("split"):
        assert_existing_full_split(config)
    registry = load_encoder_registry()
    skipped = set(comma_values(args.skip_encoders))
    included = [encoder for encoder in registry if encoder not in skipped]
    # External cohorts are inference-only. Their features are extracted before any model is
    # trained so every model scores them while it is still in memory.
    for cohort in load_external_cohorts(config):
        if not cohort.source_path("image_manifest").is_file():
            if stage_selected("external-prepare", args.from_stage, args.to_stage):
                print(f"[external] {cohort.name}: dataset not found at {cohort.dataset_root}; skipping external stages", flush=True)
            continue
        if args.run_mode == "smoke":
            # Smoke runs reuse external banks if present but never spend time extracting them.
            continue
        if stage_selected("external-prepare", args.from_stage, args.to_stage):
            run("80_prepare_external.py", *config_args, "--cohort", cohort.name, environment=environment)
        if stage_selected("external-features", args.from_stage, args.to_stage):
            run(
                "81_extract_external_features.py", *config_args, "--cohort", cohort.name,
                "--encoders", ",".join(included), "--device", args.device, "--workers", str(workers),
                environment=environment,
            )
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
    if args.run_mode == "full" and stage_selected("statistics", args.from_stage, args.to_stage):
        if primary != "siglip2_so400m" or "derm_foundation" not in encoders:
            raise ValueError("The prespecified paired comparison requires SigLIP2 as primary and Derm Foundation in this run")
    classifiers = list(load_yaml("configs/classifiers.yaml")["classifiers"])
    uploads: list[subprocess.Popen] = []

    def archive(path: Path) -> None:
        """Background upload of finished artifacts; failures are reported after compute ends."""
        if not args.archive_to_hub or args.run_mode != "full" or not path.is_dir():
            return
        uploads.append(
            subprocess.Popen(
                [sys.executable, str(REPOSITORY_ROOT / "scripts" / "86_archive_run_to_hub.py"), str(path), "--run-tag", args.mil_run_tag],
                cwd=REPOSITORY_ROOT,
                env=environment,
            )
        )
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
            archive(artifacts / "models" / "classical" / encoder)
    if args.run_mode == "full" and STAGES.index(args.to_stage) >= STAGES.index("tuning"):
        assert_classical_locked(config, encoders, classifiers)
    mil_stages = ("tuning", "cross-validation", "mil", "ensemble", "thresholds")
    selected_mil_stages = [stage for stage in mil_stages if stage_selected(stage, args.from_stage, args.to_stage)]
    common_mil = [
        *config_args, "--run-mode", args.run_mode, "--device", args.device,
        "--cpu-workers", str(workers), "--run-tag", args.mil_run_tag,
    ]
    if args.run_mode == "smoke":
        common_mil += [
            "--trials", str(config["smoke"]["trials"]),
            "--folds", str(config["smoke"]["folds"]),
            "--seeds", ",".join(map(str, config["smoke"]["seeds"])),
            "--epochs", str(config["smoke"]["epochs"]),
        ]
    if args.resume:
        common_mil.append("--resume")
    mil_stage_name = {"mil": "repeated-seeds"}
    for encoder in encoders if selected_mil_stages else []:
        run(
            "run_mil_pipeline.py", *common_mil, "--encoder", encoder,
            "--from-stage", mil_stage_name.get(selected_mil_stages[0], selected_mil_stages[0]),
            "--to-stage", mil_stage_name.get(selected_mil_stages[-1], selected_mil_stages[-1]),
            environment=environment,
        )
        archive(artifacts / "models" / "mil" / encoder / args.mil_run_tag)
    finishing_stages = ("final", "evaluation", "statistics", "tables", "figures", "verify")
    selected_finishing = [
        stage for stage in finishing_stages
        if stage_selected(stage, args.from_stage, args.to_stage) and (args.run_mode == "full" or stage in {"final", "evaluation"})
    ]
    if selected_finishing:
        finish_name = {"verify": "verification"}
        run(
            "run_mil_pipeline.py", *common_mil, "--encoder", primary,
            "--verify-encoders", ",".join(encoders),
            *(["--strict-locked-comparison"] if args.run_mode == "full" else []),
            "--from-stage", finish_name.get(selected_finishing[0], selected_finishing[0]),
            "--to-stage", finish_name.get(selected_finishing[-1], selected_finishing[-1]),
            environment=environment,
        )
    external_arguments = [*config_args, "--run-tag", args.mil_run_tag]
    if stage_selected("external-evaluate", args.from_stage, args.to_stage):
        run("82_evaluate_external.py", *external_arguments, "--workers", str(workers), environment=environment)
    if stage_selected("external-figures", args.from_stage, args.to_stage):
        run("83_external_figures.py", *external_arguments, environment=environment)
    if args.run_mode == "full" and stage_selected("bundle", args.from_stage, args.to_stage):
        run("84_build_production_bundles.py", *config_args, environment=environment)
        archive(artifacts / "production")
    if args.archive_to_hub:
        archive(path_from(config, "reports_dir"))
    for process in uploads:
        if process.wait() != 0:
            raise RuntimeError("A background Hugging Face upload failed")


if __name__ == "__main__":
    main()
