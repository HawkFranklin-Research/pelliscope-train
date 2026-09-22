#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from torch import nn

from _common import REPOSITORY_ROOT, load_context, mil_run_root

from hawk_derm.config import load_yaml, path_from
from hawk_derm.models.mil import GatedAttentionMIL
from hawk_derm.runtime import cpu_environment, resolve_cpu_workers


STAGES = (
    "preflight",
    "implementation-check",
    "tuning",
    "cross-validation",
    "repeated-seeds",
    "ensemble",
    "thresholds",
    "final",
    "evaluation",
    "statistics",
    "tables",
    "figures",
    "verification",
)


def selected(stage: str, first: str, last: str) -> bool:
    return STAGES.index(first) <= STAGES.index(stage) <= STAGES.index(last)


def run_script(
    script: str,
    arguments: list[str],
    *,
    environment: dict[str, str],
    resume: bool,
    outputs: list[Path],
) -> None:
    if resume and outputs and all(path.is_file() and path.stat().st_size > 0 for path in outputs):
        print(f"[resume] {script}: expected outputs already exist", flush=True)
        return
    command = [sys.executable, str(REPOSITORY_ROOT / "scripts" / script), *arguments]
    print(f"[run] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)


def preflight(config: dict, encoder: str, run_mode: str) -> None:
    required = [
        path_from(config, "case_manifest"),
        path_from(config, "image_manifest"),
        path_from(config, "split_manifest"),
        path_from(config, "artifacts_dir") / "features" / encoder / "feature_bank.npz",
        path_from(config, "artifacts_dir")
        / "models"
        / "classical"
        / "siglip2_so400m"
        / "random_forest"
        / "test_case_predictions.csv",
        path_from(config, "artifacts_dir")
        / "models"
        / "classical"
        / "derm_foundation"
        / "random_forest"
        / "test_case_predictions.csv",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"MIL-only pipeline prerequisites are missing: {missing}")
    cases = pd.read_csv(path_from(config, "case_manifest"))
    splits = pd.read_csv(path_from(config, "split_manifest"))
    if cases["case_id"].duplicated().any() or splits["case_id"].duplicated().any():
        raise ValueError("Case and split manifests must contain one row per case")
    if set(cases["case_id"].astype(str)) != set(splits["case_id"].astype(str)):
        raise ValueError("Case and split manifests do not contain identical case IDs")
    if run_mode == "full" and len(cases) != int(config["study"]["canonical_case_count"]):
        raise ValueError("Full MIL rerun requires the canonical 5,033-case manifest")
    if set(splits["split"]) != {"train", "validation", "test"}:
        raise ValueError("Locked split manifest must contain train, validation, and test memberships")


def implementation_check() -> None:
    config = load_yaml("configs/mil.yaml")
    if config.get("implementation") != "canonical_siglip2_mil":
        raise ValueError("configs/mil.yaml must declare implementation: canonical_siglip2_mil")
    training = config["training"]
    if training.get("scheduler") != "cosine" or training.get("checkpoint_metric") != "macro_auc_plus_lrap":
        raise ValueError("September scheduler and checkpoint selection are not configured")
    model = GatedAttentionMIL(input_dim=8, class_count=3, **config["architecture"])
    if not isinstance(model.attention_dropout, nn.Dropout):
        raise TypeError("Gated attention dropout is missing")
    modules = list(model.classifier.children())
    first_linear = next(index for index, module in enumerate(modules) if isinstance(module, nn.Linear))
    if first_linear + 1 >= len(modules) or not isinstance(modules[first_linear + 1], nn.LayerNorm):
        raise TypeError("September shared-head Linear -> LayerNorm order is missing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run only the fixed-split MIL workflow on existing features.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--encoder", default="siglip2_so400m")
    parser.add_argument("--run-mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--cpu-workers", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-tag", default="canonical")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--to-stage", choices=STAGES, default=STAGES[-1])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if STAGES.index(args.from_stage) > STAGES.index(args.to_stage):
        parser.error("--from-stage must not come after --to-stage")

    config = load_context(args.config, args.run_mode)
    workers = resolve_cpu_workers(args.cpu_workers)
    environment = cpu_environment(
        workers,
        dict(os.environ, HAWK_DERM_RUN_MODE=args.run_mode, HAWK_DERM_MIL_RUN_TAG=args.run_tag),
    )
    root = mil_run_root(config, args.encoder, args.run_tag)
    selected_config = root / "tuning" / "best_mil_config.yaml"
    common = ["--config", args.config, "--encoder", args.encoder, "--run-tag", args.run_tag]

    if selected("preflight", args.from_stage, args.to_stage):
        preflight(config, args.encoder, args.run_mode)
    if selected("implementation-check", args.from_stage, args.to_stage):
        implementation_check()

    tuning_args = [*common, "--device", args.device]
    if args.trials is not None:
        tuning_args += ["--trials", str(args.trials)]
    if args.epochs is not None:
        tuning_args += ["--epochs", str(args.epochs)]
    if selected("tuning", args.from_stage, args.to_stage):
        run_script(
            "31_tune_mil.py",
            tuning_args,
            environment=environment,
            resume=args.resume,
            outputs=[selected_config, root / "tuning" / "selection_manifest.json"],
        )

    downstream = [*common, "--mil-config", str(selected_config), "--device", args.device]
    cv_args = list(downstream)
    if args.folds is not None:
        cv_args += ["--folds", str(args.folds)]
    if args.epochs is not None:
        cv_args += ["--epochs", str(args.epochs)]
    if selected("cross-validation", args.from_stage, args.to_stage):
        run_script(
            "32_run_cross_validation.py",
            cv_args,
            environment=environment,
            resume=args.resume,
            outputs=[root / "cross_validation" / "cross_validation_metrics.csv"],
        )

    repeated_args = list(downstream)
    if args.seeds:
        repeated_args += ["--seeds", args.seeds]
    if args.epochs is not None:
        repeated_args += ["--epochs", str(args.epochs)]
    if selected("repeated-seeds", args.from_stage, args.to_stage):
        run_script(
            "33_run_repeated_seeds.py",
            repeated_args,
            environment=environment,
            resume=args.resume,
            outputs=[root / "repeated_metrics_long.csv", root / "repeated_metrics_summary.csv"],
        )

    no_device = [*common, "--mil-config", str(selected_config)]
    if selected("ensemble", args.from_stage, args.to_stage):
        run_script(
            "35_ensemble_mil.py",
            no_device,
            environment=environment,
            resume=args.resume,
            outputs=[root / "ensemble" / "test_predictions.csv", root / "ensemble" / "ensemble_manifest.json"],
        )
    if selected("thresholds", args.from_stage, args.to_stage):
        run_script(
            "40_select_thresholds.py",
            no_device,
            environment=environment,
            resume=args.resume,
            outputs=[root / "thresholds" / "validation_selected_thresholds.csv"],
        )

    final_args = list(downstream)
    if args.epochs is not None:
        final_args += ["--epochs", str(args.epochs)]
    if selected("final", args.from_stage, args.to_stage):
        run_script(
            "34_fit_final_model.py",
            final_args,
            environment=environment,
            resume=args.resume,
            outputs=[root / "final_model" / "model.pt", root / "final_model" / "test_predictions.csv"],
        )

    report_root = path_from(config, "reports_dir") / "reanalysis" / f"{args.encoder}_mil_{args.run_tag}"
    thresholds = root / "thresholds" / "validation_selected_thresholds.csv"
    if selected("evaluation", args.from_stage, args.to_stage):
        for role, predictions in (
            ("ensemble", root / "ensemble" / "test_predictions.csv"),
            ("final_model", root / "final_model" / "test_predictions.csv"),
        ):
            run_script(
                "50_evaluate.py",
                [
                    "--config",
                    args.config,
                    "--predictions",
                    str(predictions),
                    "--thresholds",
                    str(thresholds),
                    "--output-dir",
                    str(report_root / "evaluations" / role),
                ],
                environment=environment,
                resume=args.resume,
                outputs=[report_root / "evaluations" / role / "overall_metrics.csv"],
            )

    if selected("statistics", args.from_stage, args.to_stage):
        run_script(
            "51_compare_grid.py",
            [*common, "--workers", str(workers)],
            environment=environment,
            resume=args.resume,
            outputs=[report_root / "statistics" / "comparison_grid_manifest.json"],
        )
    report_args = [
        "--config",
        args.config,
        "--primary-encoder",
        args.encoder,
        "--mil-run-tag",
        args.run_tag,
    ]
    if selected("tables", args.from_stage, args.to_stage):
        run_script(
            "60_generate_tables.py",
            report_args,
            environment=environment,
            resume=args.resume,
            outputs=[report_root / "tables" / "table_generation_manifest.json"],
        )
    if selected("figures", args.from_stage, args.to_stage):
        run_script(
            "61_generate_figures.py",
            report_args,
            environment=environment,
            resume=args.resume,
            outputs=[report_root / "figures" / "figure_generation_manifest.json"],
        )
    if selected("verification", args.from_stage, args.to_stage):
        run_script(
            "70_verify_release.py",
            [
                "--config",
                args.config,
                "--primary-encoder",
                args.encoder,
                "--mil-run-tag",
                args.run_tag,
            ],
            environment=environment,
            resume=False,
            outputs=[],
        )


if __name__ == "__main__":
    main()
