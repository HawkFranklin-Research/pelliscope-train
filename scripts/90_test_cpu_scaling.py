#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _common import REPOSITORY_ROOT, load_context, load_manifests

from hawk_derm.config import load_yaml, path_from
from hawk_derm.features.bank import FeatureBank, load_feature_bank
from hawk_derm.models.classical import run_classical_experiment
from hawk_derm.models.experiments import run_mil_experiment
from hawk_derm.runtime import available_cpu_count, configure_process, cpu_environment


MODELS = ("logistic", "svm_linear", "random_forest", "gradient_boosting", "knn", "mil")


def _process_stats() -> dict[int, tuple[int, int, int]]:
    """Return pid -> (parent pid, CPU ticks, last logical CPU) from Linux /proc."""
    stats: dict[int, tuple[int, int, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "stat").read_text()
            fields = text[text.rfind(")") + 2 :].split()
            stats[int(entry.name)] = (int(fields[1]), int(fields[11]) + int(fields[12]), int(fields[36]))
        except (FileNotFoundError, IndexError, PermissionError, ValueError):
            continue
    return stats


def _descendants(root_pid: int, stats: dict[int, tuple[int, int, int]]) -> set[int]:
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _, _) in stats.items():
            if parent in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return selected


def _thread_stats(pids: set[int]) -> dict[int, tuple[int, int]]:
    """Return thread id -> (CPU ticks, last logical CPU) for a process tree."""
    threads: dict[int, tuple[int, int]] = {}
    for pid in pids:
        task_root = Path("/proc") / str(pid) / "task"
        try:
            tasks = list(task_root.iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for task in tasks:
            try:
                text = (task / "stat").read_text()
                fields = text[text.rfind(")") + 2 :].split()
                threads[int(task.name)] = (int(fields[11]) + int(fields[12]), int(fields[36]))
            except (FileNotFoundError, IndexError, PermissionError, ValueError):
                continue
    return threads


def monitor(command: list[str], environment: dict[str, str], interval: float) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=REPOSITORY_ROOT, env=environment)
    clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    previous_time = time.monotonic()
    previous_ticks: dict[int, int] = {}
    previous_thread_ticks: dict[int, int] = {}
    samples: list[dict[str, Any]] = []
    cpu_union: set[int] = set()
    while process.poll() is None:
        time.sleep(interval)
        now = time.monotonic()
        stats = _process_stats()
        pids = _descendants(process.pid, stats)
        current_ticks = {pid: stats[pid][1] for pid in pids if pid in stats}
        current_threads = _thread_stats(pids)
        delta_ticks = sum(max(0, ticks - previous_ticks.get(pid, 0)) for pid, ticks in current_ticks.items())
        elapsed = max(now - previous_time, 1e-9)
        aggregate_percent = 100.0 * delta_ticks / clock_ticks / elapsed
        active_cpus = {
            cpu
            for thread_id, (ticks, cpu) in current_threads.items()
            if ticks > previous_thread_ticks.get(thread_id, 0)
        }
        cpu_union.update(active_cpus)
        samples.append(
            {
                "elapsed_seconds": now - started,
                "process_tree_size": len(pids),
                "aggregate_cpu_percent": aggregate_percent,
                "equivalent_busy_cores": aggregate_percent / 100.0,
                "observed_logical_cpus": sorted(active_cpus),
            }
        )
        previous_ticks = current_ticks
        previous_thread_ticks = {thread_id: ticks for thread_id, (ticks, _) in current_threads.items()}
        previous_time = now
    return_code = process.wait()
    duration = time.monotonic() - started
    busy_cores = [float(sample["equivalent_busy_cores"]) for sample in samples]
    return {
        "return_code": return_code,
        "duration_seconds": duration,
        "sample_count": len(samples),
        "mean_equivalent_busy_cores": float(np.mean(busy_cores)) if busy_cores else 0.0,
        "peak_equivalent_busy_cores": max(busy_cores, default=0.0),
        "observed_logical_cpu_count": len(cpu_union),
        "observed_logical_cpus": sorted(cpu_union),
        "samples": samples,
    }


def subset_fixture(
    bank: FeatureBank,
    cases: pd.DataFrame,
    splits: pd.DataFrame,
    max_cases: int,
) -> tuple[FeatureBank, pd.DataFrame, pd.DataFrame]:
    available = set(bank.case_ids.astype(str))
    joined = splits[splits["case_id"].astype(str).isin(available)].copy()
    selected: list[str] = []
    fractions = {"train": 0.70, "validation": 0.15, "test": 0.15}
    for split_name, fraction in fractions.items():
        candidates = joined.loc[joined["split"].eq(split_name), "case_id"].astype(str).sort_values()
        limit = max(2, round(max_cases * fraction))
        selected.extend(candidates.head(limit).tolist())
    selected_set = set(selected[:max_cases])
    selected_cases = cases[cases["case_id"].astype(str).isin(selected_set)].copy()
    selected_splits = splits[splits["case_id"].astype(str).isin(selected_set)].copy()
    image_mask = np.isin(bank.case_ids.astype(str), list(selected_set))
    selected_bank = FeatureBank(
        embeddings=bank.embeddings[image_mask],
        case_ids=bank.case_ids[image_mask],
        image_ids=bank.image_ids[image_mask],
        image_paths=bank.image_paths[image_mask],
        image_sha256=bank.image_sha256[image_mask],
        encoder=bank.encoder,
        model_id=bank.model_id,
        revision=bank.revision,
        dimension=bank.dimension,
    )
    if selected_splits["split"].nunique() != 3:
        raise ValueError("CPU test fixture requires populated train, validation, and test splits")
    return selected_bank, selected_cases, selected_splits


def worker(args: argparse.Namespace) -> None:
    workers = configure_process(args.effective_workers)
    config = load_context(args.config, "smoke")
    cases, _, splits = load_manifests(config)
    bank_path = path_from(config, "artifacts_dir") / "features" / args.encoder / "feature_bank.npz"
    bank = load_feature_bank(bank_path)
    bank, cases, splits = subset_fixture(bank, cases, splits, args.max_cases)
    output_root = Path(args.output_dir)
    started = time.monotonic()
    repetition = 0
    while repetition == 0 or (
        time.monotonic() - started < args.minimum_runtime and repetition < args.max_repetitions
    ):
        output = output_root / f"repeat_{repetition:03d}"
        if args.model == "mil":
            mil = load_yaml("configs/mil.yaml")
            mil["training"]["epochs"] = args.epochs
            run_mil_experiment(
                bank,
                cases,
                splits,
                config["labels"],
                mil,
                output,
                seed=42 + repetition,
                device="cpu",
                max_images=int(config["study"]["max_images_per_case"]),
            )
        else:
            classifiers = load_yaml("configs/classifiers.yaml")["classifiers"]
            run_classical_experiment(
                bank,
                cases,
                splits,
                config["labels"],
                args.model,
                classifiers[args.model],
                output,
                seed=42 + repetition,
                workers=workers,
            )
        repetition += 1
    print(json.dumps({"model": args.model, "encoder": args.encoder, "workers": workers, "repetitions": repetition}))


def available_encoders(config: dict[str, Any]) -> list[str]:
    root = path_from(config, "artifacts_dir") / "features"
    return sorted(path.parent.name for path in root.glob("*/feature_bank.npz") if path.stat().st_size > 0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly select one trained model path and compare all-CPU (-1) with an explicit 32-worker request."
    )
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Optional selection seed for a reproducible random choice.")
    parser.add_argument("--max-cases", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minimum-runtime", type=float, default=3.0)
    parser.add_argument("--max-repetitions", type=int, default=3)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=MODELS, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--encoder", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--effective-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return

    config = load_context(args.config, "smoke")
    encoders = available_encoders(config)
    if not encoders:
        raise FileNotFoundError("No feature bank is available. Run the bounded smoke feature stage first.")
    selection_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    rng = random.Random(selection_seed)
    model = rng.choice(MODELS)
    encoder = rng.choice(encoders)
    available = available_cpu_count()
    report_path = args.report or (
        path_from(config, "reports_dir")
        / "test_logs"
        / f"cpu_scaling_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="hawk-derm-cpu-test-"))
    results = []
    print(f"Selected model={model}, encoder={encoder}, selection_seed={selection_seed}", flush=True)
    for requested in (-1, 32):
        effective = available if requested == -1 else min(requested, available)
        output = root / f"workers_{requested}"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--config",
            args.config,
            "--model",
            model,
            "--encoder",
            encoder,
            "--effective-workers",
            str(effective),
            "--max-cases",
            str(args.max_cases),
            "--epochs",
            str(args.epochs),
            "--minimum-runtime",
            str(args.minimum_runtime),
            "--max-repetitions",
            str(args.max_repetitions),
            "--output-dir",
            str(output),
        ]
        print(f"Running requested_workers={requested}, effective_workers={effective}", flush=True)
        result = monitor(command, cpu_environment(effective, os.environ), args.sample_interval)
        result.update({"requested_workers": requested, "effective_workers": effective})
        results.append(result)
    payload = {
        "complete": all(result["return_code"] == 0 for result in results),
        "selection_seed": selection_seed,
        "model": model,
        "encoder": encoder,
        "available_logical_cpus": available,
        "results": results,
    }
    report_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({**payload, "results": [{key: value for key, value in item.items() if key != "samples"} for item in results]}, indent=2))
    print(f"Report: {report_path}")
    if args.keep_output:
        print(f"Temporary training outputs retained at {root}")
    else:
        shutil.rmtree(root)
    if not payload["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
