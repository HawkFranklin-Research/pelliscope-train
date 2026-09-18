#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import load_context

from hawk_derm.statistics.comparison import compare_prediction_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired case-level bootstrap model comparisons.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--first-name", required=True)
    parser.add_argument("--second-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    config = load_context(args.config)
    run_mode = config["study"]["run_mode"]
    replicates = args.replicates or int(config[run_mode]["bootstrap_replicates"])
    compare_prediction_files(
        args.first,
        args.second,
        config["labels"],
        args.output_dir,
        replicates=replicates,
        seed=args.seed,
        first_name=args.first_name,
        second_name=args.second_name,
    )


if __name__ == "__main__":
    main()
