#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _common import load_context, load_manifests

from hawk_derm.config import load_yaml, path_from
from hawk_derm.figures.manuscript import generate_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manuscript and appendix figures from saved plot data.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--primary-encoder", default="siglip2_so400m")
    parser.add_argument("--mil-run-tag", default=None)
    parser.add_argument(
        "--only",
        help="Comma-separated groups: cohort,performance,mil,thresholds,operating,supplementary,closed",
    )
    args = parser.parse_args()
    config = load_context(args.config)
    cases, images, _ = load_manifests(config)
    selected = {value.strip() for value in args.only.split(",") if value.strip()} if args.only else None
    reports_dir = path_from(config, "reports_dir")
    if args.mil_run_tag:
        reports_dir = reports_dir / "reanalysis" / f"{args.primary_encoder}_mil_{args.mil_run_tag}"
    generate_figures(
        cases,
        images,
        reports_dir,
        path_from(config, "artifacts_dir"),
        load_yaml("configs/figures.yaml"),
        only=selected,
    )


if __name__ == "__main__":
    main()
