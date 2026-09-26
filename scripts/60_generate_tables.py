#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _common import load_context, load_manifests, mil_run_root

from hawk_derm.config import path_from
from hawk_derm.figures.tables import generate_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all machine-readable manuscript tables and plot data.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--primary-encoder", default="siglip2_so400m")
    parser.add_argument("--mil-run-tag", default=None)
    parser.add_argument("--encoders", default=None, help="Comma-separated encoders included in a tagged full run.")
    args = parser.parse_args()
    config = load_context(args.config)
    cases, images, splits = load_manifests(config)
    reports_dir = path_from(config, "reports_dir")
    primary_root = None
    if args.mil_run_tag:
        primary_root = mil_run_root(config, args.primary_encoder, args.mil_run_tag)
        reports_dir = reports_dir / "reanalysis" / f"{args.primary_encoder}_mil_{args.mil_run_tag}"
    generate_tables(
        cases,
        images,
        splits,
        config["labels"],
        path_from(config, "artifacts_dir"),
        reports_dir,
        primary_mil_root=primary_root,
        mil_run_tag=args.mil_run_tag if args.encoders else None,
        selected_encoders=args.encoders.split(",") if args.encoders else None,
    )


if __name__ == "__main__":
    main()
