#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _common import load_context, load_manifests

from hawk_derm.config import path_from
from hawk_derm.figures.tables import generate_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all machine-readable manuscript tables and plot data.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    args = parser.parse_args()
    config = load_context(args.config)
    cases, images, splits = load_manifests(config)
    generate_tables(cases, images, splits, config["labels"], path_from(config, "artifacts_dir"), path_from(config, "reports_dir"))


if __name__ == "__main__":
    main()
