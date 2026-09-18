#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from _common import load_context

from hawk_derm.config import path_from
from hawk_derm.data.splits import assert_no_split_leakage, freeze_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the shared 70/15/15 case split.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    args = parser.parse_args()
    config = load_context(args.config)
    cases = pd.read_csv(path_from(config, "case_manifest"))
    images = pd.read_csv(path_from(config, "image_manifest"))
    splits = freeze_splits(config, cases, images)
    image_audit_path = path_from(config, "audit_dir") / "image_audit.csv"
    if image_audit_path.exists():
        assert_no_split_leakage(splits, pd.read_csv(image_audit_path))


if __name__ == "__main__":
    main()
