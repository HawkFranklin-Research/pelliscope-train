#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from _common import load_context

from hawk_derm.config import path_from
from hawk_derm.data.audit import audit_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit image existence, decoding, paths, and hashes.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail when canonical full-run counts do not match the study lock.")
    args = parser.parse_args()
    config = load_context(args.config)
    cases = pd.read_csv(path_from(config, "case_manifest"))
    images = pd.read_csv(path_from(config, "image_manifest"))
    audit_dataset(config, cases, images, decode=not args.skip_decode, workers=args.workers, strict=args.strict)


if __name__ == "__main__":
    main()
