#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _common import load_context

from hawk_derm.data.manifest import build_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical 25-class case and image manifests.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--run-mode", choices=["smoke", "full"], default=None)
    args = parser.parse_args()
    config = load_context(args.config, args.run_mode)
    build_manifests(config, config["study"]["run_mode"])


if __name__ == "__main__":
    main()
