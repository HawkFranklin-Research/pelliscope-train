#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import load_context

from hawk_derm.config import path_from
from hawk_derm.features.publish import cleanup_verified_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete one uploaded feature bank after receipt verification.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--remove-encoder-directory", action="store_true")
    args = parser.parse_args()
    config = load_context(args.config)
    cleanup_verified_artifact(
        args.receipt,
        expected_root=path_from(config, "artifacts_dir") / "features",
        remove_encoder_directory=args.remove_encoder_directory,
    )


if __name__ == "__main__":
    main()
