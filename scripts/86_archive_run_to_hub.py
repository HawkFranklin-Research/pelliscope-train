#!/usr/bin/env python3
"""Archive run artifacts (every trial, fold, seed and classical model) to the private model repo.

Designed to run in the background from run_pipeline.py --archive-to-hub, or by hand. Files go
under runs/<commit>/<run-tag>/<relative path>, so each code version keeps its own copy and
earlier archives are never overwritten.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from _common import REPOSITORY_ROOT

from hawk_derm.provenance.run import git_commit

DEFAULT_REPO = "HawkFranklin-Research/pelliscope-25class-models-internal"
IGNORE = ["**/*_shards/**", "**/__pycache__/**", "**/*.tmp"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="Directories to archive (relative to the repository root).")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--run-tag", default="canonical")
    parser.add_argument("--token", default=None)
    args = parser.parse_args()
    from huggingface_hub import HfApi

    api = HfApi(token=args.token or os.getenv("HF_TOKEN"))
    prefix = f"runs/{git_commit()}/{args.run_tag}"
    for path in args.paths:
        path = path if path.is_absolute() else REPOSITORY_ROOT / path
        if not path.is_dir():
            print(f"[archive] skipped missing {path}", flush=True)
            continue
        relative = path.relative_to(REPOSITORY_ROOT) if REPOSITORY_ROOT in path.parents else Path(path.name)
        api.upload_folder(
            repo_id=args.repo,
            folder_path=str(path),
            path_in_repo=f"{prefix}/{relative.as_posix()}",
            ignore_patterns=IGNORE,
            commit_message=f"Archive {relative.as_posix()} ({prefix})",
        )
        print(f"[archive] {relative} -> {args.repo}/{prefix}/{relative.as_posix()}", flush=True)


if __name__ == "__main__":
    main()
