#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _common import load_context

from hawk_derm.config import path_from
from hawk_derm.io import write_json
from hawk_derm.runtime import resolve_cpu_workers


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public SCIN raw-image dataset snapshot.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--repo-id", default="HawkFranklin-Research/SCIN-Dermatology-Raw-Images")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Concurrent Hugging Face file downloads. Defaults to the configured CPU budget, capped at 32.",
    )
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    config = load_context(args.config)
    destination = path_from(config, "raw_dataset_root")
    max_workers = min(32, resolve_cpu_workers(args.max_workers))
    snapshot = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=destination,
        token=args.token,
        max_workers=max_workers,
    )
    write_json(
        destination / "download_receipt.json",
        {
            "repo_id": args.repo_id,
            "requested_revision": args.revision,
            "snapshot_path": snapshot,
            "max_workers": max_workers,
            "complete": True,
        },
    )


if __name__ == "__main__":
    main()
