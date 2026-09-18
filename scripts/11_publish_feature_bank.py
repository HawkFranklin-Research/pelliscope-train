#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _common import load_context

from hawk_derm.config import path_from
from hawk_derm.features.publish import publish_feature_bank
from hawk_derm.io import append_csv_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a feature bank and verify the remote checksum.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()
    config = load_context(args.config)
    local = path_from(config, "artifacts_dir") / "features" / args.encoder / "feature_bank.npz"
    remote_path = f"features/{args.encoder}/feature_bank.npz"
    receipt = publish_feature_bank(
        local,
        repo_id=config["huggingface"]["feature_dataset_repo"],
        path_in_repo=remote_path,
        repo_type=config["huggingface"].get("repo_type", "dataset"),
        private=bool(config["huggingface"].get("private", True)),
        token=args.token,
    )
    columns = [
        "artifact_id",
        "kind",
        "encoder",
        "local_path",
        "remote_repo",
        "remote_path",
        "remote_revision",
        "size_bytes",
        "sha256",
        "row_count",
        "dimension",
        "verified",
        "source_job_id",
        "notes",
    ]
    append_csv_row(
        Path(config["repository_root"]) / "coordination" / "ARTIFACT_INDEX.csv",
        columns,
        {
            "artifact_id": f"feature-bank-{args.encoder}",
            "kind": "feature_bank",
            "encoder": args.encoder,
            "local_path": receipt["local_path"],
            "remote_repo": receipt["repo_id"],
            "remote_path": receipt["path_in_repo"],
            "remote_revision": receipt["commit"],
            "size_bytes": receipt["local_size"],
            "sha256": receipt["local_sha256"],
            "row_count": receipt.get("row_count", ""),
            "dimension": receipt.get("dimension", ""),
            "verified": receipt["verified"],
            "source_job_id": "",
            "notes": "",
        },
    )


if __name__ == "__main__":
    main()
