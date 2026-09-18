from __future__ import annotations

import hashlib
import os
import platform
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hawk_derm.config import REPOSITORY_ROOT
from hawk_derm.io import append_csv_row, canonical_json, sha256_file, write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(root: Path = REPOSITORY_ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unversioned"


def environment_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "pid": os.getpid(),
    }
    try:
        import torch

        summary["torch"] = torch.__version__
        summary["cuda_available"] = torch.cuda.is_available()
        summary["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if torch.cuda.is_available():
            summary["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        summary["torch"] = None
    return summary


class RunRecorder:
    def __init__(
        self,
        stage: str,
        config: dict[str, Any],
        output_dir: str | Path,
        *,
        run_id: str | None = None,
        inputs: list[str | Path] | None = None,
    ) -> None:
        self.stage = stage
        self.config = config
        self.output_dir = Path(output_dir)
        self.run_id = run_id or f"{stage}-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.inputs = [Path(path) for path in (inputs or [])]
        self.started_at = ""
        self.manifest_path = self.output_dir / "run_manifest.json"
        self.payload: dict[str, Any] = {}

    def __enter__(self) -> "RunRecorder":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = utc_now()
        self.payload = {
            "run_id": self.run_id,
            "stage": self.stage,
            "status": "running",
            "started_at": self.started_at,
            "git_commit": git_commit(),
            "config": self.config,
            "config_sha256": hashlib.sha256(canonical_json(self.config).encode()).hexdigest(),
            "environment": environment_summary(),
            "inputs": self._hash_paths(self.inputs),
            "outputs": [],
        }
        write_json(self.manifest_path, self.payload)
        return self

    def complete(self, outputs: list[str | Path], **metadata: Any) -> None:
        self.payload.update(metadata)
        self.payload["outputs"] = self._hash_paths([Path(path) for path in outputs])
        self.payload["status"] = "complete"
        self.payload["ended_at"] = utc_now()
        self.payload["complete"] = True
        write_json(self.manifest_path, self.payload)
        self._append_ledger("complete")

    def __exit__(self, error_type: Any, error: Any, traceback: Any) -> bool:
        if error is not None:
            self.payload["status"] = "failed"
            self.payload["ended_at"] = utc_now()
            self.payload["complete"] = False
            self.payload["error"] = f"{type(error).__name__}: {error}"
            write_json(self.manifest_path, self.payload)
            self._append_ledger("failed")
        elif self.payload.get("status") == "running":
            self.complete([])
        return False

    @staticmethod
    def _hash_paths(paths: list[Path]) -> list[dict[str, Any]]:
        records = []
        for path in paths:
            records.append(
                {
                    "path": str(path),
                    "exists": path.exists(),
                    "size": path.stat().st_size if path.is_file() else None,
                    "sha256": sha256_file(path) if path.is_file() else None,
                }
            )
        return records

    def _append_ledger(self, status: str) -> None:
        input_hashes = {Path(item["path"]).name: item.get("sha256") for item in self.payload.get("inputs", [])}
        output_hashes = [item.get("sha256") for item in self.payload.get("outputs", []) if item.get("sha256")]
        columns = [
            "job_id",
            "machine",
            "agent",
            "command",
            "git_commit",
            "config_sha256",
            "dataset_sha256",
            "split_sha256",
            "started_at",
            "finished_at",
            "status",
            "artifact_revision",
            "notes",
        ]
        append_csv_row(
            REPOSITORY_ROOT / "coordination" / "RUN_LEDGER.csv",
            columns,
            {
                "job_id": self.run_id,
                "machine": socket.gethostname(),
                "agent": os.getenv("HAWK_DERM_AGENT", "unassigned"),
                "command": " ".join(sys.argv),
                "git_commit": self.payload.get("git_commit"),
                "config_sha256": self.payload.get("config_sha256"),
                "dataset_sha256": input_hashes.get("case_manifest.csv", ""),
                "split_sha256": input_hashes.get("split_manifest_v1.csv", ""),
                "started_at": self.payload.get("started_at"),
                "finished_at": self.payload.get("ended_at"),
                "status": status,
                "artifact_revision": output_hashes[0] if output_hashes else "",
                "notes": self.payload.get("error", ""),
            },
        )
