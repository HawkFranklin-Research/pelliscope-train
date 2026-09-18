from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from hawk_derm.io import read_json, sha256_file, write_json


def publish_feature_bank(
    local_path: str | Path,
    *,
    repo_id: str,
    path_in_repo: str,
    repo_type: str = "dataset",
    private: bool = True,
    revision: str = "main",
    token: str | None = None,
) -> dict[str, Any]:
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    local_hash = sha256_file(local_path)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    commit = api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        commit_message=f"Upload {path_in_repo}",
    )
    downloaded = Path(
        hf_hub_download(repo_id=repo_id, filename=path_in_repo, repo_type=repo_type, revision=commit.oid, token=token)
    )
    remote_hash = sha256_file(downloaded)
    verified = local_hash == remote_hash and local_path.stat().st_size == downloaded.stat().st_size
    local_metadata_path = local_path.with_suffix(".metadata.json")
    local_metadata = read_json(local_metadata_path) if local_metadata_path.is_file() else {}
    if local_metadata_path.is_file():
        api.upload_file(
            path_or_fileobj=str(local_metadata_path),
            path_in_repo=str(Path(path_in_repo).with_suffix(".metadata.json")),
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            commit_message=f"Upload metadata for {path_in_repo}",
        )
    receipt = {
        "local_path": str(local_path),
        "local_size": local_path.stat().st_size,
        "local_sha256": local_hash,
        "repo_id": repo_id,
        "repo_type": repo_type,
        "path_in_repo": path_in_repo,
        "commit": commit.oid,
        "remote_size": downloaded.stat().st_size,
        "remote_sha256": remote_hash,
        "row_count": local_metadata.get("row_count"),
        "dimension": local_metadata.get("dimension"),
        "verified": verified,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(local_path.with_suffix(".upload-receipt.json"), receipt)
    if not verified:
        raise RuntimeError("Uploaded feature bank did not pass size and checksum verification")
    return receipt


def cleanup_verified_artifact(
    receipt_path: str | Path,
    *,
    expected_root: str | Path,
    remove_encoder_directory: bool = False,
) -> None:
    receipt_path = Path(receipt_path)
    receipt = read_json(receipt_path)
    if receipt.get("verified") is not True:
        raise ValueError("Artifact receipt is not verified")
    target = Path(receipt["local_path"]).resolve()
    root = Path(expected_root).resolve()
    if root not in target.parents or target == root:
        raise ValueError(f"Refusing to delete path outside expected artifact root: {target}")
    if not target.is_file():
        raise FileNotFoundError(target)
    if sha256_file(target) != receipt["local_sha256"]:
        raise ValueError("Local artifact changed after upload verification")
    if remove_encoder_directory:
        import shutil

        shutil.rmtree(target.parent)
    else:
        target.unlink()
