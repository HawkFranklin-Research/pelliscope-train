#!/usr/bin/env python3
"""Download only the pinned production prediction CSVs used by this analysis."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download, list_repo_files


HERE = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((HERE / "analysis_config.yaml").read_text())
OUTPUT = HERE / "inputs" / "hf_production"
PATTERNS = (
    "classical/*/*/test_case_predictions.csv",
    "classical/*/*/test_image_predictions.csv",
    "mil/*/seed_*/test_predictions.csv",
)


def main() -> None:
    repo_id = CONFIG["hub"]["repo_id"]
    revision = CONFIG["hub"]["revision"]
    files = [
        name
        for name in list_repo_files(repo_id, revision=revision)
        if any(fnmatch(name, pattern) for pattern in PATTERNS)
    ]
    if not files:
        raise RuntimeError("No prediction files matched the pinned Hugging Face revision")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    def fetch(name: str) -> str:
        return hf_hub_download(
            repo_id=repo_id,
            filename=name,
            revision=revision,
            local_dir=OUTPUT,
        )

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(fetch, files))
    print(f"Downloaded or verified {len(files)} files from {repo_id}@{revision}")


if __name__ == "__main__":
    main()

