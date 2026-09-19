#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from _common import load_context

from hawk_derm.config import path_from


def create_public_model_card(output_dir: Path) -> None:
    readme_content = """---
license: apache-2.0
library_name: pytorch
tags:
- dermatology
- teledermatology
- multiple-instance-learning
- siglip2
- scin
datasets:
- HawkFranklin-Research/SCIN-Dermatology-25Class-Features
metrics:
- roc_auc
- pr_auc
---

# Pelliscope SigLIP-2 Gated-Attention MIL (25-Class Teledermatology)

Official model checkpoint for the 25-condition case-level teledermatology Multiple Instance Learning (MIL) model from HawkFranklin Research.

## Architecture
- **Backbone**: SigLIP-2 SO400M (image-level 1152-d frozen embeddings)
- **Aggregation**: Case-level Gated Attention Multiple Instance Learning (Ilse et al.)
- **Conditions**: 25 top dermatological conditions from the SCIN dataset
- **Input**: Bag of 1 to 3 clinic/self-captured skin images per encounter

## Included Files
- `model.pt`: Trained PyTorch Gated Attention MIL weights
- `validation_selected_thresholds.csv`: Validation-tuned operating thresholds per condition
- `test_operating_points.csv`: Out-of-sample test sensitivity, specificity, and F1 per condition
- `mil_config.yaml`: Architecture hyperparameters and training configuration
"""
    (output_dir / "README.md").write_text(readme_content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish internal model checkpoints and public release model.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--internal-repo", default="HawkFranklin-Research/pelliscope-25class-models-internal")
    parser.add_argument("--public-repo", default="HawkFranklin-Research/pelliscope-siglip2-mil")
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    from huggingface_hub import HfApi

    config = load_context(args.config)
    api = HfApi(token=args.token)

    models_dir = path_from(config, "artifacts_dir") / "models"
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    # 1. Publish all models to internal private repository
    print(f"Creating private repository: {args.internal_repo}")
    api.create_repo(repo_id=args.internal_repo, repo_type="model", private=True, exist_ok=True)
    print(f"Uploading full models directory to {args.internal_repo}...")
    api.upload_folder(
        folder_path=str(models_dir),
        repo_id=args.internal_repo,
        repo_type="model",
        commit_message="Add 25-class all-encoder trained models, seeds, and checkpoints",
    )
    print(f"Successfully uploaded internal models to {args.internal_repo}")

    # 2. Publish winning SigLIP2-MIL model to public repository
    siglip2_final_dir = models_dir / "mil" / "siglip2_so400m" / "final_model"
    siglip2_thresholds_dir = models_dir / "mil" / "siglip2_so400m" / "thresholds"

    if not (siglip2_final_dir / "model.pt").is_file():
        raise FileNotFoundError(f"SigLIP2 final model not found: {siglip2_final_dir / 'model.pt'}")

    with tempfile.TemporaryDirectory() as temp_dir:
        staging = Path(temp_dir)
        import shutil

        shutil.copy2(siglip2_final_dir / "model.pt", staging / "model.pt")
        if (siglip2_thresholds_dir / "validation_selected_thresholds.csv").is_file():
            shutil.copy2(
                siglip2_thresholds_dir / "validation_selected_thresholds.csv",
                staging / "validation_selected_thresholds.csv",
            )
        if (siglip2_thresholds_dir / "test_operating_points.csv").is_file():
            shutil.copy2(
                siglip2_thresholds_dir / "test_operating_points.csv",
                staging / "test_operating_points.csv",
            )
        if (siglip2_final_dir / "history.json").is_file():
            shutil.copy2(siglip2_final_dir / "history.json", staging / "history.json")
        if (siglip2_final_dir / "test_predictions.csv").is_file():
            shutil.copy2(siglip2_final_dir / "test_predictions.csv", staging / "test_predictions.csv")

        mil_yaml = Path(config["repository_root"]) / "configs" / "mil.yaml"
        if mil_yaml.is_file():
            shutil.copy2(mil_yaml, staging / "mil_config.yaml")

        create_public_model_card(staging)

        print(f"Creating public repository: {args.public_repo}")
        api.create_repo(repo_id=args.public_repo, repo_type="model", private=False, exist_ok=True)
        print(f"Uploading release model package to {args.public_repo}...")
        api.upload_folder(
            folder_path=str(staging),
            repo_id=args.public_repo,
            repo_type="model",
            commit_message="Release SigLIP-2 Gated Attention MIL 25-class teledermatology model",
        )
        print(f"Successfully published public release model to {args.public_repo}")


if __name__ == "__main__":
    main()
