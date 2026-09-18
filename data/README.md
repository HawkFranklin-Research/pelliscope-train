# Data directory

This repository does not commit raw clinical images. `data/manifests/`, `data/splits/`, and `data/audits/` contain reproducibility metadata derived from the configured raw-data source.

The canonical local Hawk Prime source is configured in `configs/study_25class.yaml`. External users can obtain the public data through `scripts/00_download_data.py` and must run the reconciliation audit before training.

