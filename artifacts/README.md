# Artifact directory

Feature banks and model checkpoints are generated here but ignored by Git. Raw prediction CSVs, metrics, run manifests, checksums, and upload receipts remain versioned where practical.

An artifact may be removed locally only after `scripts/11_publish_feature_bank.py` has written a verified upload receipt and `coordination/ARTIFACT_INDEX.csv` records the immutable remote revision and checksum.

