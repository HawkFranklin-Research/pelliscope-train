# Artifact schema

## Case manifest

`data/manifests/case_manifest.csv` contains one row per case. Required columns are `case_id`, the ordered `is_<label_slug>` target columns, `target_label_count`, `target_positive`, image counts, and `run_mode`. Additional source metadata may be retained.

## Image manifest and audit

`data/manifests/image_manifest.csv` contains one row per referenced image with `case_id`, `image_id`, source and resolved paths, image order, bag slot, file existence, and the bag-retention flag. The audit adds SHA-256, size, dimensions, decode status, and errors.

## Split manifest

`data/splits/split_manifest_v1.csv` contains `case_id`, `split`, `split_seed`, and `split_version`. It is the only valid source of split membership.

## Feature bank

Every `artifacts/features/<encoder>/feature_bank.npz` contains:

- `embeddings`: float32 `[images, feature_dimension]`;
- `case_ids`, `image_ids`, `image_paths`, and `image_sha256` in matching row order;
- scalar `encoder`, `model_id`, `revision`, and `dimension` fields.

The adjacent metadata JSON records model/preprocessor revision, extraction device, row count, dtype, failure count, run mode, and completion state. Per-image shards make extraction resumable.

## Raw predictions

Prediction CSVs contain one row per case for case-level outputs. The mandatory key is `case_id`. Each label has:

- `true__<label_slug>`
- `prob__<label_slug>`
- `pred__<label_slug>`

Optional columns include split, seed, model, image IDs, and MIL `attention__slot_N`. Attention values are model-internal contribution weights.

## Metrics and thresholds

Overall metrics are one row per model, seed, and split. Per-class tables include support, threshold, TP, FP, TN, FN, sensitivity, specificity, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, and calibration error.

Threshold outputs retain the complete validation sweep, the selected validation rows, and the locked-test operating points with Wilson confidence intervals.

## Run and upload records

Every major output directory contains `run_manifest.json` with the Git commit, configuration payload and hash, environment, inputs, outputs, timestamps, and status. Feature upload receipts contain the local and remote size/hash, Hugging Face commit, verification time, and `verified` flag. Cleanup accepts only a verified receipt beneath the configured feature root.
