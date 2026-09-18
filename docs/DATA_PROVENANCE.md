# Data provenance

## Public source

The raw SCIN mirror is hosted at `HawkFranklin-Research/SCIN-Dermatology-Raw-Images`. The local Hawk Prime copy is configured in `configs/study_25class.yaml`. The source contains raw images and `metadata.csv`; the download receipt must record the requested dataset revision.

The public SCIN dataset and its license remain authoritative. This repository does not redistribute raw clinical photographs.

The downloaded mirror's compact `metadata.csv` contains 6,517 image rows and 3,061 case IDs. It does not by itself reproduce the September 5,033-case multilabel universe because it exposes a compact primary-diagnosis view rather than the complete dermatologist differential-label ledger used in the September work. The canonical reconstruction therefore joins the mirror's image files and compact metadata to the versioned September case-label and image-reference manifests. The reconciliation audit reports matched, missing, and extra cases and images. A public release must include or immutably reference the derived 5,033-case label manifest if its source license permits redistribution.

## September 2026 derivation sources

The initial 25-class reconstruction uses these local legacy inputs:

- `derm-train-sep26/jul26/outputs/diff_25class_dataset.parquet`
- `derm-train-sep26/jul26/section-b/data_extraction/outputs/section_b_case_manifest.csv`
- `derm-train-sep26/jul26/section-b/data_extraction/outputs/section_b_image_manifest.csv`

`scripts/01_build_manifest.py` preserves the September case universe and target vectors while resolving each referenced image against the canonical raw-data directory. It writes portable paths and never assumes a `/Users/...` location.

## Required audit outputs

`scripts/02_audit_data.py` produces:

- `data/audits/image_audit.csv`
- `data/audits/duplicate_paths.csv`
- `data/audits/duplicate_hashes.csv`
- `data/audits/cross_case_duplicate_hashes.csv`
- `data/audits/missing_or_unreadable_images.csv`
- `data/audits/data_audit_summary.json`

The audit records source rows, resolved paths, byte sizes, SHA-256 hashes, decode status, dimensions, missing files, duplicate paths, duplicate hashes, cross-case hash reuse, bag-limit exclusions, and final target-label counts. Manuscript counts must be read from these outputs.

## Sensitive and redistribution-controlled artifacts

Raw images, bulk embeddings, checkpoints, and any restricted metadata remain outside Git. Their checksums and immutable remote locations are recorded in `coordination/ARTIFACT_INDEX.csv`. Case-level prediction release must follow the source dataset's redistribution terms and must not introduce direct identifiers.
