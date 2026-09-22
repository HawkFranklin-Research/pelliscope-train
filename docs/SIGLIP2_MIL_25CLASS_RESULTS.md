# SigLIP2 MIL 25-class results and release handoff

## Canonical analysis cohort

The canonical held-out analysis contains 750 test cases after duplicate-image conflict resolution. Every reported model comparison, confidence interval, table, and manuscript figure uses these same case IDs and target labels.

## Authoritative sources

- SigLIP2 MIL training artifacts: `artifacts/models/mil/siglip2_so400m/canonical/`
- Canonical 750-case SigLIP2 MIL predictions: `reports/reanalysis/siglip2_so400m_mil_canonical/statistics/paired_predictions/siglip2_mil.csv`
- Comparator MIL ensembles: `reports/statistics/*_mil_test_ensemble_predictions.csv`
- Full classical artifacts: private Hugging Face model repository `HawkFranklin-Research/pelliscope-25class-models-internal`
- Paired 750-case derivatives and tests: `reports/reanalysis/siglip2_so400m_mil_canonical/statistics/`

The production Random Forest artifacts in Hugging Face are authoritative. The current Hugging Face MIL directories predate the canonical revised MIL run and must be replaced during the next release upload. Some Git revisions contain six-row smoke copies at classical-result paths; do not upload or publish those copies.

## Canonical Hugging Face release

Build one merged release from the existing full classical/non-MIL artifacts and the canonical revised SigLIP2 MIL artifacts below:

- `tuning/`: trials, selected YAML/JSON configuration, and selection manifest;
- `cross_validation/`: five-fold summaries and predictions;
- `seed_42/` through `seed_51/`: train, validation, and test prediction CSVs plus metadata, histories, summaries, and model checkpoints when present;
- `ensemble/`: validation/test predictions and metrics;
- `thresholds/`: validation-selected thresholds and test operating points;
- `final_model/`: checkpoint, histories, predictions, and metrics;
- canonical 750-case predictions and paired statistical outputs under `reports/reanalysis/siglip2_so400m_mil_canonical/statistics/`.

Raw model predictions are CSV files. NPZ files in the statistical directories contain bootstrap/permutation distributions; feature-bank NPZ files are separate large inputs.

## Mandatory upload gates

Before uploading, reject an artifact set if any of these checks fails:

- any manifest says `run_mode: smoke`;
- fewer than ten seed directories exist;
- the canonical prediction under `statistics/paired_predictions/siglip2_mil.csv` does not contain exactly 750 unique case IDs;
- a comparator prediction under `statistics/paired_predictions/` does not contain those same 750 case IDs;
- case, split, feature-bank, or selected-configuration hashes disagree across seeds;
- prediction metadata is absent;
- a statistical NPZ is presented as a raw prediction file.

## Reporting rule

Use the canonical 750-case prediction files for all reported performance estimates and statistical comparisons. Training-stage files are retained as computational provenance and are not independent manuscript results.
