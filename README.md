# Hawk Derm

Reproducible code for the 25-condition case-level teledermatology experiments prepared for the *Artificial Intelligence in Medicine* revision.

The pipeline starts from SCIN case metadata and raw images, freezes one shared case-level split, extracts seven frozen feature banks, runs five classical classifier families and gated-attention MIL, exports raw case-level probabilities, computes paired uncertainty analyses, and regenerates manuscript tables and figures.

## Study contract

- Computational design: September 2026 25-condition workflow.
- Figure layouts and colours: original AIIM workflow.
- Unit of primary evaluation: patient-submitted case.
- Canonical split: one immutable 70/15/15 case split shared by every model.
- Thresholds: selected on validation predictions and applied unchanged to test predictions.
- Repeated runs: model and bag seeds vary; test membership does not.
- Raw predictions: retained for every reported model and seed.

Read [STUDY_LOCK.md](docs/STUDY_LOCK.md) before running anything.

## Repository workflow

```bash
python scripts/01_build_manifest.py --config configs/study_25class.yaml
python scripts/02_audit_data.py --config configs/study_25class.yaml
python scripts/03_freeze_splits.py --config configs/study_25class.yaml
python scripts/10_extract_features.py --config configs/study_25class.yaml --encoder resnet50
python scripts/20_train_classical.py --config configs/study_25class.yaml --encoder resnet50 --classifier random_forest
python scripts/30_train_mil.py --config configs/study_25class.yaml --encoder resnet50 --seed 42
python scripts/40_select_thresholds.py --config configs/study_25class.yaml --encoder resnet50
python scripts/50_evaluate.py --config configs/study_25class.yaml --predictions artifacts/models/mil/resnet50/seed_42/test_predictions.csv --output-dir reports/evaluations/resnet50_seed42
python scripts/60_generate_tables.py --config configs/study_25class.yaml
python scripts/61_generate_figures.py --config configs/study_25class.yaml
```

The bounded smoke workflow uses the same paths and filenames as the full workflow:

```bash
python scripts/run_pipeline.py --config configs/study_25class.yaml --run-mode smoke
```

### Smoke verification status

The complete smoke run_pipeline.py rerun passed successfully.

- All seven encoders completed.
- All classical models and MIL stages completed.
- Statistics, tables, and figures completed.
- Final release verification passed.
- No data download occurred.
- No Hugging Face upload occurred.
- Exit code: 0.

Release verification confirms:

complete: true
floating_revisions: {}
revisions_are_immutable: true

The full log is saved at /home/prime/Documents/github/derm-paper-codebases/hawk-derm/reports/test_logs/32_run_pipeline_smoke_pinned.log.

The Bit-50 and CLIP “unexpected weights” messages are expected because only their vision backbones are used. Google Derm again ran successfully through its CPU TensorFlow path.

The full workflow intentionally overwrites validated smoke artifacts:

```bash
python scripts/run_pipeline.py --config configs/study_25class.yaml --run-mode full
```

To download the raw Hugging Face dataset first, add `--download-data`. To omit an encoder, add a comma-separated skip list. For example, this runs the remaining encoders and uses Derm Foundation as the primary model:

```bash
python scripts/run_pipeline.py --config configs/study_25class.yaml --run-mode full \
  --download-data --skip-encoders siglip2_so400m --primary-encoder derm_foundation
```

Run a skipped-encoder experiment in a clean artifact directory or fresh VM so outputs from an earlier run are not mixed into its tables.

The smoke and full commands are orchestration entry points. Individual jobs can be split between Hawk Prime and the MacBook using [jobs.yaml](coordination/jobs.yaml).

To test CPU allocation with a randomly selected encoder/model pair without modifying experiment artifacts:

```bash
python scripts/90_test_cpu_scaling.py
```

## Data and large artifacts

Raw images, feature banks, checkpoints, and bulk predictions are not committed to Git. Their locations, checksums, and Hugging Face revisions are recorded in [ARTIFACT_INDEX.csv](coordination/ARTIFACT_INDEX.csv). Small manifests, split assignments, metrics, plot data, and run metadata remain versioned.

## Reproduction documentation

- [Data provenance](docs/DATA_PROVENANCE.md)
- [Legacy-code provenance](docs/LEGACY_CODE_PROVENANCE.md)
- [Statistical analysis plan](docs/STATISTICAL_ANALYSIS_PLAN.md)
- [Artifact schema](docs/ARTIFACT_SCHEMA.md)
- [Full reproduction guide](docs/REPRODUCTION.md)
- [Citation ledger](docs/CITATION_LEDGER.md)
- [Manuscript output map](docs/MANUSCRIPT_OUTPUT_MAP.md)
- [Production deployment and CPU execution](DEPLOYMENT-manual.md)

## Script inventory

- `00–03`: data acquisition, canonical manifests, audits, and the shared split.
- `10–12`: feature extraction/import, verified Hugging Face publication, and safe cleanup.
- `20`: five classical classifier families with separate image- and case-level exports.
- `30–34`: single MIL runs, tuning, development cross-validation, repeated fixed-split runs, and final frozen fitting.
- `40`: validation-only threshold selection and locked-test operating points.
- `50–53`: evaluation, manual or grid-wide paired statistics, archived closed-model normalization, and inference latency.
- `60–61`: machine-readable tables and manuscript/appendix figures.
- `70`: release completeness and checksum verification.
