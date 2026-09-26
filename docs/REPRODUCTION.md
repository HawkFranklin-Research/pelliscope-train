# Reproduction guide

## Environment

Create a Python 3.11 environment, install `requirements-lock.txt`, and install the package in editable mode. Install the `derm-foundation` optional dependency only on the machine assigned to that encoder.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e .
```

Model revisions are pinned in `configs/encoders.yaml` to the release snapshot recorded on 2026-09-18. Make the Derm Foundation model path available before running.

The included CPU container packages the public workflow. GPU and Apple MPS execution should use native environments with the corresponding PyTorch build.

```bash
docker build -t hawk-derm .
docker run --rm -v /absolute/data:/data -v "$PWD/artifacts:/workspace/hawk-derm/artifacts" hawk-derm --config configs/study_25class.yaml --run-mode smoke
```

## Staged workflow

The full pipeline order is data manifest, complete image audit, locked split, seven feature banks, classical baselines, then MIL tuning, cross-validation, repeated seeds, ensemble, validation thresholds, final model, evaluation, paired statistics, tables, figures, and release verification. The full command delegates MIL stages to `run_mil_pipeline.py`, which remains available for MIL-only runs.

The bounded run writes to `smoke_runs/` and exercises model stages without touching production outputs:

```bash
python scripts/run_pipeline.py --config configs/study_25class.yaml --run-mode smoke
```

After checking the isolated smoke outputs, run the full workflow. It requires a complete 10,407-row image audit and exactly the locked 3,529 train, 754 validation, and 750 test case IDs before model training:

```bash
python scripts/run_pipeline.py --config configs/study_25class.yaml --run-mode full
```

## Two-machine execution

1. Both agents check out the same commit and compare configuration, case-manifest, and split-manifest SHA-256 hashes.
2. Assign whole jobs in `coordination/jobs.yaml`.
3. Hawk Prime runs data preparation, five conventional feature extractors, classical models, statistics, and figures.
4. The MacBook verifies or produces the SigLIP2 feature bank, assigned MIL seeds, and Apple M4 latency measurements.
5. Each producing machine uploads large artifacts with `scripts/11_publish_feature_bank.py`.
6. The receiving machine downloads the immutable revision and verifies the receipt before use.
7. One machine consolidates metrics only after every assigned job is marked complete.

## Existing SigLIP2 bank

The historical Mac bank may be imported with `scripts/10_extract_features.py --encoder siglip2_so400m --import-bank <path> --import-metadata <metadata.json>`. The metadata JSON must record `model_id`, immutable `revision`, `dimension`, preprocessing, and preferably the bank SHA-256. Import is accepted only when case IDs, image order/path basenames, dimension 1,152, source model, revision, and checksum agree with the canonical manifests.

For orchestration, set `HAWK_DERM_IMPORT_BANK_SIGLIP2_SO400M` to the absolute bank path. Equivalent variables can be set for other encoder keys. The run manifest records the actual import argument.

## Space-safe publication

Upload feature banks while downstream jobs run. `scripts/12_cleanup_verified_artifact.py` deletes one explicit local feature bank only after its upload receipt passes remote size and SHA-256 verification. It does not delete raw images, predictions, metrics, manifests, plot data, or ledgers.

## Raw model comparison

Use `scripts/51_compare_models.py` with two case-level test prediction CSVs. The script aligns by case ID, verifies identical targets, performs paired case bootstrap comparisons for macro ROC-AUC and macro PR-AUC, applies Holm adjustment to per-condition AUC tests, and stores the full distributions.

## Final gate

`scripts/70_verify_release.py --mil-run-tag canonical --encoders <encoder-list>` checks tagged MIL seeds and ensembles, classical test membership and split provenance, the primary final model, and the paired statistics. Visual inspection remains necessary because file existence cannot detect blank or malformed plots.
