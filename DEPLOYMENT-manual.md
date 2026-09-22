# Hawk Derm production deployment

This guide runs one top-level experiment at a time and gives that experiment the configured CPU budget. It is suitable for a persistent-disk Google Cloud Spot VM and for resuming after preemption.

## 1. Required data

The repository includes the small September 2026 Parquet, case manifest, and image manifest in `data/canonical/`.

The canonical full experiment additionally requires **10,379 unique PNG files covering 5,033 cases**. The currently inspected `SCIN-Dermatology-Raw-Images` snapshot contains only 6,505 unique files and 3,061 cases. That smaller snapshot can support bounded smoke testing but cannot pass the strict full-run audit.

Before starting paid feature extraction, place the complete canonical image collection in one directory and verify:

```bash
find "$HAWK_DERM_RAW_DATA_ROOT/images" -type f -iname '*.png' | wc -l
```

The result must be at least `10379`. `01_build_manifest.py --run-mode full` now fails immediately with a clear error when this prerequisite is missing.

For faster future transfers, publish the complete image directory as a small number of compressed archive shards instead of thousands of separate Hugging Face files. The standard downloader now supports up to 32 concurrent file downloads, but an archive avoids per-file HTTP overhead.

## 2. VM and operating-system setup

Recommended baseline: 32 vCPUs, 64 GB RAM, 200 GB persistent disk. A 64-vCPU VM is useful only after profiling shows that the active stage scales beyond 32 workers.

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs tmux build-essential python3 python3-venv python3-dev libgl1
git lfs install
git clone <repository-url> hawk-derm
cd hawk-derm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements-lock.txt
pip install -e '.[dev,derm-foundation]'
```

Derm Foundation uses its official CPU-targeted TensorFlow SavedModel. The other PyTorch encoders may use a configured GPU when available.

## 3. Authentication

For private Hugging Face downloads and uploads, authenticate the HTTP client:

```bash
hf auth login
hf auth whoami
```

`ssh -T git@hf.co` verifies Hugging Face Git-over-SSH access. It does not by itself provide the HTTP token used by `snapshot_download` or dataset uploads.

GitHub deployment may use an SSH deploy key:

```bash
ssh -T git@github.com
```

## 4. Portable paths

The following environment variable moves the raw-data location without editing YAML:

```bash
export HAWK_DERM_RAW_DATA_ROOT="$PWD/data/raw/SCIN-Dermatology-Raw-Images"
```

Any configured path can be overridden with `HAWK_DERM_PATH_<CONFIG_KEY>`, for example:

```bash
export HAWK_DERM_PATH_ARTIFACTS_DIR=/mnt/disks/hawk-derm/artifacts
export HAWK_DERM_PATH_REPORTS_DIR=/mnt/disks/hawk-derm/reports
```

## 5. Fast, resumable production execution

Start inside `tmux` and use all CPUs available to the VM:

```bash
tmux new -s hawk-derm
source .venv/bin/activate
export HAWK_DERM_AGENT=gcp-production-agent
export HAWK_DERM_RAW_DATA_ROOT="$PWD/data/raw/SCIN-Dermatology-Raw-Images"
CPU_WORKERS="$(nproc)"

python scripts/run_pipeline.py \
  --config configs/study_25class.yaml \
  --run-mode full \
  --download-data \
  --cpu-workers "$CPU_WORKERS" \
  --download-workers 32 \
  --resume \
  2>&1 | tee reports/production_pipeline.log
```

The pipeline executes top-level jobs sequentially. Within a job:

- image decode, preprocessing, shard writing, and shard loading use worker pools;
- random forests use internal tree parallelism;
- logistic regression, calibrated linear SVM, and gradient boosting distribute the 25 one-vs-rest label fits across the worker pool;
- k-NN uses parallel prediction;
- PyTorch, TensorFlow, BLAS, OpenMP, NumExpr, and scikit-learn receive the same explicit CPU budget;
- bootstrap and permutation analyses use deterministic parallel work while preserving their case-level statistical definitions.

After a Spot interruption, run the same command with `--resume`. Expensive stages are skipped only when their outputs are non-empty and their run manifest is complete for the same `smoke` or `full` mode.

## 6. Stage ranges and reused artifacts

If data is already present, omit `--download-data` and start at manifest construction:

```bash
python scripts/run_pipeline.py --run-mode full --cpu-workers "$(nproc)" \
  --from-stage manifest --resume
```

To continue after feature extraction:

```bash
python scripts/run_pipeline.py --run-mode full --cpu-workers "$(nproc)" \
  --from-stage classical --resume
```

To regenerate only analysis outputs:

```bash
python scripts/run_pipeline.py --run-mode full --cpu-workers "$(nproc)" \
  --from-stage statistics --to-stage figures
```

To omit selected encoders:

```bash
python scripts/run_pipeline.py --run-mode full --cpu-workers "$(nproc)" --resume \
  --skip-encoders siglip2_so400m,derm_foundation \
  --primary-encoder vit_base
```

Use a fresh artifact directory when a reduced encoder grid must remain separate from a complete run.

## 7. Reusing historical feature banks

Provide a bank through the encoder-specific environment variable and include its metadata JSON beside the bank:

```bash
export HAWK_DERM_IMPORT_BANK_SIGLIP2_SO400M=/mnt/disks/features/siglip2/feature_bank.npz
export HAWK_DERM_IMPORT_BANK_DERM_FOUNDATION=/mnt/disks/features/derm_foundation/feature_bank.npz
```

The importer validates model ID, immutable revision, dimension, checksum, case IDs, and image ordering before use.

## 8. Monitoring

```bash
tmux attach -t hawk-derm
tail -f reports/production_pipeline.log
htop
watch -n 2 'df -h .; du -sh artifacts data/raw 2>/dev/null'
```

Perfect 100% utilization is not expected during network download, small MIL batches, synchronization, or file writes. The important condition is that compute-heavy feature, baseline, and statistical stages use the configured worker budget without running unrelated top-level models concurrently.

Run the randomized CPU scaling check after the smoke feature banks exist:

```bash
python scripts/90_test_cpu_scaling.py
```

It selects one encoder and one model family at random, trains the same selection with `requested_workers=-1` and `requested_workers=32`, monitors the entire subprocess tree through Linux `/proc`, and writes a JSON report under `reports/test_logs/`. Training outputs use a temporary directory and do not overwrite experiment artifacts. Use `--seed 42` when the random selection must be reproducible.

### 8.1 Live multi-core smoke and scaling verification results

Both the smoke pipeline and the multi-core process scaling test were executed and verified across all cores:

1. **Integrated Smoke Pipeline**:
   - Command: `python scripts/run_pipeline.py --config configs/study_25class.yaml --run-mode smoke --cpu-workers "$(nproc)" --resume`
   - Successfully completed all stages: manifest construction, data auditing, split freezing, 7 encoder feature extractions, 5 classical classifier families, 10-seed gated-attention MIL, threshold optimization, statistical bootstrap evaluations, tables, and all 31 figure panels (including 3-page paginated 25-class panels).
   - Release gate verification passed:
     - `complete: true`
     - `floating_revisions: {}`
     - `revisions_are_immutable: true`
     - Exit code: `0`
     - Saved log: `reports/test_logs/32_run_pipeline_smoke_pinned.log`.

2. **Cloud 32-vCPU Process Scaling Verification**:
   - Command: `python scripts/90_test_cpu_scaling.py`
   - Executed live on Google Cloud `c2d-highcpu-32` (32 vCPUs, 64 GB RAM, Spot in `us-central1-a`):
     - Dynamically monitored parent and child process trees via `/proc/<pid>/stat` and per-thread ticks via `/proc/<pid>/task/<tid>/stat`.
     - Observed all 32 logical cores actively engaged: `observed_logical_cpus: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]`.
     - Peak equivalent busy cores: **27.94 cores** (mean 8.16 busy cores).
     - Execution time: **6.46 seconds**.
     - Overall status: `complete: true`, exit code `0`.
     - Saved report: `reports/test_logs/cpu_scaling_20260919T172700Z.json`.

## 9. Distribution phase

Do not add `--publish-features` during the expensive compute run unless background network traffic is desired. After verification, resize to a cheap persistent-disk VM and publish each bank:

```bash
for encoder in resnet50 inception_v3 bit50 vit_base clip_vitb32 derm_foundation siglip2_so400m; do
  python scripts/11_publish_feature_bank.py --config configs/study_25class.yaml --encoder "$encoder"
done
```

Delete a local feature artifact only through `12_cleanup_verified_artifact.py` after remote checksum verification. Keep raw predictions, metrics, threshold tables, plot data, run manifests, and the run ledger.

## 10. SigLIP2 MIL-only rerun

Use the persistent production disk when the canonical manifests, locked split, SigLIP2 feature bank, and classical results already exist. This workflow does not download images, extract features, or retrain classical models.

```bash
python scripts/run_mil_pipeline.py \
  --config configs/study_25class.yaml \
  --encoder siglip2_so400m \
  --run-mode full \
  --cpu-workers 32 \
  --device cpu \
  --run-tag canonical \
  --resume \
  2>&1 | tee reports/siglip2_mil_canonical.log
```

The stages run in this order: preflight, September-behavior check, tuning, development-only cross-validation, ten fixed-split seeds, probability ensembling, validation-only threshold selection, separately labelled final-model fitting, evaluation, paired statistics, tables, figures, and strict verification.

Canonical artifacts are isolated under:

```text
artifacts/models/mil/siglip2_so400m/canonical/
reports/reanalysis/siglip2_so400m_mil_canonical/
```

The preflight deliberately fails if the complete 5,033-case manifest, locked split, existing SigLIP2 feature bank, or matched random-forest predictions are unavailable. The full run also fails if tuning has not produced `best_mil_config.yaml`; downstream stages cannot silently fall back to `configs/mil.yaml`.
