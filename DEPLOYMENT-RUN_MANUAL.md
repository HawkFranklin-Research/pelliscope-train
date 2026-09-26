# Deployment run manual (for the execution agent)

**Audience:** the AI agent (or person) that runs this pipeline on a VM or workstation.
**Scope:** which command to run for which situation, what each stage reads and writes, what the safety checks do, and what to do when one stops the run.
**Companion docs:** `DEPLOYMENT-manual.md` (VM creation, OS setup, authentication, Hugging Face publishing) and `docs/REPRODUCTION.md` (environment).

This manual reflects branch `claude-run` after the September 2026 revision: locked split, revised MIL loss, AP-based checkpoints, Platt calibration, and the pre-registered rank gate.

---

## 0. Rules that always apply

1. **Never tune on test labels.** Tuning, cross-validation, calibration and thresholds use training/validation data only. The 750 locked test cases are evaluated once per pre-registered configuration.
2. **Do not edit `configs/mil.yaml` casually.** Its contents are part of `revision-experiments/PREREGISTRATION.md`. The only allowed edit is the one forced by the rank gate (section 5.3), and it must be committed before resuming.
3. **Commit before a full run.** The pre-registration and any config change must be committed and pushed before training starts. Record the commit hash in the run log.
4. **Always run long jobs inside `tmux`**, and `tee` output to a log under `reports/`.
5. **Use `--resume` after any interruption.** It never reuses stale results. A stage is skipped only if its recorded input and output hashes still match; otherwise it re-runs.
6. **Do not publish anything unless explicitly instructed.** Releases use `scripts/publish_production_model.py --confirm` (S12). The old `scripts/publish_models.py` is retired.
7. **Smoke runs are isolated.** `--run-mode smoke` writes everything under `smoke_runs/` and cannot overwrite production files.

---

## 1. Mental model

```
raw images ─► manifest ─► audit ─► split (LOCKED 3,529/754/750) ─► features (7 encoders)
          ─► classical (7 encoders × 5 classifiers)
          ─► MIL per encoder: [rank gate] tuning ─► 5-fold CV ─► 10 seeds ─► ensemble (+Platt) ─► thresholds
          ─► final model (SigLIP2) ─► evaluation ─► paired statistics ─► tables ─► figures ─► verification
```

**Two entry points:**

| Script | Use it for |
|---|---|
| `scripts/run_pipeline.py` | The whole study, or any contiguous range of its stages, for **all included encoders** |
| `scripts/run_mil_pipeline.py` | **MIL only, one encoder**, on existing manifests, locked split, feature bank and classical results |

`run_pipeline.py` calls `run_mil_pipeline.py` for every encoder in its MIL stages, then once more for the primary encoder's finishing stages.

### Stage names

| `run_pipeline.py` stage | Script(s) | Main outputs |
|---|---|---|
| `download` (only with `--download-data`) | `00_download_data.py` | raw SCIN images + metadata |
| `manifest` | `01_build_manifest.py` | `data/manifests/case_manifest.csv`, `image_manifest.csv` |
| `audit` | `02_audit_data.py --strict` | `data/audits/image_audit.csv`, `data_audit_summary.json` |
| `split` | `03_freeze_splits.py` | `data/splits/split_manifest_v1.csv` (+ `.metadata.json`) |
| `external-prepare` | `80_prepare_external.py` | `external_runs/<cohort>/manifests/` (pipeline-schema adapter manifests) |
| `external-features` | `81_extract_external_features.py` | `external_runs/<cohort>/features/<encoder>/feature_bank.npz` |
| `features` | `10_extract_features.py` (per encoder) | `artifacts/features/<encoder>/feature_bank.npz` |
| `classical` | `20_train_classical.py` (per encoder × classifier) | `artifacts/models/classical/<encoder>/<classifier>/` |
| `tuning` | `31_tune_mil.py` | `artifacts/models/mil/<encoder>/<tag>/tuning/best_mil_config.yaml` |
| `cross-validation` | `32_run_cross_validation.py` | `…/<tag>/cross_validation/` |
| `mil` | `33_run_repeated_seeds.py` | `…/<tag>/seed_42 … seed_51/` |
| `ensemble` | `35_ensemble_mil.py` | `…/<tag>/ensemble/` (calibrated + `*_uncalibrated.csv`, `calibration_parameters.csv`) |
| `thresholds` | `40_select_thresholds.py` | `…/<tag>/thresholds/` |
| `final` | `34_fit_final_model.py` | `…/<tag>/final_model/` |
| `evaluation` | `50_evaluate.py` | `reports/reanalysis/<encoder>_mil_<tag>/evaluations/` |
| `statistics` | `51_compare_grid.py` | `reports/reanalysis/<encoder>_mil_<tag>/statistics/` |
| `tables` / `figures` | `60_generate_tables.py` / `61_generate_figures.py` | `reports/tables/`, `reports/figures/` |
| `verify` | `70_verify_release.py` | pass/fail; no files changed |
| `external-evaluate` | `82_evaluate_external.py` | `reports/external/<cohort>/` (model grid, per-label, paired comparisons) |
| `external-figures` | `83_external_figures.py` | `reports/external/<cohort>/figures/` |
| `bundle` (full mode) | `84_build_production_bundles.py` | `artifacts/production/<name>/` (self-checked bundles) |

`run_mil_pipeline.py` uses the same names, except `mil` → `repeated-seeds` and `verify` → `verification`. It also has `preflight` and `implementation-check`, which always run.

**Encoder keys:** `resnet50, inception_v3, bit50, vit_base, clip_vitb32, derm_foundation, siglip2_so400m`.
**Classifier keys:** `logistic, svm_linear, random_forest, gradient_boosting, knn`.
**Default MIL run tag:** `canonical`, so outputs go to `artifacts/models/mil/<encoder>/canonical/`.

---

## 2. Environment setup (every session)

```bash
cd ~/pelliscope-train            # repository root
source .venv/bin/activate
git status && git log -1 --oneline     # must be clean and on the intended commit
export HAWK_DERM_RAW_DATA_ROOT=/path/to/SCIN-Dermatology-Raw-Images   # if not the YAML default
CPU_WORKERS="$(nproc)"
python -m pytest -q tests              # ~5 s; must pass before any long run
```

**Path overrides** work without editing YAML: `HAWK_DERM_PATH_<CONFIG_KEY>`, e.g. `HAWK_DERM_PATH_ARTIFACTS_DIR=/mnt/disk/artifacts`.

**Reusing a feature bank from elsewhere:** set `HAWK_DERM_IMPORT_BANK_<ENCODER_UPPERCASE>=/path/feature_bank.npz`, with its `.metadata.json` beside it. The importer validates model ID, revision, dimension, checksum, case IDs and image order.

---

## 3. Safety checks and what to do when one stops the run

| Message (abbreviated) | Where | Meaning | Action |
|---|---|---|---|
| `Image audit does not cover every image-manifest row` | audit/split | The audit is partial (e.g. from a smoke run) | Re-run from `--from-stage audit` in full mode |
| `Full split counts differ from the study lock` / `case IDs differ from the locked production cohort` | split and every later stage | The regenerated split isn't 3,529/754/750 with the locked membership hash | **Stop and report.** Don't edit the lock. Check that the raw image set and manifests are complete |
| `Full split metadata has a stale or missing …` | before MIL | Split files changed after the split was frozen | Re-run from `--from-stage split` |
| `Classical model was not fitted with the locked split` / `Classical test predictions differ…` | before tuning | Classical results come from an old split or were overwritten (e.g. 6-row smoke files) | Re-run from `--from-stage classical --resume` |
| `Pre-registered rank gate requires instance_projection: …` | start of MIL tuning (full mode) | `configs/mil.yaml` disagrees with the gate result | Follow section 5.3 exactly |
| `The prespecified paired comparison requires SigLIP2 as primary and Derm Foundation…` | statistics | Statistics need both SigLIP2 and Derm Foundation in the run | Include both encoders, or stop the run before `statistics` |
| `A selected MIL configuration is required for a full run` | MIL stages after tuning | Tuning never produced `best_mil_config.yaml` for this tag | Run from `tuning` for that encoder |
| `Tagged model grid is incomplete: …` | verify | Missing or stale seed, ensemble or classical files | Resume from the stage named in the message |

**Stale-result protection:** a MIL trial, fold or seed directory is reused only if its fingerprint matches: split, feature bank, case manifest, MIL config, seed and output-file hashes. Old directories without a fingerprint are retrained automatically. Deleting them isn't necessary.

---

## 4. Scenario playbook

> **Current run order:** S1 smoke → S11 steps 1–2 (external features) → S4 → S11 step 4 → S12 bundle build.

### S1. Smoke test (always first on a new machine or after code changes)

```bash
python scripts/run_pipeline.py --config configs/study_25class.yaml --run-mode smoke \
  --device cpu --cpu-workers "$CPU_WORKERS" 2>&1 | tee reports/smoke.log
```

- Uses a tiny case subset (2 per class, plus a few no-label cases), 1 tuning trial, 2 folds, seeds 42–43, 2 epochs.
- Writes only under `smoke_runs/`.
- Stops after `final`/`evaluation`; statistics and verification need the full cohort.
- **Pass criterion:** exits 0. Its metrics are meaningless and must never be reported.

### S2. Full study from scratch (new VM, nothing on disk)

```bash
python scripts/run_pipeline.py --config configs/study_25class.yaml --run-mode full \
  --download-data --download-workers 32 --device cpu --cpu-workers "$CPU_WORKERS" \
  --resume 2>&1 | tee reports/production_pipeline.log
```

This runs every stage for all 7 encoders. It may stop at the rank gate; see section 5.3.

### S3. Full study, raw images already on disk

Same as S2 without `--download-data`, starting at `--from-stage manifest`.

### S4. **Revision rerun: reuse feature banks, refit everything on the locked split** ← current task

Use this when the raw images and all 7 feature banks are on disk and the split or model code changed.

**Step 1: audit and locked split** (a couple of minutes)
```bash
python scripts/run_pipeline.py --config configs/study_25class.yaml --run-mode full \
  --device cpu --cpu-workers "$CPU_WORKERS" --from-stage audit --to-stage split \
  2>&1 | tee reports/revision_step1_split.log
```
It must end without a split-lock error. Check: `data/splits/split_manifest_v1.csv` has 3,529 / 754 / 750 cases.

**Step 2: rank gate check (optional but recommended; seconds)**
```bash
python revision-experiments/02_loss_and_rank_diagnostics.py
```
Read `effective_rank.enable_manifold_residual` in the printed summary. If it's `true`, apply section 5.3 **now**, before step 3. The pipeline enforces the same gate anyway, but deciding here avoids a mid-run stop.

**Step 3: everything else** (about 4–5 hours on 32 vCPUs; see section 7)
```bash
python scripts/run_pipeline.py --config configs/study_25class.yaml --run-mode full \
  --device cpu --cpu-workers "$CPU_WORKERS" --from-stage features --resume \
  2>&1 | tee reports/revision_step3_models.log
```
- **Features:** skipped if each bank's recorded image-manifest and output hashes match. Otherwise that encoder is re-extracted (about 1–3 minutes each). Either outcome is correct.
- **Classical:** refits all 35 models on the locked split. This repairs the 6-row smoke-overwritten prediction files.
- **MIL:** the revised MIL (BCE with (neg/pos)^p weights, prevalence bias start, ½ macro + ½ micro AP checkpoint, Platt calibration) is tuned, cross-validated and run for 10 seeds, **for every encoder**.
- **Finishing stages:** final, evaluation, statistics, tables, figures and verification run for SigLIP2.

**Step 4: report back.** Give the commit hash, the log paths, `reports/revision/rank_gate.json`, and the ensemble `overall_metrics.csv` for SigLIP2 and Derm Foundation. Say whether `verify` passed.

### S5. MIL only, one encoder, everything else already valid

For example, re-running SigLIP2 MIL after a MIL-code change, when the locked split, SigLIP2 bank and classical results are current.

```bash
python scripts/run_mil_pipeline.py --config configs/study_25class.yaml \
  --encoder siglip2_so400m --run-mode full --device cpu --cpu-workers "$CPU_WORKERS" \
  --run-tag canonical --strict-locked-comparison --resume 2>&1 | tee reports/mil_siglip2.log
```
- Runs preflight → rank gate → tuning → CV → 10 seeds → ensemble → thresholds → final → evaluation → statistics → tables → figures → verification.
- To stop before the cross-model statistics, add `--to-stage evaluation`.
- Statistics also need a current Derm Foundation MIL ensemble under the same tag. Run S5 for `derm_foundation` first with `--to-stage thresholds`.
- **Always pass `--strict-locked-comparison` in full mode.** Without it, `51_compare_grid.py` can fall back to the old, pre-revision Derm Foundation ensemble in `reports/statistics/` if the tagged one is missing.

**Useful partial ranges** (the stage order still applies):
- Re-ensemble and recalibrate only: `--from-stage ensemble --to-stage thresholds`
- Re-evaluate only: `--from-stage evaluation --to-stage evaluation`
- Statistics → verification only: `--from-stage statistics`

### S6. One encoder end to end (features → classical → MIL), others untouched

```bash
python scripts/run_pipeline.py --config configs/study_25class.yaml --run-mode full \
  --device cpu --cpu-workers "$CPU_WORKERS" \
  --skip-encoders resnet50,inception_v3,bit50,vit_base,clip_vitb32,derm_foundation \
  --primary-encoder siglip2_so400m --from-stage features --to-stage thresholds --resume
```
- `--to-stage thresholds` avoids the statistics stage, which needs both SigLIP2 and Derm Foundation.
- Checks apply only to included encoders.
- If you need this reduced grid kept separate from the full grid, use a fresh artifacts directory (`HAWK_DERM_PATH_ARTIFACTS_DIR`).

### S7. Single scripts (debugging or targeted repairs)

```bash
# features for one encoder
python scripts/10_extract_features.py --config configs/study_25class.yaml --encoder vit_base --device cpu --workers 32
# one classical model
python scripts/20_train_classical.py --config configs/study_25class.yaml --encoder siglip2_so400m --classifier random_forest --workers 32
# one MIL training run (always uses configs/mil.yaml defaults, not a tuned config)
python scripts/30_train_mil.py --config configs/study_25class.yaml --encoder siglip2_so400m --seed 42 --device cpu
```
In full mode prefer the orchestrators: single scripts don't run the lock and provenance checks by themselves.

### S8. Resume after a Spot preemption or crash

Re-run **the same command** with `--resume`. Completed stages with matching hashes are skipped; partially written stages re-run. For MIL, completed trial, fold and seed directories with matching fingerprints are reused.

### S9. Regenerate analysis outputs only (no training)

```bash
python scripts/run_pipeline.py --config configs/study_25class.yaml --run-mode full \
  --device cpu --cpu-workers "$CPU_WORKERS" --from-stage statistics --to-stage verify
```

### S10. Diagnostics and revision analyses (read-only, no training)

```bash
python revision-experiments/01_photo_count_comparison.py   # MIL vs RF by photos per case (exploratory)
python revision-experiments/02_loss_and_rank_diagnostics.py # base-rate inflation + rank gate
python scripts/90_test_cpu_scaling.py                       # confirms all CPU cores are used
```
Outputs go to `revision-experiments/outputs/`. These never change models or configs.

---

### S11. External cohort (SD-198): features first, scored during training

The external cohort is **inference-only**. Every trained model scores it in the same process that scores the internal test set, so no model is reloaded.

1. **Download** the dataset (private) to `data/external/sd198-top25`, or point `HAWK_DERM_EXTERNAL_SD198_ROOT` at an existing copy:
   ```bash
   hf download HawkFranklin-Research/SD198-DermoLens-25Class-External-Test --repo-type dataset --local-dir data/external/sd198-top25
   ```
2. **Prepare and extract, before any model stage.** Either run the scripts directly:
   ```bash
   python scripts/80_prepare_external.py                   # validates 422 images, hashes, label order; writes adapter manifests
   python scripts/81_extract_external_features.py --device cpu --workers "$CPU_WORKERS"   # all 7 encoders, resumable
   ```
   or let the pipeline run its `external-prepare` / `external-features` stages, which sit before `features`.
3. **Train as usual (S4 step 3).** Classical models, every MIL seed and the final model then write `external_sd198_*predictions.csv` next to their test predictions. The ensemble stage averages and calibrates them with the validation-fitted Platt; the thresholds stage writes `thresholds/external_sd198_operating_points.csv`.
4. **Evaluate and plot:** the `external-evaluate` and `external-figures` stages, or `python scripts/82_evaluate_external.py` then `python scripts/83_external_figures.py`.

**Order matters:** if a model was trained *before* the external features existed, its run fingerprint and manifest hashes change once they appear, so `--resume` retrains it to add the external scores. Extract external features first to avoid paying for training twice. Smoke runs never extract external features.

### S12. Production bundles and releases (optional, manual)

- **Build and self-check:** `python scripts/84_build_production_bundles.py` (also the `bundle` stage in full mode). It packages each model in `configs/production.yaml` that has an encoder assigned, then re-scores the locked test set and every external cohort through the bundle. It fails unless the results match the evaluated ensemble to within 1e-5.
- **Release, only when explicitly instructed:**
  - `python scripts/publish_production_model.py --model pelliscope --dry-run` writes the README model card (per-condition thresholds, validation and test sensitivity/specificity with CIs, external results) into the bundle so it can be reviewed first.
  - Then `python scripts/publish_production_model.py --model pelliscope --confirm --revision-tag v2-revision` uploads it.
- **Internal archive:** add `--archive-to-hub` to `run_pipeline.py` to upload every classical model, every MIL trial, fold and seed, the production bundles and the reports to the private model repo **in the background**, under `runs/<commit>/<run-tag>/`. Or run `python scripts/86_archive_run_to_hub.py <dirs…>` by hand.
- `scripts/publish_models.py` is retired and exits immediately.

## 5. The revised MIL: what is configured and how it is enforced

### 5.1 What `configs/mil.yaml` now does

| Setting | Value | Why |
|---|---|---|
| `loss` | `bce` | One imbalance correction; focal loss removed |
| `positive_weight_power` | tuned over {0.0, 0.5} | Positive weight = (neg/pos)^p. The old setting (focal α 0.75 × full neg/pos) inflated rare-label scores (median 8× over prevalence on validation) |
| `prior_bias_init` | `true` | Output biases start at log(prev / (1 − prev)) |
| `checkpoint_metric` | `macro_micro_ap` (½ macro AP + ½ micro AP), 3-epoch average, log-loss tie-break within 0.002 | Replaces macro AUC + 0.1 × LRAP, which ignored false positives on no-label cases |
| `calibration.method` | `platt` | Per-label Platt fitted on the validation ensemble, applied unchanged to test. AP and AUC per label are unchanged |
| `instance_projection` | `dense` unless the rank gate requires `manifold_residual` | Pre-registered gate (section 5.3) |

Search: 12 random-search trials (not Optuna). Threshold rule: `balance` (sensitivity ≈ specificity), not Youden.

### 5.2 Outputs that changed shape

- `ensemble/validation_predictions.csv` and `test_predictions.csv` are now **calibrated**. The uncalibrated versions are `*_predictions_uncalibrated.csv`.
- `ensemble/calibration_parameters.csv` holds slope and offset per label.
- `ensemble/ensemble_manifest.json` has a `calibration` block.
- Seed histories record `checkpoint_score_raw`, the smoothed `checkpoint_score`, and `validation_log_loss_unweighted`.

### 5.3 Rank gate procedure (only allowed edit to `configs/mil.yaml`)

The full MIL run computes the effective rank of the L2-normalised SigLIP2 photo embeddings for training-split cases, records it in `reports/revision/rank_gate.json`, and requires:
- `instance_projection: manifold_residual` if effective rank / dimension ≤ 0.25;
- `instance_projection: dense` otherwise.

If the run stops with the rank-gate error:
1. Open `reports/revision/rank_gate.json` and note `required`.
2. Set `architecture.instance_projection` in `configs/mil.yaml` to that value. Change nothing else.
3. `git commit -am "Apply pre-registered rank gate: <value>" && git push`
4. Resume the **same** command with `--resume`. Tuning restarts for all encoders, because the MIL config hash changed.

The gate uses SigLIP2 only; the shared config then applies to every encoder. If the SigLIP2 bank is absent, the gate is recorded as not evaluated and the config stays `dense`.

---

## 6. Where results are and what to hand back

| What | Path |
|---|---|
| Locked split | `data/splits/split_manifest_v1.csv` |
| Classical predictions | `artifacts/models/classical/<encoder>/<classifier>/{train,validation,test}_case_predictions.csv` |
| MIL selected config | `artifacts/models/mil/<encoder>/canonical/tuning/best_mil_config.yaml` |
| MIL ensemble (paper headline) | `artifacts/models/mil/<encoder>/canonical/ensemble/` |
| Thresholds and operating points | `artifacts/models/mil/<encoder>/canonical/thresholds/` |
| Paired statistics | `reports/reanalysis/siglip2_so400m_mil_canonical/statistics/` |
| Rank gate record | `reports/revision/rank_gate.json` |
| Tables / figures | `reports/tables/`, `reports/figures/` |

**Hand-back checklist:**
- commit hash
- exact commands run
- log files
- whether `verify` passed
- the ensemble `overall_metrics.csv` for SigLIP2 and Derm Foundation
- the rank-gate JSON
- any stage that re-ran unexpectedly

**Don't** summarise metrics from memory; copy them from the CSVs.

---

## 7. Runtime and disk (32 vCPU `c2d-highcpu-32`, CPU only)

| Stage | Approximate time |
|---|---|
| Audit + split | ~2 min |
| Features, all 7 encoders (only if re-extracted) | ~10–12 min |
| Classical, 35 models | ~8–10 min |
| MIL per encoder (tuning ~7 + CV ~9 + 10 seeds ~23 + ensemble/thresholds ~2 min) | **~40 min** |
| MIL, all 7 encoders | **~4–5 h** |
| Final model, evaluation, statistics (10,000 bootstraps), tables, figures | ~10 min |

The earlier "~1 h 15 min" estimate counted MIL for SigLIP2 only. The merged pipeline runs MIL for every included encoder.

Disk use is about 33 GB of 200 GB (environment ~11 GB, raw images ~7.5 GB, artifacts ~2.5 GB). No cleanup is needed for a rerun.

---

## 8. Things not to do

- Don't hand-edit or delete `data/splits/split_manifest_v1.csv`, or change `locked_split_counts` / `locked_split_membership_sha256` in `configs/study_25class.yaml`.
- Don't pass `--epochs`, `--trials`, `--folds` or `--seeds` overrides in full mode unless explicitly instructed. They change the pre-registered protocol.
- Don't copy files between `smoke_runs/` and production directories.
- Don't report smoke-run metrics or partial-run metrics as results.
- Don't push model files to Hugging Face unless instructed; releases go through `publish_production_model.py --confirm` (S12).
- Don't add `--publish-features` to a compute run unless asked; publishing is a separate phase (`DEPLOYMENT-manual.md` §9).

---

## 9. Open items

1. **PelliScope Scout encoder:** `configs/production.yaml` leaves it `null` until the author names the lighter encoder. Until then only PelliScope is bundled.
2. **Manuscript update:** all MIL numbers quoted before this rerun are pre-revision and must be replaced from the new outputs, including the external SD-198 results.
