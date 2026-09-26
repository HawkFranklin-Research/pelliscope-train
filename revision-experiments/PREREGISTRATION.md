# Revised MIL training: pre-registration

Declared 26 September 2026, before the locked-split (3,529 / 754 / 750) rerun. This file is committed before any revised model is trained, so its timestamp precedes every revised result.

## Changes to MIL training (`configs/mil.yaml`, `src/hawk_derm/models/mil.py`)

1. **One imbalance correction.** Binary cross-entropy with per-label positive weights of (negatives/positives)^p. The tuning stage chooses p ∈ {0, 0.5}; focal loss and focal α are removed. The original loss was focal (α = 0.75) × full negatives/positives weights.
2. **Prevalence bias start.** Each output bias starts at log(prevalence / (1 − prevalence)) from the training split.
3. **Checkpoint rule.** 0.5 × validation macro AP + 0.5 × validation micro AP, averaged over the last 3 epochs; within 0.002, the lower unweighted validation log-loss wins. This replaces macro ROC-AUC + 0.1 × LRAP.
4. **Calibration.** Per-label Platt scaling fitted on the validation probability ensemble (`scripts/35_ensemble_mil.py`) and applied unchanged to test. Per-label AP and ROC-AUC are unchanged by construction. Uncalibrated ensemble files are kept alongside.
5. **Instance projection (gated).** The manifold-residual projection (rank 32) replaces the first dense instance layer **only if** `02_loss_and_rank_diagnostics.py` reports effective rank / dimension ≤ 0.25 on training-split photo embeddings. Otherwise it stays dense. The gate is decided before tuning. A full MIL run enforces it automatically (`hawk_derm.features.diagnostics.assert_rank_gate`): it records the result in `reports/revision/rank_gate.json` and stops if `configs/mil.yaml` disagrees.

Architecture search space, trial count (12), seeds (42–51), cross-validation folds, threshold rule and the random-forest comparators are unchanged.

## Evidence at declaration

`02_loss_and_rank_diagnostics.py` on the existing September validation ensemble: mean predicted probability exceeds prevalence for all 25 labels (median ratio 8.0; maximum 17.6), with larger ratios for rarer labels.

## Reporting

- The revised MIL is reported as one pipeline, not as a component ablation.
- The held-out 750 test cases are evaluated once, for the model produced by this configuration.
- The paired case-bootstrap comparison with the random forest is reported whatever its direction, with macro/micro ROC-AUC, macro/micro AP and calibration (ECE, Brier).
- The September MIL results remain available for reference and are described as the pre-revision model.
- The separately fitted final model (trained on train + validation) has no held-out data for calibration and is reported uncalibrated.

## Production models (declared before the rerun)

Fixed by rule in `configs/production.yaml`, never chosen by test or external performance:

- **PelliScope:** the calibrated 10-seed SigLIP2 SO400M MIL probability ensemble, with its validation-fitted Platt calibration and validation-selected thresholds.
- **PelliScope Scout:** the same recipe on a lighter encoder, to be named in `configs/production.yaml` before the rerun.

Each is packaged by `scripts/84_build_production_bundles.py`. A bundle is valid only if re-scoring the locked test set through it reproduces the evaluated ensemble predictions (max abs difference ≤ 1e-5). The separately fitted `final_model` (train + validation) is an internal artifact: it is evaluated without thresholds and is not a production model.

## External evaluation: SD-198 (declared before any external result is seen)

- **Data:** 422 SD-198 images mapped to 11 of the 25 labels (pre-mapped in the released dataset), one image per pseudo-case, 420 unique contents. Used only for inference.
- **Features:** extracted for all 7 encoders with the unchanged study extractor (`scripts/81_extract_external_features.py`) before any model is trained. Every model scores the external cohort in the same process that scores the locked test set.
- **Models evaluated:** all of them. That covers 7 encoders × (MIL ensemble, MIL final model, and 5 classical classifiers).
- **Nothing external feeds back:** no training, tuning, calibration, threshold or model choice uses external data. MIL thresholds and calibration are the validation-derived ones, applied unchanged.
- **Metrics (`scripts/82_evaluate_external.py`):**
  - top-1/3/5 accuracy over all 25 outputs;
  - one-vs-rest macro/micro ROC-AUC and AP over the represented labels (the other represented labels' images serve as negatives, assuming one diagnosis per image);
  - AP chance level;
  - for MIL ensembles, sensitivity at the locked thresholds (Wilson CIs) and the off-target alarm rate on unrepresented labels.
- **Uncertainty:** 2,000 bootstrap resamples of duplicate-content groups.
- **Paired comparisons:** MIL ensemble vs random forest for every encoder (Holm-adjusted across encoders), and each production encoder's MIL ensemble vs every other encoder's random forest.
- **Sensitivity analysis:** the same headline metrics on the 420 unique images.
- **Reporting:** all models and all metrics, whatever their direction. SD-198 evaluates single-image recognition in a different image source, not multi-photo aggregation. This limitation is stated with every external result.
