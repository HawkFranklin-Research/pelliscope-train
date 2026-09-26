# Revised MIL training: pre-registration

Declared 26 September 2026, before the locked-split (3,529 / 754 / 750) rerun. This file is committed before any revised model is trained, so its timestamp precedes every revised result.

## Changes to MIL training (`configs/mil.yaml`, `src/hawk_derm/models/mil.py`)

1. **One imbalance correction.** Binary cross-entropy with per-label positive weights of (negatives/positives)^p. The tuning stage chooses p ∈ {0, 0.5}; focal loss and focal α are removed. The original loss was focal (α = 0.75) × full negatives/positives weights.
2. **Prevalence bias start.** Each output bias starts at log(prevalence / (1 − prevalence)) from the training split.
3. **Checkpoint rule.** 0.5 × validation macro AP + 0.5 × validation micro AP, averaged over the last 3 epochs; within 0.002, the lower unweighted validation log-loss wins. This replaces macro ROC-AUC + 0.1 × LRAP.
4. **Calibration.** Per-label Platt scaling fitted on the validation probability ensemble (`scripts/35_ensemble_mil.py`) and applied unchanged to test. Per-label AP and ROC-AUC are unchanged by construction. Uncalibrated ensemble files are kept alongside.
5. **Instance projection (gated).** The manifold-residual projection (rank 32) replaces the first dense instance layer **only if** `02_loss_and_rank_diagnostics.py` reports effective rank / dimension ≤ 0.25 on training-split photo embeddings. Otherwise it stays dense. The gate is decided before tuning and recorded in `outputs/loss_and_rank/summary.json`.

Architecture search space, trial count (12), seeds (42–51), cross-validation folds, threshold rule and the random-forest comparators are unchanged.

## Evidence at declaration

`02_loss_and_rank_diagnostics.py` on the existing September validation ensemble: mean predicted probability exceeds prevalence for all 25 labels (median ratio 8.0; maximum 17.6), with larger ratios for rarer labels.

## Reporting

- The revised MIL is reported as one pipeline, not as a component ablation.
- The held-out 750 test cases are evaluated once, for the model produced by this configuration.
- The paired case-bootstrap comparison with the random forest is reported whatever its direction, with macro/micro ROC-AUC, macro/micro AP and calibration (ECE, Brier).
- The September MIL results remain available for reference and are described as the pre-revision model.
- The separately fitted final model (trained on train + validation) has no held-out data for calibration and is reported uncalibrated.
