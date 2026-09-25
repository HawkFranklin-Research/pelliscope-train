# Average-precision diagnostics

Two small, isolated analyses of saved case-level probabilities. They do not change the production pipeline or train an image encoder. Run from the repository root:

```bash
python revision-experiments/ap_diagnostics/01_calibrate_scores.py
python revision-experiments/ap_diagnostics/02_attribute_ap_errors.py
```

**Score calibration:** SigLIP2 MIL ensemble and SigLIP2 random forest are aligned on their 754 shared *validation* cases. A shared positive slope and 25 regularized condition intercepts are fitted by validation log loss. Five-fold out-of-fold predictions check validation behavior; the mapping fitted on all shared validation cases is then applied once to each model's saved **locked 750-case test** predictions. The per-condition transformation is monotone, so test macro AP and per-condition AP cannot improve by construction. Micro AP can change because scores from different conditions are rescaled. The fixed penalty is 0.01, the folds use seed 42, and none of these choices uses test outcomes. Outputs are in `outputs/calibration/`.

**AP error attribution:** The four principal saved models are compared on exactly the same 750 test cases and labels. For each condition, `1 - AP` is apportioned over negatives at or above each positive's score, treating tied scores as a group. The burden sums to that condition's AP deficit. The analysis distinguishes all-zero cases from cases positive for another target condition, and reports both total burden and burden per negative pair. High-burden case IDs are listed for internal review. This is a descriptive, **test-informed** audit; it must not be used to select a model, tune a loss, or claim a confirmatory improvement on this same test cohort. Outputs are in `outputs/error_attribution/`.

The validation overlap of 754 is separate from the immutable 750-case test cohort. This directory uses the existing case probabilities; original feature-bank NPZ files are not needed. No result here proves that calibration is the reason for a model difference or that any high-burden negative is mislabeled.
