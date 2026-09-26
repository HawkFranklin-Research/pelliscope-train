# Revision experiments

This directory contains independent analyses of the existing locked test cohort. It does not modify the production training or figure pipeline.

- `01_photo_count_comparison.py`: paired MIL versus random-forest analysis by photos per case.
- `ap_diagnostics/`: validation-fitted score calibration and test-set AP error attribution. See its README for commands and interpretation.
- `02_loss_and_rank_diagnostics.py`: the two pre-declared checks for the revised MIL training (base-rate inflation on validation predictions; effective-rank gate for the manifold-residual projection).
- `PREREGISTRATION.md`: the revised MIL training changes, declared before the locked-split rerun.

## Photo-count comparison

Run from the repository root:

```bash
python revision-experiments/01_photo_count_comparison.py
```

The script joins the **same locked 750 case IDs** to the retained-image counts in `data/manifests/image_manifest.csv`. It compares the SigLIP2 MIL ten-seed probability ensemble with SigLIP2 random forest on cases with 1, 2, or 3 photos. Both models receive identical labels and case IDs in every group. Inputs are the saved case-probability CSVs under `supplementary_analyses/label_scope_and_ranking/outputs/raw_predictions/`; neither feature-bank NPZ files nor trained weights are required.

For each photo-count group, it reports macro and micro ROC-AUC, macro and micro average precision (AP), and the paired MIL-minus-RF difference. Percentile 95% intervals come from 2,000 case bootstrap resamples **within each group**, preserving model pairing. It also reports the three-photo minus one-photo difference in the MIL advantage. A positive interaction would be consistent with a larger MIL advantage for multi-photo cases; an interval covering zero would not establish it. There is no model selection based on these results.

Macro metrics omit conditions with no positives or no negatives in that group and show the number of evaluable labels. The **one-versus-three-photo macro interaction** uses only labels evaluable in both groups, listed in `shared_macro_interaction_labels.csv`. The bootstrap re-evaluates eligibility per resample, so macro interval endpoints should be interpreted cautiously for sparse conditions. Micro metrics pool all 25 case-condition decisions and do not have that missing-label problem. AP is scikit-learn's non-interpolated average precision, not trapezoidal PR area.

Outputs in `outputs/photo_count/` include the subgroup table, paired effect and interaction intervals, label support, the locked case-to-photo-count mapping, bootstrap draws, and source-file hashes. This is an **exploratory subgroup analysis** on an already-inspected test cohort. Report every group and metric regardless of direction; confirm any apparent interaction on external data.
