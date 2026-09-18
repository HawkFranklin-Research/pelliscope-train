# Statistical analysis plan

## Primary estimands

The primary discrimination estimand is case-level macro ROC-AUC across the frozen 25-condition vocabulary. Secondary estimands are macro PR-AUC, micro ROC-AUC, micro PR-AUC, per-condition ROC-AUC and PR-AUC, precision, recall, F1, balanced accuracy, Brier score, and expected calibration error.

The primary model comparison pairs SigLIP2 MIL, Google Derm MIL, and the strongest case-aggregated classical baseline on identical case IDs and condition labels. Image-level predictions are not used for this comparison.

## Uncertainty

Case-level bootstrap sampling preserves the 25-dimensional target and prediction vectors within each sampled case. The same sampled indices are used for both models in a paired comparison. Full bootstrap distributions and seeds are saved. Percentile 95% confidence intervals are reported for each model and for paired differences in macro ROC-AUC and macro PR-AUC.

A paired Monte Carlo permutation test swaps each case's complete 25-condition prediction vectors between the two models. This preserves within-case label correlation and supplies a two-sided null test for the paired macro-metric difference. The bootstrap interval remains the primary effect-size uncertainty summary.

Per-condition paired bootstrap comparisons use the same case-level indices. Two-sided bootstrap tail probabilities are adjusted across 25 conditions with Holm's method. Classes without both positive and negative test observations are reported as not estimable.

Repeated initialization results are summarized by mean and standard deviation to describe training stability. They are not interpreted as independent patient-level replicates.

## Operating points

Per-condition thresholds are selected from validation probabilities only. The initial rule chooses the smallest grid threshold minimizing `abs(sensitivity - specificity)`. Thresholds are applied unchanged to the locked test predictions. Test sensitivity and specificity receive Wilson 95% confidence intervals from their binomial denominators.

## Sensitivity analyses

The same stored predictions are reevaluated after restricting to cases with at least one positive target among the 25 conditions. This analysis quantifies the influence of the 2,377 all-zero target cases without changing primary model training or the locked primary cohort.

## Claims

Numerical improvement without a paired confidence interval excluding zero is described as a modest numerical difference or comparable discrimination. Repeated-seed variation cannot establish clinical superiority. Clinical utility requires external and prospective validation beyond this retrospective benchmark.
