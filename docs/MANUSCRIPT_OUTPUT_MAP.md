# Manuscript output map

Every manuscript number is generated from a versioned CSV or JSON artifact. Figure scripts do not contain result constants.

| Manuscript deliverable | Generated data | Generator |
|---|---|---|
| Headline cohort numbers and Table 1 | `reports/tables/headline_study_numbers.csv` | `scripts/60_generate_tables.py` |
| Table 2 split sizes | `reports/tables/cohort_sizes_by_split.csv` | `scripts/60_generate_tables.py` |
| Table 3 encoder registry | `reports/tables/encoder_registry.csv` | `scripts/60_generate_tables.py` |
| Table 4 operating points and CIs | `reports/tables/operating_points.csv` | `scripts/40_select_thresholds.py`, then `60_generate_tables.py` |
| Hyperparameter appendix | `reports/tables/hyperparameter_trials.csv` | `scripts/31_tune_mil.py`, then `60_generate_tables.py` |
| Class-imbalance appendix | `reports/tables/class_imbalance_diagnostics.csv` | `scripts/60_generate_tables.py` |
| Inference appendix | `reports/tables/inference_latency_*` | `scripts/53_benchmark_inference.py` |
| Graphical abstract | `reports/figures/graphical_abstract.png` | `scripts/61_generate_figures.py` |
| Figure 1 cohort/preprocessing | `reports/figures/figure1_cohort_preprocessing.png` | `scripts/61_generate_figures.py` |
| Figure 2 encoder/classifier heatmap | `reports/plot_data/encoder_classifier_heatmap.csv` | `scripts/60_generate_tables.py`, then `61_generate_figures.py` |
| Figure 3A learning curves | `reports/plot_data/mil_history_long.csv` | `scripts/60_generate_tables.py`, then `61_generate_figures.py` |
| Figure 3B split-wise per-class AUC | `reports/plot_data/mil_per_class_metrics_long.csv` | `scripts/60_generate_tables.py`, then `61_generate_figures.py` |
| Figure 3C ROC/PR and sensitivity/specificity | `mil_roc_pr_curves.csv` and `test_sensitivity_specificity_curves.csv` | `scripts/60_generate_tables.py`, then `61_generate_figures.py` |
| Figure 4 threshold curves | `reports/plot_data/validation_threshold_curves.csv` | `scripts/40_select_thresholds.py`, then `60_generate_tables.py`, then `61_generate_figures.py` |
| Figure 5A closed-model heatmap | `closed_models_per_class.csv` | `scripts/52_prepare_closed_model_results.py`, then `61_generate_figures.py` |
| Figure 5B threshold accuracy comparison | `threshold_accuracy_comparison.csv` from primary-model test predictions | `scripts/60_generate_tables.py`, then `61_generate_figures.py` |
| Figure 6A operating curves | `test_sensitivity_specificity_curves.csv` | `scripts/60_generate_tables.py`, then `61_generate_figures.py` |
| Figure 6B split-wise metrics | `mil_split_per_class_metrics.csv` | `scripts/60_generate_tables.py`, then `61_generate_figures.py` |
| Condition distributions | `reports/plot_data/condition_support.csv` | `scripts/60_generate_tables.py`, then `61_generate_figures.py` |
| Split donut | `reports/plot_data/split_composition.csv` | `scripts/60_generate_tables.py`, then `61_generate_figures.py` |
| Paired statistical comparisons | `reports/statistics/**` | `scripts/51_compare_grid.py` |

The closed-model panel remains conditional because API model versions, prompts, responses, parsing, cost, and timestamps must be frozen before a 25-condition rerun. The rest of the encoder/MIL analysis is independent of that optional stage.

## Panel assembly and pagination

The submitted Figure 3, Figure 5, and Figure 6 assets were manually assembled from independent plot files. The reproduction code therefore emits each panel independently; it does not treat a generated contact sheet as a final manuscript figure.

For 25 conditions, condition-heavy panels use deterministic pages of conditions 1–10, 11–20, and 21–25 in the locked label order. This applies to Figure 3B, Figure 4, Figure 6A, Figure 6B, and the threshold/Youden appendices.

Standalone appendix outputs include:

- `mil_roc_pr_curves.png`;
- `mil_roc_with_sens_spec.png`;
- `mil_sensitivity_specificity_curves_with_thresholds_page_*.png`;
- `mil_youden_j_curves_page_*.png`;
- `mil_split_soft.png`;
- `condition_distributions.png`;
- `confidence_vs_selfreported.png` when those metadata fields are present;
- `gradability_mst.png` when those metadata fields are present;
- `images_counts_shot_types.png`.

All legacy colours are declared in `configs/figures.yaml`. Figures can be regenerated selectively without retraining, for example:

```bash
python scripts/61_generate_figures.py --config configs/study_25class.yaml --only mil,thresholds,operating
```
