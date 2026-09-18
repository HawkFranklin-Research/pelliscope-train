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
| Figure 3 MIL diagnostics | seed histories, per-class metrics, and raw test predictions | `scripts/61_generate_figures.py` |
| Figure 4 threshold curves | `validation_threshold_sweep.csv` | `scripts/40_select_thresholds.py`, then `61_generate_figures.py` |
| Figure 5 closed-model comparison | `closed_models_per_class.csv` and `closed_models_overall.csv` | `scripts/52_prepare_closed_model_results.py`, then `61_generate_figures.py` |
| Figure 6 operating points and split metrics | `operating_points.csv` and `all_model_metrics.csv` | `scripts/60_generate_tables.py`, then `61_generate_figures.py` |
| Condition distributions | `reports/plot_data/condition_support.csv` | `scripts/60_generate_tables.py`, then `61_generate_figures.py` |
| Split donut | `reports/plot_data/split_composition.csv` | `scripts/60_generate_tables.py`, then `61_generate_figures.py` |
| Paired statistical comparisons | `reports/statistics/**` | `scripts/51_compare_grid.py` |

The closed-model panel remains conditional because API model versions, prompts, responses, parsing, cost, and timestamps must be frozen before a 25-condition rerun. The rest of the encoder/MIL analysis is independent of that optional stage.
