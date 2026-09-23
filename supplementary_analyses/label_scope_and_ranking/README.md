# Label-scope and ranking analysis

This isolated supplementary analysis reuses saved case-level probabilities. It does not train models, extract features, or modify the established production pipeline.

It answers two questions:

1. How often does a true diagnosis appear among a model's top 1, 3, or 5 ranked conditions?
2. How do ROC-AUC and PR-AUC change as the evaluation expands from the 5 most prevalent training labels to 10, 15, 20, and all 25 labels?

All calculations use the canonical 750-case comparison cohort. Label scopes are selected only from training-set prevalence. Target-positive cases are used for Hit@k, Recall@k, and Precision@k because all-zero cases have no true label to retrieve.

## Reproduce

Run from the repository root:

```bash
python supplementary_analyses/label_scope_and_ranking/00_fetch_prediction_sources.py
python supplementary_analyses/label_scope_and_ranking/01_audit_predictions.py
python supplementary_analyses/label_scope_and_ranking/02_compute_metrics.py
python supplementary_analyses/label_scope_and_ranking/03_generate_figures.py
```

The audit must pass before metrics are calculated. Normalized raw probabilities, checksums, metric tables, plot data, and figures are written under `outputs/`.

Key outputs are:

- `outputs/raw_predictions/`: normalized per-case truth labels and probabilities for all 42 encoder/classifier combinations, plus individual MIL seeds;
- `outputs/manifests/`: source checksums, the locked 750 case IDs, and training-prevalence label order;
- `outputs/tables/`: Top-k, nested-scope, and per-condition metrics;
- `outputs/plot_data/`: machine-readable values and confidence bands used in the figures;
- `outputs/figures/`: thirteen publication-ready supplementary figures, including Top-5/10/15/20/25 radar and heatmap pages.

## Interpretation

The Top 5/10/15/20/25 analyses are nested evaluations of the same 25-label models; they are not separately trained classifiers. PR-AUC is prevalence-dependent, so every label scope includes its no-skill prevalence baseline and PR lift. These analyses are descriptive sensitivity analyses and should all be reported together rather than selecting the most favorable scope.
