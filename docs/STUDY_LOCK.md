# AIIM 25-class study lock

This document defines the experiment before any full training is accepted. A change to a locked item creates a new study version and requires new manifests, predictions, statistics, tables, and figures.

## Locked cohort

- Canonical raw data on Hawk Prime: `/home/prime/Documents/github/derm-paper-codebases/SCIN-Dermatology-Raw-Images`.
- Cohort and label construction follow the September 2026 `derm-train-sep26` design.
- Expected legacy universe: 5,033 cases, 10,407 image-manifest references, and 10,379 unique local PNG files.
- The 2,377 cases with no positive target among the selected 25 conditions remain in the primary reconstruction for parity with the September design.
- Every analysis reports all-zero, single-label, and multi-label case counts. A target-positive-only analysis is generated from the same predictions as a prespecified sensitivity analysis.
- A case retains at most three ordered images. Missing bag positions are masked and never enter attention pooling.

The exact label order is the `labels` list in `configs/study_25class.yaml`. That order is part of the prediction and checkpoint schema.

## Locked split

- One immutable case-level 70/15/15 train/validation/test assignment is written to `data/splits/split_manifest_v1.csv`.
- Split seed: 42.
- Method: iterative multilabel stratification in two stages.
- Every encoder, classical classifier, MIL seed, and statistical comparison uses the same case membership.
- Repeated MIL runs vary initialization and training stochasticity. They do not redraw test cases.
- No case ID or decoded image hash may cross split boundaries.

## Locked evaluation unit

The primary evaluation unit is a patient-submitted case. Classical classifiers may be fitted to image embeddings, but their probabilities are aggregated to one vector per case using the configured rule, initially the arithmetic mean. MIL produces one probability vector per case directly. Image-level results are stored separately and never treated as a paired case-level comparison.

## Locked model families

Seven frozen encoders are compared: ResNet-50, Inception-v3, BiT-50, ViT-Base, CLIP ViT-B/32, Derm Foundation, and SigLIP2 SO400M. Five classical families are fitted for every encoder: class-weighted logistic regression, calibrated linear SVM, class-weighted random forest, gradient boosting, and distance-weighted k-NN. The sixth heatmap column is gated-attention MIL.

Model repository revisions must be replaced with immutable commit revisions before a release is frozen. A floating `main` revision is allowed only during code preparation.

## Locked threshold and test policy

- Hyperparameters, early stopping, and thresholds use training and validation data only.
- Balanced operating thresholds minimize `abs(sensitivity - specificity)` on validation predictions using the configured 0.01 grid.
- Selected thresholds are applied unchanged to the locked test cases.
- The default 0.5 operating point is reported separately.
- The test set is evaluated only after the run configuration, checkpoint, and thresholds are frozen.

## Locked uncertainty interpretation

Patient-level uncertainty is estimated by case-level bootstrap resampling. Repeated-seed standard deviations describe optimization stability and are not treated as independent patient samples. Per-condition comparisons are adjusted with the Holm procedure.

## Two-agent contract

Hawk Prime and the MacBook must use the same Git commit, configuration hash, case-manifest hash, split-manifest hash, label order, model revisions, and output schema. Jobs are assigned in `coordination/jobs.yaml`; one agent owns a job at a time. Work is split by whole encoder or whole training seed. No two agents write the same artifact directory concurrently.
