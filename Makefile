PYTHON ?= python3
CONFIG ?= configs/study_25class.yaml
ENCODER ?= resnet50
CLASSIFIER ?= random_forest
SEED ?= 42

.PHONY: manifest audit split features classical mil tune cv repeated final thresholds evaluate compare tables figures verify smoke full

manifest:
	$(PYTHON) scripts/01_build_manifest.py --config $(CONFIG)

audit:
	$(PYTHON) scripts/02_audit_data.py --config $(CONFIG)

split:
	$(PYTHON) scripts/03_freeze_splits.py --config $(CONFIG)

features:
	$(PYTHON) scripts/10_extract_features.py --config $(CONFIG) --encoder $(ENCODER)

classical:
	$(PYTHON) scripts/20_train_classical.py --config $(CONFIG) --encoder $(ENCODER) --classifier $(CLASSIFIER)

mil:
	$(PYTHON) scripts/30_train_mil.py --config $(CONFIG) --encoder $(ENCODER) --seed $(SEED)

tune:
	$(PYTHON) scripts/31_tune_mil.py --config $(CONFIG) --encoder $(ENCODER)

cv:
	$(PYTHON) scripts/32_run_cross_validation.py --config $(CONFIG) --encoder $(ENCODER)

repeated:
	$(PYTHON) scripts/33_run_repeated_seeds.py --config $(CONFIG) --encoder $(ENCODER)

final:
	$(PYTHON) scripts/34_fit_final_model.py --config $(CONFIG) --encoder $(ENCODER) --seed $(SEED) --epochs 30

thresholds:
	$(PYTHON) scripts/40_select_thresholds.py --config $(CONFIG) --encoder $(ENCODER)

evaluate:
	@echo "Use scripts/50_evaluate.py with --predictions and --output-dir."

compare:
	@echo "Use scripts/51_compare_models.py with paired prediction files and model names."

tables:
	$(PYTHON) scripts/60_generate_tables.py --config $(CONFIG)

figures:
	$(PYTHON) scripts/61_generate_figures.py --config $(CONFIG)

verify:
	$(PYTHON) scripts/70_verify_release.py --config $(CONFIG)

smoke:
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --run-mode smoke

full:
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --run-mode full
