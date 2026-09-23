# Prediction source cache

`hf_production/` is a reproducible local cache of the full production prediction files downloaded from the immutable Hugging Face revision recorded in `../analysis_config.yaml`.

Run `python 00_fetch_prediction_sources.py` to rebuild it. The audited, normalized 750-case prediction tables used by the analysis are retained under `../outputs/raw_predictions/`.
