# Smoke verification status

The complete smoke run_pipeline.py rerun passed successfully.

- All seven encoders completed.
- All classical models and MIL stages completed.
- Statistics, tables, and figures completed.
- Final release verification passed.
- No data download occurred.
- No Hugging Face upload occurred.
- Exit code: 0.

Release verification confirms:

complete: true
floating_revisions: {}
revisions_are_immutable: true

The full log is saved at /home/prime/Documents/github/derm-paper-codebases/hawk-derm/reports/test_logs/32_run_pipeline_smoke_pinned.log.

The Bit-50 and CLIP “unexpected weights” messages are expected because only their vision backbones are used. Google Derm again ran successfully through its CPU TensorFlow path.
