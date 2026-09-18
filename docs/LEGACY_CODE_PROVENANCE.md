# Legacy code provenance

The public implementation was adapted from two private research snapshots. The September 2026 snapshot is authoritative for 25-class cohort construction and computational stages. The original `main` snapshot is authoritative for the submitted AIIM figure composition and palette.

| Public component | Legacy source |
|---|---|
| 25-class target construction | `derm-train-sep26/jul26/data_profiling/classify_25_classes.py` |
| Training-table preparation | `derm-train-sep26/jul26/scripts/prepare_training_datasets.py` |
| Case and image manifests | `derm-train-sep26/jul26/section-b/scripts/phase0_build_section_b_manifest.py` |
| SigLIP2 extraction | `derm-train-sep26/jul26/section-b/scripts/phase2_extract_siglip2_full_feature_bank.py` |
| SigLIP2 25-class MIL stages | `derm-train-sep26/jul26/section-b/siglip2-25class-mil/scripts/phase*.py` |
| Google Derm 25-class stages | `derm-train-sep26/jul26/section-b/google-derm-25class/scripts/phase*.py` |
| ResNet-50 extraction | `derm-train-origin-main/semi_supervised/feature_extractors/extract_resnet50.py` |
| Inception-v3 extraction | `derm-train-origin-main/semi_supervised/feature_extractors/extract_inception_v3.py` |
| BiT-50 extraction | `derm-train-origin-main/semi_supervised/feature_extractors/extract_bit.py` |
| ViT-Base extraction | `derm-train-origin-main/semi_supervised/feature_extractors/extract_vit.py` |
| CLIP extraction | `derm-train-origin-main/semi_supervised/feature_extractors/extract_clip.py` |
| Classical baselines | `derm-train-origin-main/semi_supervised/tabular_benchmarks/run_image_classifiers.py` |
| Figure layouts and palettes | `derm-train-origin-main/report_oct6/`, the manuscript `CODEBASE.map.md`, and original Draw.io assets |

The adaptation removes absolute machine paths, standardizes feature and prediction schemas, uses one locked split for every model, separates image- and case-level metrics, prevents threshold selection on test data, saves every raw probability, and records a run manifest for each major job.

Before release, record the two source Git commits and SHA-256 hashes of every consulted legacy script in a versioned provenance ledger. Legacy source files should not be copied into the public tree if their licenses or embedded data paths are unclear.
