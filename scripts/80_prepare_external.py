#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _common import load_context

from hawk_derm.data.external import build_external_manifests, load_external_cohorts
from hawk_derm.io import sha256_file, write_csv, write_json
from hawk_derm.provenance import RunRecorder


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate external cohorts and write pipeline-schema adapter manifests.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--cohort", default=None, help="Prepare only this cohort name (default: all configured).")
    args = parser.parse_args()
    config = load_context(args.config)
    cohorts = [cohort for cohort in load_external_cohorts(config) if args.cohort in (None, cohort.name)]
    if not cohorts:
        raise ValueError("No matching external cohort is configured in external_cohorts")
    for cohort in cohorts:
        if not cohort.source_path("image_manifest").is_file():
            raise FileNotFoundError(
                f"External dataset for {cohort.name} is not at {cohort.dataset_root}; "
                f"download it or set HAWK_DERM_EXTERNAL_{cohort.name.upper()}_ROOT"
            )
        with RunRecorder(
            "prepare_external",
            {**vars(args), "cohort": cohort.name},
            cohort.manifests_dir,
            inputs=[cohort.source_path("image_manifest"), cohort.source_path("label_schema")],
        ) as run:
            cases, images, summary = build_external_manifests(cohort, config["labels"])
            write_csv(cohort.case_manifest_path, cases)
            write_csv(cohort.image_manifest_path, images)
            summary_path = cohort.manifests_dir / "external_summary.json"
            write_json(
                summary_path,
                {
                    **summary,
                    "dataset_root": str(cohort.dataset_root),
                    "case_manifest_sha256": sha256_file(cohort.case_manifest_path),
                    "image_manifest_sha256": sha256_file(cohort.image_manifest_path),
                    "use": "inference_only_never_training_tuning_calibration_or_thresholds",
                    "complete": True,
                },
            )
            run.complete([cohort.case_manifest_path, cohort.image_manifest_path, summary_path])
        print(f"[external] {cohort.name}: {summary['images']} images, {summary['represented_label_count']} labels -> {cohort.manifests_dir}")


if __name__ == "__main__":
    main()
