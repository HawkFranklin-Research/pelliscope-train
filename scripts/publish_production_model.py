#!/usr/bin/env python3
"""Publish one production bundle to its Hugging Face model repo, with a generated model card.

Manual and independent: never called by the pipelines. Refuses unless the bundle exists, its
file hashes are intact and its self-check passed. Use --dry-run to write the README only.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from _common import load_context

from hawk_derm.config import path_from
from hawk_derm.data.external import load_external_cohorts
from hawk_derm.inference.bundle import MANIFEST, ProductionBundle, production_entries
from hawk_derm.io import read_json


def fmt(value: float, digits: int = 3) -> str:
    return "–" if pd.isna(value) else f"{value:.{digits}f}"


def threshold_table(bundle_dir: Path, labels: list[str]) -> str:
    validation = pd.read_csv(bundle_dir / "validation_selected_thresholds.csv").set_index("label")
    test = pd.read_csv(bundle_dir / "test_operating_points.csv").set_index("label")
    lines = [
        "| Condition | Threshold | Validation sens. | Validation spec. | Test sens. (95% CI) | Test spec. (95% CI) | Test positives |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for label in labels:
        v, t = validation.loc[label], test.loc[label]
        lines.append(
            f"| {label} | {fmt(v['threshold'], 2)} | {fmt(v['sensitivity'])} | {fmt(v['specificity'])} | "
            f"{fmt(t['sensitivity'])} ({fmt(t['sensitivity_ci_lower'])}–{fmt(t['sensitivity_ci_upper'])}) | "
            f"{fmt(t['specificity'])} ({fmt(t['specificity_ci_lower'])}–{fmt(t['specificity_ci_upper'])}) | {int(t['support_positive'])} |"
        )
    return "\n".join(lines)


def headline(bundle_dir: Path) -> str:
    test = pd.read_csv(bundle_dir / "test_operating_points.csv")
    return (
        f"- Macro ROC-AUC (mean of per-condition AUC): {fmt(test['auc'].mean())}\n"
        f"- Macro average precision: {fmt(test['pr_auc'].mean())}\n"
        "- Confidence intervals and paired comparisons are in the accompanying paper and repository reports."
    )


def external_section(config: dict, model_key: str) -> str:
    parts = []
    for cohort in load_external_cohorts(config):
        grid_path = path_from(config, "reports_dir") / "external" / cohort.name / "model_grid.csv"
        if not grid_path.is_file():
            continue
        grid = pd.read_csv(grid_path).set_index("model")
        if model_key not in grid.index:
            continue
        row = grid.loc[model_key]
        parts.append(
            f"**{cohort.display_name}** ({int(row['images'])} single images, {int(row['represented_labels'])} of 25 conditions): "
            f"macro ROC-AUC {fmt(row['macro_auc_represented'])} "
            f"({fmt(row.get('macro_auc_represented_ci_lower'))}–{fmt(row.get('macro_auc_represented_ci_upper'))}), "
            f"top-1 {fmt(row['top1_accuracy'])}, top-3 {fmt(row['top3_accuracy'])}. "
            "One diagnosis per image; evaluates single-image recognition in a different image source, not multi-photo aggregation."
        )
    return "\n\n".join(parts) if parts else "_No external evaluation has been recorded for this bundle._"


def model_card(config: dict, bundle_dir: Path, entry: dict) -> str:
    manifest = read_json(bundle_dir / MANIFEST)
    encoder = manifest["encoder"]
    labels = manifest["labels"]
    return f"""---
license: other
tags: [dermatology, teledermatology, multiple-instance-learning, medical-imaging]
---

# {manifest['display_name']}

{entry.get('description', '')}

**Intended use:** clinician-facing decision support for structured review and triage of patient-submitted
teledermatology cases (1–3 photos of one concern). **Not** an autonomous diagnostic device. Not trained on,
and must not be used for, malignant or neoplastic lesions. Not externally validated for clinical deployment.

## Model

- Case-level gated-attention multiple instance learning over frozen **{encoder['key']}** image embeddings
  (`{encoder['model_id']}` @ `{encoder['revision']}`, {encoder['dimension']}-d).
- Probability ensemble of {len(manifest['seeds'])} independently trained seeds, then per-condition Platt calibration fitted on validation data.
- 25 independent condition outputs (multi-label). Each output is compared with its own validation-selected threshold
  (rule: sensitivity ≈ specificity on validation).
- Training data: SCIN, locked case-level split (train / validation / test = 3,529 / 754 / 750 cases).

## Internal held-out test (750 cases)

{headline(bundle_dir)}

## Per-condition thresholds and operating points

Thresholds were selected on validation and applied unchanged to test. Rare conditions have wide intervals.

{threshold_table(bundle_dir, labels)}

## External evaluation

{external_section(config, f"{encoder['key']}+mil_ensemble")}

## Inference

```python
from hawk_derm.inference.bundle import ProductionBundle
bundle = ProductionBundle.load("path/to/this/repo")
out = bundle.predict(bags, masks)   # bags: [cases, 3, {encoder['dimension']}] embeddings; masks: [cases, 3] real-photo flags
out["probability"], out["decision"]  # calibrated probabilities and thresholded calls, in `labels` order
```

Embeddings must come from the exact encoder and revision above, using the preprocessing in
`hawk_derm.features.extractors`.

## Provenance

- Code commit: `{manifest['git_commit']}`
- Split manifest SHA-256: `{manifest['split_manifest_sha256']}`
- Bundle self-check (re-scoring reproduces evaluated predictions): {'passed' if manifest['self_check'].get('passed') else 'NOT PASSED'}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--model", required=True, help="Production model name from configs/production.yaml")
    parser.add_argument("--confirm", action="store_true", help="Required to upload.")
    parser.add_argument("--dry-run", action="store_true", help="Write README.md into the bundle and stop.")
    parser.add_argument("--revision-tag", default=None, help="Git tag to create on the Hub, e.g. v2-revision.")
    parser.add_argument("--token", default=None)
    args = parser.parse_args()
    config = load_context(args.config)
    entries = production_entries()
    if args.model not in entries:
        raise ValueError(f"{args.model} is not a production model with an encoder in configs/production.yaml")
    entry = entries[args.model]
    bundle_dir = path_from(config, "artifacts_dir") / "production" / args.model
    ProductionBundle.load(bundle_dir)  # verifies file hashes
    if not read_json(bundle_dir / MANIFEST)["self_check"].get("passed"):
        raise RuntimeError("Bundle self-check has not passed; run scripts/84_build_production_bundles.py")
    (bundle_dir / "README.md").write_text(model_card(config, bundle_dir, entry), encoding="utf-8")
    print(f"[publish] model card written: {bundle_dir / 'README.md'}")
    if args.dry_run:
        return
    if not args.confirm:
        raise SystemExit("Refusing to upload without --confirm")
    if not entry.get("hub_repo"):
        raise ValueError(f"{args.model} has no hub_repo in configs/production.yaml")
    from huggingface_hub import HfApi

    api = HfApi(token=args.token or os.getenv("HF_TOKEN"))
    commit = api.upload_folder(repo_id=entry["hub_repo"], folder_path=str(bundle_dir), commit_message=f"Release {args.model} bundle")
    if args.revision_tag:
        api.create_tag(entry["hub_repo"], tag=args.revision_tag, revision=commit.oid)
    print(f"[publish] uploaded {bundle_dir} -> {entry['hub_repo']} ({commit.oid})")


if __name__ == "__main__":
    main()
