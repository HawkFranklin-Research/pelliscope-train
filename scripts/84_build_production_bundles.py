#!/usr/bin/env python3
"""Package each production model as a bundle and prove it reproduces the evaluated predictions.

A bundle passes its self-check only if re-scoring the locked test set (and every external
cohort) through the packaged bundle matches the saved calibrated ensemble predictions.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _common import load_context, load_manifests

from hawk_derm.config import path_from
from hawk_derm.data.external import external_inputs
from hawk_derm.evaluation.predictions import arrays_from_prediction_frame
from hawk_derm.features.bank import load_feature_bank
from hawk_derm.inference.bundle import MANIFEST, ProductionBundle, build_bundle, mil_root, production_entries
from hawk_derm.io import read_json, write_json
from hawk_derm.models.experiments import external_dataset, prepare_bag_data

TOLERANCE = 1e-5


def compare(bundle: ProductionBundle, bags, masks, case_ids, saved_path) -> float:
    saved = pd.read_csv(saved_path)
    saved["case_id"] = saved["case_id"].astype(str)
    saved = saved.set_index("case_id").loc[[str(case) for case in case_ids]].reset_index()
    _, expected = arrays_from_prediction_frame(saved, bundle.labels)
    produced = bundle.predict(bags, masks)["probability"]
    return float(np.max(np.abs(produced - expected)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--model", default=None, help="Build only this production model name.")
    args = parser.parse_args()
    config = load_context(args.config)
    cases, _, splits = load_manifests(config)
    max_images = int(config["study"]["max_images_per_case"])
    for name, entry in production_entries().items():
        if args.model not in (None, name):
            continue
        output = path_from(config, "artifacts_dir") / "production" / name
        build_bundle(config, name, entry, output)
        bundle = ProductionBundle.load(output)
        root = mil_root(config, entry["encoder"], entry.get("run_tag", "canonical"))
        bank = load_feature_bank(path_from(config, "artifacts_dir") / "features" / entry["encoder"] / "feature_bank.npz")
        arrays, split_values = prepare_bag_data(bank, cases, splits, config["labels"], max_images)
        test = split_values == "test"
        checks = {"test": compare(bundle, arrays["bags"][test], arrays["masks"][test], arrays["case_ids"][test], root / "ensemble" / "test_predictions.csv")}
        for item in external_inputs(config, entry["encoder"]):
            saved = root / "ensemble" / f"{item.name}_predictions.csv"
            if saved.is_file():
                dataset = external_dataset(item, config["labels"], max_images)
                checks[item.name] = compare(bundle, dataset.bags.numpy(), dataset.masks.numpy(), dataset.case_ids, saved)
        passed = all(value <= TOLERANCE for value in checks.values())
        manifest = read_json(output / MANIFEST)
        manifest["self_check"] = {"passed": passed, "max_abs_difference": checks, "tolerance": TOLERANCE}
        write_json(output / MANIFEST, manifest)
        print(f"[production] {name}: bundle {'PASSED' if passed else 'FAILED'} self-check {checks} -> {output}")
        if not passed:
            raise RuntimeError(f"{name}: bundle does not reproduce the evaluated ensemble predictions")


if __name__ == "__main__":
    main()
