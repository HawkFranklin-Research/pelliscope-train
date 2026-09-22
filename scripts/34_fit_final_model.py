#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from _common import load_context, load_manifests, load_selected_mil_config, mil_run_root

from hawk_derm.config import path_from
from hawk_derm.evaluation.metrics import multilabel_metrics, per_class_metrics
from hawk_derm.evaluation.predictions import prediction_frame, save_predictions
from hawk_derm.features.bank import load_feature_bank
from hawk_derm.io import sha256_file, write_csv, write_json
from hawk_derm.models.experiments import choose_device, prepare_bag_data, selected_config_hash, subset_dataset
from hawk_derm.models.mil import predict_mil, train_mil_fixed
from hawk_derm.provenance import RunRecorder


def development_epoch_count(root, requested: int | None, fallback: int) -> tuple[int, list[int]]:
    if requested is not None:
        return requested, [requested]
    epochs: list[int] = []
    cv_path = root / "cross_validation" / "cross_validation_metrics.csv"
    if cv_path.is_file():
        frame = pd.read_csv(cv_path)
        if "selected_epoch" in frame:
            epochs.extend(frame["selected_epoch"].dropna().astype(int).tolist())
    for path in sorted(root.glob("seed_*/history.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("best_epoch"):
            epochs.append(int(payload["best_epoch"]))
    selected = max(1, int(round(float(np.median(epochs))))) if epochs else fallback
    return selected, epochs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a separately labelled final MIL model on train plus validation and export every split."
    )
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None, help="Override the median development-selected epoch count.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mil-config", default=None)
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()

    config = load_context(args.config)
    mil, mil_config_path = load_selected_mil_config(config, args.encoder, args.run_tag, args.mil_config)
    cases, _, splits = load_manifests(config)
    bank_path = path_from(config, "artifacts_dir") / "features" / args.encoder / "feature_bank.npz"
    bank = load_feature_bank(bank_path)
    arrays, split_values = prepare_bag_data(
        bank,
        cases,
        splits,
        config["labels"],
        int(config["study"]["max_images_per_case"]),
    )
    datasets = {name: subset_dataset(arrays, split_values == name) for name in ("train", "validation", "test")}
    development = subset_dataset(arrays, split_values != "test")
    root = mil_run_root(config, args.encoder, args.run_tag)
    epochs, development_epochs = development_epoch_count(root, args.epochs, int(mil["training"]["epochs"]))
    device = choose_device(args.device)
    output = root / "final_model"
    split_path = path_from(config, "split_manifest")
    with RunRecorder(
        "fit_final_mil",
        {**vars(args), "selected_epochs": epochs},
        output,
        inputs=[bank_path, path_from(config, "case_manifest"), split_path, mil_config_path],
    ) as run:
        result = train_mil_fixed(
            development,
            bank.dimension,
            len(config["labels"]),
            mil["architecture"],
            mil["training"],
            device=device,
            seed=args.seed,
            epochs=epochs,
        )
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": result.model.state_dict(),
                "architecture": mil["architecture"],
                "training": mil["training"],
                "input_dim": bank.dimension,
                "class_count": len(config["labels"]),
                "labels": config["labels"],
                "seed": args.seed,
                "epochs": epochs,
                "development_includes": ["train", "validation"],
                "selected_mil_config_sha256": selected_config_hash(mil),
            },
            output / "model.pt",
        )
        metric_rows = []
        for split_name, dataset in datasets.items():
            probabilities, attention = predict_mil(
                result.model,
                dataset,
                device=device,
                batch_size=int(mil["training"]["batch_size"]),
            )
            frame = prediction_frame(
                dataset.case_ids,
                dataset.labels.numpy(),
                probabilities,
                config["labels"],
                split=split_name,
                seed=args.seed,
                model=f"{args.encoder}+mil:final_model",
            )
            for slot in range(attention.shape[1]):
                frame[f"attention__slot_{slot + 1}"] = attention[:, slot]
            save_predictions(
                output / f"{split_name}_predictions.csv",
                frame,
                config["labels"],
                {
                    "evaluation_unit": "case",
                    "model_role": "final_model_separate_from_ten_seed_ensemble",
                    "locked_test_evaluation": split_name == "test",
                    "selected_mil_config_sha256": sha256_file(mil_config_path),
                },
            )
            metrics = multilabel_metrics(dataset.labels.numpy(), probabilities)
            metric_rows.append({"split": split_name, **metrics})
            write_csv(
                output / f"{split_name}_per_class_metrics.csv",
                per_class_metrics(dataset.labels.numpy(), probabilities, config["labels"]),
            )
        write_csv(output / "overall_metrics.csv", pd.DataFrame(metric_rows))
        write_csv(
            output / "test_overall_metrics.csv",
            pd.DataFrame([row for row in metric_rows if row["split"] == "test"]),
        )
        write_json(
            output / "history.json",
            {
                "history": result.history,
                "selected_epochs": epochs,
                "development_selected_epochs": development_epochs,
                "selection_rule": "median_cv_and_repeated_run_best_epoch",
                "complete": True,
            },
        )
        write_json(
            output / "final_model_manifest.json",
            {
                "encoder": args.encoder,
                "run_tag": args.run_tag,
                "run_mode": config["study"]["run_mode"],
                "model_role": "final_model_separate_from_ten_seed_ensemble",
                "epochs": epochs,
                "selected_mil_config": str(mil_config_path),
                "selected_mil_config_sha256": sha256_file(mil_config_path),
                "feature_bank_sha256": sha256_file(bank_path),
                "case_manifest_sha256": sha256_file(path_from(config, "case_manifest")),
                "split_manifest_sha256": sha256_file(split_path),
                "complete": True,
            },
        )
        run.complete(
            [
                output / "model.pt",
                output / "train_predictions.csv",
                output / "validation_predictions.csv",
                output / "test_predictions.csv",
                output / "overall_metrics.csv",
                output / "final_model_manifest.json",
            ]
        )


if __name__ == "__main__":
    main()
