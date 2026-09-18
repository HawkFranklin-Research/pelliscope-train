#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd
import torch

from _common import load_context, load_manifests

from hawk_derm.config import load_yaml, path_from
from hawk_derm.evaluation.metrics import multilabel_metrics, per_class_metrics
from hawk_derm.evaluation.predictions import prediction_frame, save_predictions
from hawk_derm.features.bank import load_feature_bank
from hawk_derm.io import write_csv, write_json
from hawk_derm.models.experiments import choose_device, prepare_bag_data, subset_dataset
from hawk_derm.models.mil import predict_mil, train_mil_fixed


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the frozen final MIL model on train plus validation, then evaluate locked test once.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, required=True, help="Prespecified epoch count from development runs.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_context(args.config)
    mil = load_yaml("configs/mil.yaml")
    cases, _, splits = load_manifests(config)
    bank = load_feature_bank(path_from(config, "artifacts_dir") / "features" / args.encoder / "feature_bank.npz")
    arrays, split_values = prepare_bag_data(bank, cases, splits, config["labels"], int(config["study"]["max_images_per_case"]))
    development = subset_dataset(arrays, split_values != "test")
    test = subset_dataset(arrays, split_values == "test")
    device = choose_device(args.device)
    result = train_mil_fixed(
        development,
        bank.dimension,
        len(config["labels"]),
        mil["architecture"],
        mil["training"],
        device=device,
        seed=args.seed,
        epochs=args.epochs,
    )
    output = path_from(config, "artifacts_dir") / "models" / "mil" / args.encoder / "final_model"
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "architecture": mil["architecture"],
            "input_dim": bank.dimension,
            "class_count": len(config["labels"]),
            "labels": config["labels"],
            "seed": args.seed,
            "epochs": args.epochs,
            "development_includes": ["train", "validation"],
        },
        output / "model.pt",
    )
    probabilities, attention = predict_mil(result.model, test, device=device, batch_size=int(mil["training"]["batch_size"]))
    frame = prediction_frame(test.case_ids, test.labels.numpy(), probabilities, config["labels"], split="test", seed=args.seed, model=f"{args.encoder}+mil:final")
    for slot in range(attention.shape[1]):
        frame[f"attention__slot_{slot + 1}"] = attention[:, slot]
    save_predictions(output / "test_predictions.csv", frame, config["labels"], {"evaluation_unit": "case", "locked_test_evaluation": True})
    write_csv(output / "test_overall_metrics.csv", pd.DataFrame([multilabel_metrics(test.labels.numpy(), probabilities)]))
    write_csv(output / "test_per_class_metrics.csv", per_class_metrics(test.labels.numpy(), probabilities, config["labels"]))
    write_json(output / "history.json", {"history": result.history, "complete": True})


if __name__ == "__main__":
    main()
