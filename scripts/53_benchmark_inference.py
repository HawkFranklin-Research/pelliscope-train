#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import time

import numpy as np
import pandas as pd
import torch

from _common import load_context, load_manifests

from hawk_derm.config import path_from
from hawk_derm.features.bank import load_feature_bank
from hawk_derm.io import write_csv, write_json
from hawk_derm.models.experiments import choose_device, prepare_bag_data, subset_dataset
from hawk_derm.models.mil import GatedAttentionMIL


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark frozen MIL-head inference over precomputed case bags.")
    parser.add_argument("--config", default="configs/study_25class.yaml")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-sizes", default="1,3,10")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    config = load_context(args.config)
    cases, _, splits = load_manifests(config)
    bank = load_feature_bank(path_from(config, "artifacts_dir") / "features" / args.encoder / "feature_bank.npz")
    arrays, split_values = prepare_bag_data(bank, cases, splits, config["labels"], int(config["study"]["max_images_per_case"]))
    test = subset_dataset(arrays, split_values == "test")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = GatedAttentionMIL(
        input_dim=int(checkpoint["input_dim"]),
        class_count=int(checkpoint["class_count"]),
        **checkpoint["architecture"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = choose_device(args.device)
    model.to(device).eval()
    rows = []
    for batch_size in [int(value) for value in args.batch_sizes.split(",")]:
        count = min(batch_size, len(test))
        bags = test.bags[:count].to(device)
        masks = test.masks[:count].to(device)
        durations = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            with torch.inference_mode():
                model(bags, masks)
            if device == "cuda":
                torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
        rows.append(
            {
                "encoder": args.encoder,
                "batch_cases": count,
                "device": device,
                "median_seconds": float(np.median(durations)),
                "mean_seconds": float(np.mean(durations)),
                "std_seconds": float(np.std(durations, ddof=1)) if len(durations) > 1 else 0.0,
                "repeats": args.repeats,
            }
        )
    output = path_from(config, "reports_dir") / "tables" / f"inference_latency_{platform.system().lower()}_{args.encoder}.csv"
    write_csv(output, pd.DataFrame(rows))
    write_json(output.with_suffix(".metadata.json"), {"platform": platform.platform(), "torch": torch.__version__, "complete": True})


if __name__ == "__main__":
    main()
