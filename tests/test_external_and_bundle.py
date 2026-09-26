import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from hawk_derm.config import load_yaml
from hawk_derm.constants import slugify
from hawk_derm.data.external import ExternalInput, build_external_manifests, load_external_cohort
from hawk_derm.evaluation.calibration import apply_platt, fit_platt
from hawk_derm.evaluation.external import external_metrics, first_of_each_group, group_resample_indices
from hawk_derm.features.bank import FeatureBank
from hawk_derm.io import sha256_file, write_json
from hawk_derm.models.classical import run_classical_experiment
from hawk_derm.models.experiments import run_mil_experiment

LABELS = ["Alpha", "Beta", "Gamma", "Delta"]


def make_cohort(tmp_path, schema_labels=LABELS):
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "metadata").mkdir()
    rows = []
    for index, label_index in enumerate([0, 0, 1, 2, 2]):
        path = root / "images" / f"img_{index}.jpg"
        payload = b"same" if index in (3, 4) else f"image-{index}".encode()
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        rows.append(
            {
                "image_id": f"img_{index}", "pseudo_case_id": f"case_{index}", "image_path": f"images/img_{index}.jpg",
                "label_index_25": label_index, "sha256": digest, "duplicate_group_id": f"sha256_{digest[:16]}",
            }
        )
    pd.DataFrame(rows).to_csv(root / "metadata" / "image_manifest.csv", index=False)
    schema = {"labels": [{"index": i, "label": label} for i, label in enumerate(schema_labels)]}
    (root / "metadata" / "label_schema_25class.json").write_text(json.dumps(schema))
    config = {
        "cohort": {"name": "demo"},
        "paths": {
            "dataset_root": str(root), "image_manifest": "metadata/image_manifest.csv", "case_manifest": "metadata/case_manifest.csv",
            "label_schema": "metadata/label_schema_25class.json", "output_root": str(tmp_path / "out"),
        },
        "expected": {"images": 5, "unique_images": 4, "represented_labels": 3},
    }
    config_path = tmp_path / "external.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return load_external_cohort(config_path)


def test_external_manifests_follow_pipeline_schema(tmp_path) -> None:
    cases, images, summary = build_external_manifests(make_cohort(tmp_path), LABELS)
    assert summary["images"] == 5 and summary["unique_image_contents"] == 4 and summary["represented_labels"] == ["Alpha", "Beta", "Gamma"]
    assert cases[[f"is_{slugify(label)}" for label in LABELS]].sum(axis=1).eq(1).all()
    assert images["bag_slot"].eq(0).all() and images["retained_for_bag"].all()
    assert cases.loc[3, "resampling_group"] == cases.loc[4, "resampling_group"]


def test_external_schema_order_must_match(tmp_path) -> None:
    with pytest.raises(ValueError, match="order differs"):
        build_external_manifests(make_cohort(tmp_path, ["Beta", "Alpha", "Gamma", "Delta"]), LABELS)


def test_external_metrics_and_group_resampling() -> None:
    truth = np.eye(4, dtype=np.uint8)[[0, 0, 1, 2, 2]]
    perfect = truth * 0.9 + 0.05
    metrics = external_metrics(truth, perfect, thresholds=np.full(4, 0.5))
    assert metrics["macro_auc_represented"] == 1.0 and metrics["top1_accuracy"] == 1.0
    assert metrics["represented_labels"] == 3 and metrics["off_target_alarm_rate"] == 0.0
    groups = np.array(["a", "b", "c", "d", "d"])
    for sample in group_resample_indices(groups, 20, 0):
        assert (3 in sample) == (4 in sample)
    assert first_of_each_group(groups).tolist() == [0, 1, 2, 3]


def synthetic_study(tmp_path, cases_per_split=(30, 12, 12)):
    rng = np.random.default_rng(0)
    rows, splits, embeddings, case_ids = [], [], [], []
    for split, count in zip(("train", "validation", "test"), cases_per_split, strict=True):
        for index in range(count):
            case = f"{split}_{index}"
            label = index % 3
            rows.append({"case_id": case, **{f"is_{slugify(name)}": int(position == label) for position, name in enumerate(LABELS)}})
            splits.append({"case_id": case, "split": split})
            for _ in range(1 + index % 2):
                vector = rng.normal(size=6)
                vector[label] += 3
                embeddings.append(vector)
                case_ids.append(case)
    bank = FeatureBank(
        embeddings=np.asarray(embeddings, dtype=np.float32), case_ids=np.asarray(case_ids), image_ids=np.asarray([f"i{n}" for n in range(len(case_ids))]),
        image_paths=np.asarray([""] * len(case_ids)), image_sha256=np.asarray([""] * len(case_ids)), encoder="toy", model_id="toy", revision="r", dimension=6,
    )
    external_cases = pd.DataFrame(
        [{"case_id": f"ext_{i}", **{f"is_{slugify(name)}": int(position == i % 3) for position, name in enumerate(LABELS)}} for i in range(6)]
    )
    external_vectors = rng.normal(size=(6, 6)).astype(np.float32)
    external_bank = FeatureBank(
        embeddings=external_vectors, case_ids=external_cases["case_id"].to_numpy(), image_ids=np.asarray([f"e{i}" for i in range(6)]),
        image_paths=np.asarray([""] * 6), image_sha256=np.asarray([""] * 6), encoder="toy", model_id="toy", revision="r", dimension=6,
    )
    external = ExternalInput("external_demo", external_cases, external_bank, "bank-sha", "cases-sha")
    return pd.DataFrame(rows), pd.DataFrame(splits), bank, external


MIL_CONFIG = {
    "architecture": {"instance_dim": 8, "attention_dim": 4, "shared_dim": 6},
    "training": {
        "epochs": 2, "batch_size": 8, "learning_rate": 1e-3, "weight_decay": 1e-4, "loss": "bce", "positive_weight_power": 0.5,
        "prior_bias_init": True, "checkpoint_metric": "macro_micro_ap",
    },
}


def test_mil_seed_scores_external_cohort_and_fingerprints_it(tmp_path) -> None:
    cases, splits, bank, external = synthetic_study(tmp_path)
    provenance = {"feature_bank_sha256": "a", "case_manifest_sha256": "b"}
    summary = run_mil_experiment(bank, cases, splits, LABELS, MIL_CONFIG, tmp_path / "seed", seed=1, device="cpu", provenance=provenance, external=[external])
    frame = pd.read_csv(tmp_path / "seed" / "external_demo_predictions.csv")
    assert len(frame) == 6 and summary["external_cohorts"] == ["external_demo"]
    without = run_mil_experiment(bank, cases, splits, LABELS, MIL_CONFIG, tmp_path / "seed_no_external", seed=1, device="cpu", provenance=provenance)
    assert without["experiment_fingerprint"] != summary["experiment_fingerprint"]
    cv_fold = run_mil_experiment(
        bank, cases, splits, LABELS, MIL_CONFIG, tmp_path / "fold", seed=1, device="cpu", evaluation_splits=("train", "validation"), external=[external]
    )
    assert cv_fold["external_cohorts"] == [] and not (tmp_path / "fold" / "external_demo_predictions.csv").exists()


def test_classical_model_scores_external_cohort(tmp_path) -> None:
    cases, splits, bank, external = synthetic_study(tmp_path)
    run_classical_experiment(
        bank, cases, splits, LABELS, "logistic", load_yaml("configs/classifiers.yaml")["classifiers"]["logistic"], tmp_path / "rf", external=[external]
    )
    frame = pd.read_csv(tmp_path / "rf" / "external_demo_case_predictions.csv")
    assert len(frame) == 6 and frame["split"].eq("external_demo").all()


def test_bundle_reproduces_ensemble_predictions(tmp_path) -> None:
    from hawk_derm.inference.bundle import ProductionBundle, build_bundle
    from hawk_derm.models.experiments import prepare_bag_data

    cases, splits, bank, _ = synthetic_study(tmp_path)
    config = {
        "repository_root": str(tmp_path),
        "paths": {"artifacts_dir": str(tmp_path / "artifacts"), "split_manifest": str(tmp_path / "split.csv")},
        "study": {"run_mode": "full", "max_images_per_case": 3},
        "full": {"seeds": [1, 2]},
        "labels": LABELS,
    }
    splits.to_csv(tmp_path / "split.csv", index=False)
    root = tmp_path / "artifacts" / "models" / "mil" / "siglip2_so400m" / "canonical"
    arrays, split_values = prepare_bag_data(bank, cases, splits, LABELS, 3)
    for seed in (1, 2):
        run_mil_experiment(bank, cases, splits, LABELS, MIL_CONFIG, root / f"seed_{seed}", seed=seed, device="cpu")
    from hawk_derm.evaluation.predictions import arrays_from_prediction_frame

    validation = [arrays_from_prediction_frame(pd.read_csv(root / f"seed_{s}" / "validation_predictions.csv"), LABELS) for s in (1, 2)]
    test = [arrays_from_prediction_frame(pd.read_csv(root / f"seed_{s}" / "test_predictions.csv"), LABELS) for s in (1, 2)]
    parameters = fit_platt(validation[0][0], np.mean([p for _, p in validation], axis=0), LABELS)
    (root / "ensemble").mkdir(parents=True)
    (root / "thresholds").mkdir()
    (root / "tuning").mkdir()
    parameters.to_csv(root / "ensemble" / "calibration_parameters.csv", index=False)
    write_json(root / "ensemble" / "ensemble_manifest.json", {"complete": True})
    pd.DataFrame({"label": LABELS, "threshold": 0.3}).to_csv(root / "thresholds" / "validation_selected_thresholds.csv", index=False)
    pd.DataFrame({"label": LABELS}).to_csv(root / "thresholds" / "test_operating_points.csv", index=False)
    (root / "tuning" / "best_mil_config.yaml").write_text(yaml.safe_dump(MIL_CONFIG))
    import hawk_derm.inference.bundle as bundle_module

    bundle_module.load_encoder_registry = lambda: {"siglip2_so400m": type("Spec", (), {"model_id": "m", "revision": "r", "dimension": 6})()}
    output = build_bundle(config, "demo", {"encoder": "siglip2_so400m", "model": "mil_ensemble", "run_tag": "canonical"}, tmp_path / "bundle")
    bundle = ProductionBundle.load(output)
    mask = split_values == "test"
    produced = bundle.predict(arrays["bags"][mask], arrays["masks"][mask])
    expected = apply_platt(np.mean([p for _, p in test], axis=0), parameters, LABELS)
    assert np.allclose(produced["probability"], expected, atol=1e-5)
    assert produced["decision"].shape == expected.shape
    (output / "models" / "seed_1.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed since packaging"):
        ProductionBundle.load(output)
    assert sha256_file(output / "calibration_parameters.csv")
    assert torch.__version__
