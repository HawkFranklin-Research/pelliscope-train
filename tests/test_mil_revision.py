import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from hawk_derm.evaluation.calibration import apply_platt, fit_platt
from hawk_derm.models.experiments import sample_search_configs
from hawk_derm.models.mil import (
    BagDataset,
    GatedAttentionMIL,
    ManifoldResidualProjection,
    MultilabelLoss,
    checkpoint_score,
    initialize_prior_bias,
    train_mil,
)


def test_positive_weight_power_scales_original_weights() -> None:
    labels = np.array([[1, 0], [0, 0], [0, 0], [0, 1]] * 25)
    original = MultilabelLoss(labels, "bce", use_positive_weights=True, positive_weight_power=1.0).positive_weights
    root = MultilabelLoss(labels, "bce", use_positive_weights=True, positive_weight_power=0.5).positive_weights
    off = MultilabelLoss(labels, "bce", use_positive_weights=True, positive_weight_power=0.0).positive_weights
    assert torch.allclose(root, original.sqrt())
    assert torch.allclose(off, torch.ones_like(off))


def test_prior_bias_starts_outputs_at_training_prevalence() -> None:
    labels = np.zeros((100, 3))
    labels[:2, 0], labels[:20, 1], labels[:50, 2] = 1, 1, 1
    model = GatedAttentionMIL(input_dim=8, class_count=3, instance_dim=8, attention_dim=4, shared_dim=6)
    initialize_prior_bias(model, labels)
    assert np.allclose(torch.sigmoid(model.classifier[-1].bias).detach().numpy(), [0.02, 0.2, 0.5], atol=1e-6)


def test_macro_micro_ap_checkpoint_score() -> None:
    training = {"checkpoint_metric": "macro_micro_ap", "checkpoint_macro_ap_weight": 0.5}
    assert np.isclose(checkpoint_score({"pr_auc_macro": 0.3, "pr_auc_micro": 0.2}, training), 0.25)
    assert checkpoint_score({"pr_auc_macro": float("nan"), "pr_auc_micro": 0.2}, training) == float("-inf")


def test_manifold_residual_anchor_is_frozen_and_saved() -> None:
    torch.manual_seed(0)
    layer = ManifoldResidualProjection(16, 8, rank=4)
    inputs = torch.randn(5, 16)
    assert torch.allclose(layer(inputs), inputs @ layer.anchor)
    assert "anchor" in layer.state_dict()
    assert all("anchor" not in name for name, _ in layer.named_parameters())
    model = GatedAttentionMIL(input_dim=16, class_count=3, instance_dim=8, instance_projection="manifold_residual", projection_rank=4)
    assert isinstance(model.instance[0], ManifoldResidualProjection)


def test_platt_preserves_per_label_ranking_and_moves_scale_toward_prevalence() -> None:
    rng = np.random.default_rng(0)
    truth = (rng.random((600, 3)) < [0.03, 0.1, 0.3]).astype(float)
    inflated = np.clip(0.4 + 0.3 * truth + 0.2 * rng.random((600, 3)), 0, 1)
    labels = ["a", "b", "c"]
    calibrated = apply_platt(inflated, fit_platt(truth, inflated, labels), labels)
    for index in range(3):
        assert np.isclose(
            average_precision_score(truth[:, index], inflated[:, index]),
            average_precision_score(truth[:, index], calibrated[:, index]),
        )
        assert np.isclose(roc_auc_score(truth[:, index], inflated[:, index]), roc_auc_score(truth[:, index], calibrated[:, index]))
    assert abs(calibrated.mean(axis=0) - truth.mean(axis=0)).max() < abs(inflated.mean(axis=0) - truth.mean(axis=0)).max()


def test_search_without_revision_keys_is_unchanged() -> None:
    base = {"architecture": {}, "training": {}}
    search = {"learning_rate": [1e-5, 1e-3], "weight_decay": [1e-6, 1e-3], "dropout": [0.1, 0.2], "loss": ["bce", "focal"]}
    original = sample_search_configs(base, search, 4, 7)
    assert all("positive_weight_power" not in trial["training"] for trial in original)
    assert all("instance_projection" not in trial["architecture"] for trial in original)
    extended = sample_search_configs(base, {**search, "positive_weight_power": [0.0, 0.5]}, 4, 7)
    assert original[0]["training"]["learning_rate"] == extended[0]["training"]["learning_rate"]
    assert all(trial["training"]["positive_weight_power"] in {0.0, 0.5} for trial in extended)


def test_revised_training_runs_and_records_smoothed_score() -> None:
    rng = np.random.default_rng(1)
    labels = (rng.random((80, 3)) < 0.3).astype(np.float32)
    bags = rng.normal(size=(80, 3, 8)).astype(np.float32) + labels[:, None, :].repeat(3, 1).mean(-1, keepdims=True)
    masks = np.ones((80, 3), dtype=bool)
    masks[::3, 1:] = False
    ids = np.arange(80).astype(str)
    train, validation = BagDataset(bags[:60], masks[:60], labels[:60], ids[:60]), BagDataset(bags[60:], masks[60:], labels[60:], ids[60:])
    training = {
        "epochs": 4, "batch_size": 16, "learning_rate": 1e-3, "weight_decay": 1e-4, "loss": "bce",
        "positive_weight_power": 0.5, "prior_bias_init": True, "checkpoint_metric": "macro_micro_ap",
        "checkpoint_smoothing": 3, "checkpoint_tie_tolerance": 0.002,
    }
    architecture = {"instance_dim": 8, "attention_dim": 4, "shared_dim": 6, "instance_projection": "manifold_residual", "projection_rank": 2}
    result = train_mil(train, validation, 8, 3, architecture, training, device="cpu", seed=0)
    history = result.history
    assert np.isclose(history[2]["checkpoint_score"], np.mean([row["checkpoint_score_raw"] for row in history[:3]]))
    assert "validation_log_loss_unweighted" in history[0]
    assert 1 <= result.best_epoch <= len(history)
