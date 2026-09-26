from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from hawk_derm.evaluation.metrics import multilabel_metrics


class BagDataset(Dataset):
    def __init__(self, bags: np.ndarray, masks: np.ndarray, labels: np.ndarray, case_ids: np.ndarray) -> None:
        self.bags = torch.as_tensor(bags, dtype=torch.float32)
        self.masks = torch.as_tensor(masks, dtype=torch.bool)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.case_ids = np.asarray(case_ids).astype(str)

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        return self.bags[index], self.masks[index], self.labels[index], self.case_ids[index]


class ManifoldResidualProjection(nn.Module):
    """Fixed random projection plus a zero-initialised low-rank trainable residual.

    Adapted from the Manifold Residual block (ICLR 2026), which placed it inside ABMIL's
    attention branches; here it replaces the first instance projection. The anchor is a
    persistent buffer so it is saved with the model and never redrawn at inference.
    """

    def __init__(self, input_dim: int, output_dim: int, rank: int = 32) -> None:
        super().__init__()
        bound = float(np.sqrt(6.0 / input_dim))
        self.register_buffer("anchor", (torch.rand(input_dim, output_dim) * 2 - 1) * bound, persistent=True)
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, output_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs @ self.anchor + self.up(nn.functional.gelu(self.down(inputs)))


class GatedAttentionMIL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        class_count: int,
        instance_dim: int = 768,
        attention_dim: int = 256,
        shared_dim: int = 384,
        instance_layers: int = 1,
        shared_layers: int = 1,
        dropout: float = 0.25,
        instance_projection: str = "dense",
        projection_rank: int = 32,
    ) -> None:
        super().__init__()
        if instance_layers < 1:
            raise ValueError("instance_layers must be at least 1")
        if shared_layers < 0:
            raise ValueError("shared_layers cannot be negative")
        if instance_projection not in {"dense", "manifold_residual"}:
            raise ValueError(f"Unknown instance projection: {instance_projection}")
        instance_modules: list[nn.Module] = []
        current_dim = input_dim
        for layer in range(instance_layers):
            projection: nn.Module = (
                ManifoldResidualProjection(current_dim, instance_dim, int(projection_rank))
                if layer == 0 and instance_projection == "manifold_residual"
                else nn.Linear(current_dim, instance_dim)
            )
            instance_modules.extend([projection, nn.LayerNorm(instance_dim), nn.GELU(), nn.Dropout(dropout)])
            current_dim = instance_dim
        self.instance = nn.Sequential(*instance_modules)
        self.attention_tanh = nn.Linear(instance_dim, attention_dim)
        self.attention_sigmoid = nn.Linear(instance_dim, attention_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_score = nn.Linear(attention_dim, 1)
        classifier_modules: list[nn.Module] = []
        current_dim = instance_dim
        for _ in range(shared_layers):
            classifier_modules.extend(
                [nn.Linear(current_dim, shared_dim), nn.LayerNorm(shared_dim), nn.GELU(), nn.Dropout(dropout)]
            )
            current_dim = shared_dim
        classifier_modules.append(nn.Linear(current_dim, class_count))
        self.classifier = nn.Sequential(*classifier_modules)

    def forward(self, bags: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        instances = self.instance(bags)
        gated = torch.tanh(self.attention_tanh(instances)) * torch.sigmoid(self.attention_sigmoid(instances))
        scores = self.attention_score(self.attention_dropout(gated)).squeeze(-1)
        scores = scores.masked_fill(~masks, torch.finfo(instances.dtype).min)
        attention = torch.softmax(scores, dim=1)
        attention = torch.where(masks, attention, torch.zeros_like(attention))
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-12)
        pooled = torch.sum(instances * attention.unsqueeze(-1), dim=1)
        return self.classifier(pooled), attention


def effective_number_weights(labels: np.ndarray, beta: float) -> torch.Tensor:
    counts = np.asarray(labels).sum(axis=0).clip(min=1)
    weights = (1 - beta) / (1 - np.power(beta, counts))
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)


class MultilabelLoss(nn.Module):
    def __init__(
        self,
        labels: np.ndarray,
        kind: str,
        beta: float = 0.999,
        gamma: float = 2.0,
        alpha: float = 0.25,
        use_positive_weights: bool = True,
        positive_weight_power: float = 1.0,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer("class_weights", effective_number_weights(labels, beta))
        positives = np.asarray(labels).sum(axis=0)
        negatives = len(labels) - positives
        # Power 1 is the original negatives/positives weight; 0.5 is its square root; 0 disables it.
        positive_weights = (
            np.power(negatives / np.clip(positives, 1, None), float(positive_weight_power))
            if use_positive_weights
            else np.ones_like(positives, dtype=float)
        )
        self.register_buffer("positive_weights", torch.as_tensor(positive_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.positive_weights, reduction="none")
        if "focal" in self.kind:
            probability = torch.sigmoid(logits)
            p_t = probability * targets + (1 - probability) * (1 - targets)
            alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = loss * (1 - p_t).pow(self.gamma) * alpha_factor
        if "class_balanced" in self.kind:
            loss = loss * self.class_weights.unsqueeze(0)
        return loss.mean()


@dataclass
class TrainingResult:
    model: GatedAttentionMIL
    history: list[dict[str, float]]
    best_epoch: int
    best_score: float
    checkpoint_metric: str


def _epoch(
    model: GatedAttentionMIL,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    losses = []
    correct = 0
    decisions = 0
    for bags, masks, labels, _ in loader:
        bags, masks, labels = bags.to(device), masks.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits, _ = model(bags, masks)
            loss = loss_function(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if gradient_clip_norm:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        correct += int(((logits >= 0) == labels.bool()).sum().detach().cpu())
        decisions += int(labels.numel())
    loss_value = float(np.mean(losses)) if losses else float("nan")
    accuracy = correct / decisions if decisions else float("nan")
    return loss_value, accuracy


def _prediction_metrics(
    model: GatedAttentionMIL,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    truths: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    model.eval()
    for bags, masks, labels, _ in loader:
        with torch.inference_mode():
            logits, _ = model(bags.to(device), masks.to(device))
        truths.append(labels.numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    if not truths:
        return {}
    truth, probability = np.concatenate(truths), np.concatenate(probabilities)
    return {**multilabel_metrics(truth, probability), "log_loss_unweighted": unweighted_log_loss(truth, probability)}


def unweighted_log_loss(truth: np.ndarray, probability: np.ndarray, eps: float = 1e-7) -> float:
    """Mean binary log-loss over all case-label pairs, without class or positive weights."""
    clipped = np.clip(probability, eps, 1 - eps)
    return float(-np.mean(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped)))


def checkpoint_score(metrics: dict[str, float], training: dict[str, Any]) -> float:
    metric = str(training.get("checkpoint_metric", "macro_auc_plus_lrap"))
    if metric == "validation_loss":
        return -float(metrics["validation_loss"])
    if metric == "macro_auc":
        return float(metrics.get("auc_macro", float("-inf")))
    if metric == "macro_auc_plus_lrap":
        auc = float(metrics.get("auc_macro", float("-inf")))
        lrap = float(metrics.get("label_ranking_average_precision", 0.0))
        return auc + float(training.get("checkpoint_lrap_weight", 0.1)) * lrap
    if metric == "macro_micro_ap":
        # Macro AP ranks cases within each label; micro AP ranks all case-label pairs together.
        # Unlike LRAP, both penalise false positives on cases with no target label.
        macro = float(metrics.get("pr_auc_macro", float("nan")))
        micro = float(metrics.get("pr_auc_micro", float("nan")))
        if not np.isfinite(macro) or not np.isfinite(micro):
            return float("-inf")
        weight = float(training.get("checkpoint_macro_ap_weight", 0.5))
        return weight * macro + (1 - weight) * micro
    raise ValueError(f"Unknown MIL checkpoint metric: {metric}")


def initialize_prior_bias(model: GatedAttentionMIL, labels: np.ndarray) -> None:
    """Start each output at its training prevalence so rare labels do not begin near 0.5."""
    prevalence = np.clip(np.asarray(labels, dtype=float).mean(axis=0), 1e-4, 1 - 1e-4)
    output = model.classifier[-1]
    with torch.no_grad():
        output.bias.copy_(torch.as_tensor(np.log(prevalence / (1 - prevalence)), dtype=output.bias.dtype))


def _build_model_and_loss(
    train_data: BagDataset,
    input_dim: int,
    class_count: int,
    architecture: dict[str, Any],
    training: dict[str, Any],
    device: torch.device,
) -> tuple[GatedAttentionMIL, nn.Module]:
    model = GatedAttentionMIL(input_dim=input_dim, class_count=class_count, **architecture)
    train_labels = train_data.labels.numpy()
    if training.get("prior_bias_init", False):
        initialize_prior_bias(model, train_labels)
    loss_function = MultilabelLoss(
        train_labels,
        training.get("loss", "class_balanced"),
        training.get("class_balanced_beta", 0.999),
        training.get("focal_gamma", 2.0),
        training.get("focal_alpha", 0.25),
        training.get("use_positive_weights", True),
        training.get("positive_weight_power", 1.0),
    )
    return model.to(device), loss_function.to(device)


def _scheduler(
    optimizer: torch.optim.Optimizer,
    training: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = str(training.get("scheduler", "cosine")).lower()
    if name in {"none", "constant"}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(training["epochs"])),
        )
    raise ValueError(f"Unknown MIL learning-rate scheduler: {name}")


def _optimizer(model: nn.Module, training: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(training.get("optimizer", "adamw")).lower()
    if name != "adamw":
        raise ValueError(f"Unknown MIL optimizer: {name}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )


def train_mil(
    train_data: BagDataset,
    validation_data: BagDataset,
    input_dim: int,
    class_count: int,
    architecture: dict[str, Any],
    training: dict[str, Any],
    *,
    device: str,
    seed: int,
) -> TrainingResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    target_device = torch.device(device)
    model, loss_function = _build_model_and_loss(train_data, input_dim, class_count, architecture, training, target_device)
    optimizer = _optimizer(model, training)
    scheduler = _scheduler(optimizer, training)
    train_loader = DataLoader(train_data, batch_size=int(training["batch_size"]), shuffle=True, num_workers=int(training.get("num_workers", 0)))
    validation_loader = DataLoader(validation_data, batch_size=int(training["batch_size"]), shuffle=False, num_workers=int(training.get("num_workers", 0)))
    best_state = None
    best_score = float("-inf")
    best_log_loss = float("inf")
    best_epoch = 0
    patience = 0
    history = []
    raw_scores: list[float] = []
    # Defaults (smoothing 1, tolerance 0) reproduce the original strict best-score rule.
    smoothing = max(1, int(training.get("checkpoint_smoothing", 1)))
    tolerance = float(training.get("checkpoint_tie_tolerance", 0.0))
    for epoch in range(1, int(training["epochs"]) + 1):
        train_loss, train_accuracy = _epoch(
            model, train_loader, loss_function, target_device, optimizer, training.get("gradient_clip_norm")
        )
        validation_loss, validation_accuracy = _epoch(model, validation_loader, loss_function, target_device)
        validation_metrics = _prediction_metrics(model, validation_loader, target_device)
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        score_inputs = {**validation_metrics, "validation_loss": validation_loss}
        raw_scores.append(checkpoint_score(score_inputs, training))
        score = float(np.mean(raw_scores[-smoothing:]))
        log_loss = float(validation_metrics.get("log_loss_unweighted", float("inf")))
        row["checkpoint_score_raw"] = raw_scores[-1]
        row["checkpoint_score"] = score
        history.append(row)
        improved = score > best_score + tolerance or (
            tolerance > 0 and abs(score - best_score) <= tolerance and log_loss < best_log_loss
        )
        if improved:
            best_score = score
            best_log_loss = log_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if scheduler is not None:
            scheduler.step()
        if patience >= int(training.get("early_stopping_patience", 8)):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_score=best_score,
        checkpoint_metric=str(training.get("checkpoint_metric", "macro_auc_plus_lrap")),
    )


def train_mil_fixed(
    train_data: BagDataset,
    input_dim: int,
    class_count: int,
    architecture: dict[str, Any],
    training: dict[str, Any],
    *,
    device: str,
    seed: int,
    epochs: int,
) -> TrainingResult:
    """Fit a frozen final model for a prespecified number of epochs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    target_device = torch.device(device)
    model, loss_function = _build_model_and_loss(train_data, input_dim, class_count, architecture, training, target_device)
    optimizer = _optimizer(model, training)
    fixed_training = {**training, "epochs": epochs}
    scheduler = _scheduler(optimizer, fixed_training)
    loader = DataLoader(
        train_data,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training.get("num_workers", 0)),
    )
    history = []
    for epoch in range(1, epochs + 1):
        loss, accuracy = _epoch(model, loader, loss_function, target_device, optimizer, training.get("gradient_clip_norm"))
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": loss,
                "train_accuracy": accuracy,
            }
        )
        if scheduler is not None:
            scheduler.step()
    return TrainingResult(
        model=model,
        history=history,
        best_epoch=epochs,
        best_score=float("nan"),
        checkpoint_metric="fixed_epochs",
    )


def predict_mil(model: GatedAttentionMIL, data: BagDataset, *, device: str, batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    target_device = torch.device(device)
    model.to(target_device).eval()
    probabilities, attentions = [], []
    for bags, masks, _, _ in DataLoader(data, batch_size=batch_size, shuffle=False):
        with torch.inference_mode():
            logits, attention = model(bags.to(target_device), masks.to(target_device))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        attentions.append(attention.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(attentions)
