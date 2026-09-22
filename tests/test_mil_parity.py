import numpy as np
from torch import nn

from hawk_derm.models.mil import GatedAttentionMIL, checkpoint_score


def test_sep26_attention_and_head_order() -> None:
    model = GatedAttentionMIL(input_dim=16, class_count=4, instance_dim=8, shared_dim=6, dropout=0.25)
    assert isinstance(model.attention_dropout, nn.Dropout)
    modules = list(model.classifier.children())
    linear_index = next(index for index, module in enumerate(modules) if isinstance(module, nn.Linear))
    assert isinstance(modules[linear_index + 1], nn.LayerNorm)


def test_sep26_checkpoint_score_combines_macro_auc_and_lrap() -> None:
    score = checkpoint_score(
        {"auc_macro": 0.8, "label_ranking_average_precision": 0.5, "validation_loss": 1.0},
        {"checkpoint_metric": "macro_auc_plus_lrap", "checkpoint_lrap_weight": 0.1},
    )
    assert np.isclose(score, 0.85)
