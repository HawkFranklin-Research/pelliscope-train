import numpy as np

from hawk_derm.evaluation.metrics import multilabel_metrics, per_class_metrics


def test_perfect_multilabel_predictions() -> None:
    truth = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.uint8)
    probability = truth * 0.8 + (1 - truth) * 0.2
    overall = multilabel_metrics(truth, probability)
    assert overall["auc_macro"] == 1.0
    assert overall["subset_accuracy"] == 1.0
    per_class = per_class_metrics(truth, probability, ["A", "B"])
    assert (per_class["sensitivity"] == 1.0).all()
    assert (per_class["specificity"] == 1.0).all()
