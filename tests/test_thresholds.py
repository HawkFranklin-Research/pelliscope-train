import numpy as np

from hawk_derm.evaluation.thresholds import select_thresholds, threshold_sweep


def test_balance_threshold_prefers_smallest_tie() -> None:
    truth = np.array([[0], [0], [1], [1]], dtype=np.uint8)
    probability = np.array([[0.1], [0.4], [0.6], [0.9]])
    sweep = threshold_sweep(truth, probability, ["Example"], np.array([0.4, 0.5, 0.6]))
    selected = select_thresholds(sweep, "balance")
    assert selected.iloc[0]["threshold"] == 0.5
