import pandas as pd

from hawk_derm.data.splits import assert_no_split_leakage


def test_hash_leakage_is_rejected() -> None:
    splits = pd.DataFrame({"case_id": ["a", "b"], "split": ["train", "test"]})
    audit = pd.DataFrame({"case_id": ["a", "b"], "sha256": ["same", "same"]})
    try:
        assert_no_split_leakage(splits, audit)
    except ValueError:
        return
    raise AssertionError("Expected cross-split duplicate hash to be rejected")
