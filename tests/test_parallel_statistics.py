import numpy as np

from hawk_derm.statistics.bootstrap import paired_case_bootstrap, paired_case_permutation


def test_parallel_bootstrap_matches_single_worker() -> None:
    rng = np.random.default_rng(7)
    truth = (rng.random((30, 3)) > 0.6).astype(np.uint8)
    first = rng.random((30, 3))
    second = rng.random((30, 3))

    single = paired_case_bootstrap(
        truth, first, second, metric="roc_auc", replicates=20, seed=9, workers=1
    )
    parallel = paired_case_bootstrap(
        truth, first, second, metric="roc_auc", replicates=20, seed=9, workers=2
    )
    np.testing.assert_array_equal(single.distribution, parallel.distribution)


def test_parallel_permutation_matches_single_worker() -> None:
    rng = np.random.default_rng(8)
    truth = (rng.random((30, 3)) > 0.6).astype(np.uint8)
    first = rng.random((30, 3))
    second = rng.random((30, 3))

    single_p, single_distribution = paired_case_permutation(
        truth, first, second, metric="roc_auc", replicates=20, seed=10, workers=1
    )
    parallel_p, parallel_distribution = paired_case_permutation(
        truth, first, second, metric="roc_auc", replicates=20, seed=10, workers=2
    )
    assert single_p == parallel_p
    np.testing.assert_array_equal(single_distribution, parallel_distribution)
