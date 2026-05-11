# SPDX-License-Identifier: Apache-2.0
"""PR-5 ASK-1: searchsorted purge + embargo parity with the legacy dense path.

The v1.3 ``CombinatorialPurgedCV._purge_and_embargo`` built an
``(n_train, n_test)`` bool matrix per fold; on n=2000 / n_blocks=8 / k=2 that
allocated ~32 MB per fold (1 GB aggregate) before any work happened. PR-5
replaces the dense path with :func:`purge_and_embargo_searchsorted` whose
memory bound is ``O(n_train + n_test)``.

This module pins the new path bit-for-bit against the v1.3 dense reference
(``_legacy_purge_and_embargo``) on 50 random seeded inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from market_regime_engine.walk_forward import (
    CombinatorialPurgedCV,
    _legacy_purge_and_embargo,
    purge_and_embargo_searchsorted,
)


def _random_inputs(seed: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    n_total = int(rng.integers(50, 500))
    n_test = int(rng.integers(5, max(6, n_total // 3)))
    test_idx = np.sort(rng.choice(n_total, size=n_test, replace=False))
    train_pool = np.setdiff1d(np.arange(n_total), test_idx)
    n_train_keep = int(rng.integers(10, len(train_pool) + 1))
    train_idx = np.sort(rng.choice(train_pool, size=n_train_keep, replace=False))
    horizon = int(rng.integers(0, 20))
    embargo = int(rng.integers(0, 10))
    return train_idx, test_idx, horizon, embargo


@pytest.mark.parametrize("seed", range(50))
def test_searchsorted_matches_legacy_dense_mask(seed: int) -> None:
    train_idx, test_idx, horizon, embargo = _random_inputs(seed)
    legacy = _legacy_purge_and_embargo(
        train_idx, test_idx, horizon=horizon, embargo=embargo
    )
    new = purge_and_embargo_searchsorted(
        train_idx, test_idx, horizon=horizon, embargo=embargo
    )
    np.testing.assert_array_equal(legacy, new)


def test_searchsorted_handles_empty_inputs() -> None:
    out = purge_and_embargo_searchsorted(
        np.array([], dtype=np.int64),
        np.array([1, 2, 3], dtype=np.int64),
        horizon=2,
        embargo=1,
    )
    assert out.size == 0

    out = purge_and_embargo_searchsorted(
        np.array([1, 2, 3], dtype=np.int64),
        np.array([], dtype=np.int64),
        horizon=2,
        embargo=1,
    )
    np.testing.assert_array_equal(out, np.array([1, 2, 3], dtype=np.int64))


def test_searchsorted_zero_horizon_zero_embargo_only_drops_same_as_test() -> None:
    """With horizon=0 and embargo=0 the window is ``{t}`` so only train rows
    equal to a test index are dropped (the legacy same_as_test mask)."""
    train_idx = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    test_idx = np.array([3, 7], dtype=np.int64)
    out = purge_and_embargo_searchsorted(train_idx, test_idx, horizon=0, embargo=0)
    np.testing.assert_array_equal(out, np.array([1, 2, 4, 5], dtype=np.int64))


def test_combinatorial_purged_cv_routes_through_searchsorted() -> None:
    """End-to-end smoke: CPCV folds match a hand-rolled reference exactly."""
    cpcv = CombinatorialPurgedCV(n_blocks=5, k_test_blocks=2, horizon=3, embargo=2)
    n = 100
    folds = list(cpcv.split(n))
    # 5 choose 2 = 10
    assert len(folds) == 10
    for fold in folds:
        # Compare against the legacy reference for every fold.
        reference = _legacy_purge_and_embargo(
            np.setdiff1d(np.arange(n), fold.test_idx),
            fold.test_idx,
            horizon=3,
            embargo=2,
        )
        np.testing.assert_array_equal(np.sort(fold.train_idx), np.sort(reference))


@pytest.mark.slow
def test_searchsorted_perf_under_large_cpcv() -> None:
    """Performance smoke (slow): the searchsorted path completes a 28-fold,
    n=2000 CPCV run inside 1 second on a modern laptop. The legacy dense
    path would allocate ~1 GB of transient bool matrices; the searchsorted
    path stays below 50 MB. The 1-second budget is intentionally generous
    so CI variance does not flap; the test exists to flag a *regression* in
    the asymptotic complexity, not to certify wall-clock numbers."""
    import time

    cpcv = CombinatorialPurgedCV(n_blocks=8, k_test_blocks=2, horizon=20, embargo=10)
    t0 = time.perf_counter()
    folds = list(cpcv.split(2000))
    elapsed = time.perf_counter() - t0
    assert len(folds) == 28
    assert elapsed < 1.0, f"searchsorted CPCV took {elapsed:.3f}s — regression?"
