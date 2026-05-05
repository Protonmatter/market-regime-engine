"""Tests for the purged walk-forward and CPCV splitters."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.walk_forward import (
    CombinatorialPurgedCV,
    PurgedWalkForward,
    evaluate_walk_forward,
)


def test_purged_walk_forward_emits_folds_with_purge_gap():
    n = 200
    splitter = PurgedWalkForward(min_train=60, step=10, horizon=12, embargo=2, test_block=1)
    folds = list(splitter.split(n))
    assert len(folds) > 0
    for f in folds:
        # train indices must all be at least `horizon` periods before the test point
        assert f.train_idx.max() <= f.test_idx.min() - 12
        # train indices must not overlap test indices
        assert set(f.train_idx).isdisjoint(set(f.test_idx))


def test_purged_walk_forward_respects_embargo():
    n = 100
    splitter = PurgedWalkForward(min_train=30, step=5, horizon=3, embargo=4, test_block=1)
    folds = list(splitter.split(n))
    assert len(folds) >= 2
    test_points = [int(f.test_idx[0]) for f in folds]
    for prev, curr in itertools.pairwise(test_points):
        # the next test point should leave at least the embargo gap after the prior one
        assert curr - prev >= 4


def test_purged_walk_forward_skips_when_train_too_small_after_purge():
    n = 50
    splitter = PurgedWalkForward(min_train=30, step=1, horizon=20, embargo=0)
    folds = list(splitter.split(n))
    # horizon eats too much of the prefix, so very few folds should be emitted
    assert len(folds) < 30


def test_cpcv_emits_combinatorial_folds():
    cpcv = CombinatorialPurgedCV(n_blocks=5, k_test_blocks=2, horizon=2, embargo=1)
    folds = list(cpcv.split(60))
    # 5 choose 2 = 10
    assert len(folds) == 10
    # every test set should be the union of two blocks (but allow purge-removed train rows)
    for f in folds:
        assert f.n_test > 0
        assert f.n_train > 0
        assert set(f.train_idx).isdisjoint(set(f.test_idx))


def test_evaluate_walk_forward_runs_through():
    rng = np.random.default_rng(0)
    n = 120
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=idx)
    y = pd.Series((X["a"] + 0.5 * X["b"] > 0).astype(float), index=idx)
    splitter = PurgedWalkForward(min_train=60, step=4, horizon=3, embargo=1, test_block=1)

    def predict_fn(Xtr, ytr, Xte):
        # return the fraction of historical positives as a baseline
        return np.full(len(Xte), float(ytr.mean()))

    out = evaluate_walk_forward(
        X,
        y,
        splitter=splitter,
        predict_fn=predict_fn,
        target="dummy",
        horizon="3m",
    )
    assert not out.empty
    assert {"date", "fold", "y", "p"}.issubset(out.columns)
    assert (out["p"].between(0, 1)).all()


def test_evaluate_walk_forward_propagates_predict_errors():
    n = 80
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    X = pd.DataFrame({"a": np.zeros(n)}, index=idx)
    y = pd.Series(np.zeros(n), index=idx)
    splitter = PurgedWalkForward(min_train=30, step=10, horizon=1, embargo=0)

    def predict_fn(Xtr, ytr, Xte):
        raise ValueError("boom")

    with pytest.raises(RuntimeError):
        evaluate_walk_forward(X, y, splitter=splitter, predict_fn=predict_fn, target="t", horizon="1m")
