# SPDX-License-Identifier: Apache-2.0
"""PR-5 ASK-13: fresh-per-fold model factory.

The pre-PR-5 ``evaluate_walk_forward`` accepted only a ``predict_fn`` closure
which captured the estimator from the calling scope. If the operator wired
in a stateful estimator (e.g. an sklearn ``Pipeline`` that retains
``coef_``), the same instance was refit on every fold and history bled
across folds via partial-fit-style implementations.

PR-5 adds ``model_class`` / ``model_kwargs`` so the caller can pass a
*class* and have it instantiated fresh per fold. The closure-capture
``predict_fn`` path remains available for back-compat.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.walk_forward import (
    PurgedWalkForward,
    _model_factory_default,
    evaluate_walk_forward,
)


def _synthetic_panel(n: int = 120, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="MS")
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=idx)
    y = pd.Series((X["a"] + 0.5 * X["b"] > 0).astype(float), index=idx)
    return X, y


class _CountingModel:
    """Tiny stand-in for an sklearn estimator that records its construction."""

    # Class-level instance log so the cross-fold-state rail can assert
    # one fresh instance per fold. The mutable default is intentional.
    instances: ClassVar[list[_CountingModel]] = []

    def __init__(self, *, ridge: float = 1.0) -> None:
        self.ridge = ridge
        self.coef_: np.ndarray | None = None
        type(self).instances.append(self)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _CountingModel:
        Xm = X.to_numpy(dtype=float)
        ym = y.to_numpy(dtype=float)
        # Simple closed-form ridge so we can pin the coefficient per fold.
        XtX = Xm.T @ Xm + self.ridge * np.eye(Xm.shape[1])
        self.coef_ = np.linalg.solve(XtX, Xm.T @ ym)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xm = X.to_numpy(dtype=float)
        raw = Xm @ self.coef_
        return 1.0 / (1.0 + np.exp(-raw))


@pytest.fixture(autouse=True)
def _reset_counter() -> None:
    _CountingModel.instances.clear()


def test_model_factory_class_instantiates_fresh_per_fold() -> None:
    X, y = _synthetic_panel(n=120)
    splitter = PurgedWalkForward(min_train=60, step=4, horizon=3, embargo=1)
    out = evaluate_walk_forward(
        X,
        y,
        splitter=splitter,
        target="dummy",
        horizon="3m",
        model_class=_CountingModel,
        model_kwargs={"ridge": 0.5},
    )
    assert not out.empty
    n_folds = int(out["fold"].nunique())
    # One fresh instance per fold; the constructor's instance-list assertion
    # is the cross-fold-state-leakage rail.
    assert len(_CountingModel.instances) == n_folds
    # Every instance carries the requested kwarg.
    assert all(inst.ridge == 0.5 for inst in _CountingModel.instances)


def test_model_factory_legacy_predict_fn_still_works() -> None:
    X, y = _synthetic_panel(n=120)
    splitter = PurgedWalkForward(min_train=60, step=4, horizon=3, embargo=1)
    calls = {"n": 0}

    def predict_fn(Xtr: pd.DataFrame, ytr: pd.Series, Xte: pd.DataFrame) -> np.ndarray:
        calls["n"] += 1
        return np.full(len(Xte), float(ytr.mean()))

    out = evaluate_walk_forward(
        X,
        y,
        splitter=splitter,
        target="dummy",
        horizon="3m",
        predict_fn=predict_fn,
    )
    assert not out.empty
    assert calls["n"] > 0


def test_model_factory_requires_predict_fn_or_class() -> None:
    with pytest.raises(ValueError, match="predict_fn or model_class"):
        _model_factory_default()


def test_model_factory_class_picks_positive_column_for_binary_predict_proba() -> None:
    """The factory's predict_proba convention is ``[:, -1]`` (positive
    class), mirroring v1.4 ``ProbabilityModel.predict_proba``."""

    class _Stub:
        def fit(self, X: pd.DataFrame, y: pd.Series) -> _Stub:
            return self

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            n = len(X)
            return np.stack([np.full(n, 0.3), np.full(n, 0.7)], axis=1)

    factory = _model_factory_default(model_class=_Stub)
    out = factory(
        pd.DataFrame({"x": [1.0]}),
        pd.Series([0]),
        pd.DataFrame({"x": [1.0, 2.0]}),
    )
    np.testing.assert_allclose(out, [0.7, 0.7])
