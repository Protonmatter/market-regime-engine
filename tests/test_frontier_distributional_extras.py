# SPDX-License-Identifier: Apache-2.0
"""Baseline tests for the renamed distributional heads in
:mod:`market_regime_engine.frontier.distributional`.

Phase 5.2 of v1.6.0 (REVIEW_DEEP_V1_5_2.md §4 / §1.10 / Finding #7):
the previous ``IsotonicDistributionalHead`` and ``DeepStateSpaceHead``
class names overpromised faithful Henzi-Ziegel-Gneiting 2021 IDR /
Karl-Soelch DVBF; Phase 2 renamed them to
:class:`IsotonicMarginalRegressor` and :class:`VariationalEncoderHead`
to honestly describe the implementation. The v1.5.x aliases remain
for back-compat. This file pins:

- The new names produce sensible outputs on a small fixture.
- The legacy aliases still resolve to the new classes (back-compat).
- :class:`NGBoostHead` soft-degrades to the marginal-Normal fallback when
  ngboost isn't installed.
"""

from __future__ import annotations

import numpy as np

from market_regime_engine.frontier.distributional import (
    DeepStateSpaceHead,
    IsotonicDistributionalHead,
    IsotonicMarginalRegressor,
    NGBoostHead,
    VariationalEncoderHead,
)


def _toy_xy(n: int = 60, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = X.sum(axis=1) + rng.normal(scale=0.2, size=n)
    return X, y


def test_ngboost_head_soft_degrade_marginal_normal():
    X, y = _toy_xy()
    head = NGBoostHead().fit(X, y)
    assert head.fitted is True
    assert head.backend in {"ngboost", "fallback"}
    preds = head.predict(X)
    assert preds.shape == (len(X),)
    dist = head.predict_distribution(X)
    assert len(dist) == len(X)
    # Either family must produce loc + scale fields in every row.
    for row in dist:
        assert "loc" in row and "scale" in row


def test_isotonic_marginal_regressor_smoke():
    X, y = _toy_xy()
    head = IsotonicMarginalRegressor().fit(X, y)
    assert head.fitted is True
    preds = head.predict(X)
    assert preds.shape == (len(X),)
    # CDF dict per row.
    dist = head.predict_distribution(X[:5])
    assert len(dist) == 5
    for row in dist:
        assert "levels" in row and "cdf" in row
        assert len(row["cdf"]) == len(row["levels"])


def test_variational_encoder_head_smoke():
    X, y = _toy_xy()
    head = VariationalEncoderHead(n_epochs=2).fit(X, y)
    assert head.fitted is True
    preds = head.predict(X)
    assert preds.shape == (len(X),)


def test_v15_aliases_resolve_to_new_classes():
    """REVIEW_DEEP_V1_5_2.md §1.10 / Finding #7: back-compat aliases."""
    assert IsotonicDistributionalHead is IsotonicMarginalRegressor
    assert DeepStateSpaceHead is VariationalEncoderHead
