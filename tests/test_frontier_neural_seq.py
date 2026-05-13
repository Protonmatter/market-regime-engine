# SPDX-License-Identifier: Apache-2.0
"""Baseline tests for :mod:`market_regime_engine.frontier.neural_seq`.

Phase 5.2 of v1.6.0 (REVIEW_DEEP_V1_5_2.md §4 / §1.13 / Finding #6).
The class was renamed from ``PatchTSTHead`` to
:class:`MultivariateAvgPatchHead` in Phase 2 (the prior name promised
channel-independent processing per Nie et al. 2023; the implementation
collapses channels via ``mean(axis=1)`` BEFORE patching). The
back-compat alias is preserved for v1.5.x callers and removed in v1.7.

Skip the suite when torch is missing — :class:`MultivariateAvgPatchHead`
hard-requires torch (no fallback) per the v1.2 spec.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from market_regime_engine.frontier.neural_seq import (
    MultivariateAvgPatchHead,
    PatchTSTHead,
)


def _toy_panel(n: int = 240, *, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    panel = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"), index=dates)
    target = pd.Series(panel.mean(axis=1).shift(-1).fillna(0.0), index=dates)
    return panel, target


def test_multivariate_avg_patch_head_smoke_and_predict_shape():
    panel, target = _toy_panel(n=120)
    head = MultivariateAvgPatchHead(patch_len=8, depth=1, n_epochs=2)
    head.fit(panel, target, horizon=2)
    assert head.fitted is True
    out = head.predict(panel)
    # predict returns a dataframe of horizon rows x n_quantiles columns
    # plus a horizon column.
    assert "horizon" in out.columns
    assert len(out) == 2  # one row per horizon step


def test_multivariate_avg_patch_head_raises_on_insufficient_data():
    """REVIEW_DEEP_V1_5_2.md §1.13: the prior degenerate empirical-quantile
    fallback is removed; ``fit`` raises :class:`ValueError` when n_train < 16.
    """
    panel, target = _toy_panel(n=20)  # Too short for default seq_len=48.
    head = MultivariateAvgPatchHead(patch_len=12)
    with pytest.raises(ValueError, match=r"Insufficient training data"):
        head.fit(panel, target, horizon=1)


def test_patch_tst_head_alias_preserved():
    """v1.5.x back-compat alias still resolves to MultivariateAvgPatchHead.

    Removal is tracked for v1.7.0 — this test is a tripwire for the
    deprecation window.
    """
    assert PatchTSTHead is MultivariateAvgPatchHead


def test_predict_before_fit_raises_runtime_error():
    head = MultivariateAvgPatchHead()
    panel, _ = _toy_panel(n=60)
    with pytest.raises(RuntimeError, match=r"called before fit"):
        head.predict(panel)
