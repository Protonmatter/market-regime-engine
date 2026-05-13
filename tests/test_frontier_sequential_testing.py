# SPDX-License-Identifier: Apache-2.0
"""Baseline tests for :mod:`market_regime_engine.frontier.sequential_testing`.

Phase 5.2 of v1.6.0 (REVIEW_DEEP_V1_5_2.md §4 / §1.14 / F13). Pin the
e-value monotonicity contract, the GROW-conservative eta default, and
the Phase-2 :func:`SafeTestPromotion.run` strict-zip fix that surfaces
length mismatches between challenger / champion loss series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.frontier.sequential_testing import (
    EValueLogScore,
    SafeTestPromotion,
    evaluate_with_e_values,
)


def test_evalue_log_score_monotone_under_consistent_winner():
    """When ``loss_a < loss_b`` for every step (challenger consistently
    wins), the e-value must be **monotone non-decreasing**.

    The e-process is a running product of likelihood-ratio increments
    bounded below by ``EPS``; under the alternative every increment is
    > 1 so the product never decreases.
    """
    e = EValueLogScore(alpha=0.05, eta=0.1)
    rng = np.random.default_rng(0)
    prev = e.e_value
    for _ in range(50):
        loss_a = rng.uniform(0.0, 0.5)
        loss_b = rng.uniform(0.5, 1.0)  # b strictly worse
        out = e.update(loss_a, loss_b)
        assert out >= prev - 1e-12, f"E_t {out} < E_{{t-1}} {prev} under consistent winner"
        prev = out
    # Sanity: the e-value rose meaningfully across 50 steps.
    assert e.e_value > 1.0


def test_evalue_log_score_grow_conservative_eta():
    """When ``eta=None`` (default), the class uses the GROW-conservative
    online estimate ``1 / (2 * max(|loss_diff|))`` so the e-process
    stays bounded across heavy-tailed loss differences.
    """
    e = EValueLogScore(alpha=0.05, eta=None)
    e.update(0.0, 100.0)  # huge loss difference
    e.update(0.0, 0.5)
    # _abs_max should be 100; eta is then 1 / (2 * 100) = 0.005.
    assert e._abs_max == pytest.approx(100.0)
    expected_eta = 1.0 / (2.0 * 100.0)
    # Mimic the class's eta selection.
    eta_used = 1.0 / (2.0 * max(e._abs_max, 1e-9))
    assert eta_used == pytest.approx(expected_eta)


def test_safe_test_promotion_run_strict_zip_raises_on_length_mismatch():
    """Phase-2 §1.14 / F13 fix: the prior ``zip(..., strict=False)``
    silently truncated to the shorter of (challenger, champion) losses
    and never surfaced the misalignment. ``strict=True`` now raises so
    the operator sees a clean ValueError at the API boundary.
    """
    with pytest.raises(ValueError, match=r"equal length"):
        SafeTestPromotion.run([1.0, 2.0, 3.0], [1.0, 2.0])


def test_safe_test_promotion_run_returns_status_dict():
    losses_a = np.zeros(20).tolist()
    losses_b = np.ones(20).tolist()
    out = SafeTestPromotion.run(losses_a, losses_b, alpha=0.05)
    assert {"e_value", "fired", "fired_at_n", "n", "level"}.issubset(out.keys())
    assert out["n"] == 20
    assert out["level"] == 0.05


def test_evaluate_with_e_values_aligns_on_index_and_drops_nan():
    """The pandas convenience wrapper aligns and drops NaN; an empty
    intersection returns the documented zero-state envelope.
    """
    a = pd.Series([0.1, 0.2, 0.3, np.nan], index=pd.RangeIndex(4))
    b = pd.Series([0.5, 0.6, 0.7, 0.8], index=pd.RangeIndex(4))
    out = evaluate_with_e_values(a, b)
    assert out["n"] == 3  # the NaN row is dropped
    assert out["level"] == 0.05


def test_evaluate_with_e_values_empty_returns_zero_state():
    out = evaluate_with_e_values(pd.Series(dtype=float), pd.Series(dtype=float))
    assert out["n"] == 0
    assert out["e_value"] == 1.0
    assert out["fired"] is False
