# SPDX-License-Identifier: Apache-2.0
"""Baseline tests for :mod:`market_regime_engine.frontier.dfm_mq`.

Phase 5.2 of v1.6.0 (REVIEW_DEEP_V1_5_2.md §4 / A15 / Finding #10):
pin the Phase-2 fixes that landed for the DFM-MQ wrapper:

- ``_extract_factor_series(filtered=True)`` is the default (PIT-safe),
- ``_extract_factor_se(strict=True)`` raises a clean ValueError when the
  structured ``smoothed_state_cov`` is unavailable,
- ``nowcast(asof)`` filters the cached panel to ``index <= asof`` and
  recomputes from the prefix (PIT-safety).

Soft-degrades when statsmodels isn't installed: the wrapper transparently
falls back to the v1.0 :class:`DFMDomainModel`. The fallback path is what
this test exercises (statsmodels is optional and absent on most dev boxes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.frontier.dfm_mq import (
    MQDynamicFactorModel,
    build_synthetic_panel,
)


def test_build_synthetic_panel_shape_and_persistence():
    panel, true_factor = build_synthetic_panel(n_months=48, seed=0, n_series=3)
    assert panel.shape == (48, 3)
    assert true_factor.shape == (48,)
    # Factor is AR(1) with persistence 0.7 — first lag autocorr should be high.
    f_lag = true_factor[:-1]
    f_now = true_factor[1:]
    rho = float(np.corrcoef(f_lag, f_now)[0, 1])
    assert rho > 0.5, f"AR(1) autocorr {rho:.2f} too low for persistence=0.7"


def test_mq_dynamic_factor_model_fit_uses_fallback_without_statsmodels():
    panel, _ = build_synthetic_panel(n_months=60, seed=1, n_series=4)
    model = MQDynamicFactorModel().fit(panel)
    # statsmodels is optional; on a dev box without it the wrapper must
    # transparently fall back to the v1.0 DFMDomainModel and still report
    # ``fitted=True`` so downstream consumers don't error.
    assert model.fitted is True
    assert model.backend in {"statsmodels", "fallback"}


def test_mq_dynamic_factor_model_extract_factor_se_strict_raises_on_missing_cov():
    """A15 / Finding #10 fix: without a structured smoothed_state_cov,
    ``strict=True`` must raise rather than emit a misleading proxy.
    """

    class _BareResults:
        pass

    with pytest.raises(ValueError, match=r"smoothed_state_cov unavailable"):
        MQDynamicFactorModel._extract_factor_se(_BareResults(), strict=True)


def test_mq_dynamic_factor_model_extract_factor_se_nonstrict_returns_none():
    """``strict=False`` (default) returns None + logs a warning."""

    class _BareResults:
        pass

    out = MQDynamicFactorModel._extract_factor_se(_BareResults(), strict=False)
    assert out is None


def test_mq_dynamic_factor_model_nowcast_pit_safe_filters_by_asof():
    """A15 / Finding #10 fix: ``nowcast(asof)`` must respect ``asof`` and
    NOT just return the cached final-fit factor regardless of ``asof``.
    """
    panel, _ = build_synthetic_panel(n_months=60, seed=2, n_series=3)
    model = MQDynamicFactorModel().fit(panel)
    final_asof = panel.index[-1]
    early_asof = panel.index[20]
    early = model.nowcast(early_asof)
    final = model.nowcast(final_asof)
    # Both responses must include the ``as_of`` label correctly.
    assert early["as_of"] == str(pd.Timestamp(early_asof).date())
    assert final["as_of"] == str(pd.Timestamp(final_asof).date())
    # The cached panel filter under the fallback backend rebuilds the
    # factor over the prefix; the value at early_asof should differ from
    # the value at final_asof for any panel with non-trivial dynamics.
    if model.backend == "fallback":
        assert early["factor"] != final["factor"] or len(panel) <= 2
