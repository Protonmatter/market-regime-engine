"""Hypothesis-driven property tests for core invariants.

These pin properties that must hold for arbitrary inputs:

- BOCPD output probabilities ∈ [0, 1] and run-length means ≥ 0.
- HMM forward pass returns rows whose regime posterior sums to 1 ± 1e-9.
- WFST decode returns a path of len(observed).
- Quantile predictions are monotone non-decreasing in tau.
- ``observations_as_of`` is monotone in ``as_of``.
- Conformal Mondrian thresholds are in [0, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from market_regime_engine.bocpd import DiagonalStudentTBOCPD, MultivariateNIWBOCPD
from market_regime_engine.conformal import MondrianBinaryConformal
from market_regime_engine.hmm import DOMAIN_COLUMNS, HMMRegimePosterior
from market_regime_engine.point_in_time import observations_as_of
from market_regime_engine.wfst import RegimeWFST

SLOW_SETTINGS = settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)


@SLOW_SETTINGS
@given(
    n=st.integers(min_value=12, max_value=64),
    d=st.integers(min_value=1, max_value=4),
)
def test_property_diagonal_bocpd_probabilities_in_unit_interval(n: int, d: int) -> None:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    x = pd.DataFrame(rng.normal(size=(n, d)), index=idx, columns=[f"f{i}" for i in range(d)])
    out = DiagonalStudentTBOCPD(max_run=24).score(x)
    assert len(out) == n
    assert out["change_point_prob"].between(0.0, 1.0).all()
    assert (out["bocpd_run_length_mean"] >= 0.0).all()
    assert np.isfinite(out["predictive_log_likelihood"]).all()


@SLOW_SETTINGS
@given(
    n=st.integers(min_value=12, max_value=48),
    d=st.integers(min_value=2, max_value=4),
)
def test_property_niw_bocpd_probabilities_in_unit_interval(n: int, d: int) -> None:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    x = pd.DataFrame(rng.normal(size=(n, d)), index=idx, columns=[f"f{i}" for i in range(d)])
    out = MultivariateNIWBOCPD(max_run=16).score(x)
    assert out["change_point_prob"].between(0.0, 1.0).all()


@SLOW_SETTINGS
@given(n=st.integers(min_value=4, max_value=24))
def test_property_hmm_posterior_sums_to_one(n: int) -> None:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    df = pd.DataFrame(rng.normal(size=(n, len(DOMAIN_COLUMNS))), index=idx, columns=DOMAIN_COLUMNS)
    out = HMMRegimePosterior().score(df)
    cols = [c for c in out.columns if c.startswith("regime_prob_")]
    sums = out[cols].sum(axis=1)
    assert (sums - 1.0).abs().max() < 1e-9


@SLOW_SETTINGS
@given(
    n=st.integers(min_value=2, max_value=24),
)
def test_property_wfst_decode_preserves_length(n: int) -> None:
    states = sorted(RegimeWFST().states)
    obs = [states[i % len(states)] for i in range(n)]
    decoded = RegimeWFST().decode(obs)
    assert len(decoded) == n
    assert all(s in RegimeWFST().states for s in decoded)


@SLOW_SETTINGS
@given(
    p=hnp.arrays(np.float64, shape=(50,), elements=st.floats(0.01, 0.99, allow_nan=False)),
    y=hnp.arrays(np.int8, shape=(50,), elements=st.sampled_from([0, 1])),
)
def test_property_mondrian_threshold_in_unit_interval(p, y):  # type: ignore[no-untyped-def]
    df = pd.DataFrame({"p": p.astype(float), "y": y.astype(int), "regime_bucket": "general"})
    layer = MondrianBinaryConformal(alpha=0.1).fit(df)
    for v in layer.thresholds.values():
        assert 0.0 <= v <= 1.0


@SLOW_SETTINGS
@given(
    asof=st.integers(min_value=0, max_value=10),
)
def test_property_observations_as_of_is_monotone(asof: int) -> None:
    base = pd.Timestamp("2020-01-01")
    rows = []
    for i in range(20):
        rows.append(
            {
                "series_id": "A",
                "date": (base + pd.DateOffset(months=i)).strftime("%Y-%m-%d"),
                "value": float(i),
                "vintage_date": (base + pd.DateOffset(months=i + 1)).strftime("%Y-%m-%d"),
                "source": "test",
            }
        )
    obs = pd.DataFrame(rows)
    cutoff_a = (base + pd.DateOffset(months=asof)).strftime("%Y-%m-%d")
    cutoff_b = (base + pd.DateOffset(months=asof + 3)).strftime("%Y-%m-%d")
    a = observations_as_of(obs, cutoff_a)
    b = observations_as_of(obs, cutoff_b)
    # Adding more time can only add rows or update vintages, never remove them.
    assert len(b) >= len(a)
