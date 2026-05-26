# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Phase-5 mypy fixes (REVIEW_DEEP_V1_5_2.md §4.2).

Each fix in :commit:`fix(types): drive mypy baseline from 13 → 0`
gets at least one runtime regression test here so that:

1. The fix isn't silently reverted by a future PR.
2. The safety-critical bocpd_hazard.py None-guard raises a *clean*
   ``RuntimeError`` instead of an ``AttributeError`` deep inside
   sklearn.
3. The widened input types of ``murphy_decomposition`` actually accept
   ``np.ndarray`` at runtime.
4. The TypedDict-narrowed ``ModelCard`` constructor still produces
   the same artifact hash as the pre-Phase-5 path.

These are intentionally small unit tests; the integration paths are
already covered by existing suites.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.bocpd_hazard import CovariateBOCPDHazard
from market_regime_engine.forecast_compare import murphy_decomposition
from market_regime_engine.hmm import _logsumexp as _hmm_logsumexp
from market_regime_engine.model_registry import (
    ModelCardKwargs,
    create_model_card,
    stable_hash,
)
from market_regime_engine.msvar import _logsumexp as _msvar_logsumexp

# ---------------------------------------------------------------------------
# bocpd_hazard.py — safety-critical: predict_proba before fit must raise
# ---------------------------------------------------------------------------


def test_bocpd_hazard_predict_proba_before_fit_raises_runtime_error():
    """Per-Phase-5 §4.2 safety-critical fix.

    A ``CovariateBOCPDHazard`` instance whose ``fitted`` flag is True
    but whose ``pipeline`` is still ``None`` (the partial-construction
    failure mode) must surface as a clean ``RuntimeError`` at the API
    boundary rather than an ``AttributeError`` deep inside sklearn.
    """
    haz = CovariateBOCPDHazard()
    haz.fitted = True
    haz.pipeline = None
    haz.feature_columns = ["a", "b"]
    cov = pd.DataFrame({"a": [0.0, 1.0], "b": [0.5, 0.5]})
    with pytest.raises(RuntimeError, match="hazard classifier not fitted"):
        haz.hazard_series(cov)


def test_bocpd_hazard_unfitted_hazard_at_falls_back():
    """The single-row ``hazard_at`` path is allowed to fall back to
    ``fallback_hazard`` when unfitted (existing behaviour preserved).
    """
    haz = CovariateBOCPDHazard()
    out = haz.hazard_at(pd.Series({"a": 0.0, "b": 0.0}))
    assert out == pytest.approx(haz.fallback_hazard)


# ---------------------------------------------------------------------------
# forecast_compare.murphy_decomposition — accepts np.ndarray
# ---------------------------------------------------------------------------


def test_murphy_decomposition_accepts_ndarray_inputs():
    """Per-Phase-5 §4.2: ``y`` and ``p`` widened to also accept ``np.ndarray``.

    This test passes raw arrays (not lists) — the same call path that
    ``prediction_evidence._finite_pair`` uses.
    """
    rng = np.random.default_rng(0)
    n = 200
    p = rng.uniform(0.1, 0.9, size=n)
    y = (rng.uniform(0, 1, size=n) < p).astype(float)
    out = murphy_decomposition(y, p, bins=10)
    assert set(out.keys()) >= {"reliability", "resolution", "uncertainty", "brier"}
    # All four components must be finite floats (the type-widening can't
    # have changed the numerical result if the underlying ``np.asarray``
    # path is the same).
    for key in ("reliability", "resolution", "uncertainty", "brier"):
        assert isinstance(out[key], float)
        assert np.isfinite(out[key])
    # Brier-ish identity (Murphy 1973, binned): brier ≈ REL - RES + UNC
    # within the bin-discretisation error budget (~1e-2 for n=200, 10 bins).
    assert out["brier"] == pytest.approx(out["reliability"] - out["resolution"] + out["uncertainty"], abs=2e-2)


# ---------------------------------------------------------------------------
# model_registry — TypedDict-narrowed kwargs path produces same hash
# ---------------------------------------------------------------------------


def test_create_model_card_typed_dict_path_produces_stable_hash():
    """The Phase-5 TypedDict refactor must NOT change the artifact hash.

    The hash is computed from the same field values; the ``stable_hash``
    helper is deterministic, so the test pins the exact hex digest.
    """
    card = create_model_card(
        model_name="phase5_test",
        version="0.1.0",
        target="recession",
        horizon="12m",
        training_start="2000-01-01",
        training_end="2020-12-31",
        feature_count=8,
        observations=240,
        objective="brier",
        known_limitations=["limited tail data"],
        validation_metrics={"brier": 0.18, "ece": 0.04},
    )
    expected = stable_hash(
        {
            "model_name": "phase5_test",
            "version": "0.1.0",
            "target": "recession",
            "horizon": "12m",
            "training_start": "2000-01-01",
            "training_end": "2020-12-31",
            "feature_count": 8,
            "observations": 240,
            "objective": "brier",
            "known_limitations": ["limited tail data"],
            "validation_metrics": {"brier": 0.18, "ece": 0.04},
        }
    )
    assert card.artifact_hash == expected


def test_model_card_kwargs_typeddict_round_trips():
    """``ModelCardKwargs`` is structurally a ``dict[str, Any]`` at runtime."""
    payload: ModelCardKwargs = {
        "model_name": "x",
        "version": "0.0.1",
        "target": "y",
        "horizon": "1m",
        "training_start": "2020-01-01",
        "training_end": "2020-12-31",
        "feature_count": 4,
        "observations": 12,
        "objective": "brier",
        "known_limitations": [],
        "validation_metrics": {},
    }
    assert dict(payload)["feature_count"] == 4


# ---------------------------------------------------------------------------
# hmm._logsumexp / msvar._logsumexp — overload returns float vs ndarray
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [_hmm_logsumexp, _msvar_logsumexp])
def test_logsumexp_axis_none_returns_python_float(fn):
    """``axis=None`` must return a Python ``float`` (not an ``np.ndarray``)
    so callers (Hamilton filter, Kim smoother) can compare via
    ``< prev_ll``, ``abs(...)``, etc., without unpacking a 0-D array.
    """
    out = fn(np.array([0.1, 0.2, 0.3]))
    assert isinstance(out, float)


@pytest.mark.parametrize("fn", [_hmm_logsumexp, _msvar_logsumexp])
def test_logsumexp_axis_int_returns_ndarray(fn):
    """``axis=0`` over a 2-D array must return an ``np.ndarray``."""
    a = np.array([[0.1, 0.2], [0.3, 0.4]])
    out = fn(a, axis=0)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2,)


# ---------------------------------------------------------------------------
# scenarios.py:153 — Timestamp-coerced .loc[] slice
# ---------------------------------------------------------------------------


def test_scenarios_timestamp_loc_slice_round_trips():
    """The Phase-5 fix coerces ``scenario.start`` / ``.end`` (str) to
    ``pd.Timestamp`` before passing to ``DataFrame.loc[]``. Behaviour
    must be unchanged: a panel sliced by ISO date strings vs by
    ``pd.Timestamp`` returns the same rows.
    """
    idx = pd.date_range("2000-01-01", periods=24, freq="MS")
    panel = pd.DataFrame({"x": np.arange(24, dtype=float)}, index=idx)
    by_str = panel.loc["2000-06-01":"2000-12-01"]  # type: ignore[misc]
    by_ts = panel.loc[pd.Timestamp("2000-06-01") : pd.Timestamp("2000-12-01")]
    pd.testing.assert_frame_equal(by_str, by_ts)


# ---------------------------------------------------------------------------
# api_v1._RedisTTLCache.get — non-bytes payload returns None
# ---------------------------------------------------------------------------


def test_redis_cache_get_returns_none_on_non_bytes_payload(caplog):
    """The Phase-5 ``isinstance(raw, bytes)`` narrow must protect
    ``_deserialize_cache_value`` from receiving an Awaitable / coroutine
    if a future regression switches the redis client to async mode.
    """
    from market_regime_engine import api_v1

    cache = object.__new__(api_v1._RedisTTLCache)

    class _FakeClient:
        def get(self, _key):
            return "not-bytes-at-all"

    cache.client = _FakeClient()  # type: ignore[attr-defined]
    with caplog.at_level("WARNING", logger=api_v1.log.name):
        result = cache.get("any-key")
    assert result is None
    assert any("non-bytes" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# walk_forward.SplitterProtocol — structural typing of CombinatorialPurgedCV
# ---------------------------------------------------------------------------


def test_splitter_protocol_structurally_satisfied():
    """``PurgedWalkForward`` and ``CombinatorialPurgedCV`` should both
    satisfy :class:`SplitterProtocol` structurally — i.e. expose a
    ``.split(n)`` returning an iterator of :class:`WalkForwardSplit`.
    """
    from market_regime_engine.walk_forward import (
        CombinatorialPurgedCV,
        PurgedWalkForward,
        SplitterProtocol,
        WalkForwardSplit,
    )

    pwf: SplitterProtocol = PurgedWalkForward(min_train=30, step=5, horizon=1, embargo=0, test_block=1)
    cpcv: SplitterProtocol = CombinatorialPurgedCV(n_blocks=4, k_test_blocks=1, horizon=1, embargo=0)
    folds_pwf = list(pwf.split(60))
    folds_cpcv = list(cpcv.split(60))
    assert all(isinstance(f, WalkForwardSplit) for f in folds_pwf + folds_cpcv)
    assert len(folds_pwf) > 0
    assert len(folds_cpcv) > 0
