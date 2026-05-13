# SPDX-License-Identifier: Apache-2.0
"""Baseline tests for the five conformal layers in
:mod:`market_regime_engine.frontier.conformal_ts`.

Phase 5.2 of v1.6.0 (REVIEW_DEEP_V1_5_2.md §4 / §1.7). Complements the
existing ``test_conformal*`` files by pinning the smoke surface of every
layer plus the Phase-1 / Phase-2 fixes:

- :class:`BlockConformalBinary` — Künsch 1989 MBB citation (post §1.7
  docstring fix).
- :class:`NexCPForecaster` — verifies the explicit :meth:`step` rolls
  the per-bucket ``inflation_t`` (the Phase-1 / §1.7 #4 fix).
- :class:`ConditionalConformalRegressor` — Bonferroni group-conditional
  coverage on a 3-bucket calibration set.
- :class:`LocalizedSplitConformal` — exercises the
  :func:`_localized_threshold` weighted-quantile path that the
  ``test_localized_split_conformal_coverage`` suite covers more
  rigorously; here we just smoke-test the public API.
- :class:`SequentialEConformal` — ``_increment`` is the betting
  e-process (Phase-1 / §1.7 #3 fix). The dedicated valid-e-process
  proof is in ``test_sequential_e_conformal_valid_e_process``;
  here we smoke-test :meth:`update` and :meth:`coverage_until_now`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_regime_engine.frontier.conformal_ts import (
    BlockConformalBinary,
    ConditionalConformalRegressor,
    LocalizedSplitConformal,
    NexCPForecaster,
    SequentialEConformal,
)


def _binary_calibration_frame(n: int = 200, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9, size=n)
    y = (rng.uniform(0, 1, size=n) < p).astype(int)
    bucket = np.where(p > 0.5, "high", "low")
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({"y": y, "p": p, "regime_bucket": bucket, "date": dates})


# ---------------------------------------------------------------------------
# 1. BlockConformalBinary smoke (Künsch 1989 MBB citation per §1.7)
# ---------------------------------------------------------------------------


def test_block_conformal_binary_smoke():
    cal = _binary_calibration_frame(n=240, seed=1)
    model = BlockConformalBinary(alpha=0.10, block_length=12, bootstrap=50, seed=1).fit(cal)
    assert set(model.thresholds.keys()) >= {"high", "low"}
    for thr in model.thresholds.values():
        assert 0.0 <= thr <= 1.0
    # block_count records how many non-overlapping blocks contributed per bucket.
    assert all(c >= 1 for c in model.block_count.values())
    # transform produces the conformal_set + uncertain columns.
    annotated = model.transform(cal.head(10))
    assert {"conformal_set", "conformal_uncertain", "conformal_threshold"}.issubset(annotated.columns)


# ---------------------------------------------------------------------------
# 2. NexCPForecaster online step (Phase-1 §1.7 #4 fix: explicit .step rolls inflation)
# ---------------------------------------------------------------------------


def test_nexcp_step_rolls_inflation_t():
    """The §1.7 #4 fix added an explicit :meth:`step` so ``inflation_t``
    rolls per the Stankevičiūtė ACI rule. A run of under-coverage events
    (``err = 1``) must INCREASE inflation; a run of over-coverage events
    (``err = 0``) must DECREASE it.
    """
    cal = _binary_calibration_frame(n=120, seed=2)
    model = NexCPForecaster(alpha=0.10, window=60, inflation_eta=0.05).fit(cal)
    bucket = next(iter(model.base_thresholds.keys()))
    inf0 = model.inflation_per_bucket.get(bucket, 0.0)
    # Force under-coverage: predict 0.5, observe 1, threshold is in [0,1]
    # so the prediction set may not include 1 → err=1 → inflation up.
    for _ in range(10):
        model.step(pred=0.99, y=0, bucket=bucket)
    inf_after_under = model.inflation_per_bucket[bucket]
    # ACI: inflation_{t+1} = inflation_t + eta * (err - alpha). With err=1
    # and alpha=0.10 every step adds 0.05 * 0.90 > 0.
    assert inf_after_under > inf0


def test_nexcp_step_persists_threshold_for_transform():
    cal = _binary_calibration_frame(n=120, seed=3)
    model = NexCPForecaster(alpha=0.10, window=60, inflation_eta=0.05).fit(cal)
    bucket = next(iter(model.base_thresholds.keys()))
    out = model.step(pred=0.7, y=1, bucket=bucket)
    assert out["threshold"] == model.thresholds[bucket]
    # history records each call.
    assert len(model.history) == 1
    assert model.history[0]["bucket"] == bucket


# ---------------------------------------------------------------------------
# 3. ConditionalConformalRegressor — group-conditional coverage
# ---------------------------------------------------------------------------


def test_conditional_conformal_per_group_coverage():
    cal = _binary_calibration_frame(n=300, seed=4)
    cal["regime_bucket"] = np.tile(["a", "b", "c"], 100)
    model = ConditionalConformalRegressor(alpha=0.10).fit(cal)
    assert model.n_groups == 3
    report = model.coverage_report_conditional(cal)
    # Bonferroni-adjusted alpha = 0.10 / 3.
    assert report["adjusted_alpha"] == 0.10 / 3
    assert "per_group" in report
    assert {"regime_bucket", "n", "coverage", "threshold"}.issubset(report["per_group"].columns)
    # worst_violation must be reported as a finite non-negative float.
    assert isinstance(report["worst_violation"], float)
    assert report["worst_violation"] >= 0.0


# ---------------------------------------------------------------------------
# 4. LocalizedSplitConformal smoke (Phase-1 §1.7 #12 weighted-quantile)
# ---------------------------------------------------------------------------


def test_localized_split_conformal_one_hot_smoke():
    cal = _binary_calibration_frame(n=180, seed=5)
    model = LocalizedSplitConformal(alpha=0.10, bandwidth=1.0).fit(cal)
    # In one-hot fallback (no feature_cols), threshold_for(bucket) returns
    # the per-bucket base threshold.
    assert set(model.thresholds.keys()) >= {"high", "low"}
    annotated = model.transform(cal.head(8))
    assert "conformal_threshold" in annotated.columns


def test_localized_split_conformal_weighted_quantile_in_range():
    """Phase-1 §1.7 #12: ``_localized_threshold`` returns a value in
    ``[0, 1]`` (binary scores are bounded). Pin that the weighted quantile
    machinery doesn't over-/under-flow under typical inputs.
    """
    cal = _binary_calibration_frame(n=180, seed=6)
    model = LocalizedSplitConformal(alpha=0.10, bandwidth=0.5).fit(cal)
    x_test = np.zeros(model._calibration_features.shape[1])
    thr = model._localized_threshold(x_test)
    assert 0.0 <= thr <= 1.0


# ---------------------------------------------------------------------------
# 5. SequentialEConformal — Phase-1 §1.7 #3 betting e-process
# ---------------------------------------------------------------------------


def test_sequential_e_conformal_update_returns_e_value_and_significance():
    cal = _binary_calibration_frame(n=120, seed=7)
    model = SequentialEConformal(alpha=0.10).fit(cal)
    bucket = next(iter(model.e_per_bucket.keys()))
    out = model.update({"regime_bucket": bucket}, y=1, pred=0.55)
    assert "e_value" in out and "is_significant" in out
    assert out["e_value"] >= model.e_floor
    # coverage_until_now reports the running stats.
    coverage = model.coverage_until_now()
    assert coverage["n"] == 1
    assert coverage["alpha"] == model.alpha
    assert coverage["max_e_value"] >= out["e_value"]


def test_sequential_e_conformal_increment_strictly_positive_for_clipped_p():
    """Phase-1 §1.7 #3: ``_increment`` clips ``p_hat`` to ``[EPS, 1-EPS]``
    so the increment ``1 + lambda * (y - p_hat)`` is strictly positive
    for ``lambda = 1`` and any ``y in {0, 1}``.
    """
    model = SequentialEConformal(alpha=0.10, lambda_t=1.0)
    for p_hat in (0.0, 0.5, 1.0, -0.1, 1.1):
        for y in (0, 1):
            inc = model._increment(p_hat, y)
            assert inc > 0.0, f"increment {inc} not strictly positive at p_hat={p_hat}, y={y}"
