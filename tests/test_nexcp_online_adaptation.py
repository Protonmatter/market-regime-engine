"""Regression test: NexCPForecaster.step rolls inflation online.

REVIEW_DEEP_V1_5_2.md §1.7 / Finding #4 (GA blocker): the prior
``NexCPForecaster`` computed the inflation once at ``.fit(...)`` time
and froze it; the class name promised the time-series-native online
adaptive primitive of Stankevičiūtė-Alaa-van der Schaar 2021 but the
implementation delivered a one-shot variant that degraded to plain
split conformal at test time.

This test pins the new ``.step(pred, y, bucket)`` contract:

* ``inflation_per_bucket`` actually updates after ``.step()`` (the
  defining property of online ACI; previously frozen).
* The update follows the canonical ACI rule
  ``inflation_{t+1} = inflation_t + gamma * (err_t - alpha)``
  for every step where ``err_t = 1[y not in C_t(x_t)]``
  (Stankevičiūtė et al. 2021, arXiv 2102.13066, §3 /
  Gibbs-Candès 2021).
* Persistent under-coverage drives inflation up; persistent
  over-coverage drives inflation down (toward zero from above).
* The threshold stored in ``self.thresholds[bucket]`` and returned by
  ``threshold_for()`` reflects ``base + inflation_{t+1}`` after every
  step, so subsequent ``.transform()`` / ``.threshold_for()`` calls see
  the rolled-forward value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.frontier.conformal_ts import NexCPForecaster


def _binary_calibration(n: int = 400, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = rng.beta(2.0, 5.0, size=n)
    bucket = rng.choice(["a", "b", "c"], size=n)
    bias = np.where(bucket == "a", 0.0, np.where(bucket == "b", 0.05, -0.05))
    y = (rng.uniform(size=n) < np.clip(p + bias, 1e-3, 1 - 1e-3)).astype(int)
    return pd.DataFrame({"p": p, "y": y, "regime_bucket": bucket})


# ---------------------------------------------------------------------------
# Core ACI rule: inflation_{t+1} = inflation_t + gamma * (err_t - alpha)
# ---------------------------------------------------------------------------


def test_step_actually_updates_inflation_per_bucket() -> None:
    """Inflation MUST change after .step() — defining property of online ACI."""
    layer = NexCPForecaster(alpha=0.10, window=120, inflation_eta=0.05).fit(
        _binary_calibration(n=400)
    )
    before = dict(layer.inflation_per_bucket)
    for _ in range(50):
        layer.step(pred=0.05, y=1, bucket="a")
    after = dict(layer.inflation_per_bucket)
    assert after["a"] != before["a"], (
        f"inflation for bucket 'a' must update after .step(); got "
        f"before={before['a']}, after={after['a']}"
    )


def test_step_follows_aci_update_rule_exactly() -> None:
    """``inflation_{t+1} = inflation_t + gamma * (err_t - alpha)``."""
    alpha = 0.10
    gamma = 0.05
    layer = NexCPForecaster(alpha=alpha, window=120, inflation_eta=gamma).fit(
        _binary_calibration(n=400)
    )
    inflation_t = layer.inflation_per_bucket["a"]
    res = layer.step(pred=0.05, y=1, bucket="a")
    err = res["err"]
    expected_inflation = inflation_t + gamma * (err - alpha)
    assert res["inflation"] == pytest.approx(expected_inflation, abs=1e-12)
    assert layer.inflation_per_bucket["a"] == pytest.approx(expected_inflation, abs=1e-12)


def test_step_returned_threshold_matches_persisted_threshold() -> None:
    """``self.thresholds[bucket]`` must mirror the step's returned threshold."""
    layer = NexCPForecaster(alpha=0.10, window=120, inflation_eta=0.05).fit(
        _binary_calibration(n=400)
    )
    res = layer.step(pred=0.2, y=1, bucket="b")
    assert layer.thresholds["b"] == pytest.approx(res["threshold"], abs=1e-12)
    assert layer.threshold_for("b") == pytest.approx(res["threshold"], abs=1e-12)


# ---------------------------------------------------------------------------
# Long-run direction: under-coverage drives inflation up; over-coverage down.
# ---------------------------------------------------------------------------


def test_persistent_undercoverage_drives_inflation_up() -> None:
    """When the prediction set repeatedly misses the realised label, the ACI
    rule must inflate the threshold above the calibration baseline."""
    layer = NexCPForecaster(alpha=0.10, window=120, inflation_eta=0.05).fit(
        _binary_calibration(n=400)
    )
    initial = layer.inflation_per_bucket["a"]
    # Force a long run of under-coverage: predict ~0 but realised label is 1
    # so the prediction set {0} never covers the outcome.
    for _ in range(200):
        layer.step(pred=0.02, y=1, bucket="a")
    final = layer.inflation_per_bucket["a"]
    assert final > initial, (
        f"persistent under-coverage must push inflation upward; "
        f"initial={initial}, final={final}"
    )


def test_persistent_overcoverage_drives_inflation_down() -> None:
    """When the prediction set repeatedly covers (large), the ACI rule must
    shrink the inflation (toward zero or below)."""
    layer = NexCPForecaster(alpha=0.10, window=120, inflation_eta=0.05).fit(
        _binary_calibration(n=400)
    )
    initial = layer.inflation_per_bucket["a"]
    # Force a long run of over-coverage: predict near 0.5 and realise the
    # majority outcome so the prediction set always covers.
    for _ in range(200):
        layer.step(pred=0.5, y=1, bucket="a")
    final = layer.inflation_per_bucket["a"]
    assert final <= initial + 1e-9, (
        f"persistent over-coverage must not push inflation upward; "
        f"initial={initial}, final={final}"
    )


# ---------------------------------------------------------------------------
# History accumulates and step works on a fresh (unseen) bucket too.
# ---------------------------------------------------------------------------


def test_step_accumulates_history_records() -> None:
    layer = NexCPForecaster(alpha=0.10).fit(_binary_calibration(n=400))
    layer.history.clear()
    for k in range(5):
        layer.step(pred=0.3 + 0.05 * k, y=k % 2, bucket="b")
    assert len(layer.history) == 5
    last = layer.history[-1]
    assert set(last) >= {"bucket", "pred", "y", "err", "inflation", "threshold"}
    assert last["bucket"] == "b"


def test_step_handles_unseen_bucket_via_fallback() -> None:
    """A bucket absent at .fit() time must still roll forward via the
    fallback base threshold; no KeyError on first .step()."""
    layer = NexCPForecaster(alpha=0.10).fit(_binary_calibration(n=400))
    assert "z" not in layer.base_thresholds
    res = layer.step(pred=0.4, y=1, bucket="z")
    assert "z" in layer.inflation_per_bucket
    assert "z" in layer.thresholds
    assert 0.0 <= res["threshold"] <= 1.0


def test_step_threshold_is_clipped_to_unit_interval() -> None:
    """Binary scores live in [0, 1]; the persisted threshold must too."""
    layer = NexCPForecaster(alpha=0.10, inflation_eta=10.0).fit(
        _binary_calibration(n=400)
    )
    # Even with a large step size, the threshold cannot escape [0, 1].
    for _ in range(50):
        layer.step(pred=0.5, y=1, bucket="a")
    assert 0.0 <= layer.thresholds["a"] <= 1.0
    assert 0.0 <= layer.threshold_for("a") <= 1.0
