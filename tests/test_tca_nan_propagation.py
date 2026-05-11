# SPDX-License-Identifier: Apache-2.0
"""PR-6 §D — NaN propagation in bps math (REVIEW.md §3.6 PR-11).

Pins :func:`aggregate_tca_by_regime` to:

- Drop NaN rows at the aggregation boundary (so a single bad price
  cannot poison the bucket mean).
- Emit :data:`DROPPED_ROWS_COUNTER` labelled by metric / regime label /
  liquidity label so the operator dashboard can correlate drops.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401 — register FI schema
from market_regime_engine.fixed_income.tca_segmentation import (
    DROPPED_ROWS_COUNTER,
    aggregate_tca_by_regime,
)
from market_regime_engine.observability import MetricsRegistry, metrics


def _frame_with_nan(n: int = 5, nan_indices: tuple[int, ...] = (0, 2)) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "regime_label": "Normal Liquidity" if i % 2 == 0 else "Watch / Transition",
                "liquidity_label": "Normal" if i % 2 == 0 else "Mild Stress",
                "arrival_cost_bps": float("nan") if i in nan_indices else float(i),
                "execution_success": 1.0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _reset_global_metrics(monkeypatch):
    """Each test gets a fresh global :class:`MetricsRegistry`.

    The TCA counter is module-level; tests that count increments must
    isolate from siblings or the increment counts depend on test order.
    """
    fresh = MetricsRegistry()
    monkeypatch.setattr("market_regime_engine.observability._GLOBAL", fresh)
    yield


def test_tca_aggregate_drops_nan_rows() -> None:
    trades = _frame_with_nan(n=5, nan_indices=(0, 2))
    agg = aggregate_tca_by_regime(
        trades,
        dimensions=("regime_label",),
        metrics_names=("arrival_cost_bps",),
    )
    # Only 3 non-NaN rows contributed to the aggregate.
    assert agg["sample_count"].sum() == 3
    # No NaN slipped into the final bucket means.
    assert not agg["metric_value"].apply(math.isnan).any()


def test_tca_aggregate_does_not_poison_means_with_nan() -> None:
    """The bucket mean must reflect ONLY the non-NaN observations."""
    # All trades in one bucket; values are 1, NaN, 3, NaN, 5 → mean = 3.0.
    rows = []
    values = [1.0, float("nan"), 3.0, float("nan"), 5.0]
    for v in values:
        rows.append(
            {
                "regime_label": "Normal Liquidity",
                "liquidity_label": "Normal",
                "arrival_cost_bps": v,
            }
        )
    agg = aggregate_tca_by_regime(
        pd.DataFrame(rows),
        dimensions=("regime_label",),
        metrics_names=("arrival_cost_bps",),
    )
    assert len(agg) == 1
    assert agg["metric_value"].iloc[0] == pytest.approx(3.0, abs=1e-9)
    assert agg["sample_count"].iloc[0] == 3


def test_tca_aggregate_emits_dropped_rows_counter_per_bucket() -> None:
    """One increment per metric with regime_label + liquidity_label
    bucket = ``__all__`` (the per-call summary level)."""
    trades = _frame_with_nan(n=10, nan_indices=(0, 2, 4))
    aggregate_tca_by_regime(
        trades,
        dimensions=("regime_label", "liquidity_label"),
        metrics_names=("arrival_cost_bps",),
    )
    snapshot = metrics().snapshot()
    counters = snapshot["counters"]
    # Find the counter for metric=arrival_cost_bps.
    matching = {
        k: v
        for k, v in counters.items()
        if k.startswith(DROPPED_ROWS_COUNTER) and "metric=arrival_cost_bps" in k
    }
    assert matching, f"no dropped-rows counter for arrival_cost_bps; got {counters!r}"
    # Total dropped = sum across all matching counters; must equal 3.
    assert sum(matching.values()) == 3.0, matching


def test_tca_aggregate_no_counter_increment_when_no_nan() -> None:
    """A clean frame must NOT increment the dropped-rows counter."""
    rows = [
        {
            "regime_label": "Normal Liquidity",
            "liquidity_label": "Normal",
            "arrival_cost_bps": float(i),
        }
        for i in range(5)
    ]
    aggregate_tca_by_regime(
        pd.DataFrame(rows),
        dimensions=("regime_label", "liquidity_label"),
        metrics_names=("arrival_cost_bps",),
    )
    snapshot = metrics().snapshot()
    counters = snapshot["counters"]
    matching = {
        k: v
        for k, v in counters.items()
        if k.startswith(DROPPED_ROWS_COUNTER) and "metric=arrival_cost_bps" in k
    }
    # The counter family pre-registers at 0; no metric-labelled key gets
    # an increment when the frame is clean.
    assert sum(matching.values()) == 0.0, matching


def test_dropped_rows_counter_is_pre_registered_at_zero() -> None:
    """The module pre-registration helper publishes the counter family
    with a zero baseline so a Prometheus scrape immediately after
    import returns the family with no samples rather than 404."""
    from market_regime_engine.fixed_income.tca_segmentation import _register_counter

    # The autouse fixture replaced ``_GLOBAL`` with a fresh registry,
    # which wipes the module-load registration; re-run the helper to
    # exercise the contract under test.
    _register_counter()
    snapshot = metrics().snapshot()
    assert DROPPED_ROWS_COUNTER in snapshot["counters"]
    assert snapshot["counters"][DROPPED_ROWS_COUNTER] == 0.0


def test_aggregate_drops_nan_only_for_offending_metric() -> None:
    """When metric A has NaN and metric B is clean, only A's rows are dropped."""
    rows = [
        {
            "regime_label": "Normal Liquidity",
            "liquidity_label": "Normal",
            "arrival_cost_bps": float("nan") if i == 0 else float(i),
            "execution_success": 1.0,
        }
        for i in range(4)
    ]
    agg = aggregate_tca_by_regime(
        pd.DataFrame(rows),
        dimensions=("regime_label",),
        metrics_names=("arrival_cost_bps", "execution_success"),
    )
    by_metric = agg.set_index("metric_name")
    # execution_success had no NaN → all 4 contributed.
    assert by_metric.loc["execution_success", "sample_count"] == 4
    # arrival_cost_bps had 1 NaN → 3 contributed.
    assert by_metric.loc["arrival_cost_bps", "sample_count"] == 3
