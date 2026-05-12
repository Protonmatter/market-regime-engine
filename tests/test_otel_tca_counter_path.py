# SPDX-License-Identifier: Apache-2.0
"""Regression — TCA NaN-drop counter routes through the OTel mirror.

Pre-fix (REVIEW.md Tier-2 C-AUTO-2): ``fixed_income/tca_segmentation.py``
called ``metrics().incr(DROPPED_ROWS_COUNTER, ...)`` which only writes
to the legacy in-process ``MetricsRegistry`` (``_GLOBAL``) and bypasses
the OTel meter even when ``configure_otel(enabled=True)`` was active.
The result: dashboards backed by OTel exporters showed zero NaN-drop
activity in production where they should have surfaced the metric.

Post-fix: routes through the module-level
:func:`market_regime_engine.observability.incr` which mirrors to BOTH
backends (legacy + OTel) per ``observability.py:393-409``.
"""

from __future__ import annotations

from importlib import util as importlib_util

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401  - register FI metrics
from market_regime_engine import observability
from market_regime_engine.fixed_income.tca_segmentation import (
    DROPPED_ROWS_COUNTER,
    _drop_nan_rows,
)


def _has_otel() -> bool:
    return importlib_util.find_spec("opentelemetry") is not None


@pytest.fixture
def reset_otel_state():
    yield
    observability.configure_otel(enabled=False)


def _legacy_counter_value(name: str, **labels: str) -> float:
    snap = observability.metrics().snapshot()
    if labels:
        label_part = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        key = f"{name}{{{label_part}}}"
    else:
        key = name
    return float(snap["counters"].get(key, 0.0))


def test_tca_dropped_rows_increments_via_legacy_global() -> None:
    """The default legacy ``MetricsRegistry`` snapshot must include the
    TCA NaN-drop counter — the post-fix routing through
    ``observability.incr`` MUST NOT silently break the legacy backend."""
    before = _legacy_counter_value(
        DROPPED_ROWS_COUNTER,
        metric="arrival_cost_bps",
    )
    trades = pd.DataFrame(
        {
            "arrival_cost_bps": [1.0, float("nan"), 2.0, float("nan")],
        }
    )
    cleaned, n_dropped = _drop_nan_rows(trades, metric="arrival_cost_bps", dimensions=())
    assert n_dropped == 2
    assert len(cleaned) == 2
    after = _legacy_counter_value(
        DROPPED_ROWS_COUNTER,
        metric="arrival_cost_bps",
    )
    assert after == pytest.approx(before + 2.0)


def test_tca_dropped_rows_increments_via_otel_when_configured(
    reset_otel_state,
) -> None:
    """When ``configure_otel(enabled=True)`` is active, the
    NaN-drop call site must register an OTel counter instrument so the
    OTel exporter pipeline sees the increment too."""
    if not _has_otel():
        pytest.skip("OpenTelemetry SDK not installed")
    assert observability.configure_otel(enabled=True) is True
    trades = pd.DataFrame(
        {
            "arrival_cost_bps": [1.0, float("nan"), 2.0, float("nan")],
        }
    )
    _drop_nan_rows(trades, metric="arrival_cost_bps", dimensions=())
    counter = observability._otel_counter(DROPPED_ROWS_COUNTER)  # type: ignore[attr-defined]
    assert counter is not None, (
        "OTel counter for fi_tca_dropped_rows_total must be created when "
        "the call site routes through observability.incr"
    )


def test_tca_dropped_rows_emits_with_dimension_labels() -> None:
    """Labels (``regime_label`` + ``liquidity_label`` placeholders) must
    survive through to the legacy snapshot — proving the label-kwargs
    plumbing wasn't lost when switching to ``incr``."""
    before = _legacy_counter_value(
        DROPPED_ROWS_COUNTER,
        liquidity_label="__all__",
        metric="arrival_cost_bps",
        regime_label="__all__",
    )
    trades = pd.DataFrame(
        {
            "arrival_cost_bps": [float("nan"), 1.5],
        }
    )
    _drop_nan_rows(
        trades,
        metric="arrival_cost_bps",
        dimensions=("regime_label", "liquidity_label"),
    )
    after = _legacy_counter_value(
        DROPPED_ROWS_COUNTER,
        liquidity_label="__all__",
        metric="arrival_cost_bps",
        regime_label="__all__",
    )
    assert after == pytest.approx(before + 1.0)
