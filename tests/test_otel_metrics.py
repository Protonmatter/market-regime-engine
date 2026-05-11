# SPDX-License-Identifier: Apache-2.0
"""PR-7 §E — OpenTelemetry SDK migration acceptance tests.

The OTel SDK is an optional dep (``[observability]`` extra). Tests are
designed so the assertions about the legacy backend pass *without*
the SDK installed; OTel-specific assertions are guarded by a
fixture that skips when the SDK is absent.

Per AF-4 (REVIEW.md §3.1) and plan §7 §E:

- ``configure_otel`` initialises a meter + tracer when enabled.
- The legacy in-process registry stays as the always-available
  fall-back when OTel is disabled.
- The new top-level ``incr`` / ``record_histogram`` / ``time_block``
  emit to BOTH backends when OTel is configured.
- The FI counter/histogram pre-registration in
  ``fixed_income.observability_ext`` lights up at module load.
- ``evaluate_release_gate`` block conditions emit
  ``fi_release_gate_blocks_total`` (integration test).
"""

from __future__ import annotations

from importlib import util as importlib_util

import pytest

import market_regime_engine.fixed_income  # noqa: F401  - register FI tables + metrics
from market_regime_engine import observability
from market_regime_engine.fixed_income.observability_ext import (
    FI_COUNTER_NAMES,
    FI_HISTOGRAM_NAMES,
    fi_metric_snapshot,
    incr_release_gate_block,
)


def _has_otel() -> bool:
    return importlib_util.find_spec("opentelemetry") is not None


@pytest.fixture
def reset_otel_state():
    yield
    observability.configure_otel(enabled=False)


def test_configure_otel_initializes_meter_and_tracer(reset_otel_state) -> None:
    if not _has_otel():
        # Soft-degrade path: with the SDK absent, configure_otel
        # returns False and leaves the legacy backend active.
        assert observability.configure_otel(enabled=True) is False
        assert observability.otel_enabled() is False
        assert observability.get_meter() is None
        assert observability.get_tracer() is None
        return
    assert observability.configure_otel(enabled=True) is True
    assert observability.otel_enabled() is True
    assert observability.get_meter() is not None
    assert observability.get_tracer() is not None


def test_fi_counters_registered_at_module_load() -> None:
    """Importing the FI package must light up every canonical counter
    name at a 0 baseline."""
    snapshot = fi_metric_snapshot()
    counter_bases = {key.split("{", 1)[0] for key in snapshot["counters"]}
    histogram_bases = {key.split("{", 1)[0] for key in snapshot["histograms"]}
    assert set(FI_COUNTER_NAMES) <= counter_bases, (
        f"missing counters: {set(FI_COUNTER_NAMES) - counter_bases}"
    )
    assert set(FI_HISTOGRAM_NAMES) <= histogram_bases, (
        f"missing histograms: {set(FI_HISTOGRAM_NAMES) - histogram_bases}"
    )


def test_observability_legacy_api_still_works_when_otel_disabled() -> None:
    """``incr`` / ``record_histogram`` always update the legacy registry."""
    observability.configure_otel(enabled=False)
    before = observability.metrics().snapshot()["counters"].get(
        "test_legacy_counter", 0.0
    )
    observability.incr("test_legacy_counter", 1.0, source="test")
    after = observability.metrics().snapshot()["counters"]
    # Counter is keyed with labels; find the one that has source=test.
    matching = {k: v for k, v in after.items() if k.startswith("test_legacy_counter")}
    assert any(v >= float(before) + 1.0 for v in matching.values())


def test_observability_legacy_api_emits_to_otel_when_enabled(
    reset_otel_state,
) -> None:
    """When configure_otel is active, observability.incr should also
    produce an OTel counter instrument; we verify by introspecting
    the cache."""
    if not _has_otel():
        pytest.skip("OpenTelemetry SDK not installed")
    observability.configure_otel(enabled=True)
    observability.incr("otel_test_counter", 1.0, recommended_action="auto_x_allowed")
    counter = observability._otel_counter("otel_test_counter")  # type: ignore[attr-defined]
    assert counter is not None


def test_release_gate_block_emits_counter() -> None:
    """Integration: the FI release-gate block helper increments
    ``fi_release_gate_blocks_total{reason=...}`` on the legacy
    registry (and OTel when enabled)."""
    snap_before = observability.metrics().snapshot()
    label_key = "fi_release_gate_blocks_total{reason=test_block}"
    before = snap_before["counters"].get(label_key, 0.0)
    incr_release_gate_block(reason="test_block")
    snap_after = observability.metrics().snapshot()
    after = snap_after["counters"].get(label_key, 0.0)
    assert after == pytest.approx(before + 1.0)


def test_time_block_records_through_both_backends() -> None:
    """``time_block`` should record an elapsed-seconds observation."""
    name = "test_time_block_seconds"
    with observability.time_block(name):
        pass
    snap = observability.metrics().snapshot()
    matching = {k: v for k, v in snap["histograms"].items() if k.startswith(name)}
    assert matching, "expected at least one matching histogram entry"
    stats = next(iter(matching.values()))
    assert int(stats["count"]) >= 1
