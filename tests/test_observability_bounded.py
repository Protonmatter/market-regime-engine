# SPDX-License-Identifier: Apache-2.0
"""PR-5 AF-3: bounded reservoir-backed observability histograms.

Pre-PR-5 the in-process histogram stored every observation in a Python
``list[float]``; on a long-running FastAPI worker that received millions of
execution-confidence requests the storage grew without bound and dominated
the process resident-set. PR-5 swaps the list for
:class:`BoundedHistogram`, a fixed-size reservoir-sampler-backed structure.

These tests verify:

1. ``count`` and ``sum`` remain exact under streaming inserts.
2. The reservoir replaces entries via Algorithm R once the buffer fills,
   so resident memory never grows past ``reservoir_size``.
3. Approximate quantiles match the true sample quantile within a few
   percentage points on a uniform stationary distribution.
4. The ``MetricsRegistry`` integration uses :class:`BoundedHistogram` so
   1M inserts produce a stable reservoir size (no list growth).
"""

from __future__ import annotations

import numpy as np

from market_regime_engine.observability import BoundedHistogram, MetricsRegistry


def test_bounded_histogram_records_count_and_sum_exactly() -> None:
    hist = BoundedHistogram(reservoir_size=64, seed=0)
    values = [float(i) for i in range(1, 101)]
    for v in values:
        hist.record(v)
    assert hist.count == 100
    assert hist.sum == sum(values)


def test_bounded_histogram_reservoir_replacement_after_capacity() -> None:
    """Resident memory is bounded — even 10k inserts stay at reservoir_size slots."""
    hist = BoundedHistogram(reservoir_size=32, seed=42)
    for i in range(10_000):
        hist.record(float(i))
    assert hist.count == 10_000
    assert hist.sum == sum(range(10_000))
    view = hist.reservoir_view()
    assert view.shape[0] == 32
    # Every reservoir slot must hold a value that actually appeared in the stream.
    assert all(0.0 <= v < 10_000 for v in view.tolist())


def test_bounded_histogram_quantile_approximates_truth_on_uniform_distribution() -> None:
    rng = np.random.default_rng(1234)
    samples = rng.uniform(0.0, 1.0, size=200_000)
    hist = BoundedHistogram(reservoir_size=4096, seed=7)
    for v in samples:
        hist.record(float(v))
    # Uniform(0,1) — q-quantile ≈ q. 4096-slot reservoir gives ~0.05 tolerance
    # on a stationary 200k-sample uniform stream.
    for q in (0.05, 0.5, 0.95):
        approx = hist.quantile(q)
        assert abs(approx - q) < 0.05, f"q={q} approx={approx}"


def test_bounded_histogram_quantile_on_empty_returns_zero() -> None:
    hist = BoundedHistogram(reservoir_size=16, seed=0)
    assert hist.quantile(0.5) == 0.0
    assert hist.count == 0
    assert hist.sum == 0.0


def test_metrics_registry_uses_bounded_histograms_under_1m_inserts_does_not_grow_memory() -> None:
    """1M observations on the registry must not balloon the per-key store."""
    registry = MetricsRegistry(reservoir_size=4096)
    for i in range(1_000_000):
        registry.observe("mre_test_latency_seconds", float(i % 1000), endpoint="exec")
    snap = registry.snapshot()
    key = "mre_test_latency_seconds{endpoint=exec}"
    assert snap["histograms"][key]["count"] == 1_000_000
    # Resident-set sanity: the registry must not be hoarding 1M floats per key.
    # We can't inspect the buffer through the snapshot, but we CAN verify
    # the underlying histogram object honours the reservoir bound.
    with registry._lock:
        hist = next(iter(registry._histograms.values()))
    assert isinstance(hist, BoundedHistogram)
    assert hist.reservoir_view().shape[0] == 4096


def test_observe_alias_record_histogram_is_equivalent() -> None:
    registry = MetricsRegistry()
    registry.observe("a", 1.0)
    registry.record_histogram("a", 2.0)
    snap = registry.snapshot()
    assert snap["histograms"]["a"]["count"] == 2
    assert snap["histograms"]["a"]["sum"] == 3.0


def test_metrics_registry_incr_remains_exact() -> None:
    """The counter path is unchanged; verify the bounded-histogram refactor did not touch it."""
    registry = MetricsRegistry()
    for _ in range(1000):
        registry.incr("mre_test_counter", endpoint="foo")
    snap = registry.snapshot()
    assert snap["counters"]["mre_test_counter{endpoint=foo}"] == 1000.0
