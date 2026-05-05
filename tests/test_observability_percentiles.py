"""Regression test for ``observability.prometheus_text``.

Earlier ``prometheus_text`` rebuilt histograms by replaying ``count`` copies
of the *mean* into a Prometheus ``Histogram``, so every reported percentile
collapsed to the mean. This test asserts that p50, p95, and p99 of an
observed series with a wide spread differ from each other and from the mean.
"""

from __future__ import annotations

import re

from market_regime_engine import observability
from market_regime_engine.observability import (
    MetricsRegistry,
    prometheus_text,
)


def _extract_quantile(text: str, name: str, q: str) -> float:
    pattern = rf"{re.escape(name)}\{{[^}}]*quantile=\"{re.escape(q)}\"[^}}]*\}}\s+([\d.eE+-]+)"
    match = re.search(pattern, text)
    assert match, f"could not find {name} quantile={q} in:\n{text}"
    return float(match.group(1))


def test_prometheus_text_emits_distinct_percentiles_for_wide_series(monkeypatch) -> None:
    fresh = MetricsRegistry()
    monkeypatch.setattr(observability, "_GLOBAL", fresh)

    # Use 200 monotonically increasing samples spanning four decades so the
    # p50, p95, p99 indices land on visibly different values.
    samples = [(i + 1) / 2.0 for i in range(200)]  # [0.5, 1.0, ..., 100.0]
    for v in samples:
        fresh.observe("mre_latency_seconds", v, endpoint="test")

    text = prometheus_text()
    p50 = _extract_quantile(text, "mre_latency_seconds", "0.5")
    p95 = _extract_quantile(text, "mre_latency_seconds", "0.95")
    p99 = _extract_quantile(text, "mre_latency_seconds", "0.99")

    # The series spans 200 distinct values; percentiles must differ.
    assert p50 < p95 < p99, (
        f"percentiles failed to monotonically increase across a wide series: p50={p50}, p95={p95}, p99={p99}"
    )

    mean = sum(samples) / len(samples)
    # Regression guard: the buggy version reported every percentile equal to
    # the mean. p95 must therefore be visibly above the mean.
    assert p95 > mean + 1.0, f"p95={p95} is suspiciously close to mean={mean}; prometheus_text may have regressed"


def test_prometheus_text_emits_count_and_sum_per_metric(monkeypatch) -> None:
    fresh = MetricsRegistry()
    monkeypatch.setattr(observability, "_GLOBAL", fresh)

    fresh.observe("mre_test_block_seconds", 0.10, stage="a")
    fresh.observe("mre_test_block_seconds", 0.30, stage="a")

    text = prometheus_text()
    assert "mre_test_block_seconds_count" in text
    assert "mre_test_block_seconds_sum" in text
    assert "# TYPE mre_test_block_seconds summary" in text
