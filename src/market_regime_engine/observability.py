# SPDX-License-Identifier: Apache-2.0
"""Lightweight observability primitives.

Two backends are supported:

- In-process counter / histogram registry that always works (no dependency).
  Useful for offline runs, tests, and notebooks.
- Optional Prometheus exposition via ``prometheus_client``. If the
  ``observability`` extra is installed, :func:`prometheus_text` renders the
  scrape-formatted output directly.

Metrics are intentionally narrow: latency, row counts, and the release-gate
decision. Anything richer is a dashboard concern.

Histogram exposition note (v1.1)
--------------------------------
Earlier releases pretended to be a Prometheus ``Histogram`` by replaying
``count`` copies of the *mean* into a real Prometheus histogram, which made
every reported percentile collapse to the mean — production dashboards built
on those numbers silently lied. v1.1 emits Prometheus *summary*-style text
manually using the in-process p50/p95/p99 we already record, so the scrape
output reflects the actual percentiles of the observed sample.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import defaultdict
from collections.abc import Iterator
from threading import Lock

log = logging.getLogger(__name__)


class MetricsRegistry:
    """Thread-safe in-process metrics store."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: defaultdict[tuple[str, frozenset], float] = defaultdict(float)
        self._histograms: defaultdict[tuple[str, frozenset], list[float]] = defaultdict(list)

    def incr(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = (name, frozenset(labels.items()))
        with self._lock:
            self._counters[key] += float(value)

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = (name, frozenset(labels.items()))
        with self._lock:
            self._histograms[key].append(float(value))

    def snapshot(self) -> dict:
        with self._lock:
            counters = {self._format_key(k): v for k, v in self._counters.items()}
            histograms = {
                self._format_key(k): {
                    "count": len(values),
                    "sum": sum(values),
                    "p50": _percentile(values, 0.5),
                    "p95": _percentile(values, 0.95),
                    "p99": _percentile(values, 0.99),
                }
                for k, values in self._histograms.items()
            }
        return {"counters": counters, "histograms": histograms}

    @staticmethod
    def _format_key(key: tuple[str, frozenset]) -> str:
        name, labels = key
        if not labels:
            return name
        labels_part = ",".join(f"{k}={v}" for k, v in sorted(labels))
        return f"{name}{{{labels_part}}}"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = round(q * (len(s) - 1))
    return s[idx]


_GLOBAL = MetricsRegistry()


def metrics() -> MetricsRegistry:
    return _GLOBAL


@contextlib.contextmanager
def time_block(name: str, **labels: str) -> Iterator[None]:
    """Context manager recording the elapsed seconds in a histogram."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _GLOBAL.observe(name, elapsed, **labels)


def _split_key(key: str) -> tuple[str, str]:
    """Split a snapshot key like ``name{a=b,c=d}`` into ``(name, "a=b,c=d")``."""
    base, _, label_part = key.partition("{")
    return base, label_part.rstrip("}")


def _format_label_pairs(label_part: str, *extra: tuple[str, str]) -> str:
    pairs = []
    if label_part:
        pairs.extend(label_part.split(","))
    pairs.extend(f'{k}="{v}"' for k, v in extra)
    return "{" + ",".join(p for p in pairs if p) + "}" if pairs else ""


def prometheus_text() -> str:
    """Return Prometheus-compatible scrape text.

    Counters are rendered as ``# TYPE counter`` followed by the value line.
    Histograms are rendered as Prometheus *summary* text — ``_count``,
    ``_sum``, plus a ``{quantile="..."}`` line for each of p50, p95, p99.
    The percentiles come from the in-process snapshot, so the exported
    quantiles match what the engine actually saw.
    """
    snap = _GLOBAL.snapshot()
    lines: list[str] = []

    # Group counters and histograms by base name so we emit one HELP/TYPE per
    # metric family.
    counter_families: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for key, value in snap["counters"].items():
        base, label_part = _split_key(key)
        counter_families[base].append((label_part, float(value)))
    for base, counter_items in sorted(counter_families.items()):
        lines.append(f"# HELP {base} {base}")
        lines.append(f"# TYPE {base} counter")
        for label_part, value in counter_items:
            label_block = "{" + label_part + "}" if label_part else ""
            lines.append(f"{base}{label_block} {value}")

    histogram_families: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for key, stats in snap["histograms"].items():
        base, label_part = _split_key(key)
        histogram_families[base].append((label_part, stats))
    for base, histogram_items in sorted(histogram_families.items()):
        lines.append(f"# HELP {base} {base}")
        lines.append(f"# TYPE {base} summary")
        for label_part, stats in histogram_items:
            count_label = "{" + label_part + "}" if label_part else ""
            lines.append(f"{base}_count{count_label} {int(stats['count'])}")
            lines.append(f"{base}_sum{count_label} {float(stats['sum'])}")
            for q_name, q_val in (
                ("0.5", stats["p50"]),
                ("0.95", stats["p95"]),
                ("0.99", stats["p99"]),
            ):
                base_pairs = label_part.split(",") if label_part else []
                quantile_pair = f'quantile="{q_name}"'
                merged = ",".join([p for p in base_pairs if p] + [quantile_pair])
                lines.append(f"{base}{{{merged}}} {float(q_val)}")

    if not lines:
        lines.append("# market_regime_engine in-process metrics")
    return "\n".join(lines) + "\n"


__all__ = ["MetricsRegistry", "metrics", "prometheus_text", "time_block"]
