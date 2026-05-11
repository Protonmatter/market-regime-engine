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

Bounded histograms (v1.5 PR-5, AF-3)
------------------------------------
The pre-v1.5 ``observe(name, value)`` path appended to an unbounded
``list[float]`` per ``(name, labels)`` key. On a long-running FastAPI worker
that received millions of execution-confidence requests, the histogram
storage grew without bound and dominated the process resident-set. PR-5
swaps the list for :class:`BoundedHistogram`, a fixed-size
reservoir-sampler-backed structure: exact ``count`` / ``sum``, approximate
quantiles via Algorithm R (Vitter 1985) over a 4096-slot buffer. The public
``record`` / ``observe`` / ``incr`` API and ``snapshot`` output keys are
unchanged so existing dashboards keep working.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import defaultdict
from collections.abc import Iterator
from threading import Lock

import numpy as np

log = logging.getLogger(__name__)


_DEFAULT_RESERVOIR_SIZE: int = 4096


class BoundedHistogram:
    """Fixed-size reservoir-sampler-backed histogram (v1.5 PR-5, AF-3).

    Maintains:

    - exact ``count`` (every ``record`` increments it),
    - exact ``sum`` (every ``record`` adds to it),
    - a fixed-size ``reservoir`` of ``reservoir_size`` slots for approximate
      quantile estimation via Algorithm R (Vitter 1985).

    Suitable for streaming workloads where unbounded ``list[float]`` storage
    is unacceptable. The default ``reservoir_size=4096`` matches Prometheus
    summary practice: the 95th / 99th percentile estimates stabilise within
    a few percentage points on stationary workloads while resident memory
    stays bounded at 32 KB per histogram (4096 × 8 bytes float64).
    """

    __slots__ = ("_buffer", "_count", "_sum", "_rng", "_size")

    def __init__(self, reservoir_size: int = _DEFAULT_RESERVOIR_SIZE, *, seed: int = 0) -> None:
        if reservoir_size <= 0:
            raise ValueError(f"reservoir_size must be positive; got {reservoir_size!r}")
        self._size = int(reservoir_size)
        self._buffer = np.empty(self._size, dtype=np.float64)
        self._count = 0
        self._sum = 0.0
        self._rng = np.random.default_rng(seed)

    def record(self, value: float) -> None:
        """Record ``value`` (Algorithm R reservoir sampling)."""
        v = float(value)
        self._sum += v
        if self._count < self._size:
            self._buffer[self._count] = v
        else:
            # Algorithm R: replace a random slot with probability size/(count+1).
            j = int(self._rng.integers(0, self._count + 1))
            if j < self._size:
                self._buffer[j] = v
        self._count += 1

    def quantile(self, q: float) -> float:
        """Approximate ``q``-quantile from the reservoir.

        Returns 0.0 when no samples have been recorded (mirrors the v1.1
        empty-histogram convention for back-compat).
        """
        if self._count == 0:
            return 0.0
        n = min(self._count, self._size)
        view = np.sort(self._buffer[:n])
        idx = int(round(float(q) * (n - 1)))
        idx = max(0, min(n - 1, idx))
        return float(view[idx])

    @property
    def count(self) -> int:
        return self._count

    @property
    def sum(self) -> float:
        return self._sum

    @property
    def reservoir_size(self) -> int:
        return self._size

    def reservoir_view(self) -> np.ndarray:
        """Return a copy of the currently-populated reservoir slots."""
        n = min(self._count, self._size)
        return self._buffer[:n].copy()


class MetricsRegistry:
    """Thread-safe in-process metrics store.

    v1.5 (PR-5 AF-3): histograms are bounded reservoir samplers
    (:class:`BoundedHistogram`) instead of an unbounded ``list[float]`` so a
    long-running FastAPI worker cannot leak memory through metric
    accumulation. Public ``incr`` / ``observe`` / ``snapshot`` are
    unchanged.
    """

    def __init__(self, *, reservoir_size: int = _DEFAULT_RESERVOIR_SIZE) -> None:
        self._lock = Lock()
        self._counters: defaultdict[tuple[str, frozenset], float] = defaultdict(float)
        self._reservoir_size = int(reservoir_size)
        self._histograms: defaultdict[tuple[str, frozenset], BoundedHistogram] = defaultdict(
            self._make_histogram
        )

    def _make_histogram(self) -> BoundedHistogram:
        return BoundedHistogram(reservoir_size=self._reservoir_size)

    def incr(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = (name, frozenset(labels.items()))
        with self._lock:
            self._counters[key] += float(value)

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = (name, frozenset(labels.items()))
        with self._lock:
            self._histograms[key].record(float(value))

    # Alias used in some PR-5 callers; ``observe`` is the v1.1 public name.
    record_histogram = observe

    def snapshot(self) -> dict:
        with self._lock:
            counters = {self._format_key(k): v for k, v in self._counters.items()}
            histograms = {
                self._format_key(k): {
                    "count": int(hist.count),
                    "sum": float(hist.sum),
                    "p50": hist.quantile(0.5),
                    "p95": hist.quantile(0.95),
                    "p99": hist.quantile(0.99),
                }
                for k, hist in self._histograms.items()
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
    """Legacy list-based quantile helper kept for back-compat in callers
    that pass a Python list (e.g. tests that build a synthetic registry
    snapshot)."""
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


__all__ = ["BoundedHistogram", "MetricsRegistry", "metrics", "prometheus_text", "time_block"]
