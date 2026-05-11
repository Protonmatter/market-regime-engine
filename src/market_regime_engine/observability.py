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
import os
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
        idx = round(float(q) * (n - 1))
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


# ---------------------------------------------------------------------------
# v1.5 PR-7 §E — OpenTelemetry SDK migration (AF-4 / P1)
# ---------------------------------------------------------------------------
#
# The legacy ``_GLOBAL`` registry stays as the always-available
# in-process backend. When ``configure_otel(...)`` is called and the
# OpenTelemetry SDK is installed (``[observability]`` extra), every
# ``incr`` / ``record_histogram`` / ``time_block`` call is mirrored to
# OTel counters / histograms keyed by the same name. Operators can
# scrape via the OTLP exporter, Prometheus exporter, or the legacy
# ``prometheus_text()``; cross-worker aggregation is now possible
# because OTel handles fan-in at the collector level.
#
# Pre-registered FI metric instruments live in
# :mod:`market_regime_engine.fixed_income.observability_ext` so a fresh
# warehouse without any ingest still exposes the canonical counters at
# a 0 starting value (matches the AGENT.md PR-7 dashboard contract).

_OTEL_ENABLED: bool = False
_OTEL_METER: object | None = None
_OTEL_TRACER: object | None = None
_OTEL_LOCK = Lock()
_OTEL_COUNTERS: dict[str, object] = {}
_OTEL_HISTOGRAMS: dict[str, object] = {}


def _otel_available() -> bool:
    try:
        import opentelemetry  # type: ignore[import-not-found]  # noqa: F401
    except Exception:
        return False
    return True


def configure_otel(
    *,
    service_name: str = "market-regime-engine",
    exporter_endpoint: str | None = None,
    enabled: bool | None = None,
) -> bool:
    """Initialise the OpenTelemetry SDK with an OTLP exporter.

    Parameters
    ----------
    service_name
        ``service.name`` resource attribute.
    exporter_endpoint
        OTLP collector URL. Defaults to the OTel-standard
        ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var.
    enabled
        ``True`` to force-enable, ``False`` to force-disable. ``None``
        (default) enables when the SDK is installed and otherwise
        soft-degrades to the in-process backend.

    Returns ``True`` when OTel is now active, ``False`` otherwise.
    Calling this function more than once is safe — the second call
    replaces the meter / tracer providers.
    """
    global _OTEL_ENABLED, _OTEL_METER, _OTEL_TRACER

    want_enabled = bool(enabled) if enabled is not None else _otel_available()
    if not want_enabled:
        with _OTEL_LOCK:
            _OTEL_ENABLED = False
            _OTEL_METER = None
            _OTEL_TRACER = None
            _OTEL_COUNTERS.clear()
            _OTEL_HISTOGRAMS.clear()
        return False
    if not _otel_available():
        log.warning(
            "configure_otel(enabled=True) requested but the opentelemetry "
            "package is not installed; falling back to the in-process "
            "MetricsRegistry. Install via `pip install "
            "market-regime-engine[observability]` to enable OTLP."
        )
        return False
    try:
        from opentelemetry import metrics as otel_metrics
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import (
            ConsoleMetricExporter,
            PeriodicExportingMetricReader,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
    except Exception as exc:  # pragma: no cover - import path
        log.warning("configure_otel SDK import failed: %s", exc)
        return False

    resource = Resource.create({"service.name": service_name})

    # Choose exporter: OTLP if endpoint env / arg set, else console.
    endpoint = exporter_endpoint or None
    try:
        if endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )

            metric_exporter: object = OTLPMetricExporter(endpoint=endpoint)
        else:
            metric_exporter = ConsoleMetricExporter()
    except Exception as exc:  # pragma: no cover - exporter import
        log.warning("OTLP metric exporter unavailable (%s); using console.", exc)
        metric_exporter = ConsoleMetricExporter()

    reader = PeriodicExportingMetricReader(
        metric_exporter,  # type: ignore[arg-type]
        export_interval_millis=60_000,
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    otel_metrics.set_meter_provider(meter_provider)

    tracer_provider = TracerProvider(resource=resource)
    otel_trace.set_tracer_provider(tracer_provider)

    with _OTEL_LOCK:
        _OTEL_ENABLED = True
        _OTEL_METER = otel_metrics.get_meter("market_regime_engine")
        _OTEL_TRACER = otel_trace.get_tracer("market_regime_engine")
        _OTEL_COUNTERS.clear()
        _OTEL_HISTOGRAMS.clear()
    log.info("OpenTelemetry SDK configured: service=%s endpoint=%s", service_name, endpoint)
    return True


def get_meter() -> object | None:
    """Return the configured OTel ``Meter``, or ``None`` when disabled."""
    return _OTEL_METER


def get_tracer() -> object | None:
    """Return the configured OTel ``Tracer``, or ``None`` when disabled."""
    return _OTEL_TRACER


def otel_enabled() -> bool:
    """Whether ``configure_otel`` has lit up the OTLP routing path."""
    return _OTEL_ENABLED


def _otel_counter(name: str) -> object | None:
    if not _OTEL_ENABLED or _OTEL_METER is None:
        return None
    counter = _OTEL_COUNTERS.get(name)
    if counter is not None:
        return counter
    with _OTEL_LOCK:
        counter = _OTEL_COUNTERS.get(name)
        if counter is not None:
            return counter
        try:
            counter = _OTEL_METER.create_counter(  # type: ignore[attr-defined]
                name=name,
                description=name,
            )
        except Exception as exc:  # pragma: no cover - sdk path
            log.warning("OTel counter create failed (%s): %s", name, exc)
            counter = None
        if counter is not None:
            _OTEL_COUNTERS[name] = counter
    return counter


def _otel_histogram(name: str) -> object | None:
    if not _OTEL_ENABLED or _OTEL_METER is None:
        return None
    hist = _OTEL_HISTOGRAMS.get(name)
    if hist is not None:
        return hist
    with _OTEL_LOCK:
        hist = _OTEL_HISTOGRAMS.get(name)
        if hist is not None:
            return hist
        try:
            hist = _OTEL_METER.create_histogram(  # type: ignore[attr-defined]
                name=name,
                description=name,
            )
        except Exception as exc:  # pragma: no cover - sdk path
            log.warning("OTel histogram create failed (%s): %s", name, exc)
            hist = None
        if hist is not None:
            _OTEL_HISTOGRAMS[name] = hist
    return hist


def incr(name: str, value: float = 1.0, **labels: str) -> None:
    """Increment a counter on the legacy registry AND OTel when enabled.

    Adapter shim per AF-4: existing callers continue using
    ``metrics().incr(...)`` (legacy in-process); a new top-level
    :func:`incr` is also exposed so FI components can route through one
    canonical entry point that mirrors to OTel.
    """
    _GLOBAL.incr(name, value, **labels)
    counter = _otel_counter(name)
    if counter is not None:
        try:
            counter.add(  # type: ignore[attr-defined]
                value, attributes=dict(labels) if labels else None
            )
        except Exception as exc:  # pragma: no cover - sdk path
            log.warning("OTel counter add failed (%s): %s", name, exc)


def record_histogram(name: str, value: float, **labels: str) -> None:
    """Record a histogram observation on the legacy registry AND OTel."""
    _GLOBAL.observe(name, value, **labels)
    hist = _otel_histogram(name)
    if hist is not None:
        try:
            hist.record(  # type: ignore[attr-defined]
                value, attributes=dict(labels) if labels else None
            )
        except Exception as exc:  # pragma: no cover - sdk path
            log.warning("OTel histogram record failed (%s): %s", name, exc)


@contextlib.contextmanager
def time_block(name: str, **labels: str) -> Iterator[None]:
    """Context manager recording the elapsed seconds in a histogram.

    v1.5 PR-7 §E: also routes the observation to the OTel histogram
    when ``configure_otel(...)`` is active. Legacy in-process recording
    is unchanged so dashboards built on ``prometheus_text()`` continue
    to render.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        record_histogram(name, elapsed, **labels)


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


__all__ = [
    "BoundedHistogram",
    "MetricsRegistry",
    "configure_otel",
    "get_meter",
    "get_tracer",
    "incr",
    "metrics",
    "otel_enabled",
    "prometheus_text",
    "record_histogram",
    "time_block",
]
