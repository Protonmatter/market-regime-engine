# SPDX-License-Identifier: Apache-2.0
"""Benchmark harness for hot-path kernels.

Runs the BOCPD, WFST, PSI, and rolling-statistics kernels at three problem
sizes and reports timings + memory footprint. The output is the contract that
gates Rust kernel promotion: a Rust kernel earns its place when it beats the
Python reference on this harness while passing the parity tests in
``tests/test_rust_parity.py``.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable

import numpy as np
import pandas as pd

from market_regime_engine.bocpd import DiagonalStudentTBOCPD, MultivariateNIWBOCPD
from market_regime_engine.changepoint import RollingMultivariateChangePoint
from market_regime_engine.drift import population_stability_index
from market_regime_engine.wfst import RegimeWFST


def _time_with_memory(fn: Callable[[], object]) -> tuple[float, float]:
    tracemalloc.start()
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / 1e6  # MB


def _bocpd_diag(n: int, d: int, seed: int) -> Callable[[], object]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    x = pd.DataFrame(rng.normal(size=(n, d)), index=idx, columns=[f"f{i}" for i in range(d)])
    detector = DiagonalStudentTBOCPD(max_run=min(96, n // 2))
    return lambda: detector.score(x)


def _bocpd_niw(n: int, d: int, seed: int) -> Callable[[], object]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    x = pd.DataFrame(rng.normal(size=(n, d)), index=idx, columns=[f"f{i}" for i in range(d)])
    detector = MultivariateNIWBOCPD(max_run=min(64, n // 2))
    return lambda: detector.score(x)


def _wfst_decode(n: int, d: int, seed: int) -> Callable[[], object]:
    rng = np.random.default_rng(seed)
    states = sorted(RegimeWFST().states)
    obs = [states[int(i)] for i in rng.integers(0, len(states), size=n)]
    decoder = RegimeWFST()
    return lambda: decoder.decode(obs)


def _rolling_mahalanobis(n: int, d: int, seed: int) -> Callable[[], object]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    x = pd.DataFrame(rng.normal(size=(n, d)), index=idx, columns=[f"f{i}" for i in range(d)])
    detector = RollingMultivariateChangePoint()
    return lambda: detector.score(x)


def _psi(n: int, d: int, seed: int) -> Callable[[], object]:
    rng = np.random.default_rng(seed)
    a = pd.Series(rng.normal(size=n))
    b = pd.Series(rng.normal(loc=0.2, size=n))
    return lambda: population_stability_index(a, b)


def run_bench_suite(seed: int = 0) -> pd.DataFrame:
    """Run all kernels at three sizes and return a tidy result frame."""
    rows: list[dict] = []
    sizes = [
        ("small", 60, 4),
        ("medium", 240, 8),
        ("large", 600, 12),
    ]
    kernels = [
        ("bocpd_diag", _bocpd_diag),
        ("bocpd_niw", _bocpd_niw),
        ("wfst_decode", _wfst_decode),
        ("rolling_maha", _rolling_mahalanobis),
        ("psi", _psi),
    ]
    for name, fn_factory in kernels:
        for size_name, n, d in sizes:
            fn = fn_factory(n, d, seed)
            elapsed, peak_mb = _time_with_memory(fn)
            rows.append(
                {
                    "kernel": name,
                    "size": size_name,
                    "n": int(n),
                    "d": int(d),
                    "elapsed_seconds": float(elapsed),
                    "peak_memory_mb": float(peak_mb),
                    "implementation": "python_reference",
                    "seed": int(seed),
                }
            )
    return pd.DataFrame(rows)


__all__ = ["run_bench_suite"]
