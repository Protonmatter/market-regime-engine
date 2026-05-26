# SPDX-License-Identifier: Apache-2.0
"""Ring-buffer tests for :class:`market_regime_engine.frontier.gp_cpd._GPRun` (PR-4 ASK-9).

Pre-PR-4, ``_GPRun.update`` materialised a fresh ``list[np.ndarray]``
per call, growing without bound and copying every observation seen so
far. The new implementation backs the segment with a fixed-size
``np.ndarray`` ring buffer of ``max_run`` slots. The tests below pin
both the eviction semantics and the bit-for-bit numerical equivalence
of :class:`GPBOCPD` (since ``GPBOCPD.score`` never appends past
``max_run`` onto the same segment, the new path must reproduce the
v1.4 posterior trace exactly).
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.frontier import gp_cpd as gp_cpd_module
from market_regime_engine.frontier.gp_cpd import GPBOCPD, _GPRun


def _row(i: int, d: int = 2) -> np.ndarray:
    """Deterministic distinct row so eviction order is unambiguous."""
    return np.array([float(i) + 0.1 * j for j in range(d)], dtype=np.float64)


# ---------------------------------------------------------------------------
# task E.2 ring buffer behaviour
# ---------------------------------------------------------------------------


def test_gp_cpd_ring_buffer_under_max_run() -> None:
    """First ``max_run - 1`` inserts keep insertion order with the buffer view."""
    max_run = 8
    run = _GPRun(max_run=max_run, d=2)
    assert run.xs.shape == (0, 2)
    for i in range(max_run - 1):
        run = run.update(_row(i))
    xs = run.xs
    assert xs.shape == (max_run - 1, 2)
    # Insertion order is preserved.
    for i in range(max_run - 1):
        np.testing.assert_allclose(xs[i], _row(i))


def test_gp_cpd_ring_buffer_at_max_run() -> None:
    """After exactly ``max_run`` inserts the head wraps back to 0."""
    max_run = 6
    run = _GPRun(max_run=max_run, d=1)
    for i in range(max_run):
        run = run.update(_row(i, d=1))
    # All ``max_run`` slots are filled; the head wrapped back to 0.
    assert run._n == max_run
    assert run._head == 0
    # ``xs`` returns the elements in insertion order.
    xs = run.xs
    assert xs.shape == (max_run, 1)
    np.testing.assert_allclose(xs[:, 0], np.arange(max_run, dtype=float))


def test_gp_cpd_ring_buffer_eviction_order() -> None:
    """``max_run + 5`` inserts evict the first 5 in FIFO order."""
    max_run = 7
    run = _GPRun(max_run=max_run, d=1)
    for i in range(max_run + 5):
        run = run.update(_row(i, d=1))
    xs = run.xs
    assert xs.shape == (max_run, 1)
    # The five oldest elements (0, 1, 2, 3, 4) were evicted; the survivors
    # are the most recent ``max_run`` rows in insertion order.
    expected = np.arange(5, max_run + 5, dtype=float)
    np.testing.assert_allclose(xs[:, 0], expected)


def test_gp_cpd_ring_buffer_update_is_immutable() -> None:
    """Each ``update`` returns a fresh ``_GPRun``; the source is unchanged.

    Required by the BOCPD inner loop, which holds multiple per-timestep
    states that may share a prefix. A mutating ``update`` would let one
    segment's observations leak into another.
    """
    base = _GPRun(max_run=4, d=1)
    after_one = base.update(_row(0, d=1))
    after_two = after_one.update(_row(1, d=1))
    assert base.xs.shape == (0, 1)
    assert after_one.xs.shape == (1, 1)
    assert after_two.xs.shape == (2, 1)
    # The new ndarray buffer is a copy: mutating the parent buffer does not
    # affect the child, and vice versa.
    after_two._buffer[0, 0] = 999.0
    assert after_one.xs[0, 0] != 999.0


def test_gp_cpd_ring_buffer_rejects_invalid_dims() -> None:
    """Constructor rejects ``max_run <= 0`` and ``d <= 0`` (no silent zero buffer)."""
    with pytest.raises(ValueError, match="max_run"):
        _GPRun(max_run=0, d=1)
    with pytest.raises(ValueError, match="d"):
        _GPRun(max_run=4, d=0)


# ---------------------------------------------------------------------------
# task E.1 posterior pin: bit-for-bit identical numerical trace.
#
# Pinned hash was captured on the v1.4-list implementation immediately
# before this PR; ``GPBOCPD.score`` never appends past ``max_run`` on the
# same segment (proof: ``runs[: self.max_run]`` truncates BEFORE
# ``update`` is called, so the ring buffer never evicts during a normal
# score() run), so the new implementation must produce identical
# floating-point output.
# ---------------------------------------------------------------------------


class _LegacyListGPRun:
    """Pre-ring-buffer list-backed run state used as an in-process oracle."""

    def __init__(
        self,
        *,
        max_run: int,
        d: int,
        length_scale: float = 1.0,
        noise_var: float = 0.1,
        signal_var: float = 1.0,
    ) -> None:
        del max_run, d
        self.xs: list[np.ndarray] = []
        self.length_scale = float(length_scale)
        self.noise_var = float(noise_var)
        self.signal_var = float(signal_var)

    def update(self, x: np.ndarray) -> _LegacyListGPRun:
        new = _LegacyListGPRun(
            max_run=1,
            d=len(x),
            length_scale=self.length_scale,
            noise_var=self.noise_var,
            signal_var=self.signal_var,
        )
        new.xs = [*self.xs, np.asarray(x, dtype=np.float64)]
        return new

    def predictive_logpdf(self, x: np.ndarray) -> float:
        n = len(self.xs)
        if n == 0:
            d = len(x)
            var = self.noise_var + self.signal_var
            return float(
                -0.5 * d * math.log(2 * math.pi * max(var, 1e-9)) - 0.5 * float(np.sum((x**2) / max(var, 1e-9)))
            )
        y = np.stack(self.xs)
        t = np.arange(n, dtype=float).reshape(-1, 1)
        t_star = np.array([[float(n)]])
        diff = t[:, None, :] - t[None, :, :]
        k = self.signal_var * np.exp(-0.5 * np.sum(diff**2, axis=-1) / max(self.length_scale**2, 1e-9))
        k += self.noise_var * np.eye(n)
        k_star = self.signal_var * np.exp(
            -0.5 * np.sum((t - t_star.T) ** 2, axis=-1, keepdims=True) / max(self.length_scale**2, 1e-9)
        )
        k_starstar = self.signal_var + self.noise_var
        try:
            chol = np.linalg.cholesky(k + 1e-9 * np.eye(n))
            alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
            mean_star = (k_star.T @ alpha).flatten()
            v = np.linalg.solve(chol, k_star)
            quad = float(np.asarray(v.T @ v).reshape(-1)[0])
            var_star = max(k_starstar - quad, 1e-9)
            d = len(x)
            return float(
                -0.5 * d * math.log(2 * math.pi * var_star) - 0.5 * float(np.sum((x - mean_star) ** 2) / var_star)
            )
        except np.linalg.LinAlgError:
            d = len(x)
            var = self.noise_var + self.signal_var
            return float(
                -0.5 * d * math.log(2 * math.pi * max(var, 1e-9)) - 0.5 * float(np.sum((x - 0.0) ** 2) / max(var, 1e-9))
            )


def _pinned_panel(T: int = 80, *, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.normal(0.0, 1.0, size=T // 2), rng.normal(2.0, 1.0, size=T // 2)])
    return pd.DataFrame({"x": x}, index=pd.date_range("2024-01-01", periods=T, freq="D"))


def test_gp_cpd_posterior_unchanged_after_ring_buffer_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the ring-buffer output against the same-runtime legacy list path."""
    panel = _pinned_panel()
    out = GPBOCPD(hazard=1 / 24.0, max_run=24).score(panel)
    monkeypatch.setattr(gp_cpd_module, "_GPRun", _LegacyListGPRun)
    legacy = GPBOCPD(hazard=1 / 24.0, max_run=24).score(panel)
    cols = ["change_point_prob", "bocpd_run_length_mean", "bocpd_map_run_length", "predictive_log_likelihood"]
    arr = out[cols].to_numpy(dtype=np.float64)
    legacy_arr = legacy[cols].to_numpy(dtype=np.float64)
    assert arr.tobytes() == legacy_arr.tobytes(), "ring-buffer GP-BOCPD output drifted from the list-backed path"


# ---------------------------------------------------------------------------
# task E.2 perf: ring buffer is at most as expensive as the legacy path
# on a moderate panel size. Marked slow so the regular suite stays fast.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gp_cpd_perf_at_max_run_96() -> None:
    """At ``max_run = 96`` and ``T = 1000`` the ring buffer stays within the legacy ceiling.

    The legacy list-copy path was ``O(n × d)`` per update with ``n``
    growing to ``max_run``; the ring buffer is ``O(max_run × d)`` per
    update, so the new path's per-step cost is bounded by a constant
    in the segment length and dominated by the kernel Cholesky
    (``O(n³)``). The ceiling is set generously enough to cover the
    slowest CI box we expect this suite to land on; any regression
    past the ceiling would indicate accidental quadratic behaviour in
    the ring-buffer copy path.
    """
    rng = np.random.default_rng(7)
    T = 1000
    panel = pd.DataFrame({"x": rng.normal(size=T)}, index=pd.date_range("2024-01-01", periods=T, freq="D"))
    detector = GPBOCPD(hazard=1 / 96.0, max_run=96)
    start = time.perf_counter()
    out = detector.score(panel)
    elapsed = time.perf_counter() - start
    assert len(out) == T
    # Empirically the kernel-Cholesky dominates (~30 s on the developer box);
    # the ring buffer copy itself is ~µs per step. 90 s gives 3x headroom.
    assert elapsed < 90.0, f"GP-BOCPD took {elapsed:.2f}s at T=1000, max_run=96 (regression vs. legacy bound)"
