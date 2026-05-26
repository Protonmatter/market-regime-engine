"""GP-emission Bayesian Online Change-Point Detection.

Implements Saatçi-Turner-Rasmussen 2010 "Gaussian Process Change-Point
Models" — BOCPD where each run-length segment models the data with a
Gaussian process prior instead of a parametric NIW / Student-t emission.
The default kernel is RBF with median-heuristic length-scale; an optional
``deep_kernel`` callable lets advanced users plug in a learned NN feature
embedding before the RBF (the "deep kernel learning" hook from
Wilson-Hu-Salakhutdinov-Xing 2016).

This is the cleanest 2024-grade upgrade to BOCPD. Pure numpy; no soft
dependencies.

Public API mirrors :class:`market_regime_engine.bocpd_muse.BOCPDMuse`:

- ``GPBOCPD(hazard, max_run, ...).score(panel) -> pd.DataFrame``
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_regime_engine.frontier.data_cleaning import NanPolicy, clean_with_policy

_gp_log = logging.getLogger(__name__)


def _logsumexp(arr: np.ndarray) -> float:
    m = float(np.max(arr))
    if not np.isfinite(m):
        return float("-inf")
    return float(m + math.log(float(np.sum(np.exp(arr - m)))))


class _GPRun:
    """Single GP run-length state.

    v1.5 (PR-4 ASK-9): switched the per-segment observation store from
    a Python ``list`` (copy-on-append) to a fixed-size ``np.ndarray``
    ring buffer with ``max_run`` slots. The ring keeps the same
    insertion order on read via :attr:`xs`, so the kernel matrix
    constructed in :meth:`predictive_logpdf` is bit-for-bit identical
    to the v1.4 implementation for any panel processed by
    :class:`GPBOCPD.score` (proof: the BOCPD inner loop never appends
    past ``max_run`` observations onto the same segment — at iteration
    ``t = max_run + 1`` the truncation ``runs[: self.max_run]`` already
    drops the longest run, so no eviction wraps inside ``score``).

    The ring buffer is exposed independently so tests can directly
    exercise eviction behaviour past ``max_run`` (where the legacy
    implementation would have grown unbounded).

    Cost: ``update`` is ``O(max_run × d)`` (one buffer copy + one slot
    write) versus ``O(n × d)`` in the legacy list-copy path, where
    ``n`` grows with the segment length up to ``max_run``. For the
    typical FI configuration ``max_run = 96`` and ``T = 1000`` the new
    path saves the per-step cost from quadratic-in-n to constant in
    the buffer size.
    """

    __slots__ = (
        "_buffer",
        "_head",
        "_max_run",
        "_n",
        "length_scale",
        "noise_var",
        "signal_var",
    )

    def __init__(
        self,
        *,
        max_run: int,
        d: int,
        length_scale: float = 1.0,
        noise_var: float = 0.1,
        signal_var: float = 1.0,
    ) -> None:
        if max_run <= 0:
            raise ValueError(f"max_run must be positive; got {max_run!r}")
        if d <= 0:
            raise ValueError(f"d must be positive; got {d!r}")
        self._buffer: np.ndarray = np.empty((max_run, d), dtype=np.float64)
        self._head: int = 0
        self._n: int = 0
        self._max_run: int = int(max_run)
        self.length_scale: float = float(length_scale)
        self.noise_var: float = float(noise_var)
        self.signal_var: float = float(signal_var)

    def update(self, x: np.ndarray) -> _GPRun:
        """Return a new ``_GPRun`` with ``x`` appended.

        Immutability is preserved (a fresh buffer copy is made) because
        ``GPBOCPD.score`` keeps multiple per-timestep states that may
        share a prefix. Once the ring is full, the oldest observation
        is evicted in FIFO order.
        """
        new = _GPRun.__new__(_GPRun)
        new._buffer = self._buffer.copy()
        new._head = self._head
        new._n = self._n
        new._max_run = self._max_run
        new.length_scale = self.length_scale
        new.noise_var = self.noise_var
        new.signal_var = self.signal_var
        new._buffer[new._head] = np.asarray(x, dtype=np.float64)
        new._head = (new._head + 1) % new._max_run
        new._n = min(new._n + 1, new._max_run)
        return new

    @property
    def xs(self) -> np.ndarray:
        """Return valid entries in insertion order (oldest first).

        Returns a *view* into the underlying buffer when the ring is
        still filling (``_n < _max_run``) and a freshly-concatenated
        ``(_max_run, d)`` ndarray once the ring has wrapped. Either
        way the result is a 2D ``(n, d)`` ndarray that drops in for
        the legacy ``np.stack(self.xs)`` call site.
        """
        if self._n == 0:
            return self._buffer[:0]
        if self._n < self._max_run:
            return self._buffer[: self._n]
        return np.concatenate([self._buffer[self._head :], self._buffer[: self._head]])

    def predictive_logpdf(self, x: np.ndarray) -> float:
        """Log predictive pdf of ``x`` under a GP fitted on :attr:`xs`."""
        Y = self.xs
        n = Y.shape[0]
        if n == 0:
            d = len(x)
            var = self.noise_var + self.signal_var
            return float(
                -0.5 * d * math.log(2 * math.pi * max(var, 1e-9)) - 0.5 * float(np.sum((x**2) / max(var, 1e-9)))
            )
        t = np.arange(n, dtype=float).reshape(-1, 1)
        t_star = np.array([[float(n)]])
        diff = t[:, None, :] - t[None, :, :]
        K = self.signal_var * np.exp(-0.5 * np.sum(diff**2, axis=-1) / max(self.length_scale**2, 1e-9))
        K += self.noise_var * np.eye(n)
        K_star = self.signal_var * np.exp(
            -0.5 * np.sum((t - t_star.T) ** 2, axis=-1, keepdims=True) / max(self.length_scale**2, 1e-9)
        )
        K_starstar = self.signal_var + self.noise_var
        try:
            L = np.linalg.cholesky(K + 1e-9 * np.eye(n))
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
            mean_star = (K_star.T @ alpha).flatten()
            v = np.linalg.solve(L, K_star)
            quad = float(np.asarray(v.T @ v).reshape(-1)[0])
            var_star = max(K_starstar - quad, 1e-9)
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


def _median_heuristic_lengthscale(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 3:
        return 1.0
    diffs = np.diff(arr, axis=0)
    return float(max(np.median(np.sqrt(np.sum(diffs * diffs, axis=-1))), 1e-3))


@dataclass
class GPBOCPD:
    """BOCPD with GP emissions (Saatçi-Turner-Rasmussen 2010).

    v1.4 (item B) adds the ``auto_train_deep_kernel`` flag: when set,
    the first call to :meth:`score` fits a default
    :class:`market_regime_engine.frontier.deep_kernel.MLPDeepKernel`
    on the panel before applying it. This keeps the
    operator one-liner ``GPBOCPD(auto_train_deep_kernel=True).score(panel)``
    working without manually constructing the kernel.

    v1.6.0 (REVIEW_DEEP_V1_5_2.md A16 / Finding #18) hardening:

    - ``reset_kernel_per_panel`` (default ``True``): when an
      auto-trained deep kernel was fit on panel A and ``.score`` is
      then invoked on panel B, reset the kernel before fitting on B.
      Eliminates the silent reuse bug where panel A's embedding was
      applied to panel B (heterogeneous panels).
    - Auto-train deep-kernel mode is documented as
      **retrospective-only** because the embedding is fit on the
      ENTIRE panel before the BOCPD walk-forward — every detected
      change-point benefits from kernel weights that saw observations
      after it. A causal=True per-step refit is the principled fix
      and is marked as a v1.7.0 TODO.
    - The deep-kernel transform no longer silently swallows
      exceptions via ``contextlib.suppress``; failures log a warning
      and re-raise, so the operator sees that the kernel never fired.
    """

    hazard: float = 1.0 / 48.0
    max_run: int = 96
    min_prob: float = 1e-12
    deep_kernel: Callable[[np.ndarray], np.ndarray] | None = None
    auto_train_deep_kernel: bool = False
    deep_kernel_hidden_dims: tuple[int, ...] = (64, 32)
    deep_kernel_epochs: int = 50
    reset_kernel_per_panel: bool = True
    causal: bool = False  # TODO(v1.7.0): per-step kernel refit for causal mode.

    def score(
        self,
        x: pd.DataFrame,
        *,
        nan_policy: NanPolicy = NanPolicy.NAN_TO_ZERO,
        column_policies: Mapping[str, NanPolicy] | None = None,
    ) -> pd.DataFrame:
        """Score the GP-BOCPD posterior.

        v1.5 (PR-3 ASK-5/AF-8): ``nan_policy`` defaults to
        :attr:`NanPolicy.NAN_TO_ZERO` to preserve v1.4 numerics; FI
        callers may pass :attr:`NanPolicy.NAN_FAILS_PIT_AUDIT`.
        """
        if x is None or x.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "change_point_prob",
                    "bocpd_run_length_mean",
                    "bocpd_map_run_length",
                    "predictive_log_likelihood",
                ]
            )
        frame = clean_with_policy(x, default_policy=nan_policy, column_policies=column_policies).astype(float)
        arr = frame.to_numpy(float)
        if self.causal:  # pragma: no cover - v1.7.0 TODO
            raise NotImplementedError(
                "causal=True per-step kernel refit not yet implemented "
                "(REVIEW_DEEP_V1_5_2.md A16 / v1.7.0 TODO). Set causal=False "
                "and document downstream uses as retrospective-only."
            )
        # v1.6.0 (REVIEW_DEEP_V1_5_2.md A16 / Finding #18): when
        # ``reset_kernel_per_panel`` is True and the kernel was previously
        # auto-trained, drop it so the next .score call refits on the new
        # panel. Without this reset, a kernel trained on panel A is
        # silently reused on panel B (heterogeneous panels) and downstream
        # consumers cannot tell.
        if self.reset_kernel_per_panel and self.auto_train_deep_kernel:
            self.deep_kernel = None
        # v1.4: lazily build the deep-kernel adapter when auto-training
        # is requested. We construct on first ``score`` so the fit sees
        # the operator's actual panel rather than a synthetic seed.
        if self.deep_kernel is None and self.auto_train_deep_kernel:
            try:
                from market_regime_engine.frontier.deep_kernel import make_auto_train_kernel

                # Auto-train uses the entire panel; flag the look-ahead
                # explicitly so downstream operators do not silently
                # mistake this for a causal nowcasting setup.
                _gp_log.warning(
                    "GPBOCPD auto-trained deep kernel uses the ENTIRE panel "
                    "before the BOCPD walk; treat outputs as RETROSPECTIVE "
                    "only. Set causal=True (v1.7.0 TODO) for online use."
                )
                self.deep_kernel = make_auto_train_kernel(
                    frame,
                    hidden_dims=self.deep_kernel_hidden_dims,
                    n_epochs=self.deep_kernel_epochs,
                )
            except ImportError:
                # Degrade silently to the RBF default — the operator
                # opted into a deep kernel but torch is not available.
                self.deep_kernel = None
        if self.deep_kernel is not None:
            # v1.6.0 (REVIEW_DEEP_V1_5_2.md A16 / Finding #18): no longer
            # silently suppress failures; surface to the operator so the
            # kernel-fired vs RBF-fallback path is auditable.
            try:
                arr = np.asarray(self.deep_kernel(arr), dtype=float)
            except Exception as exc:
                _gp_log.warning(
                    "GPBOCPD deep_kernel transform failed (%s); "
                    "falling back to raw features. This used to silently "
                    "no-op via contextlib.suppress; now logged.",
                    exc,
                )
                # Re-raise so the failure is visible; callers that want
                # the previous silent-fallback behaviour must wrap the
                # call in their own try/except.
                raise
        ls = _median_heuristic_lengthscale(arr)
        d = int(arr.shape[1]) if arr.ndim > 1 else 1
        runs: list[_GPRun] = [_GPRun(max_run=self.max_run, d=d, length_scale=ls, noise_var=0.1, signal_var=1.0)]
        log_joint = np.array([0.0], dtype=float)
        h = float(self.hazard)
        log_h = math.log(max(h, self.min_prob))
        log_1mh = math.log(max(1.0 - h, self.min_prob))
        rows = []
        for i, date in enumerate(frame.index):
            xt = arr[i]
            pred_logs = np.array([r.predictive_logpdf(xt) for r in runs], dtype=float)
            cp_log = _logsumexp(log_joint + pred_logs + log_h)
            growth_logs = log_joint + pred_logs + log_1mh
            new_log_joint = np.empty(min(len(growth_logs) + 1, self.max_run + 1), dtype=float)
            new_log_joint[0] = cp_log
            kept = growth_logs[: self.max_run]
            new_log_joint[1 : 1 + len(kept)] = kept
            norm = _logsumexp(new_log_joint)
            new_log_joint = new_log_joint - norm
            probs = np.exp(new_log_joint)
            probs = probs / probs.sum()
            cp_prob = float(probs[0])
            run_lengths = np.arange(len(probs), dtype=float)
            mean_run = float(np.sum(run_lengths * probs))
            map_run = int(np.argmax(probs))
            ll = float(_logsumexp(log_joint + pred_logs))
            new_runs = [_GPRun(max_run=self.max_run, d=d, length_scale=ls, noise_var=0.1, signal_var=1.0)] + [
                r.update(xt) for r in runs[: self.max_run]
            ]
            new_runs = new_runs[: len(probs)]
            log_joint = np.log(np.maximum(probs, self.min_prob))
            runs = new_runs
            rows.append(
                {
                    "date": date,
                    "change_point_prob": cp_prob,
                    "bocpd_run_length_mean": mean_run,
                    "bocpd_map_run_length": map_run,
                    "predictive_log_likelihood": ll,
                }
            )
        return pd.DataFrame(rows)


__all__ = ["GPBOCPD"]
