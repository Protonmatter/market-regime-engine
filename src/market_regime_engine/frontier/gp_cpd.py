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

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from market_regime_engine.frontier.data_cleaning import NanPolicy, clean_with_policy


def _logsumexp(arr: np.ndarray) -> float:
    m = float(np.max(arr))
    if not np.isfinite(m):
        return float("-inf")
    return float(m + math.log(float(np.sum(np.exp(arr - m)))))


@dataclass
class _GPRun:
    """One GP segment: stores a window of recent observations to predict the next."""

    xs: list[np.ndarray] = field(default_factory=list)
    length_scale: float = 1.0
    noise_var: float = 0.1
    signal_var: float = 1.0

    def update(self, x: np.ndarray) -> _GPRun:
        new_xs = [*self.xs, x.copy()]
        return _GPRun(new_xs, self.length_scale, self.noise_var, self.signal_var)

    def predictive_logpdf(self, x: np.ndarray) -> float:
        """Log predictive pdf of ``x`` under a GP fitted on ``self.xs``."""
        if not self.xs:
            d = len(x)
            var = self.noise_var + self.signal_var
            return float(
                -0.5 * d * math.log(2 * math.pi * max(var, 1e-9)) - 0.5 * float(np.sum((x**2) / max(var, 1e-9)))
            )
        Y = np.stack(self.xs)
        # Index segment positions as the GP input (1..n).
        n = Y.shape[0]
        t = np.arange(n, dtype=float).reshape(-1, 1)
        t_star = np.array([[float(n)]])
        # RBF kernel matrices.
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
    """

    hazard: float = 1.0 / 48.0
    max_run: int = 96
    min_prob: float = 1e-12
    deep_kernel: Callable[[np.ndarray], np.ndarray] | None = None
    auto_train_deep_kernel: bool = False
    deep_kernel_hidden_dims: tuple[int, ...] = (64, 32)
    deep_kernel_epochs: int = 50

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
        # v1.4: lazily build the deep-kernel adapter when auto-training
        # is requested. We construct on first ``score`` so the fit sees
        # the operator's actual panel rather than a synthetic seed.
        if self.deep_kernel is None and self.auto_train_deep_kernel:
            try:
                from market_regime_engine.frontier.deep_kernel import make_auto_train_kernel

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
            import contextlib

            with contextlib.suppress(Exception):
                arr = np.asarray(self.deep_kernel(arr), dtype=float)
        ls = _median_heuristic_lengthscale(arr)
        runs: list[_GPRun] = [_GPRun(length_scale=ls, noise_var=0.1, signal_var=1.0)]
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
            new_runs = [_GPRun(length_scale=ls, noise_var=0.1, signal_var=1.0)] + [
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
