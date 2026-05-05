"""BOCPD with Model Uncertainty (BOCPD-MUSE).

Knoblauch and Damoulas (2018) extend the Adams-MacKay BOCPD recursion to
explicitly average over a small set of *emission models*. Each timestep, the
posterior over (run-length, model) is updated jointly. Compared to picking
one emission model up-front:

- Robustness improves under regime-dependent emission shape (heavy-tail
  during stress, near-Gaussian during expansion).
- The marginal change-point probability is the model-averaged value, which
  sidesteps the ``wrong-model fights right-data`` failure mode of single-core
  BOCPD.

This module wires the same NIW and diagonal-Student-t cores as
:mod:`market_regime_engine.bocpd`, plus a third optional AR(1) emission with
Student-t innovations. The resulting posterior over emission models is also
returned, which lets downstream code surface "which emission model the data
prefers right now" — a useful diagnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_regime_engine.bocpd import (
    NIWState,
    RunningDiagState,
    _logsumexp,
    _student_t_logpdf_diag,
)


@dataclass
class _AR1State:
    """Tiny diagonal AR(1) emission with Student-t innovations.

    Each component j has its own AR(1) coefficient phi_j fitted online via
    ridge-regularized least squares on a running window. This is intentionally
    lightweight — it captures persistence without dragging in a full VAR.
    """

    n: int
    last_x: np.ndarray
    sum_x: np.ndarray
    sum_x_lag_x: np.ndarray
    sum_x_lag_sq: np.ndarray
    m2: np.ndarray
    prior_var: float = 1.0

    @classmethod
    def prior(cls, dim: int, prior_var: float = 1.0) -> _AR1State:
        return cls(
            n=0,
            last_x=np.zeros(dim, dtype=float),
            sum_x=np.zeros(dim, dtype=float),
            sum_x_lag_x=np.zeros(dim, dtype=float),
            sum_x_lag_sq=np.zeros(dim, dtype=float),
            m2=np.zeros(dim, dtype=float),
            prior_var=prior_var,
        )

    def update(self, x: np.ndarray) -> _AR1State:
        x = np.asarray(x, dtype=float)
        if self.n == 0:
            return _AR1State(
                1, x.copy(), x.copy(), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), self.prior_var
            )
        # v1.3 (item B2): the Welford m2 update is now computed BEFORE
        # ``sum_x`` is updated for the new step, using purely the prior
        # mean. The legacy code computed m2 with ``(x - new_sum_x/new_n)``
        # which numerically mixes the new sample into both terms; the
        # canonical prior-mean form is
        #
        #     M2_new = M2 + (x_new - mean_old)^2 * n / (n + 1)
        #
        # which is identical for finite-precision arithmetic to the
        # Welford ``delta_old * delta_new`` form but does not require
        # computing the new mean at all. The regression test confirms
        # ``_AR1State.m2 / (n-1)`` matches ``np.var(x[:n], ddof=1)`` to
        # ``atol=1e-10`` on a 1000-step random walk.
        new_n = self.n + 1
        prior_mean = self.sum_x / float(self.n)
        delta_old = x - prior_mean
        m2 = self.m2 + (delta_old**2) * (float(self.n) / float(new_n))
        # Now safe to update sum_x for the new step.
        new_sum_x = self.sum_x + x
        # Centered AR(1) regression: x_t - mu = phi * (x_{t-1} - mu) + eps_t.
        # Centering the cross-products with the running mean keeps phi-hat
        # unbiased even when E[x] != 0 (the un-centered version aliases the
        # mean into phi and inflates persistence on biased series).
        # Pairwise mean (n samples enter the lagged regression — index 1..n).
        mean_pair = new_sum_x / new_n
        x_c = x - mean_pair
        x_lag_c = self.last_x - mean_pair
        sum_x_lag_x = self.sum_x_lag_x + x_lag_c * x_c
        sum_x_lag_sq = self.sum_x_lag_sq + x_lag_c**2
        return _AR1State(
            new_n,
            x.copy(),
            new_sum_x,
            sum_x_lag_x,
            sum_x_lag_sq,
            m2,
            self.prior_var,
        )

    def predict(self, x: np.ndarray, *, min_df: float = 3.0) -> float:
        if self.n < 4:
            # not enough samples; defer to a Student-t with prior variance
            return float(np.sum(_student_logpdf_scalar(x, np.zeros_like(x), np.full_like(x, self.prior_var), min_df)))
        mean_overall = self.sum_x / max(self.n, 1)
        denom = np.maximum(self.sum_x_lag_sq, 1e-9)
        phi = np.clip(self.sum_x_lag_x / denom, -0.99, 0.99)
        # E[x_t | x_{t-1}] = mu + phi * (x_{t-1} - mu)
        mean = mean_overall + phi * (self.last_x - mean_overall)
        var = np.maximum(self.m2 / max(self.n - 1, 1), 1e-8)
        return float(np.sum(_student_logpdf_scalar(x, mean, var, max(min_df, self.n + 1))))


def _student_logpdf_scalar(x: np.ndarray, mean: np.ndarray, var: np.ndarray, df: float) -> np.ndarray:
    scale = np.sqrt(np.maximum(var * (1.0 + 1.0 / max(df, 1)), 1e-8))
    z = (x - mean) / scale
    c = math.lgamma((df + 1.0) / 2.0) - math.lgamma(df / 2.0) - 0.5 * math.log(df * math.pi)
    return c - np.log(scale) - ((df + 1.0) / 2.0) * np.log1p((z * z) / df)


@dataclass
class BOCPDMuse:
    """BOCPD averaged across {NIW, diagonal-Student-t, diagonal-AR1-Student-t}.

    The recursion runs three independent BOCPD trackers and folds them into a
    Bayesian model average using an online posterior over the model index
    seeded with a flat Dirichlet prior.
    """

    hazard: float = 1.0 / 48.0
    max_run: int = 96
    min_prob: float = 1e-12
    prior_kappa: float = 1.0
    prior_psi_scale: float = 1.0
    prior_var_diag: float = 1.0

    def score(self, x: pd.DataFrame) -> pd.DataFrame:
        if x is None or x.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "change_point_prob",
                    "bocpd_run_length_mean",
                    "bocpd_map_run_length",
                    "predictive_log_likelihood",
                    "model_post_niw",
                    "model_post_diag",
                    "model_post_ar1",
                ]
            )
        # ``.ffill().fillna(0.0)`` assumes the caller has already enforced
        # point-in-time / vintage discipline upstream (the as-of materializer
        # in :mod:`market_regime_engine.asof` produces NaN-free frames audited
        # by the ``audit-vintage`` CLI). Forward-filling here only patches
        # numerical artifacts (e.g. an exact 0.0 from a robust-z divisor); it
        # MUST NOT be used to backfill missing vintages — that would leak
        # post-publication information into change-point detection.
        frame = x.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).astype(float)
        arr = frame.to_numpy(float)
        dim = arr.shape[1]

        niw_states: list[NIWState] = [NIWState.prior(dim, kappa0=self.prior_kappa, psi_scale=self.prior_psi_scale)]
        diag_states: list[RunningDiagState] = [RunningDiagState.prior(dim, self.prior_var_diag)]
        ar1_states: list[_AR1State] = [_AR1State.prior(dim, self.prior_var_diag)]
        log_joint_per_model = [
            np.array([0.0], dtype=float),
            np.array([0.0], dtype=float),
            np.array([0.0], dtype=float),
        ]
        log_model_post = np.log(np.array([1.0, 1.0, 1.0]) / 3.0)

        h = min(max(float(self.hazard), self.min_prob), 1.0 - self.min_prob)
        log_h = math.log(h)
        log_1mh = math.log(1.0 - h)

        rows = []
        for i, date in enumerate(frame.index):
            xt = arr[i]
            ll_per_model = []
            cp_probs = []
            run_means = []
            map_runs = []
            new_states_niw = None
            new_states_diag = None
            new_states_ar1 = None
            new_joints = []

            for m_idx, (states, joint, predict_fn) in enumerate(
                [
                    (niw_states, log_joint_per_model[0], lambda st, x=xt: st.predictive_logpdf(x)),
                    (diag_states, log_joint_per_model[1], lambda st, x=xt: _student_t_logpdf_diag(x, st)),
                    (ar1_states, log_joint_per_model[2], lambda st, x=xt: st.predict(x)),
                ]
            ):
                pred_logs = np.array([predict_fn(st) for st in states], dtype=float)
                pred_norm = _logsumexp(joint + pred_logs)
                cp_log = _logsumexp(joint + pred_logs + log_h)
                growth_logs = joint + pred_logs + log_1mh
                new_log_joint = np.empty(min(len(growth_logs) + 1, self.max_run + 1), dtype=float)
                new_log_joint[0] = cp_log
                kept = growth_logs[: self.max_run]
                new_log_joint[1 : 1 + len(kept)] = kept
                norm = _logsumexp(new_log_joint)
                new_log_joint = new_log_joint - norm
                probs = np.exp(new_log_joint)
                probs = probs / probs.sum()

                cp_probs.append(float(probs[0]))
                run_lengths = np.arange(len(probs), dtype=float)
                run_means.append(float(np.sum(run_lengths * probs)))
                map_runs.append(int(np.argmax(probs)))
                ll_per_model.append(float(pred_norm))
                new_joints.append(np.log(np.maximum(probs, self.min_prob)))

                if m_idx == 0:
                    prior = NIWState.prior(dim, kappa0=self.prior_kappa, psi_scale=self.prior_psi_scale)
                    new_states_niw = [prior.update(xt)] + [st.update(xt) for st in states[: self.max_run]]
                    new_states_niw = new_states_niw[: len(probs)]
                elif m_idx == 1:
                    prior_diag = RunningDiagState.prior(dim, self.prior_var_diag)
                    new_states_diag = [prior_diag.update(xt)] + [st.update(xt) for st in states[: self.max_run]]
                    new_states_diag = new_states_diag[: len(probs)]
                else:
                    prior_ar = _AR1State.prior(dim, self.prior_var_diag)
                    new_states_ar1 = [prior_ar.update(xt)] + [st.update(xt) for st in states[: self.max_run]]
                    new_states_ar1 = new_states_ar1[: len(probs)]

            niw_states = new_states_niw  # type: ignore[assignment]
            diag_states = new_states_diag  # type: ignore[assignment]
            ar1_states = new_states_ar1  # type: ignore[assignment]
            log_joint_per_model = new_joints

            ll_arr = np.array(ll_per_model, dtype=float)
            log_model_post = log_model_post + ll_arr
            log_model_post -= _logsumexp(log_model_post)
            model_post = np.exp(log_model_post)
            cp_avg = float(np.sum(model_post * np.array(cp_probs)))
            mean_avg = float(np.sum(model_post * np.array(run_means)))
            map_avg = int(np.argmax(np.array(map_runs)))
            ll_avg = float(_logsumexp(log_model_post + ll_arr))
            rows.append(
                {
                    "date": date,
                    "change_point_prob": cp_avg,
                    "bocpd_run_length_mean": mean_avg,
                    "bocpd_map_run_length": map_avg,
                    "predictive_log_likelihood": ll_avg,
                    "model_post_niw": float(model_post[0]),
                    "model_post_diag": float(model_post[1]),
                    "model_post_ar1": float(model_post[2]),
                }
            )
        return pd.DataFrame(rows)


__all__ = ["BOCPDMuse"]
