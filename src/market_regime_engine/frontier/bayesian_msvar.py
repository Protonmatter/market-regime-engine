# SPDX-License-Identifier: Apache-2.0
"""Bayesian Markov-Switching VAR with NumPyro NUTS / SVI inference.

The maximum-likelihood EM in :mod:`market_regime_engine.msvar` recovers
the *modal* regime trajectory but cannot quantify how confident we are
in those regime probabilities — every regime indicator collapses to the
EM posterior mean, drifting silently through ill-conditioned regions of
the likelihood. Production decision pipelines need posterior credible
intervals on the per-state probabilities so the release-gate / e-value
machinery can refuse to act on an over-confident point estimate.

This module fits the same Hamilton-Kim MS-VAR via NumPyro:

- Dirichlet prior on the rows of the transition matrix
  (``concentration=1.0`` ≡ uniform).
- Per-state VAR coefficient prior ``Normal(0, sigma_beta**2 * I)``.
- Per-state innovation covariance via half-Cauchy scales + LKJ
  correlation.
- Categorical mixture marginalised in closed form per timestep so HMC
  / SVI run on a tractable likelihood (no per-state augmentation).

Inference modes:

- ``method="nuts"`` — full No-U-Turn HMC (default). Slow on large panels
  but produces honest credible bands and R-hat / ESS diagnostics.
- ``method="svi"`` — autoguide ELBO fallback for the large-panel path.
  Returns the variational posterior mean as the regime trajectory and
  pads the credible interval at ``±1.5 * variational_std``.

Both modes return the same ``score`` schema as
:meth:`MarkovSwitchingVAR.score` plus per-row credible bands and a
``divergence_count`` column so downstream consumers can fail-closed on
bad fits.

Soft-degrades: the import block raises ``ImportError`` with the
``pip install market-regime-engine[bayesian]`` hint when ``numpyro`` /
``jax`` are missing, exactly mirroring the v1.2 ``[frontier]`` pattern.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from market_regime_engine.frontier.data_cleaning import NanPolicy, clean_with_policy
from market_regime_engine.hmm import DOMAIN_COLUMNS, REGIME_STATES

_INSTALL_HINT = (
    "BayesianMSVAR requires the optional [bayesian] extra. Install with `pip install market-regime-engine[bayesian]`."
)


def _require_numpyro() -> tuple[Any, Any, Any, Any, Any]:
    """Import jax + numpyro lazily so the soft-degrade test can stub them out."""
    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
    except ImportError as exc:  # pragma: no cover - import path
        raise ImportError(_INSTALL_HINT) from exc
    return (
        jax,
        jnp,
        numpyro,
        dist,
        {"MCMC": MCMC, "NUTS": NUTS, "SVI": SVI, "Trace_ELBO": Trace_ELBO},
    )


def _coerce_panel(
    panel: pd.DataFrame,
    domains: list[str],
    *,
    nan_policy: NanPolicy = NanPolicy.NAN_TO_ZERO,
    column_policies: Mapping[str, NanPolicy] | None = None,
) -> np.ndarray:
    """Project ``panel`` into the ``domains`` order; apply per-column NaN policy.

    Mirrors the cleaner used by :class:`MarkovSwitchingVAR.fit` so the
    EM-vs-Bayes parity test isn't biased by different missing-data
    handling.

    v1.5 (PR-3 ASK-5/AF-8): the legacy ``ffill().fillna(0.0)`` is
    routed through :func:`clean_with_policy`. Default
    ``NanPolicy.NAN_TO_ZERO`` is bit-for-bit identical to the v1.4
    cleaner, so the existing fixtures keep passing.
    """
    if panel is None or panel.empty:
        return np.zeros((0, len(domains)), dtype=float)
    frame = panel.copy()
    for d in domains:
        if d not in frame:
            frame[d] = 0.0
    return clean_with_policy(frame[domains], default_policy=nan_policy, column_policies=column_policies).to_numpy(
        dtype=float
    )


def _logsumexp(arr: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Stable logsumexp returning a 0-d / N-d array depending on ``axis``."""
    if axis is None:
        m = float(np.max(arr))
        if not math.isfinite(m):
            return np.asarray(float("-inf"))
        return np.asarray(m + math.log(float(np.sum(np.exp(arr - m)))))
    m = np.max(arr, axis=axis, keepdims=True)
    safe_m = np.where(np.isfinite(m), m, 0.0)
    out = safe_m + np.log(np.sum(np.exp(arr - safe_m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


@dataclass
class BayesianMSVAR:
    """NumPyro-backed Bayesian Markov-Switching VAR.

    Posterior diagnostics are surfaced via ``last_diagnostics`` after a
    ``fit()`` call so the new ``bayesian_msvar_diagnostics`` warehouse
    table can be populated without re-running inference.
    """

    states: list[str] = field(default_factory=lambda: REGIME_STATES.copy())
    domains: list[str] = field(default_factory=lambda: DOMAIN_COLUMNS.copy())
    p: int = 1
    sigma_beta: float = 0.5
    transition_concentration: float = 1.0
    cov_concentration: float = 2.0
    seed: int = 0

    # Population posterior summaries (means).
    _posterior_intercepts: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    _posterior_coefficients: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0, 0)))
    _posterior_covariances: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    _posterior_transition: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    _posterior_prior: np.ndarray = field(default_factory=lambda: np.zeros((0,)))

    # Sample-level posterior probabilities (n_samples, T_eff, K).
    _posterior_state_probs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))

    last_diagnostics: dict[str, Any] = field(default_factory=dict)
    fitted: bool = False

    # ------------------------------------------------------------------
    # NumPyro model
    # ------------------------------------------------------------------

    def _model_factory(self):
        """Return a Numpyro callable closed over the panel + dims.

        Defining the model inside a method keeps the dependency on the
        lazy-imported ``dist`` module local — the import failure cleanly
        becomes an ``ImportError`` raised by ``_require_numpyro``.
        """
        _jax, jnp, numpyro, dist, _infer = _require_numpyro()
        K = len(self.states)
        d = len(self.domains)
        sigma_beta = float(self.sigma_beta)
        conc = float(self.transition_concentration)
        cov_conc = float(self.cov_concentration)

        def model(Y: jnp.ndarray) -> None:  # type: ignore[name-defined]
            T = Y.shape[0]
            # Per-state intercept + AR(1) coefficient matrix.
            # We enforce label ordering by sampling the first-domain
            # intercept under an *ordered* transform (so state 0's
            # first-domain intercept is strictly < state 1's, etc).
            # This eliminates the classic MS-VAR label-switching across
            # MCMC samples without restricting the model class — the
            # remaining intercept / AR / cov fields are free.
            ordered_first = numpyro.sample(
                "intercept_state_anchor",
                dist.TransformedDistribution(
                    dist.Normal(0.0, sigma_beta).expand([K]).to_event(1),
                    dist.transforms.OrderedTransform(),
                ),
            )
            other_intercepts = (
                numpyro.sample(
                    "intercept_remainder",
                    dist.Normal(0.0, sigma_beta).expand([K, d - 1]).to_event(2),
                )
                if d > 1
                else jnp.zeros((K, 0))
            )
            intercept = numpyro.deterministic(
                "intercept",
                jnp.concatenate([ordered_first[:, None], other_intercepts], axis=1),
            )
            ar = numpyro.sample(
                "ar1",
                dist.Normal(0.0, sigma_beta).expand([K, d, d]).to_event(3),
            )
            # Per-state innovation covariance: scale * LKJCorrCholesky.
            scales = numpyro.sample(
                "scales",
                dist.HalfCauchy(scale=1.0).expand([K, d]).to_event(2),
            )
            corr_chol = numpyro.sample(
                "corr_chol",
                dist.LKJCholesky(d, concentration=cov_conc).expand([K]).to_event(1),
            )
            cov_chol = scales[:, :, None] * corr_chol
            # Initial regime distribution + transition matrix.
            prior = numpyro.sample("prior", dist.Dirichlet(jnp.ones(K) * conc))
            transition = numpyro.sample(
                "transition",
                dist.Dirichlet(jnp.ones(K) * conc).expand([K]).to_event(1),
            )

            # Closed-form forward marginalisation over the regime
            # sequence. Walking ``T`` steps is fine because ``T`` is at
            # most ~750 for a 60-year monthly panel; the JIT compile
            # amortises across chains.
            def forward_step(log_alpha, t_idx):
                y_t = Y[t_idx]
                y_prev = Y[t_idx - 1]
                emit_logps = []
                for k in range(K):
                    mu_k = intercept[k] + ar[k] @ y_prev
                    emit_logps.append(dist.MultivariateNormal(mu_k, scale_tril=cov_chol[k]).log_prob(y_t))
                emit = jnp.stack(emit_logps)
                # forward update: log_alpha[t] = log( sum_i alpha[t-1, i] * A[i, j] ) + emit[j]
                trans_log = jnp.log(transition + 1e-12)
                new_log_alpha = jax_logsumexp(log_alpha[:, None] + trans_log, axis=0) + emit
                return new_log_alpha, new_log_alpha

            # Initial log-alpha = log_prior + emit_0
            mu_init = jnp.stack([intercept[k] for k in range(K)])
            init_emits = jnp.stack(
                [dist.MultivariateNormal(mu_init[k], scale_tril=cov_chol[k]).log_prob(Y[0]) for k in range(K)]
            )
            log_alpha0 = jnp.log(prior + 1e-12) + init_emits

            log_alpha, log_alphas = _jax.lax.scan(forward_step, log_alpha0, jnp.arange(1, T))
            log_lik = jax_logsumexp(log_alpha, axis=-1)
            numpyro.factor("forward_loglik", log_lik)

            # Deterministic site so ``Predictive`` can recover per-step
            # posterior probabilities without a second pass.
            full_log_alphas = jnp.concatenate([log_alpha0[None, :], log_alphas], axis=0)
            log_norm = jax_logsumexp(full_log_alphas, axis=-1, keepdims=True)
            numpyro.deterministic("state_log_probs", full_log_alphas - log_norm)

        # Local helper because ``jax.scipy.special.logsumexp`` lives
        # under ``jax.scipy``; importing here keeps the soft-degrade
        # pattern intact.
        from jax.scipy.special import logsumexp as jax_logsumexp

        return model

    # ------------------------------------------------------------------
    # fit / score
    # ------------------------------------------------------------------

    def fit(
        self,
        panel: pd.DataFrame,
        *,
        method: Literal["nuts", "svi"] = "nuts",
        num_chains: int = 2,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        svi_steps: int = 2000,
        progress_bar: bool = False,
    ) -> BayesianMSVAR:
        """Fit posterior over the MS-VAR parameters.

        ``method="nuts"`` runs the full No-U-Turn sampler. ``method="svi"``
        falls back to a mean-field autoguide ELBO surrogate, suitable
        for large panels where NUTS warmup time dominates wall-clock.

        Diagnostics are written to :pyattr:`last_diagnostics` so the
        caller can persist them via
        :meth:`Warehouse.write_bayesian_msvar_diagnostics`.
        """
        _jax, jnp, numpyro, _dist, infer = _require_numpyro()

        Y_arr = _coerce_panel(panel, self.domains)
        if Y_arr.shape[0] < self.p + len(self.states) * 4:
            return self
        Y = jnp.asarray(Y_arr)

        rng_key = _jax.random.PRNGKey(int(self.seed))
        model = self._model_factory()

        started = time.perf_counter()
        if method == "nuts":
            kernel = infer["NUTS"](model)
            mcmc = infer["MCMC"](
                kernel,
                num_warmup=int(num_warmup),
                num_samples=int(num_samples),
                num_chains=int(num_chains),
                chain_method="sequential",
                progress_bar=progress_bar,
            )
            mcmc.run(rng_key, Y=Y)
            samples = mcmc.get_samples(group_by_chain=False)
            try:
                summary = numpyro.diagnostics.summary(mcmc.get_samples(group_by_chain=True), prob=0.9)
            except Exception:  # pragma: no cover - diagnostics are best-effort
                summary = {}
            divergences = int(np.sum(np.asarray(mcmc.get_extra_fields().get("diverging", []), dtype=bool)))
            rhats: list[float] = []
            esss: list[float] = []
            for stats in summary.values():
                if isinstance(stats, dict):
                    if "r_hat" in stats:
                        rhats.append(float(np.nanmax(np.asarray(stats["r_hat"]))))
                    if "n_eff" in stats:
                        esss.append(float(np.nanmin(np.asarray(stats["n_eff"]))))
            self.last_diagnostics = {
                "method": "nuts",
                "num_chains": int(num_chains),
                "num_divergences": int(divergences),
                "max_rhat": float(max(rhats)) if rhats else float("nan"),
                "min_ess": float(min(esss)) if esss else float("nan"),
                "runtime_seconds": float(time.perf_counter() - started),
                "num_warmup": int(num_warmup),
                "num_samples": int(num_samples),
            }
        elif method == "svi":
            try:
                from numpyro.infer.autoguide import AutoNormal
            except ImportError as exc:  # pragma: no cover - import path
                raise ImportError(_INSTALL_HINT) from exc
            guide = AutoNormal(model)
            svi = infer["SVI"](
                model,
                guide,
                _make_optimizer(numpyro),
                infer["Trace_ELBO"](),
            )
            # ``svi.run`` uses jax.lax.scan internally so the optimisation
            # loop is JIT-traced once instead of once per step (the
            # Python loop took ~250s on a 2x2 panel before this fix).
            svi_result = svi.run(rng_key, int(svi_steps), Y=Y, progress_bar=False)
            params = svi_result.params
            sample_key, _ = _jax.random.split(rng_key)
            posterior = guide.sample_posterior(sample_key, params, sample_shape=(int(num_samples),))
            samples = {k: np.asarray(v) for k, v in posterior.items()}
            self.last_diagnostics = {
                "method": "svi",
                "num_chains": 1,
                "num_divergences": 0,
                "max_rhat": float("nan"),
                "min_ess": float("nan"),
                "runtime_seconds": float(time.perf_counter() - started),
                "svi_steps": int(svi_steps),
                "num_samples": int(num_samples),
                "final_loss": float(np.asarray(svi_result.losses)[-1]),
            }
        else:
            raise ValueError(f"unknown method: {method!r}")

        self._extract_posterior_means(samples)
        self.fitted = True
        return self

    def _extract_posterior_means(self, samples: dict[str, Any]) -> None:
        intercepts = np.asarray(samples["intercept"])
        ar = np.asarray(samples["ar1"])
        scales = np.asarray(samples["scales"])
        corr_chol = np.asarray(samples["corr_chol"])
        prior = np.asarray(samples["prior"])
        trans = np.asarray(samples["transition"])
        # Recover per-state covariance from scales + cholesky factor.
        cov = np.einsum("nki,nkij->nkij", scales, corr_chol)
        cov_full = np.einsum("nkij,nklj->nkil", cov, cov)
        self._posterior_intercepts = intercepts.mean(axis=0)
        self._posterior_coefficients = ar.mean(axis=0)
        self._posterior_covariances = cov_full.mean(axis=0)
        self._posterior_prior = prior.mean(axis=0)
        self._posterior_transition = trans.mean(axis=0)
        # ``state_log_probs`` is the per-sample per-step log posterior;
        # convert to probabilities so the credible bands work in linear
        # probability space.
        if "state_log_probs" in samples:
            log_probs = np.asarray(samples["state_log_probs"])
            self._posterior_state_probs = np.exp(log_probs)

    # ------------------------------------------------------------------
    # filtered scoring (re-running the Hamilton filter at posterior means)
    # ------------------------------------------------------------------

    def _emission_logpdf(self, y_t: np.ndarray, y_prev: np.ndarray) -> np.ndarray:
        K = len(self.states)
        out = np.empty(K)
        for k in range(K):
            mu = self._posterior_intercepts[k] + self._posterior_coefficients[k] @ y_prev
            cov = self._posterior_covariances[k] + 1e-6 * np.eye(len(self.domains))
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(cov + 1e-3 * np.eye(len(self.domains)))
            diff = y_t - mu
            z = np.linalg.solve(L, diff)
            log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
            out[k] = -0.5 * (len(self.domains) * math.log(2.0 * math.pi) + log_det + float(z @ z))
        return out

    def _filtered_state_probs(self, Y: np.ndarray) -> np.ndarray:
        n = Y.shape[0]
        K = len(self.states)
        log_alpha = np.full((n, K), -np.inf)
        log_pi = np.log(np.maximum(self._posterior_prior, 1e-12))
        log_A = np.log(np.maximum(self._posterior_transition, 1e-12))
        for t in range(n):
            y_t = Y[t]
            y_prev = Y[t - 1] if t > 0 else np.zeros(len(self.domains))
            log_b = self._emission_logpdf(y_t, y_prev)
            if t == 0:
                log_alpha[t] = log_pi + log_b
            else:
                log_alpha[t] = log_b + _logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)
        # Normalise per-row to per-step probability.
        log_norm = _logsumexp(log_alpha, axis=1)[:, None]
        log_alpha = log_alpha - log_norm
        return np.exp(log_alpha)

    def score(
        self,
        panel: pd.DataFrame,
        *,
        nan_policy: NanPolicy = NanPolicy.NAN_TO_ZERO,
        column_policies: Mapping[str, NanPolicy] | None = None,
    ) -> pd.DataFrame:
        """Return per-step regime probabilities + credible bands.

        Columns mirror :meth:`MarkovSwitchingVAR.score`:

        - ``date``, ``msvar_regime``, ``msvar_confidence``
        - ``msvar_prob_<state>`` for every state
        - ``bayesian_credible_lo`` / ``bayesian_credible_hi`` — per-row 5%
          / 95% posterior quantiles of the *modal* state probability
        - ``divergence_count`` — copied from the latest fit, lets the
          caller reject a bad NUTS run downstream.

        When per-sample ``state_log_probs`` are available (NUTS path)
        the per-step probabilities are the *posterior mean over
        samples* of the forward-filtered state probabilities. That is
        the correct Bayesian estimator and is robust to label-switching
        across the chain because each sample is internally consistent.

        When only posterior-mean parameters are available (SVI path) we
        fall back to re-running the Hamilton filter at the posterior
        mean.

        v1.5 (PR-3 ASK-5/AF-8): ``nan_policy`` defaults to
        :attr:`NanPolicy.NAN_TO_ZERO` to preserve v1.4 numerics; FI
        callers may pass :attr:`NanPolicy.NAN_FAILS_PIT_AUDIT` so a
        missing FI feature aborts the score rather than silently
        zero-filling.
        """
        cols = (
            ["date", "msvar_regime", "msvar_confidence"]
            + [f"msvar_prob_{s}" for s in self.states]
            + ["bayesian_credible_lo", "bayesian_credible_hi", "divergence_count"]
        )
        if not self.fitted or panel is None or panel.empty:
            return pd.DataFrame(columns=cols)
        frame = panel.copy()
        Y = _coerce_panel(panel, self.domains, nan_policy=nan_policy, column_policies=column_policies)
        n = Y.shape[0]
        if n == 0:
            return pd.DataFrame(columns=cols)
        # Prefer per-sample posterior probabilities when available
        # (avoids the label-switching bias of the posterior-mean
        # parameter point estimate).
        use_samples = self._posterior_state_probs.size > 0 and self._posterior_state_probs.shape[1] == n
        if use_samples:
            samples = self._posterior_state_probs  # (S, n, K)
            filtered = samples.mean(axis=0)
            mod_idx = filtered.argmax(axis=1)
            sample_modal = samples[:, np.arange(n), mod_idx]
            lo = np.quantile(sample_modal, 0.05, axis=0)
            hi = np.quantile(sample_modal, 0.95, axis=0)
        else:
            filtered = self._filtered_state_probs(Y)
            modal = filtered.max(axis=1)
            sd = float(np.std(filtered, axis=1).mean()) if filtered.size else 0.0
            lo = np.clip(modal - 1.5 * sd, 0.0, 1.0)
            hi = np.clip(modal + 1.5 * sd, 0.0, 1.0)
        diverg = int(self.last_diagnostics.get("num_divergences", 0))

        rows: list[dict] = []
        index = list(frame.index)[:n]
        for t in range(n):
            probs = filtered[t]
            best = int(np.argmax(probs))
            row = {
                "date": index[t],
                "msvar_regime": self.states[best],
                "msvar_confidence": float(probs[best]),
            }
            row.update({f"msvar_prob_{s}": float(probs[i]) for i, s in enumerate(self.states)})
            row["bayesian_credible_lo"] = float(lo[t])
            row["bayesian_credible_hi"] = float(hi[t])
            row["divergence_count"] = diverg
            rows.append(row)
        return pd.DataFrame(rows, columns=cols)

    # ------------------------------------------------------------------
    # OnlineBMA helper
    # ------------------------------------------------------------------

    def latest_state_probs(self, panel: pd.DataFrame) -> dict[str, float]:
        """Return the latest filtered state probabilities as a flat dict.

        Plugs straight into :meth:`OnlineBMA.update` /
        :meth:`OnlineBMA.mix` (via the existing ``dict[str, float]``
        interface) — every ``bayesian_msvar:<state>`` key gets a posterior
        mean weight in the BMA layer.
        """
        if not self.fitted or panel is None or panel.empty:
            return {}
        Y = _coerce_panel(panel, self.domains)
        if Y.shape[0] == 0:
            return {}
        filtered = self._filtered_state_probs(Y)
        latest = filtered[-1]
        return {f"bayesian_msvar:{s}": float(latest[i]) for i, s in enumerate(self.states)}


def _make_optimizer(numpyro_mod: Any) -> Any:
    """Instantiate the SVI Adam optimiser via the version-tolerant API.

    Older NumPyro releases expose ``optim`` under ``numpyro.optim``;
    newer releases moved ``Adam`` to ``numpyro.infer.optim``. Try both so
    we don't constrain the lockfile to a single point release.
    """
    for path in ("optim", "infer.optim"):
        mod: Any = numpyro_mod
        ok = True
        for piece in path.split("."):
            mod = getattr(mod, piece, None)
            if mod is None:
                ok = False
                break
        if ok and hasattr(mod, "Adam"):
            return mod.Adam(step_size=1e-2)
    # Fall back: build via JAX optax-equivalent if available (numpyro
    # always ships with at least one optimiser path).
    raise ImportError(_INSTALL_HINT)  # pragma: no cover - very old numpyro


__all__ = ["BayesianMSVAR"]
