# Review Hardening Implementation State

This document records the build state after the end-to-end code and mathematics hardening pass. It is written as an operator/model-risk contract, not a marketing summary.

## Bayesian MS-VAR(p)

`src/market_regime_engine/frontier/bayesian_msvar.py` now implements a NumPyro Bayesian Markov-switching VAR with an explicit lag stack for `p >= 1`:

\[
y_t = c_{s_t} + \sum_{j=1}^{p} A_{s_t,j} y_{t-j} + \epsilon_t,
\qquad \epsilon_t \sim \mathcal{N}(0, \Sigma_{s_t})
\]

Model details:

- state transition rows use a Dirichlet prior;
- state intercepts use an ordered first-domain anchor to reduce label switching;
- coefficient tensor is sampled as `(state, lag, output_dim, input_dim)`;
- innovation covariance uses half-Cauchy marginal scales plus LKJ Cholesky correlation;
- the latent regime path is marginalized with the Hamilton forward recursion;
- first `p` rows have no complete lag stack and expose the normalized prior in `state_log_probs`;
- posterior diagnostics include companion-matrix spectral radius by regime and a `stationary` boolean.

This supersedes the earlier fail-closed AR(1)-only contract. Documentation that says `BayesianMSVAR` only supports AR(1) is now stale.

## MS-VAR EM stability and weak-regime covariance shrinkage

`src/market_regime_engine/msvar.py` now includes two fail-safe layers for classical EM MS-VAR:

1. **Companion-matrix stability checks** for each regime. If `enforce_stability=True`, explosive coefficient stacks are scaled so the companion spectral radius is no greater than `stability_radius`.
2. **Weak-regime covariance shrinkage**. When posterior regime weight is low or covariance is ill-conditioned, the innovation covariance is shrunk toward a spherical target:

\[
\hat\Sigma_{k,shrunk} = (1 - \alpha) \hat\Sigma_k + \alpha \bar\sigma_k^2 I
\]

where `alpha` is raised under small effective sample size or excessive condition number. `fit_log` records:

- `max_companion_radius`
- `companion_radius_by_regime`
- `stabilized_updates`
- `covariance_shrinkage_events`
- `max_covariance_condition`
- `max_covariance_shrinkage_intensity`

The shrinkage path is a numerical/model-risk guard. It is not a replacement for full Bayesian uncertainty over weak regimes.

## Native D/W/M mixed-frequency state-space backend

`src/market_regime_engine/frontier/dfm_mq.py` now has two supported mixed-frequency routes:

| Frequency layout | Backend | Contract |
|---|---|---|
| `M` / `Q` | statsmodels `DynamicFactorMQ` when available; fallback otherwise | Bańbura-Modugno-style monthly/quarterly DFM-MQ wrapper |
| `D` / `W` / `M` | native `custom_state_space` backend | daily latent AR(1) factor with time-varying observation aggregation |

The native custom backend uses a daily state vector:

\[
x_t = [f_t, f_{t-1}, \dots, f_{t-L+1}]^\top
\]

with AR(1) transition for `f_t` and deterministic lag shifting. Observation rows average the latent factor over the release window:

- `D`: current day only;
- `W`: trailing 7 calendar days;
- `M`: month-to-date, capped at 31 days.

Unsupported layouts fail closed. In particular, `Q` cannot currently be mixed with `D`/`W` in the custom backend.

## Giacomini-White covariance estimation

`src/market_regime_engine/forecast_compare.py::giacomini_white` now returns a full HAC sandwich covariance path for conditional predictive ability tests:

\[
\widehat{\mathrm{Var}}(\hat\beta) = \frac{1}{T} Q^{-1} S Q^{-1}
\]

where `S` is the Newey-West long-run covariance of vector moment conditions `z_t u_t`. The result dictionary now includes:

- `covariance`
- `covariance_estimator="hac_sandwich"`
- `lag`

This replaces the older scalar-HAC shortcut that was only adequate for the unconditional constant-only case.

## Experimental / retrospective frontier fence

`src/market_regime_engine/frontier/experimental.py` centralizes the opt-in flag:

```bash
MRE_ENABLE_EXPERIMENTAL_FRONTIER=1
```

The flag is required for paths that are useful in retrospective research but unsafe as silent production defaults:

- DFM-MQ smoothed latent-factor extraction (`filtered=False`) because it uses future observations;
- release-gate `promotion_method="e_values"` because the build treats this as an experimental frontier promotion rail while production defaults remain MCS-based.

Absent the flag, these paths raise `RuntimeError` with a concrete reason.

## Adversarial tests added

`tests/test_review_math_hardening.py` now covers:

- Bayesian MS-VAR `p > 1` lag-stack scoring;
- exact assignment counterexample for label symmetry;
- native D/W/M custom mixed-frequency backend;
- unsupported D/W/Q layout failure;
- explosive MS-VAR coefficient stabilization;
- collinearity-induced covariance shrinkage / positive definiteness;
- full HAC sandwich result for Giacomini-White;
- experimental flag fences for smoothed DFM-MQ and e-value release gates.

## Validation status

Targeted local validation passed:

```bash
pytest -q tests/test_review_math_hardening.py
pytest -q tests/test_frontier_dfm_mq.py tests/test_forecast_compare.py tests/test_frontier_bayesian_msvar.py tests/test_bayesian_msvar.py
```

Observed result: all targeted tests passed; Bayesian optional-dependency tests skipped where NumPyro/JAX were unavailable; statsmodels emitted an expected short-fixture EM convergence warning.
