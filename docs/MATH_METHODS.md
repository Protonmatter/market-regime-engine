# Mathematical Method Inventory

This document is the single-source map from implementation modules to model assumptions. It is intentionally conservative: if a path is retrospective-only, optional-dependency-bound, or research-grade, it is marked as **experimental frontier** rather than production core.

## Package boundary

| Boundary | Modules | Release-gate treatment |
|---|---|---|
| Stable core | `storage`, `models`, `walk_forward`, `forecast_compare`, `release_gates`, `fixed_income`, `validation` | Default target for production release gates. Uses `profile="production"` unless overridden by `MRE_ENV` or explicit CLI flags. |
| Experimental frontier | `market_regime_engine.frontier.*` | Requires `MRE_ENABLE_EXPERIMENTAL_FRONTIER=1` for fenced paths and release-gate evaluation with `gate_boundary="experimental_frontier"`. Not production-eligible by default. |

## Regime and change-point models

### HMM regime classifier

The HMM path estimates latent regimes over feature vectors and pins labels through a deterministic minimum-cost assignment, not greedy label matching. This reduces label-switching instability between fits. Label names remain semantic only after post-fit assignment; raw state indexes are not stable across refits.

Production status: stable core when used through the standard regime scoring path.

### MS-VAR and Bayesian MS-VAR(p)

The frequentist MS-VAR path models regime-dependent vector autoregression. Stability is checked through the companion-matrix spectral radius. Regimes with explosive companion radius are shrunk conservatively before downstream diagnostics are emitted.

The Bayesian MS-VAR supports true AR(p) lag stacking. The coefficient tensor is indexed as:

```text
[state, lag, output_dim, input_dim]
```

For a K-dimensional process with p lags, the state equation is:

```text
y_t = c_s + A_{s,1} y_{t-1} + ... + A_{s,p} y_{t-p} + epsilon_t,
epsilon_t ~ N(0, Sigma_s)
```

Bayesian diagnostics include posterior coefficient summaries and companion-radius checks. Weakly identified regimes use covariance shrinkage toward a spherical target when sample support is small or the empirical covariance is ill-conditioned.

Production status: Bayesian MS-VAR remains experimental frontier because it depends on NumPyro/JAX and requires operator review of convergence diagnostics, posterior stability, R-hat/ESS, divergences, and regime support.

## Mixed-frequency nowcasting

### DFM-MQ monthly/quarterly backend

Monthly/quarterly panels use the statsmodels `DynamicFactorMQ` backend when available. This path is valid only for M/Q frequency layouts.

### Native D/W/M state-space backend

Daily/weekly/monthly panels route through the custom Kalman backend. The observation matrix maps latent daily factors into release windows for D/W/M observations. Unsupported D/W plus Q layouts fail closed because the current custom backend does not yet model quarter aggregation consistently with daily/weekly release windows.

Smoothed latent factors are retrospective-only because smoothing can condition on future observations. Calls that expose smoothed factors are fenced behind `MRE_ENABLE_EXPERIMENTAL_FRONTIER=1`.

Production status: D/W/M nowcasting is experimental frontier until release-calendar alignment, live nowcast-only filtering, and business-day aggregation rules are separately certified.

## Forecast comparison and release gates

### Giacomini-White conditional predictive ability

Conditional GW tests use moment conditions of the form:

```text
g_t = z_t * (L_1,t - L_2,t)
```

where `z_t` are conditioning instruments and `L_i,t` are forecast losses. The implemented conditional path estimates a full HAC sandwich covariance over the moment vector rather than a scalar Newey-West shortcut. This matters when instruments are correlated or heteroskedastic.

Production status: stable core for validation reports, subject to minimum sample-size checks and finite HAC covariance.

### Release-gate profiles

The stable core defaults to the production profile. The production profile gates on confidence, drift, MCS membership, conformal coverage when supplied, DSR/PBO when supplied, Brier/ECE when supplied, and positive-direction TCA-lift when supplied.

The experimental frontier boundary is separate. It requires explicit opt-in and records the boundary in `metadata_json`. Passing a frontier research gate does not by itself certify a model for production promotion.

## Fixed-income signal layer

The fixed-income layer remains stable-core API surface, but each output remains fail-closed when required point-in-time inputs are missing. Execution-confidence scoring is calibrated against realized outcomes when calibration data is available; raw scores and calibrated scores must be treated separately in downstream evaluation.

API implementation is split by responsibility:

- `fixed_income.api_schemas`: Pydantic request validation and response serializers.
- `fixed_income.api_handlers`: FastAPI route handlers.
- `fixed_income.api_middleware`: body cap and rate-limit startup checks.
- `fixed_income.api_cache`: versioned cache keyed by latest signal identity.

## Storage and evidence contracts

Storage is split by responsibility:

- `storage_registry`: table DDL registry and legacy schema aggregate shims.
- `storage_backends`: SQLite/DuckDB backend selection and write adapters.
- `storage_repositories`: `Warehouse` read/write repository API.
- `storage_pool`: per-process warehouse pool and write locks.

The historical `market_regime_engine.storage` module remains a compatibility facade. New code should import focused modules directly when it needs internals, and import the facade only for public `Warehouse` operations.

## v1.7 certification hardening addendum

The certification profile is a fail-closed release-gate profile for stable-core model-risk review. Unlike the production profile, it does not treat validation evidence as opportunistic. It requires the confidence frame to carry machine-auditable fields for PIT leakage, walk-forward validation, calibration metrics, positive-direction TCA lift, model-card location, validation artifact hash, evidence-pack HMAC, DSR, PBO, and minimum regime support.

Execution-confidence validation is implemented in `market_regime_engine.fixed_income.execution_validation`. It joins `execution_confidence_predictions` to later `execution_outcomes` through the PIT-safe calibration join, then emits Brier/log-loss/ECE, calibration by regime, confidence-decile lift, positive-direction TCA lift by regime, invalid probability-score rejection, and an artifact hash suitable for the certification gate.

Frontier diagnostics are implemented in `market_regime_engine.frontier.diagnostics`. Bayesian MS-VAR experimental gates fail closed on divergences, missing or excessive R-hat, insufficient ESS, unstable companion radius, and weak posterior regime mass. Mixed-frequency online safety is tested by prefix invariance: an online-safe nowcast at time `t` must not change when future rows are appended. Retrospective smoothed factors remain research-only.

Method cards under `docs/method_cards/` are now the canonical method-level documentation. The docs audit test fails if a required method lacks production status, equations, assumptions, diagnostics, release-gate requirements, concrete test references, or known limitations.
