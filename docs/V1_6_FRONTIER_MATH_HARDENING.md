# V1.6 Frontier Math Hardening

## Current build state

The build now distinguishes production-safe filtered paths from retrospective/experimental paths, implements Bayesian MS-VAR(p), adds weak-regime covariance shrinkage, and supports a native D/W/M mixed-frequency state-space backend.

## Production-safe defaults

- Classical MS-VAR uses EM with covariance ridge, weak-regime shrinkage, label pinning, and companion-matrix stability diagnostics.
- Bayesian MS-VAR supports `p >= 1` with a true lag-stack likelihood.
- DFM-MQ nowcasting uses filtered factors by default.
- Release gates default to the production profile and MCS promotion.
- Giacomini-White uses a vector HAC sandwich covariance for conditional predictive ability.

## Retrospective-only / experimental paths

The following require `MRE_ENABLE_EXPERIMENTAL_FRONTIER=1`:

1. smoothed DFM-MQ factor extraction (`filtered=False`);
2. e-value release-gate promotion (`promotion_method="e_values"`).

The flag is intentionally coarse-grained. It prevents operators from accidentally running research-oriented paths in real-time decisioning.

## Mathematical limitations still present

- The native D/W/M custom backend is a single-factor Gaussian linear state-space model. It is not yet a multi-factor, stochastic-volatility, or nonlinear release-calendar model.
- Weekly aggregation is trailing seven calendar days. If the economic series is business-week or release-calendar-specific, the caller must align the index accordingly before fitting.
- Monthly aggregation is month-to-date capped at 31 daily lags. It does not yet implement true flow-vs-stock transformations per series.
- MS-VAR covariance shrinkage is a guardrail, not a full posterior over covariance uncertainty.
- Bayesian MS-VAR uses ordered intercept anchoring for label-switching mitigation; symmetric regimes can still require domain-specific identification constraints.
