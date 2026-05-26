# Giacomini-White Forecast Comparison Method Card

## Production status
Stable core validation method.

## Module path
`market_regime_engine.forecast_compare`

## Mathematical equation
Tests conditional predictive ability using moment conditions `g_t = z_t * d_t`, where `d_t` is the loss differential and `z_t` are instruments. Covariance uses a HAC sandwich estimator over vector moments.

## Inputs
Competing forecast losses, instrument matrix, HAC lag choice.

## Outputs
Test statistic, p-value, HAC covariance metadata, and diagnostics.

## Assumptions
Moment process is weakly dependent; instruments are not rank-deficient or diagnostics explicitly report repair.

## Failure modes
Rank-deficient instruments, too few observations, ill-conditioned HAC covariance, excessive serial dependence.

## Diagnostics
HAC lag, moment rank, condition number, covariance repair/pinv/ridge status, sample sufficiency.

## Release-gate requirements
Certification should reject insufficient samples or unrepaired singular covariance.

## Tests that validate it
`tests/test_forecast_compare.py`, `tests/test_review_math_hardening.py`.

## Known limitations
Asymptotic p-values can be fragile in small samples; block bootstrap should be preferred for high-stakes promotions.
