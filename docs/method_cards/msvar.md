# Markov-Switching VAR Method Card

## Production status
Stable core with stability diagnostics. Promotion requires companion-radius and weak-regime support evidence.

## Module path
`market_regime_engine.msvar`

## Mathematical equation
For regime `s_t`, `y_t = c_{s_t} + Σ_{l=1}^p A_{s_t,l} y_{t-l} + ε_t`, with
`ε_t ~ N(0, Σ_{s_t})` and Markov transition matrix `P(s_t | s_{t-1})`.

## Inputs
Dated multivariate panel, regime count, lag order, and nan policy.

## Outputs
Filtered regime probabilities, semantic regime labels, companion-radius diagnostics, covariance shrinkage metadata.

## Assumptions
Regime transitions are Markovian; VAR roots must be stable for production use; weak regimes require shrinkage or rejection.

## Failure modes
Explosive VAR roots, label switching, singular covariance, collinearity, low effective sample size per regime.

## Diagnostics
`max_companion_radius`, per-regime radius, covariance condition number, shrinkage intensity, label assignment cost.

## Release-gate requirements
Fail if post-stabilization radius remains `>= 1`, if regime support is below floor, or if shrinkage/stability repair is excessive.

## Tests that validate it
`tests/test_review_math_hardening.py`, `tests/test_phase2_phase3.py`, `tests/test_refactor_boundaries.py`.

## Known limitations
Current stabilization is a conservative repair path, not a fully constrained maximum-likelihood optimizer.
