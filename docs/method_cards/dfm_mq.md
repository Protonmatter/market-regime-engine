# Mixed-Frequency Dynamic Factor Method Card

## Production status
Experimental frontier for D/W/M custom state-space and retrospective smoothed factors; M/Q statsmodels path remains constrained by input contract.

## Module path
`market_regime_engine.frontier.dfm_mq`, `market_regime_engine.frontier.diagnostics`

## Mathematical equation
Latent factor evolves as a state process; observations map daily, weekly, or monthly releases to the latent state through frequency-specific observation equations.

## Inputs
Dated panel with declared frequency layout. D/W/M layouts must be predeclared and cannot be mixed with unsupported quarterly layouts.

## Outputs
Filtered/nowcast factor values, standard errors when available, backend metadata, and experimental flags for smoothed paths.

## Assumptions
Online production decisions use filtered/prefix-safe values only; smoothed factors are retrospective and fenced.

## Failure modes
Unsupported frequency layout, release-calendar leakage, holiday/partial-week mismatch, statsmodels fallback failure.

## Diagnostics
Frequency mix, backend, missingness, update count, prefix-safety check, statsmodels failure log when applicable.

## Release-gate requirements
Experimental flag required; prefix-safety diagnostics required before any online use claim.

## Tests that validate it
`tests/test_frontier_dfm_mq.py`, `tests/test_review_math_hardening.py`, `tests/test_certification_frontier_diagnostics.py`.

## Known limitations
The custom backend is intentionally simplified and should be extended with business-day and release-calendar exactness before production use.
