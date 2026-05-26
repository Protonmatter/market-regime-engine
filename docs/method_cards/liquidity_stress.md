# Fixed-Income Liquidity Stress Method Card

## Production status
Stable core fixed-income signal.

## Module path
`market_regime_engine.fixed_income.liquidity_stress`

## Mathematical equation
Composite liquidity stress index combines spread, depth, dealer-response, TRACE/RFQ, and scope-specific features with governed weights and hysteresis.

## Inputs
PIT market data, dealer response stats, TRACE/RFQ events, scope identifier.

## Outputs
Liquidity stress score, label, release-gate flag, artifact hash, signal age.

## Assumptions
Feature vintages are valid as of the score timestamp and scope-specific data is not collapsed across unrelated bonds without declared hierarchy.

## Failure modes
Missing critical features, stale market data, scope mismatch, hysteresis misconfiguration.

## Diagnostics
Missing-feature report, signal age seconds, release gate, score component metadata.

## Release-gate requirements
Fail closed on critical missing features, stale signal, or unset release gate for production use.

## Tests that validate it
`tests/test_liquidity_stress.py`, `tests/test_liquidity_stress_missing_features_degrades_safely.py`, `tests/test_liquidity_stress_per_scope.py`.

## Known limitations
Composite index weights require empirical review and should not be treated as universal liquidity laws.
