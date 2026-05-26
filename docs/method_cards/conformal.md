# Conformal Coverage Method Card

## Production status
Stable core validation method; some online/frontier conformal variants remain experimental.

## Module path
`market_regime_engine.conformal`, `market_regime_engine.multi_horizon_conformal`, `market_regime_engine.frontier.online_conformal`

## Mathematical equation
Prediction sets/intervals are calibrated from nonconformity scores so empirical coverage approaches `1 - alpha` under exchangeability or the stated online assumptions.

## Inputs
Predictions, realized targets, alpha, bucket/group definitions.

## Outputs
Coverage report by target, horizon, bucket/group, and realized coverage.

## Assumptions
Calibration data is exchangeable or the online method's sequential validity assumptions hold; calibration rows are PIT-safe.

## Failure modes
Coverage data missing, bucket underpower, leakage, nonstationarity, adaptive misuse.

## Diagnostics
Worst coverage, target coverage, bucket n, coverage drop, missing-data flags.

## Release-gate requirements
Production/certification fail when required coverage report is missing or below floor.

## Tests that validate it
`tests/test_conformal.py`, `tests/test_conformal_coverage.py`, `tests/test_release_gates_empty_coverage.py`.

## Known limitations
Coverage guarantees do not imply directional accuracy or economic value.
