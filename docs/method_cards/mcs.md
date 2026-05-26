# Hansen Model Confidence Set Method Card

## Production status
Stable core promotion method.

## Module path
`market_regime_engine.promotion`, `market_regime_engine.forecast_compare`

## Mathematical equation
Sequentially eliminates inferior models from a candidate set under loss-differential uncertainty until the remaining set cannot be statistically distinguished at the configured level.

## Inputs
Out-of-sample losses, target, horizon, model identifiers, confidence level.

## Outputs
MCS membership status, promoted flag, evidence label.

## Assumptions
Losses are out-of-sample and point-in-time; model universe is declared before evaluation.

## Failure modes
Stale promotion rows, in-sample losses, untracked model universe changes.

## Diagnostics
Latest-date filtering, MCS evidence value, loss sample size.

## Release-gate requirements
Production and certification profiles require `mcs_evidence == in_set` for promoted models.

## Tests that validate it
`tests/test_promotion_mcs.py`, `tests/test_release_gates_promotion_latest_filter.py`.

## Known limitations
MCS does not prove economic usefulness; it only controls statistical model-set comparison under the provided loss function.
