# Fixed-Income Credit Spread Regime Method Card

## Production status
Stable core fixed-income signal.

## Module path
`market_regime_engine.fixed_income.credit_spread_regime`

## Mathematical equation
Credit spread regime score aggregates spread/rate/curve features into a normalized stress score with hysteresis label assignment.

## Inputs
PIT credit-spread and macro/curve features.

## Outputs
Regime score, label, release-gate flag, artifact hash, signal age.

## Assumptions
Input spreads and curves are timestamped and available as of the score time; hysteresis thresholds are versioned.

## Failure modes
Feature leakage, stale source data, overcollapse onto unrelated HMM states, weak threshold calibration.

## Diagnostics
Feature contribution metadata, PIT audit, hysteresis band, signal age, release gate.

## Release-gate requirements
Fail closed on PIT audit failure, stale source data, or missing release gate when consumed by execution confidence.

## Tests that validate it
`tests/test_credit_spread_regime.py`, `tests/test_credit_regime_no_collapse_onto_hmm_states.py`, `tests/test_credit_regime_pit_audit_vectorized.py`.

## Known limitations
Score is a decision-support signal, not a structural credit-risk model.
