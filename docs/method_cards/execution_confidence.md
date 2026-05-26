# Fixed-Income Execution Confidence Method Card

## Production status
Stable core baseline with empirical calibration and realized-outcome validation gates.

## Module path
`market_regime_engine.fixed_income.execution_confidence`, `market_regime_engine.fixed_income.execution_calibration`, `market_regime_engine.fixed_income.execution_validation`

## Mathematical equation
Baseline score is logistic: `p = sigmoid(β0 + β_liq x_liq + β_regime x_regime + β_notional x_notional + protocol/rating/urgency terms)`. Empirical Platt calibration maps `logit(p_raw)` to realized fill-success probability.

## Inputs
Execution request, PIT credit-regime score, PIT liquidity-stress score, optional bond/dealer/outcome context, realized outcomes for validation.

## Outputs
Confidence score, expected slippage, recommendation, artifact hash, metadata, realized validation report.

## Assumptions
All features are available at decision time; outcome rows occur strictly after decision time; realized outcome labels are consistently defined.

## Failure modes
Stale signals, missing release gate, leakage through outcomes, underpowered regime buckets, poor calibration, negative TCA lift; invalid probability scores outside [0, 1].

## Diagnostics
Brier, log-loss, ECE, calibration by regime, lift by confidence decile, positive-direction TCA lift by regime, outcome sample size, invalid probability-score count, artifact hash.

## Release-gate requirements
Certification requires PIT pass, walk-forward pass, Brier/ECE, DSR/PBO when applicable, positive-direction TCA lift payload, model card, validation artifact hash, and evidence-pack HMAC.

## Tests that validate it
`tests/test_execution_confidence.py`, `tests/test_execution_confidence_calibration.py`, `tests/test_certification_release_and_execution_validation.py`.

## Known limitations
The baseline is explainable but hand-tuned; production promotion should be based on realized outcome lift, not coefficient aesthetics.
