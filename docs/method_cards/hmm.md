# HMM Regime Classifier Method Card

## Production status
Stable core. Production-eligible when PIT inputs and validation artifacts pass the selected release gate.

## Module path
`market_regime_engine.hmm`

## Mathematical equation
A discrete latent-state hidden Markov model uses transition probabilities
`P(z_t=j | z_{t-1}=i)` and state-conditional emissions `p(x_t | z_t)` to infer
filtered regime probabilities `P(z_t | x_1:t)`.

## Inputs
Macro/market feature panel with dates, numeric features, and explicit missing-data handling.

## Outputs
Regime label, regime probabilities, confidence score, and metadata.

## Assumptions
Filtered probabilities are online-safe; labels are semantic after deterministic assignment; features are point-in-time valid.

## Failure modes
Label symmetry, degenerate state support, missing features, and non-PIT feature construction.

## Diagnostics
State support, confidence distribution, transition matrix, label stability, PIT audit status.

## Release-gate requirements
MCS promotion evidence, PIT leakage pass, walk-forward validation, calibration/coverage evidence when used downstream.

## Tests that validate it
`tests/test_review_math_hardening.py`, `tests/test_core.py`, `tests/test_golden_trace.py`.

## Known limitations
HMM emissions are not a structural causal model and should not be interpreted as policy-invariant dynamics.
