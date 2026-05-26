# Bayesian MS-VAR(p) Method Card

## Production status
Experimental frontier. Not production-eligible by default; requires `MRE_ENABLE_EXPERIMENTAL_FRONTIER=1` and posterior diagnostics.

## Module path
`market_regime_engine.frontier.bayesian_msvar`, `market_regime_engine.frontier.diagnostics`

## Mathematical equation
Bayesian regime-switching VAR(p): `y_t = c_s + Σ_l A_{s,l} y_{t-l} + ε_s`, with priors over transition rows, intercepts, lag coefficients, and covariance factors.

## Inputs
Feature panel, lag order `p`, priors, NUTS/SVI settings, random seed.

## Outputs
Posterior regime probabilities, credible bands, posterior means, convergence diagnostics, companion radii.

## Assumptions
NUTS/SVI diagnostics are reliable; ordered anchor reduces label switching; posterior mass exists for every operationally interpreted regime.

## Failure modes
Divergences, high R-hat, low ESS, posterior label instability, unstable companion radius, weak regime mass.

## Diagnostics
`num_divergences`, `max_rhat`, `min_ess`, `max_companion_radius`, `min_state_mass`, runtime, posterior predictive checks when supplied.

## Release-gate requirements
Experimental gates fail closed on divergences, missing/poor R-hat or ESS, unstable radius, or weak posterior regime mass.

## Tests that validate it
`tests/test_frontier_bayesian_msvar.py`, `tests/test_bayesian_msvar.py`, `tests/test_certification_frontier_diagnostics.py`.

## Known limitations
Full Bayesian inference is computationally expensive and remains research/frontier until posterior predictive checks are institutionalized.
