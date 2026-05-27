# Mathematics Used

This document is the concise mathematical map for the build. For module-level
production status and assumptions, see `docs/MATH_METHODS.md` and the method
cards under `docs/method_cards/`.

## Forecast target

The engine estimates conditional predictive distributions:

```text
P(Y_{t+h} | F_t)
```

where `F_t` is the information set available at forecast time. The point-in-time
constraint is part of the mathematics, not just an engineering detail:

```text
observation_date <= t
vintage_date     <= t
```

If this condition is violated, validation metrics are contaminated by
look-ahead information and the model is not eligible for release.

## Composite forecast distribution

The README's core forecast equation is:

```text
F_hat_{t,h}(y) =
  C_{h,R}[ sum_m w_{m,t,h}(CP_t, gamma_t, R_t, Loss_m, CalErr_m)
                 F_{m,t,h}(y) ]
```

Terms:

- `CP_t`: online change-point probability.
- `gamma_t`: latent regime posterior.
- `R_t`: decoded regime state.
- `w_{m,t,h}`: model weights from validation loss, calibration error,
  regime fit, change-point intensity, staleness, and online BMA evidence.
- `F_{m,t,h}`: model-specific predictive distribution.
- `C_{h,R}`: conformal calibration layer, optionally regime-conditional.

## Regime models

| Method | Mathematical role | Production posture |
|---|---|---|
| HMM | Latent finite-state posterior over feature vectors using Baum-Welch-style estimation | Stable core |
| MS-VAR | Regime-dependent vector autoregression with companion-radius stability checks | Stable core |
| Bayesian MS-VAR | Bayesian AR(p) regime model with posterior diagnostics, R-hat/ESS/divergence checks | Frontier |
| WFST | Constrained Viterbi decoding over regime states | Stable core |

MS-VAR state equation:

```text
y_t = c_s + A_{s,1} y_{t-1} + ... + A_{s,p} y_{t-p} + epsilon_t
epsilon_t ~ N(0, Sigma_s)
```

## Change-point models

The engine uses several online change-point signals:

- rolling Mahalanobis distance;
- Normal-Inverse-Wishart BOCPD;
- BOCPD-MUSE;
- covariate-conditioned hazard;
- GP-BOCPD and deep-kernel GP-BOCPD in the frontier lane.

These methods estimate the probability that recent observations no longer
belong to the same run-length regime as prior observations. Run-length
truncation is an approximation and should be monitored in frontier diagnostics
when long-memory behavior matters.

## Nowcasting and mixed frequency

Mixed-frequency nowcasting uses dynamic factor ideas:

- M/Q layouts can route through a DynamicFactorMQ backend when available.
- D/W/M layouts use a native Kalman state-space backend.
- Smoothed factors can depend on future observations and are retrospective;
  filtered/prefix-safe values are required for live decisioning.

## Validation statistics

Release evidence combines point forecasts, probabilistic calibration,
distributional comparison, and model-selection controls:

| Statistic | Role |
|---|---|
| Brier score | Mean squared error for binary probabilities |
| Log loss | Strict scoring rule for binary probability forecasts |
| ECE | Calibration gap across probability bins |
| Pinball loss | Quantile forecast loss |
| Diebold-Mariano / HLN | Forecast loss comparison with small-sample correction |
| Giacomini-White | Conditional predictive ability using instruments |
| Hansen MCS | Model Confidence Set selection using `T_R` and `T_SQ` statistics |
| Christoffersen UC/CC | Unconditional and conditional coverage checks |
| Knuppel PIT tests | Raw-moment and autocorrelation diagnostics for PIT values |
| Murphy decomposition | Reliability, resolution, and uncertainty |
| CRPS-DM | Distributional forecast comparison using CRPS loss |
| DSR / PBO | Overfit-control evidence for strategy/model selection |

## Conformal methods

Conformal outputs convert model scores into finite-sample coverage controls
when exchangeability or the chosen time-series variant's assumptions are
credible.

Implemented families include:

- Mondrian split conformal by regime or bucket;
- block conformal for dependent sequences;
- conformalized quantile regression;
- adaptive conformal inference;
- NexCP online adaptation;
- conditional and localized conformal variants;
- e-conformal / sequential e-value promotion paths;
- Bonferroni multi-horizon coverage.

## Fixed-income execution confidence

The fixed-income execution-confidence layer scores whether an order context is
eligible for a given execution protocol. The model combines:

- credit-regime context;
- liquidity-stress context;
- order side and notional;
- protocol label;
- signal age and upstream release-gate state;
- calibrated empirical evidence when outcome data exists.

Track B adds counterfactual protocol ranking:

```text
score(request with Auto-X)
score(request with RFQ)
score(request with Manual)
        |
        v
rank release-gate-passing candidates by confidence
        |
        v
tie-break RFQ, Auto-X, Manual unless caller supplies explicit order
```

If every candidate fails governance or staleness, the recommendation is Manual
with human review.

## Numeric contract

Internal model math may use floats where existing scorers require them. New
XPro decision artifacts publish deterministic fixed-point values:

| Quantity | Representation |
|---|---|
| Probability | integer ppm |
| Basis points | integer q4 |
| Price | integer q6 |
| Money/notional | integer cents |
| Timestamp | UTC epoch nanoseconds as string |

Canonical JSON hashing uses the project RFC8785/JCS v2 encoder. Raw NaN,
Infinity, naive timestamps, and raw floats in new XPro artifacts fail closed.

## Certification mathematics

Certification release profiles treat validation evidence as mandatory. For
execution confidence, realized-outcome validation joins decision-time
predictions to later outcomes and emits:

- Brier, log-loss, and ECE;
- calibration by regime;
- confidence-decile fill-rate lift;
- positive-direction TCA lift by regime;
- minimum regime sample-size support;
- validation artifact hash for release-gate consumption.

Missing or non-finite certification metrics are release blockers.
