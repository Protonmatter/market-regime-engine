# Pragma-grade Target: How This Engine Becomes Institutional-Quality

This project is not trying to clone a private execution engine. The target
is to exceed a typical execution-model stack on **macro-regime intelligence,
probability calibration, point-in-time discipline, and explainability**.

## Public benchmark context

Public MarketAxess/Pragma material describes Pragma as algorithmic trading
intelligence and quantitative technology. MarketAxess has also publicly
described Pragma's deep-neural-network execution engine as controlling
routing, sizing, pricing, and timing of orders. That is an **execution
modeling** problem.

This application is a **macro/market regime probability engine**. It should
be judged against different targets:

- recession probability calibration
- drawdown probability calibration
- forward return distribution quality
- regime transition detection
- historical analog usefulness
- scenario sensitivity
- model-risk governance readiness

## Differentiators required (with v1.2 status)

| Capability | MVP | Institutional target | v1.2 status |
|---|---:|---:|---|
| Point-in-time data alignment | partial | required on every feature | **done** — `audit-vintage --enforce`, `feature_asof_values` is the default training input |
| Vintage macro support | scaffold | ALFRED/FRED vintage ingestion | **done** — `alfred_real.py` uses `series/vintagedates`; observation-by-vintage retrieval |
| Walk-forward validation | added in v0.2 | required for every model | **done** — `walk_forward.PurgedWalkForward` with `horizon` purge + `embargo`; `CombinatorialPurgedCV`; wired into `backtest.benchmark_report` (v1.1) |
| Calibration reporting | added in v0.2 | model-card gated release | **done** — Platt + 6 conformal backends (split / block / NexCP / conditional / localized / e-conformal) + Bonferroni multi-horizon |
| Regime-aware weighting | scaffold | learned and monitored | **done** — `stacking_v2` regime-conditioned grid + `bma.OnlineBMA` exponentially-discounted log-score (post-norm floor in v1.2) |
| Online change-point detection | rolling detector | BOCPD / Student-t multivariate core | **done** — `bocpd.MultivariateNIWBOCPD` + `bocpd_muse.BOCPDMuse` + `bocpd_hazard.CovariateBOCPDHazard` + v1.2 `frontier.gp_cpd.GPBOCPD` |
| WFST regime grammar | path smoother | formal constrained decoder | **done** — `wfst.RegimeWFST` with prior arcs, learnable empirical re-costing, event-bonus grid search |
| Stress testing | manual | scenario library + adversarial tests | **done** — `scenarios.SCENARIOS` covers 1973 oil → 2022 inflation; per-scenario pass/fail |
| Model registry | added in v0.2 | release-gated governance workflow | **done** — `release_gates`, `alerts`, `promotion_workflow`, immutable `model_runs` with reproducibility envelope; v1.2 supports both Hansen MCS and sequential e-value safe-testing (experimental flag required for release-gate promotion) |
| Explainability | basic | local attribution + analog evidence | **done** — z-score (`attribution`) + counterfactual (`counterfactual.counterfactual_delta`) + permutation Owen-style + optional SHAP + regime-weighted analogs |
| Forecast comparison | naive deltas | DM / GW / Hansen MCS / PIT / Christoffersen / Murphy / CRPS-DM | **done** — `forecast_compare` with HLN, T_R + T_SQ, autocorrelation moments, and v1.2 `crps_diks_panchenko` |
| Latent regime model | hand-prior HMM | Baum-Welch + MS-VAR | **done** — `hmm.HMMRegimePosterior.fit` (Baum-Welch + label pinning) and `msvar.MarkovSwitchingVAR` (Hamilton-Kim) |
| Domain factor model | hand-tuned linear | learned DFM | **done (v1.2)** — `dfm.DFMDomainModel` (Watson-Engle EM Kalman + RTS smoother with v1.2 *true marginal* likelihood) plus `frontier.dfm_mq.MQDynamicFactorModel` (M/Q Bańbura-Modugno wrapper plus native D/W/M Kalman state-space backend) |
| Nowcasting | none | mixed-frequency / ragged-edge | **done (v1.2)** — `MQDynamicFactorModel` + `frontier.midas.MIDASRegressor` Almon-polynomial; `mre nowcast` writes `nowcast_factors` |
| Distributional regression | per-quantile HGBR | parametric + non-parametric + neural | **done (v1.2)** — `frontier.distributional`: NGBoost / Henzi-Ziegel-Gneiting IDR / Karl-Soelch DVBF deep state-space |
| Neural sequence baseline | none | transformer / state-space | **done (v1.2)** — `frontier.neural_seq.PatchTSTHead` with CPU-friendly defaults and torch soft-degrade |
| Anytime-valid promotion | none | sequential e-values | **done (v1.2)** — `frontier.sequential_testing.EValueLogScore` + `SafeTestPromotion`; reachable via `release_gate(promotion_method="e_values")` |
| Robust statistics | mean/std z | MAD / winsorized z | **done** — `robust_stats` |
| Hot paths | pure Python | validated Rust kernels | **done** — `rust_ext` (NIW BOCPD update, WFST Viterbi, PSI, rolling Mahalanobis) with parity tests at `atol=1e-9` |
| Reproducibility | hash of features only | full envelope (git, lockfile, payloads, RNG) | **done** — `model_runs.ReproEnvelope` + `mre verify-run` |
| Observability | print-to-stdout | structured logs + Prometheus summary | **done** — `logging_setup` (`json` + `human`), `observability.prometheus_text` (v1.1 fix: real percentiles, not mean) |
| API hardening | read-only | versioned + auth + cache + metrics | **done** — `api_v1.app`, `MRE_API_KEY`, TTL cache (lock-protected v1.1), `/v1/metrics` (auth-gated v1.1) |
| Orchestration | CLI subcommands | scheduler-ready flow | **done** — `orchestration.daily_flow` (v1.2 adds nowcast / conditional-coverage / e-value steps) |
| Multi-horizon coherence | independent intervals | joint conformal | **done** — `multi_horizon_conformal.BonferroniMultiHorizonConformal` (Stankevičiūtė et al. 2021 + v1.1 rsuffix-bug fix) |

## Release gate

A model cannot be promoted unless it beats these baselines out-of-sample on
purged + embargoed walk-forward, and survives Hansen MCS at the configured
confidence level *or* the sequential e-value safe-test:

1. historical event-rate probability
2. yield-curve-only recession model
3. previous-month same-probability naive model
4. historical median return / unconditional quantile model
5. simple macro-stress logistic model

The promotion gate (`promotion.PromotionGate`) thresholds Brier / log-loss /
ECE deltas; the multi-model gate
(`forecast_compare.hansen_mcs(statistic="T_R" | "T_SQ")`) requires
MCS membership; the anytime-valid gate
(`frontier.sequential_testing.SafeTestPromotion`) requires the e-value to
cross `1/α`.

## Promotion criteria

For each horizon:

- Lower log loss than baseline (DM-significant at the configured level)
- Lower Brier score than baseline
- Better or equal expected calibration error (`max_ece` ≤ 0.12 by default)
- Tail quantile coverage within tolerance (CQR / NexCP / conditional /
  localized / e-conformal-conformalised, marginal coverage `1 - α`)
- Conformal coverage by regime bucket within ±2% of target (Mondrian
  per-bucket coverage report)
- Conditional conformal coverage within target on the
  `conditional_coverage_report` table (v1.2)
- No look-ahead leakage found (`audit-vintage --enforce` PASS)
- Stable across at least three historical regimes
- Generates a model card with limitations
- Reproducibility envelope verifies (`mre verify-run` exits 0)
- Sequential e-value (when `promotion_method="e_values"`) ≥ `1/α`

## Kill criteria

Demote a model if:

- calibration error doubles over the rolling window
- conformal coverage drifts more than 5 percentage points away from
  target on the latest hold-out
- performance only works in one era
- predictions swing without feature evidence (counterfactual deltas
  near zero)
- feature importance is unstable
- change-point periods cause systematic underestimation
- it cannot beat a naive baseline after walk-forward testing
- the reproducibility envelope no longer verifies
- the running e-value test crosses `1/α` *against* the candidate
- the DFM-MQ nowcast factor diverges from the engine's domain stress
  score by > 3σ for an extended period

## Engineering rule

The engine should not claim exact market levels. It should emit:

- forecast distribution (with conformal coverage guarantee — choose the
  right backend for the regime)
- drawdown probability (regime-conditional Mondrian / conditional / localized)
- recession probability (path-aware horizon survival)
- regime posterior (HMM and / or MS-VAR)
- mixed-frequency nowcast factors (DFM-MQ)
- dominant drivers (z-score + counterfactual delta)
- historical analogs (regime-weighted)
- model confidence, drift, invalidation triggers
- release-gate / alert / promotion decisions (Hansen MCS *or* sequential
  e-value)
- reproducibility envelope (git SHA, lockfile hash, payload hashes)

## v1.2 institutional delta

v1.2 closes the gap from "post-v1.1 institutionally defensible" to
"2026-2027 SOTA frontier" with seven additions:

1. **Time-series-native conformal**. The marginal-coverage guarantee no
   longer leans on exchangeability; six backends cover every modern
   finite-sample relaxation.
2. **Mixed-frequency DFM-MQ + MIDAS**. Bańbura-Modugno M/Q is the production
   architecture at the New York Fed and the ECB.
3. **Distributional regression heads**. NGBoost (parametric SOTA), IDR
   (non-parametric calibration gold-standard), and a small DVBF-style
   deep state-space head — together they cover the full spectrum of
   "give me a calibrated predictive density".
4. **Neural sequence baseline**. PatchTST is the 2023-2024 winner of
   long-horizon benchmarks. Ships as a small CPU-friendly default with
   torch soft-degrade.
5. **Sequential safe-testing**. e-values + safe testing supersede
   fixed-horizon MCS for online deployments because they give *anytime-
   valid* type-I-error control.
6. **CRPS-DM**. The binary Murphy decomposition was the only
   loss-difference test in v1.0/v1.1; CRPS-DM
   (Diks-Panchenko-van Dijk) is the distributional counterpart so the
   new distributional heads can be compared properly.
7. **GP-BOCPD**. Saatçi-Turner-Rasmussen 2010 BOCPD with GP emissions
   and a deep-kernel hook for Wilson-Hu-Salakhutdinov-Xing 2016-style
   learned embeddings.

The current model is still not production approved without a real
ingestion footprint and out-of-sample track record. v1.2 is the
*correct* algorithmic frontier, the validated modeling and conformal
layers, and the audit + governance plumbing — not the final alpha
engine for any one strategy.
