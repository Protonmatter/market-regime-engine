# Model Risk and Validation Standard

## Purpose

This document defines the minimum validation standard for models in the
Market Regime Engine. Every promoted artifact must pass these gates; the
pipeline encodes them as fail-closed CLI commands.

## Required controls

1. **Point-in-time data**
   - No feature may use a value unavailable on the forecast date.
   - Vintage data must be used wherever revisions materially affect the
     series.
   - Enforced by `mre audit-vintage --enforce` and the
     `audit_feature_asof_lineage` invariant `observation_date <= as_of_date
     AND vintage_date <= as_of_date`.
   - `mre train-baseline` and `mre validate` route through
     `feature_asof_values` by default. `--legacy-features` is allowed only
     for back-compat regression work.

2. **Walk-forward testing**
   - No random train/test split for time-series targets.
   - Use the `walk_forward.PurgedWalkForward` (expanding or rolling) with
     an explicit `horizon` purge and `embargo` gap.
   - Use `walk_forward.CombinatorialPurgedCV` (López de Prado 2018) for
     variance-stable Sharpe / accuracy estimation.
   - The legacy expanding-window split in `backtest.py` was migrated to
     `PurgedWalkForward` in v1.1; new model evaluation must use the
     purged splitter.

3. **Baseline comparison**
   - Every model must beat simple baselines before promotion.
   - Baselines are not optional decorations.
   - Required baselines: historical event-rate, previous-event naive,
     expanding-quantile, yield-curve-only recession, simple
     macro-stress logistic.
   - Promotion is decided by `promotion.PromotionGate` (Brier / log-loss /
     ECE deltas) **and** by either Hansen MCS membership
     (`forecast_compare.hansen_mcs(statistic="T_R" | "T_SQ")`) when more
     than two models are compared, or by sequential safe-testing
     (`frontier.sequential_testing.SafeTestPromotion`) when an
     anytime-valid gate is preferred.

4. **Calibration**
   - Binary probabilities require Brier score, log loss, and expected
     calibration error.
   - Quantile forecasts require pinball loss and realized coverage.
   - Distributional forecasts require CRPS and CRPS-DM
     (`forecast_compare.crps_diks_panchenko`).
   - Calibration error is measured by:
     - `validation.expected_calibration_error` (binary).
     - `forecast_compare.murphy_decomposition` (REL / RES / UNC).
   - Coverage is measured by:
     - `forecast_compare.pit_uniformity(autocorrelation=True | False)`
       (Diebold-Gunther-Tay + Knüppel raw moments and / or Knüppel
       autocorrelation moments).
     - `forecast_compare.christoffersen_coverage` (UC + CC).
   - Conformal layers must be applied **on top of** Platt-calibrated
     outputs. The full menu of backends:
     - `conformal.MondrianBinaryConformal(backend="split")` per regime
       bucket — Vovk-Gammerman-Shafer 2005 (assumes exchangeability).
     - `conformal.MondrianBinaryConformal(backend="block")` —
       Politis-Romano stationary block bootstrap; recovers a finite-sample
       coverage guarantee under stationary β-mixing.
     - `conformal.MondrianBinaryConformal(backend="nexcp")` —
       Stankevičiūtė-Alaa-van der Schaar 2021 NexCP, time-series-native
       split conformal.
     - `conformal.MondrianBinaryConformal(backend="conditional")` —
       Gibbs-Cherian-Candès 2023 group-conditional finite-class.
     - `conformal.MondrianBinaryConformal(backend="localized")` —
       Lin-Trivedi-Sun 2023 RBF-localized split conformal (test-point
       conditional quantile).
     - `conformal.MondrianBinaryConformal(backend="e_conformal")` —
       Vovk-Wang 2021 sequential e-conformal with anytime-valid coverage.
     - `conformal.ConformalizedQuantileRegression` (Romano-Patterson-Candès
       2019) per quantile pair.
     - `conformal.AdaptiveConformalInference` (Gibbs-Candès 2021) for
       online drift.
     - `multi_horizon_conformal.BonferroniMultiHorizonConformal` for
       joint multi-horizon coverage.

5. **Regime validation**
   - Performance must be reported by regime segment.
   - A model that only works in one era is a regime-specific tool, not a
     general engine.
   - The Mondrian and conditional conformal layers encode this:
     per-bucket thresholds and per-bucket coverage diagnostics are
     persisted alongside the calibrated outputs (see
     `conditional_coverage_report` warehouse table).

6. **Forecast-comparison statistics**
   - For any pairwise comparison: `forecast_compare.diebold_mariano` with
     HLN small-sample correction (5%-direction).
   - For any conditional question ("does model A beat B given regime
     state?"): `forecast_compare.giacomini_white`.
   - For multi-model evaluation: `forecast_compare.hansen_mcs` with
     stationary block bootstrap; report both `statistic="T_R"` and
     `statistic="T_SQ"` for high-power coverage of the alternative space.
   - For distributional forecasts: `forecast_compare.murphy_decomposition`
     plus `pit_uniformity` and `crps_diks_panchenko`.
   - For online deployments: prefer
     `frontier.sequential_testing.EValueLogScore` /
     `SafeTestPromotion` (anytime-valid) over fixed-window MCS.

7. **Explainability**
   - Every forecast must expose top contributing domains.
   - Every risk change must identify which feature groups moved the
     output.
   - In addition to z-score attribution (`attribution.py`), every
     promoted model emits **counterfactual deltas**
     (`counterfactual.counterfactual_delta`): "if feature X were where it
     was 12 months ago, the prediction would change by Δ." Optional
     `shap_attribution_if_available` for tree / linear heads when the
     `shap` extra is installed.

8. **Reproducibility envelope**
   - Every promoted model is tied to an immutable `model_run` row whose
     `metadata_json` contains the full
     `model_runs.ReproEnvelope`: short and long git SHA, working-tree
     dirty bit, lockfile SHA-256, platform / Python version,
     feature / output / vintage payload SHA-256s, and named RNG seeds.
   - `mre verify-run` re-derives the envelope and exits non-zero on any
     drift.

9. **Change management**
   - Every promoted model gets a model card.
   - Every model card records objective, horizon, training window,
     metrics, limitations, and artifact hash.
   - The release gate
     (`release_gates.evaluate_release_gate(promotion_method="mcs" |
     "e_values")`) consumes confidence, drift, invalidation triggers,
     promotion outcome, *and* conformal-coverage drift, and emits a
     single fail-closed decision (`release` or `hold`).

10. **Scenario replay**
    - Models are exercised against the canonical historical episodes
      (`scenarios.SCENARIOS`): 1973 oil shock, Volcker disinflation, S&L
      crisis, dotcom bust, GFC, COVID, 2022 inflation. The replay reports
      per-scenario pass/fail on regime, change-point, and hazard
      directionality.

11. **Mixed-frequency nowcast cross-check**
    - When the `[frontier]` or `[nowcast]` extra is installed,
      `frontier.dfm_mq.MQDynamicFactorModel` is run nightly and its
      per-domain factor estimates are persisted in the `nowcast_factors`
      warehouse table. M/Q inputs use the Bańbura-Modugno-style
      statsmodels backend when available; D/W/M inputs use the native
      filtered Kalman state-space backend. Smoothed factor extraction is
      retrospective-only and requires `MRE_ENABLE_EXPERIMENTAL_FRONTIER=1`.

## Validation artifacts

Required outputs (written by the pipeline into `data/validation` and the
warehouse):

- model card JSON (`mre model-card`)
- walk-forward predictions (`binary_predictions_*.csv`,
  `quantile_predictions_*.csv`)
- binary validation metrics (`binary_validation.csv`)
- quantile validation metrics (`quantile_validation.csv`)
- benchmark metrics + best-benchmark selection
  (`binary_benchmark_validation.csv`, `binary_best_benchmark.csv`)
- promotion decisions (`model_promotion.csv` plus the new
  `mcs_evidence` column)
- reliability table (`calibration_table` from `validation.py`)
- conformal coverage report — per-bucket, per-horizon, per-method (the
  `conditional_coverage_report` warehouse table)
- driver attribution sample + counterfactual deltas (top-K)
- known limitations
- release-gate decision row + alert routing rows + promotion-workflow row
- vintage / as-of audit rows
- `e_value_log` rows when sequential testing is run
- `nowcast_factors` rows when `mre nowcast` is run
- reproducibility envelope (in `model_runs.metadata_json`)

## Prohibited practices

- random cross-validation on time-series data
- centered rolling windows
- training on the legacy `features` table when `feature_asof_values`
  exists (use `--legacy-features` only with explicit owner sign-off)
- revised macro data used as if known historically
- exact price-level claims without a confidence distribution
- promotion based only on in-sample fit
- promotion based on a single beating-of-baseline without DM / MCS or
  sequential-e-value evidence at the requested confidence level
- opaque model output without driver evidence
- publishing the institutional report when `mre verify-run` exits
  non-zero
- silencing the staleness gate on `label-recessions` without a recorded
  exemption
- using `MondrianBinaryConformal(exchangeable=True)` on a non-exchangeable
  time-series stream — set `exchangeable=False` (auto-bumps to
  `block`) or pick `nexcp` / `conditional` / `localized` / `e_conformal`
  explicitly

## Hard kill criteria

Demote a model if any of the following holds for two consecutive
release-gate cycles:

- calibration error doubles over the rolling window
- conformal coverage drifts more than 5 percentage points away from
  target on the latest hold-out
- predictions swing without feature evidence (counterfactual deltas
  near zero)
- feature importance is unstable across regimes
- change-point periods cause systematic underestimation
- it cannot beat a naive baseline after walk-forward testing
- the reproducibility envelope no longer verifies under `mre verify-run`
- the sequential e-value test crosses `1/α` *against* the candidate
  (i.e. the champion's evidence dominates)
- the DFM-MQ nowcast factor and the engine's domain stress score
  diverge by > 3 standard deviations and stay there
