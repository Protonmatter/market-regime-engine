# Architecture

```text
                    Ingestion
   ┌─────────────────────────────────────────┐
   │ sample (deterministic synthetic panel)  │
   │ FRED observations / FRED USREC          │
   │ ALFRED real vintage dates + observations│
   └────────────────┬────────────────────────┘
                    ↓
                SQLite warehouse (WAL + busy_timeout)
        observations · series_vintages
        vintage_observations
                    ↓
        Point-in-time materialization
       feature_asof_values (audited, fail-closed)
                    ↓
        ┌──────────────────────────────┐
        │ Feature builder              │
        │  - transforms                │
        │  - robust_stats (MAD z)      │
        │  - DFM domain factor (EM)    │
        │  - DFM-MQ mixed-frequency    │
        │  - MIDAS Almon-poly          │
        └────────────┬─────────────────┘
                    ↓
   ┌────────────────────────┐ ┌──────────────────────────┐
   │ Change-point           │ │ Regime posterior         │
   │  - rolling Mahalanobis │ │  - HMM (Baum-Welch)      │
   │  - NIW BOCPD           │ │  - MS-VAR (Hamilton-Kim) │
   │  - BOCPD-MUSE          │ │  - WFST decoded path     │
   │  - covariate hazard    │ │                          │
   │  - GP-BOCPD            │ │                          │
   └─────────────┬──────────┘ └─────────────┬────────────┘
                 ↓                          ↓
                 └────────────┬─────────────┘
                              ↓
        ┌─────────────────────────────────────────┐
        │ Heads                                   │
        │  - logistic probability (drawdown)      │
        │  - HGBR quantile (non-crossing)         │
        │  - discrete-time hazard (path-aware)    │
        │  - cross-sectional FF/sector/curve      │
        │  - NGBoost distributional               │
        │  - IDR isotonic distributional          │
        │  - DVBF deep state-space                │
        │  - PatchTST neural sequence             │
        └────────────────────┬────────────────────┘
                             ↓
        ┌─────────────────────────────────────────┐
        │ Calibration / ensembling / coverage     │
        │  - Platt logit                          │
        │  - regime-conditioned simplex stacking  │
        │  - online BMA (post-norm floor)         │
        │  - Mondrian split conformal             │
        │  - CQR / ACI / multi-horizon Bonferroni │
        │  - block conformal                      │
        │  - NexCP                                │
        │  - Gibbs-Cherian-Candès conditional     │
        │  - Lin-Trivedi-Sun localized            │
        │  - Vovk-Wang sequential e-conformal     │
        └────────────────────┬────────────────────┘
                             ↓
        ┌─────────────────────────────────────────┐
        │ Validation                              │
        │  - purged + embargoed walk-forward      │
        │  - combinatorial purged CV              │
        │  - DM/HLN with 5%-direction             │
        │  - GW conditional                       │
        │  - Hansen MCS (T_R + T_SQ)              │
        │  - PIT / Knüppel raw + autocorrelation  │
        │  - Christoffersen UC + CC               │
        │  - Murphy CRPS decomposition            │
        │  - Diks-Panchenko-van Dijk CRPS-DM      │
        │  - sequential e-value safe testing      │
        └────────────────────┬────────────────────┘
                             ↓
        ┌─────────────────────────────────────────┐
        │ Governance                              │
        │  - confidence grade                     │
        │  - PSI drift monitor                    │
        │  - invalidation triggers                │
        │  - release gate (fail-closed)           │
        │       MCS or sequential-e-value variant │
        │  - conformal coverage gate              │
        │  - alert routing                        │
        │  - promotion workflow                   │
        │  - immutable model_runs + repro envelope│
        └────────────────────┬────────────────────┘
                             ↓
        ┌─────────────────────────────────────────┐
        │ Surfaces                                │
        │  - CLI (41 subcommands, JSON logs)      │
        │  - legacy /api + hardened /v1/api       │
        │  - Streamlit dashboard                  │
        │  - institutional report (markdown)      │
        │  - DuckDB / Parquet warehouse exports   │
        │  - Prometheus / OTel metrics            │
        │  - scenario replay (1973 → 2022)        │
        │  - mixed-frequency nowcast factors      │
        └─────────────────────────────────────────┘
```

## Fixed-Income lane (v1.5)

Per [V1_5_FIXED_INCOME_RCIE.md](V1_5_FIXED_INCOME_RCIE.md), v1.5
adds a Fixed-Income RCIE / X-Pro Auto-X adapter that sits alongside
the macro lane:

```text
                 (vendor feeds)
                       |
                       v
     IngestContract.validate            (TRACE / MarketAxess RFQ -
                       |                 schema-drift + bounds)
                       v
            Warehouse.write_*           (DuckDB - 13 FI tables)
                       |
        +--------------+--------------+
        v              v              v
 score_credit   score_liquidity  score_execution
   _regime        _stress         _confidence
        |              |              |
        v              v              v
  credit_regime   liquidity_   execution_
    _scores        stress_       confidence_
                   scores        predictions
        |              |              |
        +--------------+--------------+
                       v
            tag_trade_with_regime_context
                       |
                       v
                tca_regime_segments
                       |
                       v
            build_evidence_pack ->
            sign_pack (HMAC-SHA-256) ->
            write_evidence_pack
                       |
                       v
             fixed_income_evidence_packs
                       |
                       v
                  GET /v1/evidence-pack/{id}
                  mre verify-run --model-run-id <id>
                  mre fi-report
```

The FI lane shares the warehouse (`storage.register_tables` adds 13
FI tables on top of the 33 macro tables), the v1 API app (FI router
mounted on `api_v1.app`), the CLI dispatcher (`mre fi-*` routes
through `cli_dispatch._FI_COMMANDS`), the observability registry
(legacy + OTel adapters), and the model registry (three FI baselines
registered per FLAG F-20).

## Formal forecast target

```text
P(Y_{t+h} | F_t)
```

where `F_t` is the information set available at forecast time. v1.0
introduces and v1.2 enforces the point-in-time invariant on every feature:
any row used to construct `F_t` must satisfy `observation_date <= t` *and*
`vintage_date <= t`. The audit gate (`audit_feature_asof_lineage`) fails
closed when this is violated.

### v1.2.1 vectorised PIT materialisation

`asof.materialize_feature_asof_values` previously rebuilt the full vintage
panel and re-applied every feature transform once per as-of date — an
O(N_asof × N_series × N_transforms) Python loop that timed out the
reviewer's smoke pipeline. v1.2.1 detects whether the input has any
revisions per `(series_id, observation_date)` pair and dispatches to a
set-based pipeline. In the no-revisions fast path (the synthetic sample,
ALFRED snapshots before any revision lands) the panel and feature
transforms are computed exactly once on the full data; per-as-of work
collapses to a slice + `drop_duplicates` over the small per-feature
frame, with lineage attached via a single vectorised `merge_asof`. In
the revisions path the per-as-of rebuild is preserved but the inner
per-feature `iterrows` is replaced by a single vectorised `merge_asof`
for lineage and the panel build is cached when consecutive as-of dates
share the same legal table. Output is byte-identical to the
pre-v1.2.1 frame (modulo float rounding within `1e-9`); see
`tests/test_asof_perf.py`.

## Active targets

- `dd10_3m`, `dd10_6m`, `dd10_12m` — 10% drawdown probability over horizon h.
- `ret_3m`, `ret_6m`, `ret_12m` — forward log-return quantiles
  (5/10/25/50/75/90/95) with non-crossing repair and conformalised
  intervals (CQR / NexCP / conditional / localized / e-value).
- `recession_next_3m`, `recession_next_6m`, `recession_next_12m` — forward
  recession indicator (NBER + USREC-derived).
- `monthly_recession_hazard` — discrete-time hazard with path-aware
  horizon survival (`DiscreteTimeHazardModel.horizon_probability_path`).
- Cross-sectional: factor returns (Fama-French + extensions), sector
  dispersion, curve level / slope / curvature — all regime-conditioned.
- Mixed-frequency nowcast factors per domain (Bańbura-Modugno DFM-MQ).

## Composite forecast distribution

The engine combines per-model forecasts as:

```text
F_hat_{t,h}(y) = C_{h,R}[ Σ_m w_{m,t,h} · F_{m,t,h}(y) ]
```

where:

- `w_{m,t,h}` are **online BMA** weights (`bma.OnlineBMA`) updated with an
  exponentially-discounted log-score, biased by validation loss,
  calibration error, regime fit, change-point intensity, and staleness.
  v1.2: floor is applied *after* normalization so minority-model weights
  are not silently inflated.
- `C_{h,R}[·]` is a **regime-conditional conformal layer** dispatching to
  one of six backends:
  - `split` (`MondrianBinaryConformal`)
  - `block` (`frontier.conformal_ts.BlockConformalBinary`,
    Politis-Romano stationary block bootstrap)
  - `nexcp` (`frontier.conformal_ts.NexCPForecaster`,
    Stankevičiūtė-Alaa-van der Schaar 2021)
  - `conditional` (`frontier.conformal_ts.ConditionalConformalRegressor`,
    Gibbs-Cherian-Candès 2023)
  - `localized` (`frontier.conformal_ts.LocalizedSplitConformal`,
    Lin-Trivedi-Sun 2023)
  - `e_conformal` (`frontier.conformal_ts.SequentialEConformal`,
    Vovk-Wang 2021)
- It is optionally extended with
  `multi_horizon_conformal.BonferroniMultiHorizonConformal` for joint
  multi-horizon coverage (Stankevičiūtė et al. 2021 Bonferroni adjustment).

## Promotion gating

Two complementary mechanisms:

- **Hansen MCS** (`forecast_compare.hansen_mcs(statistic="T_R" | "T_SQ")`)
  with stationary block bootstrap, recentered under the equal-mean null;
  used by the default `release_gate` when `require_mcs_membership=True`.
- **Sequential e-value safe-testing**
  (`frontier.sequential_testing.EValueLogScore` +
  `SafeTestPromotion`); used by
  `release_gate(promotion_method="e_values")`. Anytime-valid: an operator
  can stop sampling whenever the e-value crosses `1/α`.

## Reproducibility envelope

Every `model_run` row carries the full envelope:

```text
{ code_sha, code_dirty, lockfile_hash, platform, python_version,
  feature_payload_sha256, output_payload_sha256, vintage_payload_sha256,
  rng_seeds, extra }
```

`mre verify-run` re-derives the envelope from the current environment and
fails non-zero if any field drifted. Run pinning is therefore not "trust
me, this number came from the script" but "here is the bit pattern of the
inputs that produced it."

## Cardinal rule

The engine emits **distributions, not point forecasts**, and **probabilities
with explicit coverage guarantees**, not probabilities without owners. The
release gate is fail-closed: high-severity invalidation triggers, severe
PSI drift, calibration error above the configured ceiling, conformal
coverage drift below the configured floor, missing promoted challenger,
and (when configured) lack of MCS membership / failing e-value all block
the gate. This is the design.
