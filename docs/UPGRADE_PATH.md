# Upgrade Path

## v0.1-v0.4 baseline

- Sample data pipeline
- Feature matrix
- Regime scoring (rule-based + WFST + HMM scaffold + BOCPD scaffold)
- Backtesting and benchmark validation
- Historical analogs and attribution
- Institutional report shell

## v0.5 governance layer

- Release-calendar metadata
- Point-in-time audit table
- Calibrated probability outputs
- Immutable model-run IDs
- Forecast invalidation triggers
- Confidence score
- Regime-weighted analogs

## v0.6 forecast governance layer

- Drift monitor (PSI)
- Release gate
- Stacking diagnostics
- DuckDB / Parquet warehouse export

## v0.7 hazard + ingestion layer

- Discrete-time recession hazard model
- ALFRED/FRED real-time observation matrix (synthetic vintage grid)
- Regime-conditioned stacking optimizer
- Alert routing
- Promotion workflow record

## v0.8 real point-in-time vintage layer

- Real ALFRED vintage-date request planning via `series/vintagedates`.
- Observation-by-vintage storage in `vintage_observations`.
- As-of feature snapshots in `feature_asof_values`.
- Hard vintage/as-of leakage audits (`audit-vintage --enforce`).

## v1.0 SOTA-grade engineering, statistics, and modeling — completed

Detailed in [`V1_0_UPGRADE.md`](V1_0_UPGRADE.md). Headline:

- Engineering floor: GitHub Actions CI matrix, ruff, lockfile,
  pre-commit, structured logging, reproducibility envelope, golden-master
  regime trace.
- Statistical correctness: purged + embargoed walk-forward,
  Diebold-Mariano with HLN, Giacomini-White, Hansen MCS, PIT / Knüppel,
  Christoffersen, Murphy.
- Modern regime modeling: MS-VAR (Hamilton-Kim), BOCPD-MUSE,
  covariate-conditioned BOCPD hazard, online BMA, cross-sectional heads.
- Real PIT everywhere: `feature_asof_values` is the default training
  input.
- Validated Rust hot-paths: NIW BOCPD update, WFST Viterbi, PSI,
  rolling Mahalanobis (parity-tested).
- Operations: `daily_flow`, hardened `api_v1`, Streamlit dashboard, in-
  process metrics + Prometheus exposition.
- Distributional / scenario forecasting: scenario library,
  counterfactual attribution, Bonferroni multi-horizon conformal.

## v1.1 highest-ROI fix bundle — completed

Detailed in [`V1_1_FIXES.md`](V1_1_FIXES.md). Headline:

- `git init` + initial commit so the reproducibility envelope round-trips
  cleanly.
- `validate` and `backtest.benchmark_report` route through
  `walk_forward.PurgedWalkForward(horizon=H, embargo=1)`.
- Hansen MCS column on `PromotionGate` outputs and a
  `require_mcs_membership` flag on `release_gate`.
- `MondrianBinaryConformal` + CQR + a new `conformal_coverage` warehouse
  table wired into `daily_flow`, plus a `min_coverage` gate on
  `release_gate`.
- SQLite `journal_mode=WAL` + `synchronous=NORMAL` + `busy_timeout=10000`
  + `foreign_keys=ON`.
- API: `hmac.compare_digest` for the API-key compare, `/v1/metrics`
  auth, lock-protected `_TTLCache`.
- `prometheus_text` records true percentiles instead of collapsing to
  the mean.
- `BonferroniMultiHorizonConformal.joint_coverage` rsuffix collision
  fix.
- `training_data` PIT-router test coverage, audit dict surfaced into the
  reproducibility envelope.
- `label-recessions --max-stale-months` evaluated *before* the warehouse
  write, so a stale fetch can no longer poison the table.
- Lint debt collapse: 416 ruff errors → 0; 73/83 → 0/90 files
  needing format.

## v1.2 SOTA frontier — completed

Detailed in [`V1_2_FRONTIER.md`](V1_2_FRONTIER.md). The math-correctness
floor and the 2026-2027 frontier modeling layer in one release:

- **All 13 math fixes** from the post-v1.1 second-opinion math review.
- **`market_regime_engine.frontier.*` package** with five time-series-
  native conformal predictors (block / NexCP / Gibbs-Cherian-Candès
  conditional / Lin-Trivedi-Sun localized / Vovk-Wang sequential
  e-conformal), Bańbura-Modugno mixed-frequency DFM-MQ +
  Almon-polynomial MIDAS, three distributional heads (NGBoost /
  Henzi-Ziegel-Gneiting IDR / Karl-Soelch DVBF deep state-space), a
  CPU-friendly PatchTST baseline, sequential e-value safe-testing
  (Howard-Ramdas + Grünwald-de Heide-Koolen) wired into the release
  gate, CRPS-DM (Diks-Panchenko-van Dijk), and a Saatçi-Turner-Rasmussen
  GP-BOCPD.
- **Three new warehouse tables**: `e_value_log`, `nowcast_factors`,
  `conditional_coverage_report`.
- **Three new CLI commands**: `mre nowcast`, `mre e-value-test`,
  `mre conformal-conditional`.
- **Three new `daily_flow` summary keys**: `nowcast_factors`,
  `worst_conditional_coverage`, `e_value_promotion_pending`.

Optional dependencies (`statsmodels`, `ngboost`, `torch`) live behind a
new `[frontier]` extra; everything degrades gracefully when the optional
deps are missing.

162 tests pass (+45 vs. v1.1), ruff is clean, mypy improved 38 → 35.

## v1.2 deliberately deferred

These are tracked but not in v1.2:

- **DuckDB-primary `Warehouse`.** The export path covers analytics; full
  primary-store polymorphism is operations work that should follow real
  concurrency pain.
- **Consolidating `report_writer_v{2..5}`.** Cosmetic, and risks
  regressing the institutional report contract. Defer until a real
  refactor of the report itself.

## v1.3 recommended target (next quarter)

- DuckDB-primary `Warehouse` with feature parity to the SQLite store and
  a migration path.
- Wheel-distributed Rust extension for cp311 / cp312 across manylinux,
  win64, and macOS arm64.
- Wheel-distributed `[frontier]` bundle (and a docker image) so
  statsmodels / ngboost / torch are available by default.
- Daily/intraday vintage support: hourly `feature_asof_values`
  materialisation keyed off real release calendars.
- BLS / BEA / Census / Fed exact release-calendar ingestion.
- Live alert sinks (Slack / PagerDuty / email) gated behind env-var
  credentials.
- `mre verify-data` companion to `verify-run`: detect silent warehouse
  drift between a run-id record and the present database state.
- Consolidated `report_writer.py` with versioned section selection.

## v1.5 stretch target

- Sparse Bayesian regime-switching factor model (NumPyro variant of
  `msvar`) for small-N inference.
- Deep-kernel `GPBOCPD` with learned NN feature embeddings
  (Wilson-Hu-Salakhutdinov-Xing 2016 style).
- Multi-asset cross-sectional forecast layer with conformal guarantees
  per asset / per regime.
- Regime-aware stress-scenario *generation* (not just replay) using a
  generative model conditioned on regime posterior.
- Continuous deployment: signed wheels, container images, scheduled
  orchestration, alert smoke tests on every deploy.

## Operating discipline (unchanged across versions)

The engine should never make naked point forecasts. It should output
distributions, regime probabilities, dominant drivers, historical
analogs, model confidence, invalidation triggers, release-gate
decisions, mixed-frequency nowcast factors, and a verifiable
reproducibility envelope. Every promotion is contingent on PIT audit
PASS, conformal coverage at target, and DM / MCS-significant or
sequential-e-value-significant superiority over the relevant baseline
set.
