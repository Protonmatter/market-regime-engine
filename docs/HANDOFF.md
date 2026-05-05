# Handoff

## Build status

v1.2. The engine runs end-to-end on synthetic sample data (no API keys
required) and is wired for live FRED/ALFRED ingestion when keys are set.
The full pytest suite passes:

```text
162 passed (was 117 pre-v1.2 → +45 new tests),
1 skipped (Rust parity, requires `maturin develop`),
1 deselected (slow), 0 failed.
```

`ruff check src tests` is clean. `ruff format --check` is clean across 100
files. `mypy src/market_regime_engine` is at 35 errors in 18 files (down
from 38 pre-v1.2). End-to-end smoke flow → `mre verify-run` exits 0 with
`approved: true`.

The CI matrix (`.github/workflows/ci.yml`) runs ruff, the pytest suite on
Python 3.11/3.12 across Linux and Windows, an end-to-end smoke flow, and
the golden-trace regression on every push.

## Quick verification

```bash
pip install -e ".[dev,dashboard,analytics]"

# Optional: build the Rust hot paths.
pip install maturin && (cd rust_ext && maturin develop --release)

# Optional: install the v1.2 frontier modeling extras.
pip install -e ".[frontier]"

pytest tests/ -q -m "not slow"

# Smoke the full pipeline.
mre bootstrap-sample --db data/mre.db
mre seed-vintage-from-observations --db data/mre.db
mre materialize-asof-features --db data/mre.db --write-features
mre audit-vintage --db data/mre.db --enforce
mre build-features --db data/mre.db
mre score-regime --db data/mre.db
mre label-recessions --db data/mre.db --max-stale-months 24
mre train-baseline --db data/mre.db
mre train-fitted-hazard --db data/mre.db --oos
mre validate --db data/mre.db --out data/validation
mre calibrate-probabilities --db data/mre.db --validation-dir data/validation

# v1.2 frontier steps (degrade gracefully without the extras).
mre nowcast --db data/mre.db
mre conformal-conditional --db data/mre.db --validation-dir data/validation
mre e-value-test --db data/mre.db --challenger candidate_logistic

mre score-confidence --db data/mre.db --validation-dir data/validation
mre release-gate --db data/mre.db --validation-dir data/validation
mre model-run --db data/mre.db --purpose "v1.2 handoff smoke"
mre verify-run --db data/mre.db
mre report --db data/mre.db
mre bench --out data/bench.csv
```

`mre verify-run` should exit `0` with `"approved": true`. If it ever exits
non-zero, the reproducibility envelope drifted; treat that as a release
blocker.

## What v1.0 added

See [`docs/V1_0_UPGRADE.md`](V1_0_UPGRADE.md). Headline modules:
purged + embargoed walk-forward, Mondrian / CQR / ACI conformal,
DM / GW / Hansen MCS / Christoffersen / Knüppel / Murphy comparison
statistics, Watson-Engle DFM, MS-VAR, BOCPD-MUSE, covariate-conditioned
BOCPD hazard, online BMA, cross-sectional heads, training-data PIT
router, FRED USREC + staleness gate, reproducibility envelope, structured
logging, in-process metrics with Prometheus exposition, hardened
`/v1/...` API, scenario replay, counterfactual attribution, multi-horizon
coherent conformal, real PyO3 Rust kernels.

## What v1.1 fixed

See [`docs/V1_1_FIXES.md`](V1_1_FIXES.md). The highest-ROI bundle from
the post-v1.0 SOTA roadmap and second-opinion review:
`git init` + initial commit so the reproducibility envelope is real,
purged walk-forward in `validate`, Hansen MCS column in `PromotionGate`,
conformal coverage gate in `daily_flow`, SQLite WAL + busy_timeout, API
`hmac.compare_digest` + `/v1/metrics` auth + lock-protected TTL cache,
fixed `prometheus_text` percentiles, Bonferroni `joint_coverage` rsuffix
fix, `training_data` PIT-router tests + audit envelope wiring,
`label-recessions` gate-before-write reorder, and the lint-debt collapse
(416 ruff errors → 0).

## What v1.2 added

See [`docs/V1_2_FRONTIER.md`](V1_2_FRONTIER.md) for the per-fix and
per-module table. Headline:

- **All 13 math correctness fixes** from the post-v1.1 math review (true
  marginal DFM likelihood via Sherman-Morrison-Woodbury, cached
  training-time `(mu, sd)`, AR(1) centering, `MondrianBinaryConformal`
  `backend=` dispatch, hazard horizon-path mode, BMA floor placement,
  Hansen MCS T_SQ statistic, Knüppel autocorrelation moments, DM
  5%-direction, dead-line cleanup, etc.).
- **`market_regime_engine.frontier.*` package**: five time-series-native
  conformal predictors, mixed-frequency DFM-MQ + MIDAS, three
  distributional heads (NGBoost / IDR / DVBF deep state-space), the
  PatchTST neural sequence baseline, sequential e-value safe-testing
  wired into the release gate, CRPS-DM, and GP-BOCPD.
- **Three new warehouse tables**: `e_value_log`, `nowcast_factors`,
  `conditional_coverage_report`.
- **Three new CLI commands**: `mre nowcast`, `mre e-value-test`,
  `mre conformal-conditional`.
- **Three new `daily_flow` summary keys**: `nowcast_factors`,
  `worst_conditional_coverage`, `e_value_promotion_pending`.

Optional dependencies live behind `[frontier]`; everything degrades
gracefully when statsmodels / ngboost / torch are missing.

## Next implementation priorities (post-v1.2)

The open items, ordered by leverage:

1. **DuckDB-primary `Warehouse`.** The export path already produces a
   queryable analytical store; making DuckDB the primary backend removes
   the SQLite-single-writer concurrency ceiling for live deployments.
   Estimate: 1 week.
2. **Consolidate `report_writer_v{2..5}`.** Five additive shims
   accumulated across versions; merge into a single `report_writer.py`
   under a stable contract before the next institutional-report change.
3. **Daily/intraday vintage support.** With exact release calendars
   already loaded, materialise `feature_asof_values` at hourly
   granularity so day-of-release forecasts use the closest snapshot.
4. **Real BLS / BEA / Census / Fed exact release calendars.** Replace the
   conservative `release_calendar_exact.py` rule table with live calendar
   ingestion + reconciliation against `vintage_observations.realtime_start`.
5. **Wheel-distributed Rust extension.** The kernels are in place; build
   and publish wheels (cp311 + cp312 × manylinux + win64 + macOS arm64)
   so callers don't need the Rust toolchain.
6. **Wheel-distributed `[frontier]` extras.** Same idea — ship a docker
   image and a binary distribution that bundles statsmodels / ngboost /
   torch so the frontier features light up by default.
7. **Live alert sinks.** `alerts.route_alerts` writes structured rows;
   add real Slack / PagerDuty / email transports gated behind env-var
   credentials.
8. **`mre verify-data`.** Companion to `verify-run` that checks the
   warehouse has not silently mutated since a model run was recorded
   (e.g. someone re-ingested ALFRED with `--max-vintages-per-series` set
   differently).
9. **Sparse Bayesian regime-switching factor model.** A NumPyro / Stan
   variant of `msvar` for the small-N case. The current implementation
   is the dense MLE.

## Operating discipline notes

- **Always run `audit-vintage --enforce` before training.** The flag
  fails closed if any feature row violates the PIT invariant.
- **Always run `verify-run` before publishing the institutional report.**
  Drifted envelope ⇒ block the release.
- **Never set `--legacy-features` in production.** The flag exists to
  unblock back-compat work; it routes training around the PIT pipeline.
- **`--max-stale-months` on `label-recessions` is a release gate.**
  Staleness is silent unless you ask it to fail.
- **`promotion_method="e_values"` is anytime-valid.** Operators can stop
  sampling whenever the e-value crosses `1/α`; the gate fires on the
  cumulative evidence, not on a fixed window.
- **`MondrianBinaryConformal(exchangeable=False)`** auto-bumps to the
  block backend. For long-memory series prefer `nexcp` or
  `e_conformal` explicitly.
- **Rust kernels need parity tests, not just speed.**
  `tests/test_rust_parity.py` asserts `atol=1e-9` against the Python
  reference. Promotion criterion: parity passes *and* `mre bench` shows
  a real speedup.

## Things that look broken but aren't

- **`verify-run` returning `code_version: "unknown"`** when run outside
  a git checkout. The envelope still verifies; only the SHA is missing.
  This is the right behaviour in shipped tarballs.
- **`PatchTSTHead` raising `ImportError`** when torch isn't installed.
  This is the documented soft-degrade path; the rest of the engine is
  unaffected.
- **`MQDynamicFactorModel.backend == "fallback"`** when statsmodels isn't
  installed. The class falls back to the v1.0 `DFMDomainModel` and the
  `nowcast_factors` table records the fallback explicitly.
- **Parity test `test_bocpd_diag_update_parity`** skips if
  `mre_rust_ext` is not built. That's the gate, not a failure.
