# Market Regime Engine v1.6.1

Python-first, Rust-ready probabilistic macro/market regime intelligence
engine with a 2026-2027-frontier modeling layer **and the v1.5
Fixed-Income RCIE / X-Pro Auto-X adapter**.

> **v1.5.0 — Fixed-Income RCIE layer:** the deterministic credit-spread
> regime scorer, per-scope liquidity-stress index, fail-closed
> execution-confidence service, regime-aware TCA segmentation, and
> tamper-evident HMAC-signed evidence pack are the headline additions.
> 13 new FI tables, 6 FI API endpoints, and 7 `mre fi-*` CLI commands
> sit alongside the existing macro/regime engine.
>
> Note: "regime" in this product refers to a **market** regime (a
> probabilistic state of a market). It is distinct from the
> *regulatory* regime (FCA / ESMA / MIFID) usage in adjacent
> MarketAxess documentation. The PR-9 audit doc disambiguates the
> two terms; the credit-spread regime label, the liquidity-stress
> label, and the execution-confidence bucket are all
> market-state classifications.
>
> **v1.5.1 (PR-9 audit hardening):** a patch release on top of
> v1.5.0 that closes the eight priority audit gaps flagged by a
> third-party review of the v1.5.0 stack. The full PR-9 fix list
> is in the v1.5.0 commit table below and in
> [`docs/V1_5_FIXED_INCOME_RCIE.md`](docs/V1_5_FIXED_INCOME_RCIE.md)
> §"Fail-Closed Contract" (PR-9 FIX 8), plus
> [`docs/V1_5_HMAC_OPERATIONS.md`](docs/V1_5_HMAC_OPERATIONS.md)
> §8 (PR-9 FIX 3).
>
> **v1.5.2 (validation-primitive bug fixes):** a patch release on top
> of v1.5.1 that closes three Bailey-López de Prado (2014, 2017)
> conformance bugs in the validation primitives flagged by an
> out-of-band code review of `validation.py`:
>
> - **A1** — `deflated_sharpe` and `minimum_track_record_length` were
>   evaluating BLP eq. 5's Pearson-kurtosis form `(γ_4 − 1)/4` on the
>   *excess* kurtosis returned by `_sample_skew_kurt`. Off by 3/4 in
>   the kurt term. The corrected excess-form coefficient is
>   `(γ_4_excess + 2)/4`; for Gaussian iid returns the var_term now
>   correctly collapses to `1 + 0.5·SR²`.
> - **A2** — `probability_of_backtest_overfitting`'s `_purge_and_embargo`
>   was applying max-style overlap so embargo was silently subsumed
>   when `embargo <= purge`. The fix makes embargo additive on the
>   right side of every OOS block (`purge + embargo` total drop),
>   matching the union semantics in
>   `walk_forward.purge_and_embargo_searchsorted`.
> - **A3** — the DSR multiplicity threshold was scaling by
>   `1/sqrt(T−1)` instead of the *moment-corrected* stderr
>   `sqrt(Var(SR_hat))` per BLP eq. 9. For skewed or fat-tailed
>   inputs the v1.5.1 threshold was systematically biased by a factor
>   of `sqrt(var_term)`.
>
> The behavioural surface is otherwise the v1.5.1 release; v1.5.2 only
> tightens the validation primitives' BLP conformance. The PR-9 audit
> tests in `tests/test_validation_dsr_mtrl_audit.py` were updated to
> encode the BLP-correct expectations and four new property-style
> anchors were added (one per bug plus a Gaussian-iid sanity).
>
> **v1.6.1 (release-identity sync):** behaviorally identical to v1.6.0
> (PR #23, tag `v1.6.0`); a patch release that fixes a source-vs-tag
> identity mismatch where `pip install` and runtime `__version__`
> reported `1.5.2` despite the v1.6.0 git tag and GitHub Release.
> No code changes; only the version string in `pyproject.toml`,
> `src/market_regime_engine/__init__.py`, the README banner, and a
> hardening to `scripts/check_readme_version.py` to also verify the
> `pyproject.toml` version (which the v1.6.0 release would have caught
> if Actions weren't billing-blocked).
>
> **v1.6.0** is the substantive release (frontier hardening + governed
> signal layer + software-engineering quality - 64 commits, ~+300
> tests, mypy 13 -> 0). See PR #23 and the v1.6.0 release notes for the
> full release narrative.
>
> v1.5.0 references:
> [`docs/V1_5_FIXED_INCOME_RCIE.md`](docs/V1_5_FIXED_INCOME_RCIE.md)
> for the full release notes,
> [`docs/V1_5_AUTOX_CONTRACT.md`](docs/V1_5_AUTOX_CONTRACT.md) for
> the Auto-X consumer contract,
> [`docs/V1_5_HMAC_OPERATIONS.md`](docs/V1_5_HMAC_OPERATIONS.md) for
> the HMAC operating playbook, and
> [`docs/V1_5_BREAKING_CHANGES.md`](docs/V1_5_BREAKING_CHANGES.md)
> for the (small) v1.4 → v1.5 behavioural deltas. The seven `mre
> fi-*` commands are: `fi-build-features`, `fi-score-credit-regime`,
> `fi-score-liquidity`, `fi-score-execution-confidence`,
> `fi-tca-segment`, `fi-evidence-pack`, `fi-report` (plus the
> rotation-tooling extra `mre fi-evidence-resign`).
>
> **v1.4.1** is a release-integrity patch on top of v1.4.0. It closes the
> audit-grade hygiene gaps a third-party reviewer found in v1.4: the README
> identity drift (and the matching wheel-METADATA `Description` first-line
> drift), the `verify_run` skip set that quietly let arbitrary `extra` and
> `rng_seeds` drift past the gate, the permissive release-gate defaults that
> regressed to the v1.2.1 baseline when an operator ran `mre release-gate`
> with no flags, and the platform lockfile coverage hole for the
> `[bayesian]` / `[scraping]` / `[frontier]` / `[dashboard]` extras. See
> [`docs/V1_4_1_FIXES.md`](docs/V1_4_1_FIXES.md). The behavioural surface is
> the v1.4.0 release; v1.4.1 only tightens the release-integrity rails.
>
> **v1.4.0** lands four substantial frontier additions and a
> default-warehouse swap:
>
> 1. **Bayesian NumPyro MS-VAR**
>    (`market_regime_engine.frontier.bayesian_msvar`) — NUTS + SVI fits
>    with Dirichlet/LKJ priors, `OrderedTransform` anchor to kill MCMC
>    label-switching, R-hat / ESS diagnostics persisted to a new
>    `bayesian_msvar_diagnostics` warehouse table, and a clean
>    NumPyro-missing soft-degrade path. Plug-in to BMA via the
>    `enable_bayesian` flag in `daily_flow`.
> 2. **Deep-kernel GP-BOCPD**
>    (`market_regime_engine.frontier.deep_kernel`) — learned MLP feature
>    embedding for the GP change-point detector, restoring resolution
>    over the v1.2 RBF length-scale heuristic on correlated feature
>    clusters. Falls back cleanly when torch is missing.
> 3. **DuckDB-primary swap with appender rewrite** — `Warehouse` now
>    defaults `backend="auto"` (suffix-routed), the CLI default `--db`
>    flipped to `data/mre.duckdb`, and the bulk-load path is `register`
>    + `INSERT … SELECT … ON CONFLICT` wrapped in an explicit
>    transaction. 10k row write went from 427s (v1.3 executemany cliff)
>    to 0.064s — DuckDB is now ~95× faster than SQLite executemany on
>    the same panel.
> 4. **Real BLS / BEA / Census / Fed release calendars**
>    (`market_regime_engine.frontier.release_calendars`) — four agency
>    fetchers with a deterministic YAML cache under
>    `config/release_calendars/`, `audit-release-calendar
>    --tolerance-days` reconciliation, and a clean BS4-missing
>    soft-degrade path. Replaces the 9-domain `DEFAULT_LAGS` heuristic.
>
> The v1.4.0 source-of-truth doc is
> [`docs/V1_4_RELEASE.md`](docs/V1_4_RELEASE.md).
>
> v1.3 closed the v1.2 review's structural fixes (audit-zip slimming,
> stable `_hash_frame`, vectorised purge-and-embargo, real DuckDB
> warehouse parity, alert sinks, `verify-data`, production release-gate
> profile, supply-chain hardening with SBOM + license + bandit, real
> ALFRED recorded fixtures, Rust wheel matrix, report-writer
> consolidation, multi-backend API cache).
> [`docs/V1_3_RELEASE.md`](docs/V1_3_RELEASE.md).
>
> v1.2.1 closed the v1.2 production-readiness gaps (real package
> metadata, fail-closed PIT training, hardened legacy API, Apache-2.0
> LICENSE, vectorised as-of materialization, dual release artifacts,
> package-sanity CI). [`docs/V1_2_1_FIXES.md`](docs/V1_2_1_FIXES.md).
>
> v1.2 lands the math-correctness floor and the SOTA frontier modeling
> package: five time-series-native conformal predictors, Bańbura-Modugno
> mixed-frequency DFM-MQ / native D/W/M state-space nowcasting + Almon-polynomial MIDAS, three distributional
> heads (NGBoost / IDR / DVBF deep state-space), CPU-friendly PatchTST,
> sequential e-value safe-testing, CRPS-DM, and Saatçi-Turner-Rasmussen
> GP-BOCPD. [`docs/V1_2_FRONTIER.md`](docs/V1_2_FRONTIER.md).

This build estimates and stores:

- macro regime scores (rule-based + HMM + MS-VAR + **Bayesian MS-VAR
  with credible bands** + GP-BOCPD with **learned deep-kernel
  embedding**)
- BOCPD-style change-point probability (NIW / BOCPD-MUSE / GP / MLP
  deep-kernel GP)
- WFST-constrained decoded regime path
- drawdown probability with conformal coverage guarantee
- recession probability and fitted discrete-time hazard with path-aware
  horizon survival
- forward-return quantiles with non-crossing repair + CQR / NexCP /
  conditional / localized / e-conformal intervals
- distributional forecasts via NGBoost, IDR, DVBF deep state-space, and
  PatchTST heads
- mixed-frequency nowcast factors (Bańbura-Modugno M/Q DFM-MQ plus native D/W/M Kalman state-space backend)
- NBER recession labels (FRED USREC by default with explicit staleness)
- historical analogs (regime-weighted)
- domain / feature driver attribution + counterfactual deltas
- calibrated probabilities (Platt) and online Bayesian-averaged
  ensembles
- model confidence, invalidation triggers, drift, release-gate, alert
  routing, promotion workflow (production MCS by default; sequential e-value safe-testing is fenced behind `MRE_ENABLE_EXPERIMENTAL_FRONTIER=1`), and full point-in-time vintage / as-of feature lineage
- DuckDB / Parquet / CSV analytical warehouse exports (DuckDB is the
  default warehouse backend in v1.4)
- immutable model runs with a full reproducibility envelope (git SHA,
  lockfile hash, feature / output / vintage payload hashes, RNG seeds,
  arbitrary `extra` envelope)
- forecast-comparison statistics: DM, GW, Hansen MCS (T_R *and* T_SQ),
  Christoffersen UC+CC, Knüppel raw-moment + autocorrelation moment,
  Murphy reliability/resolution/uncertainty, CRPS-DM
  (Diks-Panchenko-van Dijk)
- scenario replays (1973 oil → 2022 inflation) and multi-horizon
  coherent conformal sets
- real BLS / BEA / Census / Fed release calendars reconciled against
  observed vintages (`audit-release-calendar --enforce`)

## Build status

<!-- ci-status-start -->
- **Tests: 1438 passed / 0 failed / 0 errored / 31 skipped (junit `tests/`).**
- **Ruff: 0 offences (`ruff check src tests`).**
- **Mypy: 0 errors (`mypy src/market_regime_engine`).**
- **Bench: `mre bench` recorded 15 measurements.**
- Smoke: end-to-end `bootstrap-sample → … → verify-run` reports `approved: true` on the latest green CI run.
- Initial commit `741e51a`; v1.1 commit `79249df`; v1.2 commit `904d058`; v1.2.1 commit on `v1.1-fixes`.
<!-- ci-status-end -->

> The build-status block above is regenerated by
> `scripts/refresh_build_status.py` from the CI artifacts uploaded by
> `.github/workflows/ci.yml`. Run that script locally or let CI commit
> the refreshed numbers on the default branch. Hand-edits inside the
> `<!-- ci-status-start -->` / `<!-- ci-status-end -->` sentinel block
> will be overwritten.

## Pinned dependencies

`requirements-lock.txt` is the canonical pinned dependency manifest for
the **core** install (the v1.2.1+ `[dev,dashboard,analytics,nowcast,
observability,security]` superset). `mre verify-run` hashes its bytes
into every reproducibility envelope.

v1.4.1 ships **four** platform-conditional lockfiles alongside the core
manifest. Each one covers a single optional extra so reproducibility
is no longer partial when an operator opts into the frontier / Bayesian
/ scraping / dashboard surface:

| Lockfile | Extras covered | Generated for |
|---|---|---|
| `requirements-lock.core.txt` (= `requirements-lock.txt`) | core: `dev`, `dashboard`, `analytics`, `nowcast`, `observability`, `security` | Linux, py3.11 |
| `requirements-lock.frontier-cpu-linux.txt` | `frontier` (statsmodels, ngboost, torch CPU) | Linux, py3.11 |
| `requirements-lock.bayesian-cpu-linux.txt` | `bayesian` (numpyro, jax[cpu], arviz) | Linux, py3.11 |
| `requirements-lock.dashboard.txt` | `dashboard` only (streamlit, plotly) | Linux, py3.11 |

`requirements-lock.txt` is preserved as a duplicate of
`requirements-lock.core.txt` so existing tooling (CI lockfile-sanity
job, `verify-run` envelope hash) keeps working without modification.

**Regenerate** any lockfile via `pip-compile` from a clean Linux
py3.11 environment:

```bash
# Core (canonical lockfile that verify-run hashes).
pip-compile pyproject.toml \
    --extra dev --extra dashboard --extra analytics \
    --extra nowcast --extra observability --extra security \
    --output-file requirements-lock.core.txt
cp requirements-lock.core.txt requirements-lock.txt

# Frontier extras (CPU torch).
pip-compile pyproject.toml \
    --extra frontier \
    --output-file requirements-lock.frontier-cpu-linux.txt

# Bayesian extras (CPU JAX).
pip-compile pyproject.toml \
    --extra bayesian \
    --output-file requirements-lock.bayesian-cpu-linux.txt

# Dashboard only (Streamlit + Plotly).
pip-compile pyproject.toml \
    --extra dashboard \
    --output-file requirements-lock.dashboard.txt
```

The CI `lockfile-sanity` job blocks any line containing `-e `, `c:\`,
`/Users/`, `/home/`, or `file://` in any of the five lockfiles. The
new `lockfile-platform-sanity` job (v1.4.1) additionally pins each
extra-lockfile's content hash so CI fails if the lockfile is shipped
without regenerating it from a drifted `pyproject.toml`. Editable
installs leak the developer's directory tree into the lockfile, which
is the v1.2 bug we are still keeping closed.

The `[frontier]` / `[bayesian]` / `[scraping]` / `[redis]` extras stay
**unlocked** in the canonical `requirements-lock.txt` because their
wheels are platform-conditional (CUDA / CPU / Apple-silicon for torch,
JAX, etc.). Install on demand:

```bash
pip install -e ".[frontier]"   # statsmodels + ngboost + torch
pip install -e ".[bayesian]"   # numpyro + jax[cpu] + arviz
pip install -e ".[scraping]"   # beautifulsoup4 + lxml (release-calendar fetchers)
pip install -e ".[redis]"      # redis-py for the shared /v1 API cache
```

The frontier modules degrade to NumPy fallbacks when their extras are
missing.

## API surfaces and the legacy-gate breaking change

The engine ships two FastAPI mounts:

```bash
# v1 hardened — versioned routes, optional X-API-Key auth, TTL cache, metrics.
MRE_API_KEY="rotate-me" uvicorn market_regime_engine.api_v1:app --reload

# Legacy /api — UNAUTHENTICATED, gated behind an explicit env-var ack.
MRE_LEGACY_API_ALLOW_UNAUTH=1 uvicorn market_regime_engine.api:app --reload
```

> **Breaking change in v1.2.1:** `market_regime_engine.api:app` (the
> legacy mount) refuses to import unless
> `MRE_LEGACY_API_ALLOW_UNAUTH=1`. The legacy app exposes the same
> governance / model-output surface as `/v1` with no API-key check;
> production deployments should mount `market_regime_engine.api_v1:app`
> instead. The env-var gate exists so a misconfigured deployment fails
> fast at uvicorn import time rather than silently serving governance
> artifacts to the public internet.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e ".[dev,dashboard,analytics]"

# Optional: build Rust hot-path kernels.
# v1.3 NOTE: ``pip install market-regime-engine[frontier]`` does NOT
# install the Rust extension; the Rust wheels are platform-specific and
# distributed separately. Two ways to install them:
#
#   (a) Build locally via maturin (requires a Rust toolchain):
pip install maturin
(cd rust_ext && maturin develop --release)
#
#   (b) Or install one of the published wheels for your platform from
#       the v1.3 ``rust-wheels`` CI artifacts (cp311+cp312 ×
#       manylinux/win64/macos-arm64). The wheel name pattern is
#       ``mre_rust_ext-<version>-cp31x-...whl``:
#   pip install ./mre_rust_ext-1.3.0-cp311-cp311-manylinux_2_28_x86_64.whl

# Optional: install the v1.2 frontier modeling extras (statsmodels, ngboost, torch).
pip install -e ".[frontier]"

# Optional v1.3 / v1.4 extras:
#   [security] — cyclonedx-bom + pip-licenses + bandit (CI hardening).
#   [redis]    — redis-py for the shared /v1 API cache.
#   [bayesian] — numpyro + jax[cpu] + arviz (Bayesian MS-VAR, v1.4).
#   [scraping] — beautifulsoup4 + lxml (release-calendar fetchers, v1.4).
pip install -e ".[security]"
pip install -e ".[bayesian]"
pip install -e ".[scraping]"

# 1. Bootstrap a deterministic warehouse on synthetic sample data.
#    v1.4 default backend is DuckDB; pass --db data/mre.db to keep SQLite.
mre bootstrap-sample --db data/mre.duckdb
mre audit-release-calendar --db data/mre.duckdb --enforce
mre build-exact-release-calendar --db data/mre.duckdb --enforce
mre pit-check --db data/mre.duckdb

# v1.4: refresh real release calendars from BLS/BEA/Census/Fed.
mre refresh-release-calendars

# 2. Materialize the point-in-time feature snapshots and audit them.
mre seed-vintage-from-observations --db data/mre.duckdb   # local smoke-test only
mre materialize-asof-features --db data/mre.duckdb --write-features
mre audit-vintage --db data/mre.duckdb --enforce

# 3. Build features, label recessions, score regimes.
mre build-features --db data/mre.duckdb
mre label-recessions --db data/mre.duckdb --max-stale-months 12
mre score-regime --db data/mre.duckdb

# 4. Train, validate, calibrate, ensemble.
mre train-baseline --db data/mre.duckdb                   # PIT mode by default
mre train-survival --db data/mre.duckdb
mre train-fitted-hazard --db data/mre.duckdb --oos
mre validate --db data/mre.duckdb --out data/validation --min-train 120 --step 6
mre calibrate-probabilities --db data/mre.duckdb --validation-dir data/validation
mre optimize-stacking --db data/mre.duckdb --out data/stacking
mre optimize-regime-stacking --db data/mre.duckdb --validation-dir data/validation

# 5. v1.2 / v1.4 frontier steps (optional — degrade gracefully without the extras).
mre nowcast --db data/mre.duckdb
mre conformal-conditional --db data/mre.duckdb --validation-dir data/validation
mre e-value-test --db data/mre.duckdb --challenger candidate_logistic
mre bayesian-msvar-fit --db data/mre.duckdb              # v1.4: NumPyro Bayesian MS-VAR
mre deep-kernel-train --db data/mre.duckdb               # v1.4: train MLP deep-kernel for GP-BOCPD

# 6. Analogs, attribution, governance.
mre analogs --db data/mre.duckdb --regime-weighted --out data/analogs.csv
mre attribute --db data/mre.duckdb --out data/attribution
mre invalidation-triggers --db data/mre.duckdb
mre monitor-drift --db data/mre.duckdb
mre score-confidence --db data/mre.duckdb --validation-dir data/validation
# v1.4.1: ``mre release-gate`` defaults to the production profile when
# called with no flags AND no MRE_ENV env var. Use ``--profile default``
# (or ``MRE_ENV=dev``) to opt back into the v1.2.1 looser baseline.
mre release-gate --db data/mre.duckdb --validation-dir data/validation
mre route-alerts --db data/mre.duckdb --validation-dir data/validation
mre promotion-workflow --db data/mre.duckdb --validation-dir data/validation

# 7. Record the run, verify reproducibility, report, export warehouse.
mre model-run --db data/mre.duckdb --purpose "v1.4.1 governed runtime run"
# v1.4.1: verify-run now compares the full ``extra`` envelope and
# ``rng_seeds`` (use ``--ignore-rng-seeds`` for stochastic-seed reruns).
mre verify-run --db data/mre.duckdb
mre institutional-report --db data/mre.duckdb --out data/reports/institutional_report.md
mre export-warehouse --db data/mre.duckdb --out data/lake --duckdb data/mre.duckdb
mre report --db data/mre.duckdb

# Optional: bench the BOCPD / WFST / PSI hot paths.
mre bench --out data/bench.csv
```

Global flags applied to every subcommand:

```text
--json-logs              Emit one JSON object per line (override via MRE_LOG_FORMAT).
--log-level LEVEL        DEBUG | INFO | WARNING | ERROR (also MRE_LOG_LEVEL).
```

## API and dashboard

Two API surfaces ship:

```bash
# Legacy read-only (v0.8 routes). Unauthenticated by design; v1.2.1 gates
# the legacy mount behind an explicit env-var so a misconfigured deploy
# fails fast at uvicorn import time. Prefer api_v1 in production.
MRE_LEGACY_API_ALLOW_UNAUTH=1 uvicorn market_regime_engine.api:app --reload

# v1 hardened: versioned routes, optional API-key auth, TTL cache, metrics.
MRE_API_KEY="rotate-me" uvicorn market_regime_engine.api_v1:app --reload
```

Streamlit dashboard (cached reads, Plotly regime ribbon, release-gate
banner):

```bash
streamlit run src/market_regime_engine/dashboard.py
```

v1 routes (require `X-API-Key` when `MRE_API_KEY` is set):

```text
/health                              public
/v1/health                           release-gate-aware
/v1/metrics                          Prometheus exposition (or in-process snapshot, auth-gated)
/v1/regime/latest                    auth-gated
/v1/model-outputs/latest             auth-gated
/v1/calibrated-outputs/latest        auth-gated
/v1/release-gate/latest              auth-gated
/v1/analogs/latest                   auth-gated
```

Legacy v0.8 routes (still served by the read-only `api.py` mount):

```text
/regime/latest    /model-outputs/latest    /calibrated-outputs/latest
/analogs/latest   /attribution/latest      /confidence/latest
/invalidation/latest    /model-runs/latest    /drift/latest
/release-gate/latest    /ensemble-weights/latest    /alerts/latest
/promotion-workflow/latest    /hazard/latest    /vintage-audit/latest
/feature-asof/latest    /vintage-observations/coverage
```

## ALFRED/FRED vintage ingestion

Plan the request footprint first:

```bash
mre alfred-plan \
  --series UNRATE CPIAUCSL FEDFUNDS \
  --vintage-start 2000-01-01 \
  --vintage-end 2001-01-01
```

Run live ingestion only after setting an API key:

```bash
export FRED_API_KEY="..."
mre ingest-alfred \
  --db data/mre.duckdb \
  --series UNRATE CPIAUCSL FEDFUNDS \
  --observation-start 1960-01-01 \
  --vintage-start 2000-01-01 \
  --vintage-frequency QS
```

Monthly vintages across many series can be expensive and slow. Start
quarterly, because hammering an API and calling it research is how
monitoring dashboards become evidence.

## Real point-in-time vintage workflow

For production, use true ALFRED/FRED vintage dates rather than synthetic
vintage grids:

```bash
mre alfred-real-plan \
  --series UNRATE CPIAUCSL PAYEMS \
  --vintage-start 2000-01-01 \
  --vintage-end 2001-01-01 \
  --max-vintages-per-series 10

export FRED_API_KEY="..."
mre ingest-alfred-real \
  --db data/mre.duckdb \
  --series UNRATE CPIAUCSL PAYEMS \
  --observation-start 1960-01-01 \
  --vintage-start 2000-01-01 \
  --max-vintages-per-series 20

mre materialize-asof-features --db data/mre.duckdb --write-features
mre audit-vintage --db data/mre.duckdb --enforce
```

The hard point-in-time invariant is:

```text
observation_date <= as_of_date
vintage_date     <= as_of_date
```

If a feature cannot prove that lineage, it should not enter training,
validation, backtesting, or inference. `audit-vintage --enforce` fails
closed when the invariant is violated, and `train-baseline` /
`validate` route through `feature_asof_values` by default. Pass
`--legacy-features` to opt back into the pre-v1.0 (non-PIT) training
input. This is annoying. So is look-ahead bias pretending to be alpha.

## Core math

The engine estimates a calibrated forecast distribution:

```text
F_hat_{t,h}(y) = C_{h,R}[ sum_m w_{m,t,h}(CP_t, gamma_t, R_t, Loss_m, CalErr_m) F_{m,t,h}(y) ]
```

where:

- `CP_t` is online change-point probability (NIW BOCPD, BOCPD-MUSE,
  GP-BOCPD, or **MLP deep-kernel GP-BOCPD**, optionally with a
  covariate-conditioned hazard)
- `gamma_t` is the latent regime posterior (HMM Baum-Welch, MS-VAR, *or
  **Bayesian NumPyro MS-VAR with credible bands***)
- `R_t` is the WFST-decoded regime state
- `w_{m,...}` are online BMA weights (exponentially-discounted log-score)
- `Loss_m` is rolling or validation loss
- `CalErr_m` is calibration error
- `C_{h,R}[·]` is a regime-conditional conformal layer with backend
  ∈ `{split, block, nexcp, conditional, localized, e_conformal}`,
  optionally extended with `BonferroniMultiHorizonConformal` for joint
  multi-horizon coverage

Promotion can be gated by Hansen MCS (T_R or T_SQ) *or* by Howard-Ramdas
sequential e-value safe-testing (`promotion_method="e_values"`).

## Modules at a glance

| Layer | Module | What it does |
|---|---|---|
| Ingestion | `alfred.py`, `alfred_real.py`, `fred_recession.py` | Synthetic vintage grid, real ALFRED vintage-date ingestion, FRED USREC |
| Storage | `storage.py` (DuckDB-primary in v1.4 with SQLite back-compat, 35 tables incl. `bayesian_msvar_diagnostics` + `release_calendar_refreshes`), `analytics_warehouse.py` (DuckDB/Parquet export) | Warehouse |
| Point-in-time | `asof.py`, `point_in_time.py`, `release_calendar.py`, `release_calendar_exact.py`, `training_data.py` | PIT lineage, audits, training-data router |
| Features | `features.py`, `robust_stats.py`, `dfm.py` | Transforms, MAD-based robust z, Watson-Engle DFM |
| Regime | `regimes.py`, `changepoint.py`, `bocpd.py`, `bocpd_muse.py`, `bocpd_hazard.py`, `hmm.py`, `msvar.py`, `wfst.py` | NIW BOCPD, BOCPD-MUSE, covariate hazard, HMM, MS-VAR, WFST decoder |
| Heads | `models.py`, `survival.py`, `hazard_model.py`, `cross_sectional.py` | Logistic + non-crossing quantile, transparent + fitted hazards, FF/sector/curve |
| Validation | `validation.py`, `walk_forward.py`, `forecast_compare.py`, `backtest.py`, `baselines.py`, `promotion.py` | Brier/log-loss/ECE/pinball, purged WF + CPCV, DM/GW/MCS(T_R+T_SQ)/PIT/Christoffersen/Murphy/CRPS-DM |
| Calibration | `calibration.py`, `conformal.py`, `multi_horizon_conformal.py` | Platt, Mondrian split conformal (with backend dispatch), CQR, ACI, Bonferroni multi-horizon |
| Ensembling | `ensemble.py`, `ensemble_v2.py`, `stacking.py`, `stacking_v2.py`, `bma.py` | Dynamic / regime-conditioned / online BMA |
| Governance | `confidence.py`, `invalidation.py`, `drift.py`, `release_gates.py`, `alerts.py`, `promotion_workflow.py`, `model_runs.py`, `model_registry.py` | Confidence grade, drift PSI, release gate (MCS *or* e-value variant; **production-profile default in v1.4.1**), alerts, promotion, immutable runs with reproducibility envelope (full `extra` + `rng_seeds` verified in v1.4.1) |
| Explain | `analogs.py`, `analogs_v2.py`, `attribution.py`, `counterfactual.py`, `explain.py` | Regime-weighted analogs, z-score + counterfactual + permutation + optional SHAP |
| Reporting | `report_writer.py` … `report_writer_v5.py` | Institutional report (v0.4 + v0.5 + v0.6 + v0.7 + v0.8 sections) |
| Operations | `cli.py`, `api.py`, `api_v1.py`, `dashboard.py`, `orchestration.py`, `logging_setup.py`, `observability.py`, `bench.py` | CLI (44 subcommands incl. v1.4 `bayesian-msvar-fit`, `deep-kernel-train`, `refresh-release-calendars`), legacy + v1 APIs, Streamlit, daily flow, structured logs, metrics, bench harness |
| Scenarios | `scenarios.py` | 1973 / Volcker / S&L / dotcom / GFC / COVID / 2022 inflation replay |
| **Frontier (v1.2 + v1.4)** | `frontier/conformal_ts.py`, `frontier/dfm_mq.py`, `frontier/midas.py`, `frontier/distributional.py`, `frontier/neural_seq.py`, `frontier/sequential_testing.py`, `frontier/gp_cpd.py`, **`frontier/bayesian_msvar.py`** (v1.4), **`frontier/deep_kernel.py`** (v1.4), **`frontier/release_calendars.py`** (v1.4) | 5 time-series-native conformal layers, MQ-DFM + MIDAS nowcasting, NGBoost / IDR / DVBF / PatchTST heads, e-value safe-testing, GP-BOCPD, **Bayesian NumPyro MS-VAR**, **MLP deep-kernel for GP-BOCPD**, **real BLS/BEA/Census/Fed release-calendar fetchers** |
| Rust hot paths | `rust_ext/src/lib.rs`, `rust_kernels.py` | NIW BOCPD update, WFST Viterbi, PSI, rolling Mahalanobis (parity-tested) |

## Validation standard

Models are not promoted because they look clever in a chart. The
validation workflow writes candidate / benchmark predictions, Brier /
log-loss / ECE metrics, quantile (pinball) loss, model promotion
results, release gates, alerts, and promotion workflow rows. v1.2 layers
on top:

- **Purged + embargoed walk-forward** (`walk_forward.PurgedWalkForward`,
  `CombinatorialPurgedCV`) — the default `validate` step uses these so
  overlapping forward-target windows can not leak.
- **Forecast-comparison statistics** (`forecast_compare`):
  - `diebold_mariano` with HLN small-sample correction (5%-direction).
  - `giacomini_white` conditional predictive ability.
  - `hansen_mcs(statistic="T_R" | "T_SQ")` with stationary block bootstrap.
  - `pit_uniformity(autocorrelation=True | False)` Diebold-Gunther-Tay +
    Knüppel raw-moment + autocorrelation moments.
  - `christoffersen_coverage` unconditional + conditional coverage.
  - `murphy_decomposition` reliability / resolution / uncertainty.
  - `crps_diks_panchenko` distributional CRPS-DM with HAC test.
- **Conformal coverage guarantees** (`conformal`,
  `multi_horizon_conformal`, `frontier.conformal_ts`):
  - `MondrianBinaryConformal(backend="split" | "block" | "nexcp" |
    "conditional" | "localized" | "e_conformal")` per-regime split
    conformal with time-series-native backends.
  - `ConformalizedQuantileRegression` (Romano-Patterson-Candès 2019).
  - `AdaptiveConformalInference` (Gibbs-Candès 2021).
  - `BonferroniMultiHorizonConformal` for joint multi-horizon coverage.
- **Sequential safe-testing** (`frontier.sequential_testing`):
  - `EValueLogScore` Howard-Ramdas confidence-sequence e-value.
  - `SafeTestPromotion` Grünwald-de Heide-Koolen anytime-valid promotion
    gate; enable via `release_gates.evaluate_release_gate(promotion_method="e_values")`.

## Rust boundary

Python remains the research and orchestration layer. Rust hosts four
parity-tested hot-path kernels:

- multivariate diagonal-Student-t BOCPD update
- WFST log-space Viterbi decoder
- population-stability-index (PSI)
- ridge-stabilised rolling Mahalanobis distance

Build with `maturin develop --release` from `rust_ext/`. The Python
module auto-detects the compiled extension and falls back to the
reference implementation when missing. Promotion criterion: parity
tests in `tests/test_rust_parity.py` pass at `atol=1e-9`, **and** `mre
bench` shows a measurable speedup at every problem size. Fast wrong is
still wrong, just with better posture.

## Reproducibility envelope

Every immutable model run captures:

```text
code_version (short SHA)   code_sha (long SHA)   code_dirty
lockfile_hash (sha256 of requirements-lock.txt)
platform                   python_version
feature_payload (sha256)   output_payload (sha256)   vintage_payload (sha256)
rng_seeds                  extra (training_audit + arbitrary fields)
artifact_hash (sha256 of the whole envelope)
```

`mre verify-run --db data/mre.duckdb [--run-id <id>]` re-derives the
envelope from the current environment and exits non-zero if any field
drifted. Hook this into change-management gates.

> **v1.4.1 verify-run hardening:** `verify_run` now compares the full
> `extra` envelope structurally (not just the `training_audit`
> sub-key), and `rng_seeds` is no longer in the unconditional skip set.
> Both sides are canonicalised via JSON sort-keys round-trip so dict
> insertion order is not a false-drift signal. Two opt-outs ship for
> the legitimate stochastic-rerun and arbitrary-extra workflows: pass
> `--ignore-rng-seeds` to `mre verify-run` to keep the v1.2.1 skip
> behaviour, and move arbitrary per-run metadata into a sibling
> `metadata` dict instead of `extra` if it legitimately drifts
> run-to-run.

## v1.4.1 breaking-change advisory

1. **`mre release-gate` default profile flipped from permissive to
   `production`.** The v1.4.0-and-earlier behaviour with no flags was
   the v1.2.1 looser baseline (`min_confidence=0.55`,
   `require_mcs_membership=False`, `min_coverage=None`) — exactly the
   thresholds a hands-off operator running `mre release-gate` got. v1.4.1
   resolves the default by:
   1. Explicit `--profile <value>` argument wins.
   2. Else `MRE_ENV` env var: `MRE_ENV=production` → production
      profile, `MRE_ENV=dev` → default profile.
   3. Else fall back to `production`.

   Use `--profile default` or `MRE_ENV=dev` to opt back into the
   permissive thresholds (e.g. for legitimate debugging / staging
   environments). Explicit `--min-confidence` / kwargs always win over
   profile-resolved defaults so an operator can relax a single rail in
   production without tearing down the others.

2. **`verify_run` now compares the full `extra` envelope and
   `rng_seeds`.** Operators with stochastic-seed reruns must pass
   `--ignore-rng-seeds`. Operators who legitimately stash arbitrary
   per-run metadata in `extra` should move it into a sibling
   `metadata` dict instead of `extra` so it does not trigger
   verify-drift.

3. **Default warehouse backend** (carried over from v1.4.0): SQLite →
   DuckDB. Existing `data/mre.db` continues to work via the `*.db` →
   SQLite auto-route. `pip install -e ".[bayesian,scraping]"` if you
   want the v1.4 NumPyro / release-calendar surfaces.

## Current limitations

- **Synthetic sample data is included** so the app runs immediately.
  Live official ingestion still requires `FRED_API_KEY` and operating
  discipline.
- **The Rust extension is a build-on-demand artifact.** CI ships a
  matrix wheel under the v1.3 `rust-wheels` artifact set; parity tests
  skip cleanly when the extension is missing.
- **The `[frontier]` / `[bayesian]` / `[scraping]` extras are
  optional** because their wheels are platform-conditional. Without
  them the relevant frontier modules raise a clean `ImportError` with
  the install hint, or degrade to NumPy fallbacks where possible.
- **Five `report_writer_v{2..5}` modules still exist** as additive
  shims emitting `DeprecationWarning`. Consolidation is queued behind
  a real refactor of the institutional report.
- **Release-calendar metadata** is now real for BLS/BEA/Census/Fed
  (v1.4) reconciled against observed vintages; the YAML cache under
  `config/release_calendars/` is hand-curated for the 16 catalog
  series.
- **Historical analogs explain similarity, not causality.**

## Operating principle

The engine should never make naked point forecasts without uncertainty
bands. It should output:

```text
Current regime
Recession / drawdown probabilities (with conformal coverage guarantee)
Forward return quantiles (non-crossing, optionally CQR-conformalised)
Top drivers (z-score + counterfactual deltas)
Historical analogs (regime-weighted)
Model confidence, drift, invalidation triggers
Release gate / alert routing / promotion decision (MCS or sequential e-value;
  production-profile default in v1.4.1)
Mixed-frequency nowcast factors
Reproducibility envelope (git SHA + lockfile hash + payload hashes +
  full extra + rng_seeds; verified in v1.4.1)
```

Not:

```text
The S&P will be exactly X.
```

That is spreadsheet astrology wearing a blazer.

### Architecture and mathematics map

The implementation is now split around explicit review boundaries. See [`docs/ARCHITECTURE_REFACTOR_BOUNDARY.md`](docs/ARCHITECTURE_REFACTOR_BOUNDARY.md) for the module decomposition and stable-core/frontier split. See [`docs/MATH_METHODS.md`](docs/MATH_METHODS.md) for the mathematical method inventory, assumptions, production status, and retrospective-only fences.


## v1.7 certification hardening

The repository now includes an explicit certification layer for the stable core:

- `mre release-gate --profile certification` / `evaluate_release_gate(profile="certification")` fail closed when validation artifacts are missing.
- Fixed-income execution confidence has a realized-outcome validation path that reports Brier/log-loss/ECE, regime calibration, confidence-decile lift, and positive-direction TCA lift by regime.
- Experimental frontier diagnostics remain behind the `experimental_frontier` boundary and `MRE_ENABLE_EXPERIMENTAL_FRONTIER=1`.
- Method cards under `docs/method_cards/` document equations, assumptions, diagnostics, release-gate requirements, tests, and limitations for each major method.

For CI-grade local validation of the certification additions:

```bash
pytest -q tests/test_certification_release_and_execution_validation.py   tests/test_certification_frontier_diagnostics.py   tests/test_certification_import_boundary.py   tests/test_method_cards_docs_audit.py
```
