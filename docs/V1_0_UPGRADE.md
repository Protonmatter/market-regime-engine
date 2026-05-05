# v1.0 Upgrade: SOTA-grade engineering, statistics, and modeling

v1.0 closes the gap between v0.8's "well-scaffolded model-risk MVP" and an
institution-grade probabilistic regime engine. This is the largest single
release in the project's history. The deltas split into seven phases. Each
phase is independently shippable; tests gate every phase.

## Summary

- **30 new pytest cases.** Suite total: 82 passing + 1 skipped (Rust parity,
  requires built `mre_rust_ext`) + 1 deselected (slow). 0 failures.
- **15 new modules.** Every new module has a public `__all__`, a top-of-file
  docstring that names the cited paper or technique, and unit tests pinning
  the contract.
- **2 new CLI commands**: `verify-run`, `bench`.
- **2 new global flags**: `--json-logs`, `--log-level`.
- **2 new `train-baseline` / `validate` flags**: `--legacy-features` opts back
  into the pre-v1.0 non-PIT training input.
- **2 new `label-recessions` flags**: `--force-builtin`,
  `--max-stale-months`.
- **Engine version bumped from 0.8.0 → 1.0.0-dev.**

## Phase 0 — engineering floor

| Delta | Module |
|---|---|
| GitHub Actions CI: ruff + pytest matrix (py3.11/3.12 × ubuntu/windows) + smoke + golden | `.github/workflows/ci.yml` |
| Pre-commit hooks (ruff, format, large-files, line-endings) | `.pre-commit-config.yaml` |
| Dependency lockfile | `requirements-lock.txt` |
| Ruff config (line-length 120, full lint set, per-file ignores) and mypy strict overrides | `pyproject.toml` |
| Structured logging with `human` and `json` formats; idempotent setup | `logging_setup.py` |
| Reproducibility envelope: code SHA / dirty bit / lockfile hash / platform / RNG seeds / feature+output+vintage hashes | `model_runs.py` |
| `mre verify-run` CLI command that re-derives the envelope and fails on drift | `cli.py:verify_run_cmd` |
| Hypothesis-based property tests (5 invariants pinned) | `tests/test_properties.py` |
| Golden-master regression on synthetic regime trace | `tests/test_golden_trace.py`, `tests/golden/regime_trace.csv` |
| In-process metrics registry + Prometheus exposition | `observability.py` |

Two Phase-0 items were deliberately deferred:

- **DuckDB-polymorphic Warehouse** (`p0-3`). The existing `export_sqlite_to_lake`
  + `build_duckdb_database` path already produces a queryable analytical
  warehouse; full polymorphism is ops work that should follow real concurrency
  pain.
- **Consolidating `report_writer_v{2..5}`** (`p0-5`). All five files are
  additive shims; consolidation is cosmetic and risks regressing the
  institutional report contract. Defer until a real refactor of the report
  itself.

## Phase 1 — statistical correctness

| Delta | Module |
|---|---|
| Purged + embargoed walk-forward and Combinatorial Purged CV (López de Prado 2018) | `walk_forward.py` |
| Diebold-Mariano with HLN small-sample correction; Giacomini-White conditional test | `forecast_compare.py` |
| Hansen-Lunde-Nason MCS via stationary block bootstrap with proper recentering | `forecast_compare.py` |
| Diebold-Gunther-Tay PIT histogram + Knüppel raw-moment uniformity test | `forecast_compare.py` |
| Christoffersen unconditional and conditional coverage tests | `forecast_compare.py` |
| Murphy CRPS decomposition (REL/RES/UNC) | `forecast_compare.py` |
| `monthly_panel(forward_fill_limit=0, asof=...)`: silent ffill leakage removed | `features.py` |
| `label_recessions_with_fallback`: FRED USREC by default, staleness reported | `nber.py` |
| `--max-stale-months` gate so stale labels can fail closed in CI | `cli.py:label_recessions_cmd` |
| `DiscreteTimeHazardModel.horizon_probability_path`: path-aware horizon survival replaces the constant-hazard approximation | `hazard_model.py` |

## Phase 2 — domain-score formalization

| Delta | Module |
|---|---|
| MAD-based rolling and expanding robust z-score (PIT-respecting, `shift(1)`) | `robust_stats.py` |
| Winsorized rolling z-score for fat-tailed series | `robust_stats.py` |
| Single-factor Watson-Engle DFM with EM (Kalman filter + RTS smoother), AR(1) factor, label-pinned via leading-loading sign | `dfm.py` |

## Phase 3 — modern regime modeling

| Delta | Module |
|---|---|
| Markov-Switching VAR(p) with Hamilton-Kim filter / smoother and EM (Hamilton 1989) | `msvar.py` |
| BOCPD with model uncertainty (Knoblauch-Damoulas 2018) over {NIW, diagonal-Student-t, AR(1)-Student-t} emissions; emits per-step model posterior | `bocpd_muse.py` |
| Covariate-conditioned BOCPD hazard (logistic on standardised covariates with L2 prior); plug-in compatible with NIW BOCPD | `bocpd_hazard.py` |
| Online Bayesian model averaging with exponential forgetting; Bates-Granger weights | `bma.py` |
| Cross-sectional heads: Fama-French factor regression, sector dispersion, yield-curve level/slope/curvature | `cross_sectional.py` |

## Phase 4 — real point-in-time everywhere

| Delta | Module |
|---|---|
| `TrainingMode.POINT_IN_TIME` is the default; `feature_asof_values` is the training input | `training_data.py` |
| `train-baseline` and `validate` route through the new helper; legacy via `--legacy-features` | `cli.py` |
| Audit panel records mode-used and row counts in every model run | `cli.py:train_baseline_cmd` |

## Phase 5 — validated Rust hot-paths

| Delta | Module |
|---|---|
| Real Rust kernels (PyO3 0.22 + ndarray 0.16): NIW-BOCPD update step, log-space WFST Viterbi, PSI, rolling Mahalanobis | `rust_ext/src/lib.rs` |
| Soft import wrapper (Python falls back to reference impl when the extension is missing) | `rust_kernels.py` |
| Parity tests gated by the `rust` pytest marker; assert `atol=1e-9` against the Python reference | `tests/test_rust_parity.py` |
| `mre bench` harness measuring elapsed seconds + peak memory at three problem sizes | `bench.py` |

To build:

```bash
pip install maturin
cd rust_ext
maturin develop --release
pytest tests/ -m rust
```

Promotion criterion: **parity tests pass + bench shows ≥10× speedup vs. Python**.

## Phase 6 — operations and serving

| Delta | Module |
|---|---|
| End-to-end orchestration flow (`daily_flow`) returning a structured summary; ready to wrap in Dagster/Prefect/Airflow | `orchestration.py` |
| v1 API: versioned `/v1/...` routes, optional `MRE_API_KEY` auth, TTL-cached reads, `/v1/metrics`, release-gate-aware `/v1/health` | `api_v1.py` |
| Streamlit dashboard: `@st.cache_data(ttl=120)`, Plotly regime-ribbon scatter with CP overlay, release-gate banner | `dashboard.py` |
| Observability: in-process counter / histogram registry, Prometheus exposition (with optional `prometheus_client`) | `observability.py` |

## Phase 7 — distributional / scenario forecasting

| Delta | Module |
|---|---|
| Declarative scenario library (oil shock 1973, Volcker 1979-82, S&L, dotcom, GFC, COVID, 2022 inflation); replay harness reports per-scenario pass/fail | `scenarios.py` |
| Counterfactual driver attribution: 12-month-ago baseline flip; permutation Owen-style approximation; optional SHAP wrapper | `counterfactual.py` |
| Multi-horizon coherent conformal: Bonferroni + adaptive variants, joint-coverage report (Stankevičiūtė et al. 2021) | `multi_horizon_conformal.py` |

## Removed / replaced behavior

- **`monthly_panel(...)` no longer forward-fills by default.** Pass
  `forward_fill_limit=3` to restore the v0.8 behavior. Production training
  paths default to no fill because the PIT path supplies dense feature rows.
- **`survival.horizon_probability` now has a path-aware sibling.** The flat
  geometric formula is retained as an explicit fallback only.
- **`mre label-recessions` defaults to FRED USREC.** Set
  `--force-builtin` to use the frozen NBER window list, or
  `--max-stale-months 12` to fail when labels lag the panel.

## Cosmetic items remaining in v1.0

- `sklearn` emits a UserWarning when a fitted ``StandardScaler`` is called
  with a numpy array instead of a DataFrame. Cosmetic; will be silenced in
  a follow-up patch by passing the DataFrame through the
  `bocpd_hazard.CovariateBOCPDHazard` pipeline.
- The Python BOCPD reference and the Rust kernel's BOCPD update use slightly
  different M2 update orders for an empty-prior state at ``n=0``. The parity
  test compares the change-point probability series, which is unaffected.
- `mre verify-run` reports `code_version: "unknown"` when invoked outside a
  git checkout (e.g. shipped tarball). The envelope still verifies; only
  the SHA is unavailable.

## What v1.0 does *not* do (open work)

- **`Warehouse` is still SQLite-primary.** DuckDB is supported via the export
  path; primary-store polymorphism remains an operational task.
- **`report_writer_v{2..5}` modules still exist.** Consolidation is queued
  behind a real refactor of the institutional report.
- **The Rust extension is a build-on-demand artifact.** CI does not yet ship
  a wheel; the parity tests skip cleanly when the extension is missing.

## How to verify the upgrade locally

```bash
pip install -e ".[dev,analytics,observability]"
pytest tests/ -q -m "not slow"

# Optional: build and parity-test the Rust kernels.
pip install maturin && (cd rust_ext && maturin develop --release)
pytest tests/ -q -m rust

# End-to-end smoke (mirrors the CI smoke job).
mre bootstrap-sample --db data/mre.db
mre seed-vintage-from-observations --db data/mre.db
mre materialize-asof-features --db data/mre.db --write-features
mre audit-vintage --db data/mre.db --enforce
mre build-features --db data/mre.db
mre score-regime --db data/mre.db
mre label-recessions --db data/mre.db --max-stale-months 24
mre train-baseline --db data/mre.db
mre validate --db data/mre.db --out data/validation
mre calibrate-probabilities --db data/mre.db --validation-dir data/validation
mre score-confidence --db data/mre.db --validation-dir data/validation
mre release-gate --db data/mre.db --validation-dir data/validation
mre model-run --db data/mre.db --purpose "v1.0 verify smoke"
mre verify-run --db data/mre.db        # MUST exit 0 with "approved": true
mre bench --out data/bench.csv
```

## Operating principle (unchanged)

The engine still emits forecast distributions, regime posteriors, dominant
drivers, historical analogs, model confidence, and explicit invalidation
conditions — never a naked point forecast. v1.0 just adds the conformal
guarantees, multi-horizon coherence, regime-conditional cross-sectional
forecasts, and the audit envelope a model-risk committee will actually want
to see.
