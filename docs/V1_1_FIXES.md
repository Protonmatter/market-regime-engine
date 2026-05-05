# v1.1 Fix Bundle — Production-Readiness Closure

> Highest-ROI fix bundle that closes the consequential gaps surfaced by both
> the SOTA roadmap (P0-1..P0-4) and the second-opinion review at
> `docs/SECOND_OPINION_REVIEW.md`. Implemented on the `v1.1-fixes` branch
> over a single coherent diff.

This document is the canonical changelog for what landed in v1.1. The
historical upgrade docs (`V1_0_UPGRADE.md`, `V0_*_UPGRADE.md`, etc.) are
intentionally untouched.

---

## Per-fix table

| ID | What changed | Files (key lines) | Severity addressed | New regression test(s) |
|---|---|---|---|---|
| **A** | Real reproducibility envelope: initial git commit on `main`, local-only `user.email/name`, `.gitattributes` enforcing `* text=auto eol=lf` plus `*.png binary` / `*.csv text` so the lockfile sha and golden-trace fixture stay deterministic on Windows checkouts. | `.gitattributes`, `.gitignore`, initial commit `741e51a` | P0-1 — `mre verify-run` had no commit to point at | E2E smoke now produces `verify-run → approved: true` (criterion 5) |
| **B** | Backtests routed through purged walk-forward. `expanding_window_binary_backtest` and `expanding_window_quantile_backtest` parse `H` from the horizon label (`"3m" → 3`) and call `PurgedWalkForward(min_train, step, horizon=H, embargo=1)` + `evaluate_walk_forward`. Public signatures unchanged. | `src/market_regime_engine/backtest.py:1-260` | P0-2 — naive expanding split leaked overlapping forward-target windows | `tests/test_backtest_purged.py` (4 tests; one verifies `train_idx.max() ≤ test_idx.min() − H` on every fold) |
| **C** | Hansen MCS gates promotion + release. Added `forecast_compare.mcs_promotion_filter()` thin wrapper, attached `mcs_evidence ∈ {"in_set","out_of_set","absent"}` to every `PromotionGate.evaluate_binary` row, and added `release_gates.evaluate_release_gate(require_mcs_membership=False)` that blocks the gate when True and the latest promotion row is not `in_set`. Default off → backward compatible. | `src/market_regime_engine/forecast_compare.py:435-485`, `src/market_regime_engine/promotion.py:1-90`, `src/market_regime_engine/release_gates.py:6-115` | P0-3 — promotion lacked statistical evidence of dominance | `tests/test_promotion_mcs.py` (8 tests including `require_mcs_membership=True` block path and back-compat default) |
| **D** | Conformal coverage as a first-class warehouse table + release gate. `Warehouse.write_conformal_coverage` / `read_conformal_coverage` (PK `(as_of_date, target, horizon, bucket, method)`); `orchestration.daily_flow` fits `MondrianBinaryConformal(alpha=0.10)` per `(target, horizon)` over the OOS preds in `validation_dir`, persists the per-bucket realized-coverage report, and surfaces the worst per-bucket coverage on `summary["worst_coverage"]`. `evaluate_release_gate(min_coverage=None, coverage_alpha=0.10, coverage_drop_pp=0.05)` blocks the gate when worst-bucket realized coverage drops more than 5pp below `1 − alpha`. | `src/market_regime_engine/storage.py:344-365,633-674`, `src/market_regime_engine/orchestration.py:38-58,150-180,232-280`, `src/market_regime_engine/release_gates.py:14-115` | P0-4 — conformal coverage was un-tracked and not gated | `tests/test_conformal_coverage.py` (5 tests; covers round-trip, `compute_conformal_coverage`, `min_coverage` block + back-compat) |
| **E** | SQLite concurrency hardening. `Warehouse.__post_init__` now opens with `check_same_thread=False` and immediately runs `PRAGMA journal_mode=WAL`, `PRAGMA synchronous=NORMAL`, `PRAGMA busy_timeout=10000`, `PRAGMA foreign_keys=ON`. | `src/market_regime_engine/storage.py:13-32` | Second-opinion #3 (HIGH) — concurrent writers immediately raised `database is locked` | `tests/test_conformal_coverage.py::test_conformal_coverage_round_trip` (two `Warehouse` instances against the same path write distinct tables sequentially) |
| **F** | API hardening quick-wins. `require_api_key` now uses `hmac.compare_digest` (constant-time). `/v1/metrics` is gated behind the same `MRE_API_KEY` dependency; `/v1/health` stays public for liveness probes. `_TTLCache` adds a `threading.Lock` and a `# TODO(MRE-API-1)` note that the cache is still process-local — multi-worker uvicorn deployments need a shared cache. | `src/market_regime_engine/api_v1.py:21-145` | Second-opinion #5 (HIGH) — `==` compare, auth-free `/v1/metrics`, lock-free TTL cache | `tests/test_api_v1_hardening.py` (5 tests including the `hmac.compare_digest` source guard and a lock concurrency storm) |
| **G** | `prometheus_text` percentile fix. The buggy version replayed `count` copies of the *mean* into a `Histogram`, collapsing every percentile to the mean. v1.1 emits Prometheus *summary*-style text manually using the in-process p50/p95/p99 we already record, so the scrape output reflects the actual percentiles. | `src/market_regime_engine/observability.py:96-170` | Second-opinion #5 (HIGH) — production dashboards silently lied | `tests/test_observability_percentiles.py` (2 tests; one asserts p50<p95<p99 on a wide series) |
| **H** | `BonferroniMultiHorizonConformal.joint_coverage` join fix. The chained `DataFrame.join(rsuffix=f"_{h}")` collided with the renamed `q_lo_{h}`/`q_hi_{h}` columns. Replaced with `pd.concat([... .add_suffix(f"_{h}")], axis=1, join="inner")`, and surfaced `horizons_used` so the caller knows which horizons contributed. | `src/market_regime_engine/multi_horizon_conformal.py:73-145` | Second-opinion #4 (HIGH) — joint coverage was wrong on ≥2 horizons | `tests/test_conformal.py::test_bonferroni_joint_coverage_*` (3 tests covering 3 horizons, partial overlap, and missing calibrators) |
| **I** | Lint debt collapsed. Reformatted the entire `cli.py` argparse table from semicolon one-liners to a proper multi-line form (kills ~150 `E702`s on its own), then ran `ruff check --fix --unsafe-fixes` + `ruff format`. Hand-fixed residual `B023` (closure capture in `regimes.domain_scores`) and `SIM102` in `alerts.py`. Pyproject `tool.ruff.lint.ignore` now also lists `RUF002` (math γ/σ in docstrings) and `PERF401` (readability beats `list.extend` micro-optimisation for our N). Tests carry an extra `E402` ignore because `tests/test_core.py` legitimately interleaves imports between blocks. | `pyproject.toml:64-90`, `src/market_regime_engine/cli.py:807-1085`, ~80 files reformatted | P0 quick win — CI lint gate was dead (416 errors → 0) | Verified by `ruff check src tests` exit 0 + `ruff format --check src tests` exit 0 (criteria 2/3) |
| **J** | PIT-router test coverage + audit propagation. `training_data.py` now emits a real `DeprecationWarning` in LEGACY mode (so the project's `filterwarnings=["error::DeprecationWarning:market_regime_engine"]` rule actually trips). `model_runs.create_model_run(training_audit=...)` is a new keyword arg that embeds the audit into both `metadata.training_audit` and `repro_envelope.extra.training_audit`. CLI `train_baseline_cmd` and `validate_cmd` stash the audit at `<db_dir>/training_audit.json`; `model_run_cmd` reads it back and forwards it. | `src/market_regime_engine/training_data.py:16-100`, `src/market_regime_engine/model_runs.py:170-235`, `src/market_regime_engine/cli.py:175-360` | Second-opinion #4 (CRITICAL) — `training_data.py` had 0% direct coverage and the LEGACY-fallback was un-auditable | `tests/test_training_data.py` (6 tests covering all four cases plus the `create_model_run(training_audit=...)` round-trip) |
| **K** | Stale-label gate ordering fix. `label_recessions_cmd` now evaluates `--max-stale-months` BEFORE writing to the warehouse. The earlier ordering wrote the stale rows then exited 2, leaving the warehouse poisoned. | `src/market_regime_engine/cli.py:140-175` | Second-opinion #14 (HIGH) — gate fired but warehouse already polluted | `tests/test_label_recessions_gate.py` (2 tests; one asserts the table stays unchanged when the gate trips) |

---

## Test count and lint status

| | Before | After |
|---|---|---|
| `pytest tests/ -q -m "not slow"` | 82 passed, 1 skipped, 1 deselected | **117 passed, 1 skipped, 1 deselected** (+35 tests) |
| `ruff check src tests` | 416 errors | **0** |
| `ruff format --check src tests` | 73/83 files would be reformatted | **0** (90 files already formatted) |
| `mypy src/market_regime_engine` | 39 errors | **38 errors** (one fewer; observability rewrite cleared an `unused-ignore`) |
| End-to-end smoke ⇒ `verify-run.approved` | not exercised | **true** |

`pytest -k slow` is intentionally not in the required gate; the only `slow`-marked test is the synthetic `test_daily_flow_runs_on_synthetic_pipeline` end-to-end.

---

## Out-of-scope deferrals

These items appeared in the SOTA roadmap or the second-opinion review but were
explicitly deferred per the user's "highest-ROI cohesive bundle" instruction.
Each one is captured here as a `TODO` for a follow-up PR rather than rolled
into this diff.

- `# TODO(MRE-API-1)`: replace `_TTLCache` with a shared cache (Redis) for
  multi-worker uvicorn deployments. The lock-protection in F is sufficient
  for `--workers 1`; multi-worker still has independent caches per worker.
- DFM EM likelihood correctness (`dfm.py:116`, second-opinion #C/8). Math-heavy
  fix that touches the Watson-Engle approximation; needs a separate review pass.
- BOCPD-MUSE M2 ordering edge case (no test coverage gap currently observable;
  parity tests pass).
- DuckDB-primary `Warehouse` (Phase B / scale-out work).
- Five `report_writer_v{1..5}.py` consolidation (code-rot smell, second-opinion
  #20). Pure cleanup, no functional impact.
- `_hash_frame` dtype-fragility (second-opinion #10). Documented; needs a
  switch from `astype(str)+csv` to `pd.util.hash_pandas_object` and a
  golden-trace migration.
- `apply_release_lag` only knows 8 series (second-opinion #B). Needs a
  release-rule extension across the catalog; out of scope for the bundle.
- `MondrianBinaryConformal` docstring contradiction + non-binary `y`
  validation (second-opinion #7). Cosmetic + small validation; not blocking.
- `_purge_and_embargo` O(|train|·|test|) Python loop (second-opinion #G). Fine
  for monthly data; NumPy `searchsorted` migration deferred until daily
  workloads land.

---

## How to consume this PR

1. `git checkout v1.1-fixes`
2. `pip install -e ".[dev]"`
3. `pytest tests/ -q -m "not slow"` → expect `117 passed`.
4. `ruff check src tests && ruff format --check src tests` → both exit 0.
5. End-to-end smoke (the canonical reproducibility flow):
   ```
   mre bootstrap-sample
   mre seed-vintage-from-observations
   mre materialize-asof-features --write-features
   mre audit-vintage --enforce
   mre build-features
   mre score-regime
   mre label-recessions --max-stale-months 24 --force-builtin
   mre train-baseline
   mre validate --min-train 60 --step 12
   mre calibrate-probabilities --validation-dir data/validation
   mre release-gate --validation-dir data/validation --min-confidence 0.0
   mre model-run --validation-dir data/validation --purpose "v1.1 smoke"
   mre verify-run
   ```
   `verify-run` should print `"approved": true` and exit 0.

The v1.0 upgrade docs and reviews are kept untouched. This file is the source
of truth for v1.1.
