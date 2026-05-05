# Second-Opinion Adversarial Review (Claude Opus 4.7, vs. GPT-5.5)

> Independent cross-model audit of the v1.0 upgrade at
> `C:\Users\mkang\market-regime-engine-v0.8\market-regime-engine`. Read +
> evidence pass; no source modifications. All claims below are backed by
> file:line references and / or shell output captured during this review.
> Reviewed against the GPT-5.5 v1.0 upgrade narrative summarized in the
> conversation history.

---

## Executive verdict

The engine's *mathematical* core is largely sound and aligns with what GPT-5.5
claims: HLN-corrected DM, NIW-BOCPD with Murphy 2007 predictive density, MS-VAR
Hamilton-Kim, purged walk-forward, finite-sample conformal quantile, and Rust
parity all pass close inspection. End-to-end `mre model-run` → `mre verify-run`
round-trips with `approved: true` and lockfile drift correctly fails the gate
with exit code 2. **However, the build is _not_ production-ready as advertised.**
The CI lint gate is dead (a fresh CI run on `main` would fail before any test
runs), the v1.0 PIT routing module has *zero* direct unit-test coverage, the
multi-horizon Bonferroni `joint_coverage` helper has a join-bug that no test
exercises, the API's TTL cache and Prometheus exporter have correctness defects,
and the SQLite warehouse has no concurrency configuration. Silent fallbacks
(stale labels, FRED failure, legacy training mode) produce auditable artefacts
in the *log* but never in the *warehouse*, which means a Monday-morning operator
could ship calibrated outputs trained on the wrong source and never know.

### Top 5 issues GPT-5.5 missed or overstated

| # | Severity | Issue | Where |
|---|---|---|---|
| 1 | **CRITICAL** | CI lint gate cannot pass: `ruff check src tests` reports **416 errors**, `ruff format --check` would reformat **73 of 83 files**. The advertised CI workflow `.github/workflows/ci.yml:19-20` runs both — the lint job is a hard fail on a fresh push. | `pyproject.toml:64-83`, ruff stats below |
| 2 | **HIGH** | `BonferroniMultiHorizonConformal.joint_coverage` produces ambiguous columns when joining horizons because `cqr.transform()` leaves the original `q_lo`/`q_hi` in the frame and the rsuffix collides with the renamed `q_lo_{h}` columns. No test exercises it. | `src/market_regime_engine/multi_horizon_conformal.py:90-109` |
| 3 | **HIGH** | SQLite warehouse opened without `journal_mode=WAL` or `busy_timeout`; CLI / API / Streamlit each open separate connections. Two writers (e.g. `mre score-regime` while the API or another CLI runs) immediately raise `database is locked`. | `src/market_regime_engine/storage.py:16` |
| 4 | **HIGH** | `training_data.py` (the entire v1.0 PIT routing core) has **0% pytest coverage**. The silent LEGACY fallback when `feature_asof_values` is empty only emits a `log.warning`; no warehouse marker, no SystemExit. | `src/market_regime_engine/training_data.py:57-83`, `cov_out.txt` |
| 5 | **HIGH** | `prometheus_text()` rebuilds the histogram by `for _ in range(count): hist.observe(mean)`; every reported percentile collapses to the mean — Prometheus dashboards will silently lie about p95/p99 latency. `_TTLCache` is also process-local *and* not lock-protected. `/v1/metrics` is auth-free. `/v1/health` opens a SQLite connection per probe. | `src/market_regime_engine/observability.py:118-120`, `src/market_regime_engine/api_v1.py:60-100,111-126` |

---

## Claims-vs-reality table

| # | GPT-5.5 claim | Verdict | Evidence (file:line) | Consequence |
|---|---|---|---|---|
| 1 | "82 passed, 1 skipped (Rust parity), 1 deselected (slow), 0 failed in ~6m." | **VERIFIED** (timing partial) | `pytest tests/ -q -m "not slow" --tb=line --maxfail=20` → `82 passed, 1 skipped, 1 deselected, 1 warning in 251.59s` (terminal `617054.txt`). Actual time **~4 min 12 s**, not ~6 min. One ungated `UserWarning` from sklearn (`tests/test_phase2_phase3.py::test_covariate_hazard_fits_and_predicts_in_unit_interval`). | Tests do pass; timing claim slightly inflated. Coverage is only **58%** (see #11). |
| 2 | "ruff is configured with `select = [E, F, W, I, B, UP, SIM, C4, RET, PERF, RUF]` and the CI lint gate is enforceable." | **FALSIFIED** | Config is present (`pyproject.toml:64-83`). But `ruff check src tests` → **416 errors** (top categories: E702 multi-statement-on-one-line ×151 in `cli.py`, I001 unsorted-imports ×66, E402 module-import-not-at-top ×30, RUF046 unnecessary-cast-to-int ×28, F401 unused-import ×25, B905 zip-without-strict ×18). Worst file: `src/market_regime_engine/cli.py` (153 errors). `ruff format --check src tests` → **73 of 83 files would be reformatted**. CI workflow runs both (`.github/workflows/ci.yml:19-20`) — the **lint job would fail on every push to main right now**. | The advertised CI gate is essentially not running. Pre-commit auto-fixes locally (`.pre-commit-config.yaml:18-20`), which is why the developer never sees the failure. |
| 3 | "Reproducibility envelope round-trips; `mre verify-run` would fail if the lockfile changed." | **PARTIAL** | End-to-end check executed: `mre bootstrap-sample → … → train-baseline → validate → model-run → verify-run` produced `approved: true`. After appending one byte to `requirements-lock.txt`, `mre verify-run` exited 2 with `differences.lockfile_hash` showing both the stored and current sha256. ✓ However: <br/>(a) `_hash_frame` (`src/market_regime_engine/model_runs.py:81-93`) coerces every column with `astype(str)`. Empirical: identical data with int vs float dtype produces **different** hashes (`python -c …` → `int dtype vs float dtype: False`). A schema/dtype change between runs (e.g. NaN promoting int→float) drifts the hash silently. <br/>(b) `verify_run` (`model_runs.py:248-249`) contains a no-op bool coercion: `if isinstance(current_value, bool) and not isinstance(stored_value, bool): current_value = bool(current_value)` — both come out as `bool` because JSON loads booleans as Python booleans, so the branch is dead. Should have coerced `stored_value` instead. <br/>(c) `lockfile_present` (line 257) checks the CWD via `os.path.exists(_LOCKFILE_NAME)`, but `_lockfile_hash` (line 118-123) uses `_project_root() / _LOCKFILE_NAME`. Inconsistent — running `verify-run` from a different cwd would report `lockfile_present=False` while still computing the correct hash. <br/>(d) `_git_revision` and `_git_dirty` (lines 96-111) invoke git in CWD, returning `"unknown"`/`False` on failure. A non-git checkout produces a valid envelope where `code_version` is just `"unknown"`. | The headline claim holds for the lockfile, but the dtype-fragile hash and the inconsistent project-root resolution are silent reproducibility hazards. |
| 4 | "PIT path is the real default; `feature_asof_values empty` fallback has an audit-trail consequence." | **PARTIAL** | `_resolve_training_mode` defaults to PIT (`cli.py:175-178`). The empty-vintage fallback emits `audit["fallback_reason"] = "feature_asof_values empty"` (`training_data.py:64`) and a `log.warning` (line 59-62). **No SystemExit, no warehouse row, no model_run marker.** The `audit` dict is forwarded only to `log.info(extra=)` and a `print()` (`cli.py:201-202`). `train_baseline_cmd` does not write a `ModelRun` row at all — that happens in a separate `mre model-run` command, and the `repro_envelope` it writes does not include `audit.fallback_reason`. The entire `training_data.py` module has **0% test coverage** (`cov_out.txt`). | If the operator forgets to run `materialize-asof-features --write-features`, the engine silently trains on revised macro data and the only audit trail is one INFO log line. |
| 5 | "Hansen MCS recentering is correct (HLN, T_R variant, eliminate-worst loop)." | **PARTIAL** | Standard T_R variant: per-model SE via Newey-West (`forecast_compare.py:227-230`), `td = (mean_loss - mean_loss.mean()) / se`, `observed = max(td)`, recentered bootstrap `(sm - sm.mean()) - (mean_loss - mean_loss.mean())` (line 243), p-value `mean(boot_max >= observed)`. Eliminate-worst is `argmax(td)` (line 246) — correct for "lower loss = better". <br/>**Two issues**: (a) Comment claims "Stationary block bootstrap" (line 235) but the code is the **moving block bootstrap** with fixed `block_size`; stationary block (Politis-Romano) requires geometric block lengths. Documentation drift. (b) Bootstrap sample length is `min(blocks * block_size, n)` not `n` (line 239 truncates `[:n]` after `np.arange(s, s + block_size) % n`); when `n % block_size != 0`, the bootstrap is up to `block_size - 1` samples shorter than the original, which biases SE. | T_R algorithm is correct; the bootstrap variant is misnamed and slightly biased on non-divisible n. Tests only check that the worst model is eliminated when one is clearly best — none verifies the bootstrap recentering math. |
| 6 | "Diebold-Mariano with HLN: `sqrt((n + 1 - 2h + h(h-1)/n) / n)`, NW lag `max(0, h-1)`, two-sided p-value via \|stat\|." | **VERIFIED** | All three formulas match the implementation (`forecast_compare.py:119,126,128`). For `h=0`: `lag=0`, `_newey_west_var(d, 0)` returns just `gamma0` (lines 38-50), `se` finite, `stat=0.0` for identical inputs, `p=1.0` (`tie`) — verified by inspection. **One small risk**: `scale = math.sqrt(...)` (line 126) has no guard if the argument turns negative; the f(h) function `n + 1 - 2h + h(h-1)/n` reaches a minimum of `-0.25/n` at `h ≈ n + 0.5`. For all integer `h`, the value is ≥ 0 in practice, but pathological floats could raise `ValueError`. Negligible. | Math is correct. |
| 7 | "Conformal coverage: `ceil((n+1)*(1-alpha))/n` rank, Mondrian binary semantics agree, `prediction_set` valid." | **PARTIAL** | Quantile rank formula is correct (`conformal.py:62`); the `min(max(rank, 1), n)` clamp handles edge cases by silently degrading coverage when `alpha < 1/(n+1)` (no warning emitted — operator never knows). <br/>**Docstring is wrong**: `MondrianBinaryConformal` (`conformal.py:87-88`) claims "score `s = min(p, 1 - p)` when `y` is unknown is below the threshold" — the code never computes `min(p, 1-p)`. The actual scores are `_score(p, 1) = 1-p` and `_score(p, 0) = p` (lines 188-192). The label-1-included condition is `1 - p ≤ threshold ⇒ p ≥ 1 - threshold`; the label-0-included condition is `p ≤ threshold`. <br/>**Robustness gaps**: `fit` requires both `y` and `p` columns; if `y` is missing entirely, `dropna(subset=["y", "p"])` raises `KeyError`. `astype(int)` on `y` (line 118, 176) silently truncates non-binary `y` (e.g. 2.0 → 2), leaving the row outside the {0,1} prediction set so coverage drops to 0 with no warning. | Math is correct, but the docstring is misleading and the `y`-validation is thin. |
| 8 | "Multi-horizon Bonferroni: per-horizon `alpha/H`, `joint_coverage` correctly merges by date." | **PARTIAL — bug found** | Per-horizon alpha math is correct (`multi_horizon_conformal.py:51-53`). <br/>**`joint_coverage` JOIN BUG** (lines 90-98): `cqr.transform(df).rename({"q_lo_conformal": f"q_lo_{h}", ...})` leaves the *original* `q_lo`, `q_hi`, `y` columns un-renamed in `adj`. When the second horizon is joined into `merged`, the unrenamed `q_lo` collides with the existing one and is suffixed by pandas' `rsuffix=f"_{h}"` to `q_lo_6m` — which **already exists** in `adj` from the rename step. Result: pandas keeps both columns, and `merged.get(f"q_lo_{h}")` (line 103) becomes ambiguous. <br/>Also: if any horizon's `y_{h}` column is missing from `merged`, the loop silently `continue`s (line 106-107) and `joint` only reflects the horizons that *do* have all three cols — making the reported "joint coverage" cover fewer than `H` horizons without telling the caller. <br/>No tests in `tests/test_phase6_phase7.py` exercise `joint_coverage`. | The reported joint coverage number can be wrong or computed over a subset of horizons; consumers (release gate, dashboards) get an inflated coverage estimate. |
| 9 | "Walk-forward purge handles `horizon=12`, `expanding=False`, no off-by-one at fold boundaries." | **VERIFIED** | For test point `i` and `horizon=12`, `train_upper = i - 12` (`walk_forward.py:96-97`); train rows therefore stop at `t = i - 13`, whose forward window `[i-12, i-1]` excludes `i`. ✓ For `expanding=False`, `train_lower = max(0, train_upper - min_train)` gives the trailing `min_train` rows ending at `train_upper` (line 98). ✓ The `train_upper - train_lower < min_train // 2` check (line 99-102) gracefully skips folds whose window has shrunk too far. CPCV `_purge_and_embargo` correctly drops `t` if any `tau` in test satisfies `t < tau ≤ t + horizon` (line 159) and embargo neighbours `tau < t ≤ tau + embargo` (line 162). ✓ | Algorithm is correct. **Performance flag**: `_purge_and_embargo` is a Python `for t in train_idx` loop with two nested `any()` over `test_idx` — O(\|train\|·\|test\|) per fold. For daily-frequency CPCV (n≈5000, k=2 of 6 blocks), that's ~25M comparisons per fold × 15 folds ≈ 400M Python ops. |
| 10 | "Rust kernels exist with parity at atol=1e-9 (BOCPD diag, WFST Viterbi, PSI, rolling Mahalanobis)." | **VERIFIED** | All four kernels (`rust_ext/src/lib.rs:114, 258, 321, 342`) match their Python references. The BOCPD parity test (`tests/test_rust_parity.py:91-148`) reconstructs `RunningDiagState` from kernel outputs by reading back `state_n`, `state_mean`, `state_m2`. The Rust kernel correctly handles the `n=0`/empty-prior edge case at `lib.rs:209` (`if old_n == 0 { mean = xv; m2 = 0 }`), matching `RunningDiagState.update` (`bocpd.py:81-90`). The "fresh prior + xv" output at position 0 (`lib.rs:196-199`) sets `mean=xv, m2=0, n=1`, which equals `RunningDiagState.prior(d, 1.0).update(xv)`. | Kernels are correct. The Rust parity test is `pytestmark = pytest.mark.rust` and is skipped (`is_available()` returns False) unless `maturin develop` was run; the local environment skipped this test (1 skipped in pytest output). |
| 11 | "MS-VAR Hamilton-Kim filter / Kim smoother / EM with `gamma` weights." | **VERIFIED** with caveats | Forward filter uses `_logsumexp` correctly (`msvar.py:107-115`); the smoother uses Kim's update `log_gamma[t] = log_alpha[t] + logsumexp(log_A + log_gamma[t+1] - denom)` (line 125). The EM M-step uses `gamma` consistently (lines 173, 184, 195). Numerical safeguard `if w.sum() < d + 2: keep old params` ✓ (line 174). The convergence test `abs(ll - prev_ll) < tol * max(abs(prev_ll), 1)` is fine. <br/>**Performance flag**: `W = np.diag(w)` (line 184) is O(n²) memory. Should use `sqrt_w[:, None] * X` and `sqrt_w * Y` for the standard `(W^{1/2} X)^T (W^{1/2} X)` trick. Same with `(resid.T * w) @ resid` (line 195) which is fine because it uses broadcast. | Math is correct; Python loops dominate runtime. |
| 12 | "NIW BOCPD predictive matches Murphy 2007 (`scale = psi*(kappa+1)/(kappa·df)`, `df = nu - d + 1`); run-length truncation does not bias near boundary." | **PARTIAL** | The predictive log-pdf matches Murphy 2007 Eq. 232 (`bocpd.py:251-279`): `df = nu_n - d + 1` ✓, `scale = psi_n * (kappa_n + 1.0) / (kappa_n * df)` ✓, `log_norm` and Cholesky-based `quad` correctly computed. The Cholesky-fallback ladder (lines 264-272) is a reasonable robustness layer. <br/>**Truncation IS biased**: `growth_logs[: self.max_run]` is kept and renormalized via `_logsumexp(new_log_joint)` (lines 369-373). This redistributes the lost tail mass over the kept run lengths, **inflating** `cp_prob` and growth probs near `max_run`. The docs flag this; it is what the code does. Whether the bias is acceptable is a design decision, not a bug — but it should be surfaced in monitoring. | Predictive density is correct; truncation behaviour is the documented (biased) variant. |
| 13 | "`monthly_panel.ffill` leakage fix; `forward_fill_limit=0` default; legacy fallback flag exists." | **VERIFIED** | `features.py:10` defaults `forward_fill_limit=0`; the ffill only runs when `forward_fill_limit and forward_fill_limit > 0` (lines 47-48). `rolling_z` (line 57-60) uses `.shift(1)` on both `mean` and `std` so the standardisation never sees `s_t` itself. ✓ `cli.py:142` calls `monthly_panel(..., forward_fill_limit=0)` explicitly. <br/>I checked all internal callers (`Grep "monthly_panel\("`): every call inside `cli.py` and `training_data.py` uses the default. The legacy positive-int opt-in is preserved for any external consumer. | Fix is in place. |
| 14 | "Recession label staleness gate fails closed; FRED-fail-back-to-builtin is auditable." | **PARTIAL** | `--max-stale-months` does fail closed when set: `cli.py:156-159` raises `SystemExit(2)` if `staleness.months_stale > args.max_stale_months`. ✓ <br/>**But**: (a) the flag's default is `None` (`cli.py:818`), so the gate is **off by default**; (b) the labels are `db.write_recession_labels(labels)` (cli.py:147) **before** the gate fires (line 156), so the warehouse keeps the stale rows even when the CLI exits 2; (c) the FRED fallback path (`nber.py:175-181`, `pragma no cover`) silently swallows any exception into `fetch_err`, which is captured into `staleness.metadata` but never raised — the operator only sees it if they read the `staleness.fetch_error` print line. | Gate exists but defaults open and persists stale data before the gate fires. Worse: the FRED path is a `# pragma: no cover` branch, so test coverage for the most operationally important fallback is zero. |
| 15 | "API hardening: `MRE_API_KEY` enforced, TTL cache, `/v1/health` reports release-gate, `/v1/metrics` exposed." | **PARTIAL — multiple defects** | (a) `require_api_key` enforces only when `MRE_API_KEY` is set (`api_v1.py:41-46`) ✓. (b) `/v1/metrics` (line 124) and `/v1/health` (line 111) **lack** the auth dependency — confirmed bugs (operator may have intended at least `/v1/metrics` to be auth-free, but if so the README does not say so; both leak engine internals). (c) `_TTLCache` is a module-global instance (`api_v1.py:84`) — confirmed process-local, *not* shared across uvicorn workers. Worse: it has **no lock** (line 60-81); concurrent `move_to_end` + `popitem` calls under load can race in CPython. (d) `/v1/health` opens a fresh SQLite connection (`api_v1.py:113-120`) and reads ALL release_gates rows per request. FastAPI runs sync route handlers in a threadpool so the event loop is not blocked, but high-frequency liveness probes will create connection churn and serial table scans. (e) `prometheus_text` (`observability.py:96-130`) tries the real Prometheus registry path and, on the way, does `for _ in range(int(stats["count"])): hist.observe(stats["sum"] / max(stats["count"], 1))` — every observation is the **mean**, so the resulting Prometheus histogram has degenerate p50/p95/p99 (all equal to the mean). Dashboards built on `/v1/metrics` will silently lie. | API hardening is shallower than the README claims. The metrics defect is the most severe because it's invisible to anyone reading the JSON snapshot directly. |

---

## Cross-model analysis (three-category)

### Overlap (both reviews agree) — high confidence

These items are mentioned both in GPT-5.5's narrative and in this review:

- **PIT-by-default routing**: both agree the wiring is in place (`_resolve_training_mode` defaults to `POINT_IN_TIME`).
- **Lockfile-pin reproducibility envelope**: both agree the envelope captures git rev, lockfile sha, payload hashes, and that `mre verify-run` exits non-zero on lockfile drift (this review confirmed empirically with `exit 2`).
- **DM/HLN, NIW-BOCPD predictive density, MS-VAR Hamilton-Kim**: both agree these match the canonical references.
- **Walk-forward purge for `horizon=h`**: both agree the train upper bound is `i - h`, no off-by-one.
- **Rust kernel parity at atol=1e-9**: both agree the parity tests are correctly structured (skipped without `maturin develop`).

### Unique to GPT-5.5's review

- **"Test suite green; CI lint enforceable"** — overstated by GPT-5.5. The test suite is green; the lint gate is not.
- **Hansen MCS as "stationary block bootstrap"** — adopted from the docstring without inspection. Actual implementation is moving block.
- **"`audit-trail consequence`" for the LEGACY-fallback in `train_baseline`** — GPT-5.5 said the fallback is auditable; in practice the audit dict only reaches stdout/log, never the warehouse or the `repro_envelope`.

### Unique to this second-opinion review

- **416 ruff errors and 73 unformatted files**, with `cli.py` carrying 153 errors (mostly `E702` multi-statement-on-one-line because the entire `parser()` function packs many `add_argument` calls onto one line). CI lint job currently impossible to pass.
- **`BonferroniMultiHorizonConformal.joint_coverage` rename/join collision bug** (`multi_horizon_conformal.py:90-98`).
- **`prometheus_text()` percentile destruction** (`observability.py:118-120`).
- **`_TTLCache` is not lock-protected** and is process-local.
- **SQLite warehouse opens with default isolation, no WAL, no `busy_timeout`** (`storage.py:16`). Confirmed concurrency hazard.
- **`_hash_frame` is dtype-fragile** (int↔float of the same numerical value produce different hashes; verified empirically).
- **`verify_run` has dead bool-coercion code** (`model_runs.py:248-249`) and **inconsistent project-root resolution** for `lockfile_present` (line 257) vs. `_lockfile_hash` (line 118-123).
- **`MondrianBinaryConformal` docstring contradicts the code** (`conformal.py:87-88`): claims `s = min(p, 1-p)` but the code uses label-conditioned `_score(p, label)`.
- **`MondrianBinaryConformal` silently degrades to 0 coverage on non-binary `y`** because `astype(int)` truncates and the label is never in `{0, 1}`.
- **Stale recession labels are written to the warehouse BEFORE the staleness gate fires** (`cli.py:147` runs before `cli.py:156`).
- **5 `report_writer_v{,2,3,4,5}.py` files** all at 0% coverage — code-rot smell.
- **`training_data.py` (the entire v1.0 PIT routing core) is at 0% pytest coverage** (`cov_out.txt`).
- **DFM `f_filt[t] = fp + np.sum(K * resid * sigma_eps2 / np.maximum(gain_den, 1e-12) ** 0)`** (`dfm.py:121`) is dead code (`** 0 = 1`) immediately overwritten by the information-form posterior at lines 128-131.
- **DFM log-likelihood is wrong** (`dfm.py:116`): assumes columns of `y_t` are conditionally independent given the integrated factor, but the integrated covariance has cross-terms `lambda_i * lambda_j * pp`. Convergence is judged on this approximate likelihood.
- **Bootstrap-sample-length bias in Hansen MCS** when `n % block_size != 0` (`forecast_compare.py:239`).
- **Latent SQL-injection surface in `_write`** (`storage.py:360,372`): `f"INSERT OR {mode} INTO {table} ({col_sql}) VALUES ..."`. All current call sites pass hardcoded literals, so there is no live vulnerability — but a future caller with a user-controlled `mode` would be vulnerable.

---

## Adversarial probe findings

### A. Concurrency / race conditions

`Warehouse.__post_init__` (`src/market_regime_engine/storage.py:13-17`) calls `sqlite3.connect(str(self.path))` with **no PRAGMAs** (no `journal_mode=WAL`, no `busy_timeout`, default `isolation_level="deferred"`). The CLI, the v1 API (`api_v1.py:113`, `api_v1.py:93`), the Streamlit dashboard (`dashboard.py`), and the orchestration daily-flow all instantiate fresh `Warehouse` objects against the same `data/mre.db` file. With default journal mode (delete/rollback), SQLite serialises all writers and *also* blocks readers during a write transaction. Without `busy_timeout`, conflicting writes raise `sqlite3.OperationalError: database is locked` immediately. Concretely: running `mre score-regime` while a uvicorn `api_v1.app` worker is mid-`/v1/regime/latest` cache-miss read can deadlock; running two `mre …` commands in parallel (e.g. a cron `materialize-asof-features` and a manual `train-baseline`) will sporadically lose writes and then fail downstream commands that read partial tables. Every `_write` call wraps `executemany + commit` (`storage.py:372-373`) but has no retry. **Fix**: enable WAL (`PRAGMA journal_mode=WAL`) and set `busy_timeout=5000` on connect.

### B. Lookahead leakage in `apply_release_lag` and `rolling_z`

`apply_release_lag` (`src/market_regime_engine/point_in_time.py:43-63`) is a per-row `frame.apply(effective_vintage, axis=1)` that conservatively computes the latest of `(actual vintage, observation_date + release_rule_lag)`. `DEFAULT_RELEASE_RULES` only covers 8 series; everything else falls through to `ReleaseRule(series_id, lag_days=0, lag_months=0)` (`apply_release_lag.effective_vintage` line 55) — **most series therefore have no release lag applied**, so the v1.0 release-lag layer is a stub for series outside the eight named macros. This is *not* leakage but it does mean PIT-correctness depends on the upstream `feature_asof_values` materialisation, not on this function.

`rolling_z` (`src/market_regime_engine/features.py:57-60`) is correct: both `mean` and `std` are computed over the rolling window then `.shift(1)` is applied, so the standardisation at time `t` uses statistics from `[t-window, t-1]`. Confirmed no leakage.

### C. DFM identifiability and EM correctness

Sign anchoring is applied at init (`dfm.py:87-88`) AND after fit (`dfm.py:178-180`) — robust. ✓ The Kalman filter / RTS smoother loop (lines 96-141) uses the *current* iteration's parameters consistently, and the M-step (lines 143-165) computes new parameters from the smoothed expectations before assigning them; this is **standard EM**, not the "uses previous iteration's parameters" anti-pattern.

Two real bugs:

- Line 121: `f_filt[t] = fp + np.sum(K * resid * sigma_eps2 / np.maximum(gain_den, 1e-12) ** 0)` — the operator precedence makes `np.maximum(...) ** 0` resolve to `1`, so this is literally `fp + sum(K * resid * sigma_eps2)`. The line is then immediately **overwritten** by the information-form posterior at lines 128-131. Dead code that betrays an unfinished refactor (the comment on line 122-123 says the gain-form was buggy and replaced).
- Line 116: `ll += -0.5 * np.sum(np.log(2 * pi * S) + (resid ** 2) / np.maximum(S, 1e-12))` integrates over each observation column independently using its marginal predictive variance `S_j = lambda_j^2 * pp + sigma_eps2_j`. But conditional on the integrated factor, the columns of `y_t` are **correlated** with cross-covariance `lambda_i * lambda_j * pp`. The log-likelihood is therefore a Watson-Engle approximation, not the true marginal — and the EM convergence test `abs(ll - prev_ll) < tol * max(abs(prev_ll), 1.0)` (line 167) is judging convergence on this approximation. M-step updates remain valid (they only need smoothed factor moments), but EM may terminate early or oscillate.

### D. Reproducibility hash drift (end-to-end test)

Executed: `mre bootstrap-sample → seed-vintage-from-observations → materialize-asof-features --write-features → build-features → score-regime → train-baseline → validate → model-run → verify-run` on a fresh `data/mre_audit_test.db`. Result: `mre verify-run` produced `{"approved": true, "differences": {}, "lockfile_present": true, …}` with exit 0. Then I appended one byte to `requirements-lock.txt` and re-ran `mre verify-run`; it produced `differences.lockfile_hash` showing both stored and current sha256 and exited with **code 2**. ✓ The headline lockfile-drift behaviour is correct.

Additional empirical check on `_hash_frame` determinism (`python -c "..."` scratch script): two consecutive reads of the same warehouse `features` table produced the same hash (`True`). But constructing two equivalent dataframes with int vs float dtypes for the same numerical values produced different hashes (`False`) — `astype(str)` of `1` is `"1"`, of `1.0` is `"1.0"`. If pandas ever promotes a column from int to float between runs (e.g. because a NaN was introduced by a missing observation), the envelope hash drifts silently.

### E. Trust boundaries on user input

- `release_calendar.py:27` and `config.py:15` use `yaml.safe_load`. ✓
- `data_sources.py:53,108`: `float(obs["value"])` with no overflow / `inf` / `nan` guard. A pathological FRED response could land `inf` or `nan` in the warehouse via the `value REAL NOT NULL` column.
- Catalog YAML (`config.load_catalog`, `config.py:18-19`) returns `list(load_yaml(path).get("series", []))` with **no schema validation**. If a malformed entry has `series_id` as a list/dict, downstream `item["series_id"]` raises a confusing TypeError deep inside `build_features`.
- `storage.py:360,372`: `_write(self, table, df, cols, mode="REPLACE")` formats `table`, `cols`, and `mode` directly into the SQL string. All current call sites pass hardcoded literals (verified by `Grep _write\(`), so this is **not a live vulnerability** — but the `mode` parameter is a latent SQL-injection sink the moment any caller starts forwarding user input.

### F. Test coverage gaps (effective coverage estimate)

Total reported coverage is **58%** (`pytest --cov=src/market_regime_engine --cov-report=term`, full output captured in `cov_out.txt`). Modules at 0% direct test coverage (i.e. the test suite never imports them or only imports without exercising):

- `cli.py` (651 stmts, 0%) — entire CLI is untested by pytest. CI's smoke job exercises it end-to-end, but unit-level coverage is zero.
- `training_data.py` (44 stmts, 0%) — **the v1.0 PIT routing core**. This is the most concerning gap because the entire claim "PIT-by-default" rests on these 44 statements.
- `targets.py` (32 stmts, 0%) — target construction; tested only indirectly through `make_targets` calls in other tests.
- `dashboard.py` (128 stmts, 0%) — Streamlit UI.
- `report_writer.py`, `report_writer_v2.py`, `report_writer_v3.py`, `report_writer_v4.py`, `report_writer_v5.py` (5 versions, all 0%) — five iterations of the same module, all dead from the test suite's POV.
- `bench.py` (58 stmts, 0%) — bench harness.
- `analogs_v2.py`, `api.py`, `backtest.py`, `data_sources.py`, `fred_recession.py`, `fred_vintage.py` — all 0%.

Significantly under-covered modules: `orchestration.py` (25%), `explain.py` (29%), `api_v1.py` (36%), `alfred.py` (38%), `alfred_real.py` (37%), `logging_setup.py` (38%), `rust_kernels.py` (50%), `storage.py` (50%).

The headline claim "82 tests, all green" looks thinner once you ask "tests of *what*" — the math primitives (`forecast_compare`, `walk_forward`, `conformal`, `bocpd`) are well tested, but the orchestration / governance layers (`cli`, `training_data`, `orchestration`, `release_gates`) live mostly through the smoke job.

### G. Performance / scaling

The Python hot loops that will not scale to daily-frequency or multi-asset workloads:

- `bocpd.py:142-173` (Diagonal) and `337-390` (NIW): per-step `pred_logs = np.array([... for st in states])` — Python list comprehension over up to `max_run` states, each doing a Cholesky / lgamma. For NIW with `max_run=96` and `d=8`, each step is ~96 Cholesky solves. The Rust `bocpd_diag_update` exists for the diagonal kernel, but the NIW core has no Rust fallback.
- `walk_forward.py:153-164` `_purge_and_embargo`: `for t in train_idx: ... any(t < tau <= t + horizon for tau in test_idx)` — O(\|train\| × \|test\|) per fold. For daily CPCV (`n=5000, k_test_blocks=2 of 6`), ≈ 25M comparisons per fold × 15 folds ≈ 400M Python ops.
- `dfm.py:103-131`: per-iteration Python Kalman loop. With `max_iter=50` and large `n`, this dominates `fit_domain_dfm` runtime.
- `msvar.py:184` `W = np.diag(w)`: O(n²) memory; should use `sqrt(w)` weighting.
- `point_in_time.py:60` `frame.apply(effective_vintage, axis=1)`: row-by-row Python apply.
- `storage.py:466-472` `init_release_gates_severe_column`: every `Warehouse.__init__` runs an `ALTER TABLE` migration check — repeated unnecessarily on every connect.

### H. Silent exception-eating

21 files contain `except Exception` blocks. Real risks:

- `nber.py:175-181` (`# pragma: no cover - network branch`): FRED USREC fetch failure silently falls back to the built-in NBER table frozen at `2020-04-01`. The error is captured into `staleness.fetch_error` but never raised. With the default `--max-stale-months=None`, the fallback is invisible in production.
- `regimes.py:128`: HMM fit failure silently falls back to a default `HMMRegimePosterior()`. Comment is brief; could mask data-quality regressions.
- `model_runs.py:100,110,240`: git invocations swallow all exceptions and return `"unknown"` / `False`. Operator never sees the actual error.
- `storage.py:471-472`: `init_release_gates_severe_column` ALTER TABLE migration silently swallows any DDL error.
- `report_writer_v2.py:25` (and several others under `report_writer_v*`): bare `pass`. Module has 0% coverage so the dead path is never exercised.
- `analogs_v2.py:24,47`, `analogs.py:111`, `explain.py:13`, `release_calendar.py:70`: all `json.loads(metadata_json)` wrapped in `except Exception → {}`. Acceptable for forward-compat but masks true JSON corruption.
- `models.py:158`: gradient-boosting fit failure → linear fallback. Comment justifies; acceptable.

---

## Severity-ranked production blockers

### CRITICAL

1. **CI lint gate is dead** — `ruff check src tests` returns 416 errors, `ruff format --check` would reformat 73/83 files. The advertised CI workflow at `.github/workflows/ci.yml:19-20` cannot pass on a fresh push to `main`. Either run `ruff check --fix && ruff format src tests` and re-commit, or relax the CI gate to match reality.
2. **`training_data.py` (the PIT routing core) has 0% direct test coverage** — the empty-vintage LEGACY fallback is silent (only `log.warning`, no warehouse marker). Operator who forgets `materialize-asof-features --write-features` ships calibrated outputs trained on revised data with no audit trail.

### HIGH

3. **SQLite warehouse has no WAL / no busy_timeout** (`storage.py:16`). Concurrent writers (CLI + API + cron) will hit `database is locked` errors immediately.
4. **`BonferroniMultiHorizonConformal.joint_coverage` join bug** (`multi_horizon_conformal.py:90-98`): `cqr.transform()` leaves original `q_lo`/`q_hi` columns un-renamed; pandas `rsuffix` collides with the renamed `q_lo_{h}` columns when joining horizon 2+. Reported joint coverage is wrong; no test.
5. **`prometheus_text` percentile destruction** (`observability.py:118-120`): emits N copies of the mean → all percentiles equal the mean. Production dashboards built on `/v1/metrics` will silently lie.
6. **`_TTLCache` is process-local *and* lock-free** (`api_v1.py:60-84`): with `--workers > 1`, cache hits are randomized; under load, OrderedDict races can corrupt the eviction order or skip TTLs.
7. **`/v1/metrics` and `/v1/health` are auth-free** (`api_v1.py:111-126`): `/v1/health` reads ALL release_gates rows per request and opens a fresh SQLite connection.
8. **DFM EM converges on the wrong likelihood** (`dfm.py:116`): the integrated covariance is non-diagonal but the LL treats columns as conditionally independent given the integrated factor.
9. **Stale recession labels are written to the warehouse before the gate fires** (`cli.py:147` then 156): even when `--max-stale-months` triggers `SystemExit(2)`, the warehouse already contains the stale rows.

### MEDIUM

10. **`_hash_frame` is dtype-fragile**: int→float promotion (a routine pandas behaviour when NaN appears) silently changes the reproducibility hash.
11. **`verify_run` dead bool-coercion** (`model_runs.py:248-249`) — minor but a bug-magnet on JSON round-trips.
12. **Hansen MCS docstring vs implementation mismatch** ("stationary" vs moving block) plus bootstrap-length bias when `n % block_size != 0`.
13. **`MondrianBinaryConformal` docstring contradicts implementation** and `astype(int)` silently zeros coverage on non-binary `y`.
14. **`apply_release_lag` only knows 8 series** — every other series gets zero lag.
15. **Latent SQL-injection surface in `Warehouse._write`** — current call sites are safe, but the `mode` parameter is a footgun.
16. **`_purge_and_embargo` is O(n²) in Python** — fine for monthly, painful for daily.
17. **`init_release_gates_severe_column` swallows ALTER TABLE errors** silently (`storage.py:471-472`).

### LOW

18. **DFM dead code** at `dfm.py:121` (`** 0 = 1`).
19. **`_git_revision`/`_git_dirty`** silently return `"unknown"`/`False` from any non-git CWD.
20. **Five `report_writer_v*.py`** modules all at 0% coverage — code-rot.
21. **`prometheus_text` rebuilds the entire registry on every request** — wasteful but correct.
22. **`pyproject.toml`'s `filterwarnings = ["error::DeprecationWarning:market_regime_engine"]`** doesn't escalate `log.warning`s — the legacy-mode warning is just a log line.

---

## Recommended next actions (ordered by leverage)

1. **Repair CI lint** (`pyproject.toml`, `cli.py` formatting). Either run `ruff check --fix --unsafe-fixes && ruff format src tests` and commit the diff, or weaken the ruff selection to what the codebase already passes. Either choice is fine — pretending the gate works is not.
2. **Add direct unit tests for `training_data.py`**. At minimum: (a) PIT mode with non-empty `feature_asof_values` returns the converted matrix; (b) PIT mode with empty `feature_asof_values` falls back to LEGACY *and* emits `audit["fallback_reason"]`; (c) LEGACY mode emits the deprecation warning. Then escalate the warning to `DeprecationWarning` so the existing `filterwarnings = ["error::DeprecationWarning:market_regime_engine"]` rule actually trips on legacy use.
3. **Persist the training audit dict** alongside the `ModelRun` row (e.g. into `metadata_json.training_audit`). Right now the audit only goes to stdout/log; promote it to a first-class envelope field so `verify-run` can detect a fallback after the fact.
4. **Configure SQLite for concurrency**: at the top of `Warehouse.__post_init__`, run `self.conn.execute("PRAGMA journal_mode=WAL")` and `self.conn.execute("PRAGMA busy_timeout=5000")`. Add a small write-with-retry helper for the `executemany` paths.
5. **Fix `BonferroniMultiHorizonConformal.joint_coverage`**: drop the unrenamed `q_lo`, `q_hi`, `y` columns before the rename, or join on the renamed columns explicitly. Add a regression test that joins three horizons with overlapping dates and asserts the column names + the expected joint-coverage value.
6. **Fix `prometheus_text`**: either compute proper buckets from the in-process histograms (snapshot retains `count` and `sum` only; switch the in-process histogram to retain a fixed-size reservoir or histogram buckets) or stop emitting fake Prometheus histograms and just emit `*_count` + `*_sum` counters. Document the limitation.
7. **Lock-protect `_TTLCache`** (use a `threading.Lock` for `set`/`pop`/`move_to_end`) and gate `/v1/metrics` (and possibly `/v1/health`) behind `require_api_key` if metrics are not meant to be public.
8. **Remove the `int → str` coercion in `_hash_frame`**: hash the dataframe via `pd.util.hash_pandas_object(df, index=True)` reduced through sha256, which is dtype-aware and faster.
9. **DFM**: delete the dead line `dfm.py:121`. Replace the diagonal LL with the correct Gaussian likelihood `-0.5 (d log 2π + log\|S\| + diff^T S^{-1} diff)` where `S = lambda lambda^T pp + diag(sigma_eps2)` (use Sherman-Morrison since rank-1 update). Or document the Watson-Engle approximation explicitly so reviewers don't read it as a bug.
10. **Reverse the order of "write recession labels" and "check staleness gate"** in `label_recessions_cmd` (`cli.py:147` ↔ `:156`). If the gate is going to fail, do not pollute the warehouse first.
11. **Hansen MCS**: either implement Politis-Romano stationary blocks (geometric block lengths) or update the docstring to "moving block bootstrap"; pad bootstrap samples to length `n` exactly.
12. **`MondrianBinaryConformal`**: rewrite the docstring to match the code; validate `y ∈ {0, 1}` at fit time and raise (or coerce + log) on non-binary input.
13. **Lock down `_write`'s `mode`** to a `Literal["REPLACE", "IGNORE"]` (or an Enum) so the latent SQL-injection sink can never grow up.
14. **Vectorise `_purge_and_embargo`** with `np.searchsorted` on a sorted `test_idx` to drop the inner Python `any()` loops.
15. **Delete or consolidate the five `report_writer_v*.py` modules** — pick one canonical writer and remove the rest.

---

## Appendix — empirical evidence captured during this review

```
$ pytest tests/ -q -m "not slow" --tb=line --maxfail=20
82 passed, 1 skipped, 1 deselected, 1 warning in 251.59s (0:04:11)

$ ruff check src tests --statistics
…
Found 416 errors.
[*] 188 fixable with the `--fix` option (37 hidden fixes can be enabled with the `--unsafe-fixes` option).
top filenames: cli.py=153, test_core.py=40, conformal.py=10, bocpd.py=9, training_data.py=9, …

$ ruff format --check src tests
73 files would be reformatted, 10 files already formatted

$ pytest tests/ -m "not slow" --cov=src/market_regime_engine --cov-report=term
TOTAL  6227 stmts  2645 missed  58% covered

$ mre model-run --db data/mre_audit_test.db --validation-dir data/validation_audit_test --purpose "second-opinion-test"
Wrote immutable model run rows: 1   (run_id f96d5b78dd17d4a7, code_dirty=true)

$ mre verify-run --db data/mre_audit_test.db
{"approved": true, "differences": {}, "lockfile_present": true, "missing_envelope": false, "run_id": "f96d5b78dd17d4a7"}
EXIT=0

$ # mutate requirements-lock.txt
$ mre verify-run --db data/mre_audit_test.db
{"approved": false, "differences": {"lockfile_hash": {"current": "0b53fe...", "stored": "d2a897..."}}, …}
EXIT=2

$ python -c "from market_regime_engine.model_runs import _hash_frame; ..."
df1 vs df2 (different index): True
NaN as float vs np.nan: True
int dtype vs float dtype: False
tiny float diff: False
```
