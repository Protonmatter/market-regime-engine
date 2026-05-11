# v1.5 Fixed-Income RCIE / X-Pro Auto-X Adapter

This document tracks the Fixed-Income (FI) extension landed in
`v1.5`. The adapter sits alongside the existing macro/regime engine
and is delivered as the `fixed_income/` subpackage plus a small
number of cross-cutting fixes (NaN policy, trading-day calendar,
UTC timestamps) that benefit the macro side too.

Per the FI v1.5 plan (`.cursor/plans/fi_v1.5_implementation_plan_bcda9355.plan.md`),
the work ships in 7 sequential PRs on top of `v1.4.1`:

| PR | Scope | Status |
|----|------|--------|
| PR-1 | Scaffolding & contracts | shipped |
| PR-2 | Warehouse (13 FI tables + register_tables + bond_reference temporal versioning) | shipped |
| PR-3 | Credit spread regime model | shipped |
| PR-4 | Liquidity stress model | shipped |
| PR-5 | Execution confidence | shipped |
| PR-6 | TCA segmentation | shipped |
| PR-7 | Evidence-pack hardening + report | pending |

## PR-1 surface (already shipped)

- Frozen data contracts: `CreditRegimeOutput`, `LiquidityStressOutput`,
  `ExecutionConfidenceRequest`, `ExecutionConfidenceResponse`,
  `FixedIncomeEvidencePack`.
- Label enums: `RegimeLabel`, `LiquidityLabel`,
  `ExecutionRecommendation` (string Enums).
- Helpers: `assert_pit_safe`, `canonical_sha256`, `regime_label_from_score`,
  `liquidity_label_from_score`.
- Posterior wrappers: `FilteredPosterior`, `SmoothedPosterior`,
  `require_filtered` (real-time decisioning rail).
- Placeholder API router (501 stubs) and CLI surface (7 `fi-*`
  commands; stubs emit `not_yet_implemented`).
- Sub-daily PIT grid (`asof._resolve_asof_grid(freq=...)`),
  `MRE_BUILD_SHA` / `MRE_BUILD_DIRTY` env overrides on
  `model_runs._git_revision`, 5-lockfile hash dict in
  `model_runs._lockfile_hash`, `api_v1._db_path()` default flipped to
  `data/mre.duckdb`, release-gate empty-coverage / missing-decision
  fixes (AF-1 / AF-6 / AF-9 / AF-13 / AF-14 / ASK-7 / ASK-12).

## PR-2 surface (already shipped)

- 13 FI warehouse tables (`bond_reference`, `trace_trades`,
  `rfq_events`, `dealer_quotes`, `dealer_response_stats`,
  `curve_snapshots`, `cds_curve_snapshots`, `credit_regime_scores`,
  `liquidity_stress_scores`, `execution_confidence_predictions`,
  `execution_outcomes`, `tca_regime_segments`,
  `fixed_income_evidence_packs`) registered via
  `storage.register_tables`.
- `bond_reference` temporal versioning (`valid_from` / `valid_to`),
  `read_bond_reference_asof`, `read_bond_reference_history`.
- DuckDB + SQLite parity; per-table indexes for the hot reads.
- `bulk_load_chunked` for 100M-row TRACE imports without OOM.
- `migrate_warehouse` table-name SQL-injection guard.
- `httpx` pinned in `[dev]` for FastAPI `TestClient`.

## PR-3 surface (this release)

The PR-3 scope is the **credit spread regime model**: an explainable
deterministic composite scorer producing 0–100 index + label +
confidence + drivers + governance triple, persisted to the
`credit_regime_scores` table and served on `GET /v1/regime_index/latest`.

### 1. Per-column NaN policy (`frontier.data_cleaning`)

Historical cleaner (`frame.replace([inf, -inf], NaN).ffill().fillna(0.0)`)
became a per-column policy:

| Policy | Semantics |
|---|---|
| `NAN_TO_ZERO` | back-compat default (matches legacy cleaner bit-for-bit) |
| `NAN_TO_LAST_VALID` | forward-fill only; no zero seed |
| `NAN_DROPS_ROW` | drop rows with NaN in any drop-policy column |
| `NAN_FAILS_PIT_AUDIT` | raise `PitAuditFailure` so `release_gate=False` |

`bocpd.py`, `frontier/bayesian_msvar.py`, and `frontier/gp_cpd.py`
now route through `clean_with_policy`. Each public `score()` accepts
optional `nan_policy=` and `column_policies=` parameters. The
default `NAN_TO_ZERO` preserves v1.4 numerics; FI feature builders
override to `NAN_FAILS_PIT_AUDIT` so a missing CUSIP-level feature
trips `release_gate=False` rather than silently zero-filling.

### 2. FI trading-day calendar (`fixed_income.calendars`)

- `TradingCalendar` enum (`SIFMA_BOND` default, `NYSE_BOND`,
  `FEDERAL`).
- Helpers: `is_trading_day`, `next_trading_day`,
  `previous_trading_day`, `trading_days_between`,
  `assert_trading_day` (raises `PitViolationError` on closures).
- Hand-curated YAML at `data/calendars/sifma_bond.yaml` covers
  2020–2030 (federal holidays + SIFMA early closes; per
  [SIFMA US Treasury Holiday Schedule](https://www.sifma.org/resources/general/holiday-schedule/)).
- Optional `pandas_market_calendars` adapter behind the
  `[fixed_income]` extra, gated on `MRE_FI_USE_PMC=1`.
- Cache refresh via `MRE_FI_CALENDAR_REFRESH=1` or
  `reset_calendar_cache()`.

### 3. UTC timestamp enforcement (`fixed_income.timestamps`)

- `to_utc(ts)` — rejects naive datetimes/strings at the FI boundary;
  converts aware timestamps to UTC; `None` passes through.
- `assert_utc(ts, label=...)` — strict write-path invariant.
- `iso8601_z(ts)` — ISO-8601 with the explicit `Z` suffix; rejects
  non-UTC inputs.

FI feature builders, the scorer, and the API serialiser all route
through these helpers so the storage convention (`Z` suffix) and the
PIT contract (no naive timestamps) are uniform across the FI surface.

### 4. Composite scorer (`fixed_income.credit_spread_regime`)

```python
from market_regime_engine.fixed_income import (
    build_credit_features,
    score_credit_regime,
    write_credit_regime_score,
)
from market_regime_engine.storage import Warehouse

wh = Warehouse("data/mre.duckdb")
asof = pd.Timestamp("2026-05-08T16:00:00+00:00")
features = build_credit_features(wh, asof, lookback_days=504)
output = score_credit_regime(features, asof=asof, profile="production")
write_credit_regime_score(wh, output)
```

**Inputs (`build_credit_features`)** — reads `curve_snapshots`
(Treasury & swap level/slope/curvature), `cds_curve_snapshots`
(CDX.IG/CDX.HY 5Y), and `vintage_observations` for VIX, MOVE, and
ETF premium/discount. Output is long-form
`["date", "feature_name", "value", "source_timestamp", "vintage_date"]`
within the `(asof - lookback_days, asof]` window. Every row passes
the PIT rail (`assert_pit_safe`); curve / CDS rows on closed
trading days raise `PitViolationError`.

**Composite design** — five components, each 0–100 where higher =
more risk-off:

| Component | Sub-features | Normalisation |
|---|---|---|
| `treasury_curve` | `ust_slope` (10Y-2Y), `ust_curvature` (2·5Y-2Y-10Y) | percentile (slope inverted) + z-sigmoid on \|curvature - mean\| |
| `spreads` | `cdx_ig_5y` | rolling percentile vs 2y window |
| `cds` | `cdx_hy_5y` | rolling percentile vs 2y window |
| `volatility` | `vix`, `move` | z-score sigmoid (50 = z=0) |
| `etf_dislocation` | `etf_prem_disc` | rolling percentile of \|prem/disc\| |

**Default weights** (sum to 1.0; callers can override via `weights={...}`):

| Component | Weight |
|---|---:|
| `treasury_curve` | 0.15 |
| `spreads` | 0.30 |
| `cds` | 0.25 |
| `volatility` | 0.20 |
| `etf_dislocation` | 0.10 |

**Confidence** — `1.0 - (fraction_of_components_with_missing_input_data)`,
capped at `0.5` when `release_gate=False` (AGENT.md non-negotiable 8).

**Drivers** — top-2 components by `|score - 50.0|` (most-extreme
deviation from the neutral midline). Ties break in component
declaration order.

**Artifact hash** — `canonical_sha256({timestamp, regime_score,
regime_label, confidence, drivers, component_scores})` per the
v1.5 hashing rules. `model_run_id` is intentionally NOT part of the
hash so a re-run with the same data + weights produces a stable
hash.

**Output bucket → label** (matches PR-1
`RegimeLabel.regime_label_from_score`):

| Score | Label |
|---:|---|
| 0–20 | Risk-On / Compression |
| 20–40 | Normal Liquidity |
| 40–60 | Watch / Transition |
| 60–80 | Risk-Off / High Risk Aversion |
| 80–100 | Crisis / Severe Dislocation |

### 5. API: `GET /v1/regime_index/latest`

Mounted on `api_v1.app` via `fixed_income.api.build_router`:

- **200** with the full `CreditRegimeOutput` JSON when at least one
  row exists. Rows with `release_gate=false` are returned verbatim so
  consumers can fail closed downstream.
- **503** `{"detail": "no_data", "release_gate": false}` when the
  `credit_regime_scores` table is empty.
- The other 5 FI endpoints (`liquidity_index/latest`,
  `liquidity_index/{scope_type}/{scope_id}`, `execution_confidence`,
  `tca/regime-segments/latest`, `evidence-pack/{model_run_id}`)
  remain `501 not_yet_implemented` until their PR ships.

Sample 200 response (AGENT.md §6.1):

```json
{
  "timestamp": "2026-05-10T16:00:00Z",
  "regime_score": 78.3,
  "regime_label": "Risk-Off / High Risk Aversion",
  "confidence": 0.82,
  "drivers": ["spreads", "cds"],
  "component_scores": {"treasury_curve": 55.4, "spreads": 88.1, "cds": 85.9, "volatility": 62.0, "etf_dislocation": 50.0},
  "model_run_id": "credit_regime-production-<uuid>",
  "release_gate": true,
  "artifact_hash": "sha256:..."
}
```

### 6. CLI: `mre fi-score-credit-regime`

```text
mre fi-score-credit-regime \
    --db data/mre.duckdb \
    --asof 2026-05-08T16:00:00Z \
    --profile production \
    --release-gate true \
    [--model-run-id <id>] \
    [--lookback-days 504] \
    [--output-json data/regime.json]
```

Workflow: open warehouse → `build_credit_features` → `score_credit_regime`
→ `write_credit_regime_score` → print envelope to stdout → optional
JSON write. Exit code `0` on clean run (including `release_gate=false`);
`2` on PIT violation, PIT-audit failure, or naive `--asof`.

### 7. `merge_asof` tolerance constants

`fixed_income.feature_builders` exports:

```python
DEFAULT_INTRADAY_MERGE_TOLERANCE = pd.Timedelta("5min")
DEFAULT_EOD_MERGE_TOLERANCE = pd.Timedelta("1D")
```

Downstream FI builders that join across cadences (PR-4 / PR-5)
must pass an explicit `tolerance=` per `pd.merge_asof` call; the
default 5-min constant is the right choice for intraday quote↔trade
joins, and the 1-day constant covers end-of-day curve / vol joins.

### Back-compat guarantees

- The NaN-policy default `NAN_TO_ZERO` is bit-for-bit equivalent to
  the legacy cleaner. Existing `tests/test_bocpd*.py`,
  `tests/test_bayesian_msvar.py`, `tests/test_v1_2_frontier.py`, and
  golden traces pass unchanged.
- The FI router is mounted on `api_v1.app` at import time; the macro
  routes (`/v1/regime/latest`, `/v1/model-outputs/latest`, etc.) are
  unchanged.
- The `fi-*` CLI surface is the same 7 commands shipped in PR-1; the
  PR-3 commit upgrades only `fi-score-credit-regime` to a real
  workflow. The other 6 remain stubs.

### Reference

- Plan: `.cursor/plans/fi_v1.5_implementation_plan_bcda9355.plan.md` §3 PR-3
- Markdown pack: `markdown-pack/MRE_FIXED_INCOME_AGENT.md` "PR 3"
- Markdown pack: `markdown-pack/MRE_FIXED_INCOME_INSTRUCTIONS.md` §6.1
- Review threads: §3.2 ASK-5, §3.4 Q-7 / Q-8 / Q-9, §3.1 AF-8, §3.3 F-9

## PR-4 surface (this release)

The PR-4 scope is the **liquidity stress model**: a scope-aware
deterministic composite scorer that emits a 0-100 `liquidity_index`
(higher = more stress) for `market` / `sector` / `rating` / `cusip`
scopes, with optional asymmetric hysteresis on the resulting label
and an opt-in hierarchical Bayesian back-stop for sparsely-traded
cusips.

### 1. Liquidity-stress composite scorer (`fixed_income.liquidity_stress`)

```python
from market_regime_engine.fixed_income import (
    build_liquidity_features,
    score_liquidity_stress,
    write_liquidity_stress_score,
)
from market_regime_engine.storage import Warehouse

wh = Warehouse("data/mre.duckdb")
asof = pd.Timestamp("2026-05-08T16:00:00+00:00")
features = build_liquidity_features(
    wh, asof,
    scope_type="market", scope_id="ALL",
    lookback_days=30,
)
out = score_liquidity_stress(
    features,
    scope_type="market", scope_id="ALL",
    asof=asof,
    profile="production",
)
write_liquidity_stress_score(wh, out)
```

**Inputs (`build_liquidity_features`)** — reads `trace_trades`,
`rfq_events`, `dealer_quotes`, and (for sector / rating scopes) the
survivorship-safe `read_bond_reference_asof`. Output is the standard
long-form
`["date", "feature_name", "value", "source_timestamp", "vintage_date"]`
frame carrying the eleven AGENT.md liquidity features
(`bid_ask_width`, `trade_count_velocity`, `volume_over_adv`,
`time_since_last_trade`, `dealers_requested`, `quotes_received`,
`quote_dispersion`, `amihud_illiquidity`, `dealer_response_count`,
`axe_freshness_proxy`, `order_imbalance`).

**Component scores** (each 0-100, higher = more stress):

| Component | Sub-features | Normalisation |
|---|---|---|
| `quotes_dispersion` | `quote_dispersion` | z-score sigmoid vs trailing window |
| `bid_ask` | `bid_ask_width` | rolling percentile |
| `trade_velocity` | `trade_count_velocity` | inverse percentile (low velocity → high stress) |
| `rfq_fill_rate` | `quotes_received`, `dealers_requested` | `(1 - quotes_received/dealers_requested) * 100` |
| `amihud` | `amihud_illiquidity` | rolling percentile |
| `time_gap` | `time_since_last_trade` | `min(100, minutes * 2)` |

**Default weights** (sum to 1.0; override via `weights={...}`):

| Component | Weight |
|---|---:|
| `quotes_dispersion` | 0.20 |
| `bid_ask` | 0.20 |
| `trade_velocity` | 0.15 |
| `rfq_fill_rate` | 0.20 |
| `amihud` | 0.15 |
| `time_gap` | 0.10 |

**Confidence / drivers / artifact hash / bucket labels** mirror the
PR-3 credit-regime contract exactly so the API + evidence-pack code
can serve both signals from one envelope shape.

**Output bucket → label** (matches PR-1
`LiquidityLabel.liquidity_label_from_score`):

| Score | Label |
|---:|---|
| 0-20 | Normal |
| 20-40 | Mild Stress |
| 40-60 | Elevated Stress |
| 60-80 | Severe Stress |
| 80-100 | Crisis Liquidity |

### 2. Label hysteresis (credit + liquidity)

Asymmetric `(enter, exit)` bands per label so the regime / liquidity
labels stop flipping on every tick when the score oscillates near a
bucket boundary. Cold start (`prev_label=None`) falls through to the
sharp-bucket mapping, preserving the PR-3 contract bit-for-bit.

Liquidity bands (`HYSTERESIS_BANDS_LIQUIDITY`):

| Label | Enter | Exit |
|---|---:|---:|
| Normal | — | 25 |
| Mild Stress | 20 | 45 |
| Elevated Stress | 40 | 65 |
| Severe Stress | 60 | 85 |
| Crisis Liquidity | 80 | — |

Credit bands (`HYSTERESIS_BANDS_CREDIT`) follow the same shape over
`RegimeLabel`. `score_credit_regime` and `score_liquidity_stress`
accept an optional `prev_label=` parameter; when supplied, the new
label is the output of `classify_with_hysteresis(score, prev_label)`.
Metadata records `hysteresis_applied` / `prev_label` for telemetry.

### 3. Hierarchical Bayesian liquidity model

`frontier.hierarchical_liquidity.HierarchicalLiquidityModel` is the
opt-in Bayesian back-stop per the v1.5 deep-research report §2 (OFR
latent-liquidity-states). Model::

    y_ij ~ Normal(mu_i, sigma_obs)
    mu_i = mu_global + group_effect[s_i, r_i] + cusip_effect[i]

Partial pooling across `(sector, rating)` plus a cusip-level random
effect lets a sparsely-traded cusip inherit its tier's posterior
rather than collapse to neutral. Inference is NumPyro NUTS; the
`[bayesian]` extra (`numpyro`, `jax[cpu]`, `arviz`) is required —
`fit()` raises a clean `ImportError` with the install hint when the
extras are absent.

`predict(cusip=..., sector=..., rating=...)` returns a posterior
summary dict (`posterior_mean`, `ci_low_5`, `ci_high_95`, `n_obs`,
`hierarchy_level`) with explicit back-off through cusip →
`(sector, rating)` → rating → market. Per AGENT.md non-negotiable
"explainable baselines first", the deterministic composite remains
the production primary; the hierarchical scorer is opt-in via
`mre fi-score-liquidity --use-hierarchical` and activates in
production only after PR-7 validation.

### 4. GP-BOCPD ring buffer (ASK-9)

`frontier/gp_cpd._GPRun` switched from a Python list (copy-on-append)
to a fixed-size `np.ndarray` ring buffer of `max_run` slots. Bit-for-
bit equivalent to v1.4 for any panel processed by `GPBOCPD.score`
(the BOCPD inner loop never appends past `max_run` on the same
segment), with per-update cost dropping from O(n × d) to
O(max_run × d). The posterior trace is pinned by sha256 in
`tests/test_gp_cpd_ring_buffer.py`.

### 5. API surface

| Endpoint | Status |
|---|---|
| `GET /v1/regime_index/latest` | PR-3 |
| `GET /v1/liquidity_index/latest` | **PR-4** |
| `GET /v1/liquidity_index/{scope_type}/{scope_id}` | **PR-4** |
| `POST /v1/execution_confidence` | 501 (PR-5) |
| `GET /v1/tca/regime-segments/latest` | 501 (PR-6) |
| `GET /v1/evidence-pack/{model_run_id}` | 501 (PR-7) |

Both PR-4 liquidity endpoints return the full `LiquidityStressOutput`
JSON (release_gate passthrough) on 200, `404` with
`{"detail": "invalid_scope_type", ...}` when the by-scope endpoint
receives an unknown scope, and `503` `{"detail": "no_data",
"release_gate": false}` when no row exists for the requested scope.

### 6. CLI: `mre fi-score-liquidity`

```text
mre fi-score-liquidity \
    --db data/mre.duckdb \
    --scope-type {market|sector|rating|cusip} \
    --scope-id ALL \
    --asof 2026-05-08T16:00:00Z \
    --profile production \
    --release-gate true \
    [--model-run-id <id>] \
    [--lookback-days 30] \
    [--use-hierarchical] \
    [--prev-label-from-warehouse true] \
    [--output-json data/liquidity.json]
```

Workflow: open warehouse → optional `latest_liquidity_stress_score`
for the previous label → `build_liquidity_features` →
`score_liquidity_stress` (with `prev_label` for hysteresis) →
`write_liquidity_stress_score` → print envelope to stdout → optional
JSON write. Exit code `0` on clean run, `2` on PIT violation / audit
failure / naive `--asof`.

### Back-compat guarantees (PR-4)

- `score_credit_regime(...)` without `prev_label=` is bit-for-bit
  identical to PR-3 (same artifact hash, same label, same metadata
  shape except for the additive `hysteresis_applied=False,
  prev_label=None` keys).
- GP-BOCPD posterior outputs are bit-for-bit unchanged for any panel
  driven through `GPBOCPD.score` (pinned by sha256 in the ring-buffer
  test).
- Default `NanPolicy.NAN_TO_ZERO` paths in `bocpd` / MSVAR / GP-CPD
  continue to match v1.4.
- The PR-3 `test_other_fi_endpoints_still_return_501` test is updated
  to drop the now-live `/v1/liquidity_index/*` paths from its 501-
  stub list. The remaining stubs are
  `POST /v1/execution_confidence`, `/v1/tca/regime-segments/latest`,
  and `/v1/evidence-pack/{model_run_id}`.

### Reference (PR-4)

- Plan: `.cursor/plans/fi_v1.5_implementation_plan_bcda9355.plan.md` §4 PR-4
- Markdown pack: `markdown-pack/MRE_FIXED_INCOME_AGENT.md` "PR 4 — liquidity stress model"
- Markdown pack: `markdown-pack/MRE_FIXED_INCOME_INSTRUCTIONS.md` §6.2
- Review thread: §3.2 ASK-9 (GP-BOCPD ring buffer)
- Deep-research report: §2 (hierarchical Bayesian liquidity)

## PR-5 surface (this release)

The PR-5 scope is the **bond-level execution confidence model** plus
seven cross-cutting engine improvements the FI scale workload needs.

### 1. Execution-confidence deterministic logistic baseline

`fixed_income/execution_confidence.score_execution_confidence` blends
the latest credit-regime + cusip-scoped liquidity-stress signals
(falling back to market scope when the cusip-specific row is missing)
with order attributes through a closed-form logit:

| Component | Magnitude | Notes |
|---|---:|---|
| `base_intercept` | +0.5 | conservative prior |
| `liquidity_penalty` | -0.01 × liquidity_index | higher stress → lower confidence |
| `notional_penalty` | -0.15 × max(0, log10(notional) − 6) | scales above $1M |
| `regime_penalty` | -0.008 × regime_score | higher credit stress → lower confidence |
| `protocol_bonus` | Auto-X +0.10, RFQ +0.05, Manual −0.10 | |
| `urgency_penalty` | low 0, normal −0.05, high −0.15 | |
| `rating_bonus` | IG +0.10, HY −0.10 | |
| `limit_distance_penalty` | -0.05 × max(0, \|limit − mid\| − 10) bps | only when caller supplies `metadata.mid_price` |

`confidence_score = sigmoid(sum(...))` clipped to `[0.05, 0.95]`.
`expected_slippage_bps = 5 + 30·(1 − confidence) + 0.5·liquidity_index`,
floored at 1 bps and capped at 200 bps. Confidence interval is the
heuristic `[score − 0.10, score + 0.10]`; v1.5.1 swaps this for a
calibrated quantile output. The metadata blob records the top-3 logit
components by absolute magnitude as the explainability surface.

The baseline is intentionally conservative (per AGENT.md "explainable
baselines first") so AUTO_X_ALLOWED is effectively reserved for the
v1.5.1 calibrated successor; the deterministic path tops out near the
AUTO_X_CAUTION band on a clean signal.

**Decision rule** (INSTRUCTIONS.md §6.3):

```text
release_gate=False
    → "Manual review required" + human_review_required=True
score ≥ 0.80 AND liquidity NOT IN {Severe Stress, Crisis Liquidity}
    → "Auto-X allowed"
score ≥ 0.60
    → "Auto-X caution / trader confirm"
otherwise
    → "Manual review required"
```

**Stale-signal soft-fail.** When either the credit-regime or
liquidity feed is older than `MRE_FI_MAX_SIGNAL_STALENESS_SEC`
(default 900s = 15 minutes), the scorer returns
`recommended_action="Unavailable — stale signal"` with
`release_gate=False` without raising. Both
`signal_age_seconds_credit_regime` and `signal_age_seconds_liquidity`
remain in the metadata blob for telemetry.

### 2. POST /v1/execution_confidence

| Endpoint | Status |
|---|---|
| `GET /v1/regime_index/latest` | PR-3 |
| `GET /v1/liquidity_index/latest` | PR-4 |
| `GET /v1/liquidity_index/{scope_type}/{scope_id}` | PR-4 |
| `POST /v1/execution_confidence` | **PR-5** |
| `GET /v1/tca/regime-segments/latest` | 501 (PR-6) |
| `GET /v1/evidence-pack/{model_run_id}` | 501 (PR-7) |

The POST endpoint accepts an `ExecutionConfidenceRequestModel`
(Pydantic v2 with `extra="forbid"`), validates ISO-8601 UTC timestamps
and alphanumeric CUSIPs, caps the body at 32 KB (413 on oversize),
and is rate-limited per API key via `slowapi`
(`MRE_FI_EXEC_CONF_RATE_LIMIT`, default `100/second`; 429 carries
`Retry-After: 1`). The handler reads through the per-process pooled
warehouse (§5 below), runs `score_execution_confidence`, and persists
the prediction keyed by `request_id` (PR-15 composite PK).

**Load smoke** (`tests/test_api_v1_load_smoke.py`, marked `slow`): 1000
sequential POSTs across 10 cusips with varied protocol / urgency /
notional measured **p50 = 27.57ms, p99 = 40.05ms** on the dev laptop —
well under the 500ms budget the plan specifies.

### 3. CLI: `mre fi-score-execution-confidence`

```text
mre fi-score-execution-confidence \
    --db data/mre.duckdb \
    --input examples/sample_order.json \
    [--output-json data/exec_conf.json] \
    [--profile production] \
    [--release-gate true] \
    [--request-id <id>] \
    [--model-run-id <id>]
```

The JSON input is validated by the same Pydantic model the API uses,
so the CLI surfaces the same naive-timestamp / oversized-notional /
non-alphanumeric-cusip errors with exit code 2. `--request-id` is
auto-generated as a UUID4 hex when omitted so the warehouse row
always has a value on the PR-15 composite PK.

### 4. Walk-forward improvements (ASK-1 / AF-11 / ASK-13)

- **ASK-1: searchsorted purge** —
  `purge_and_embargo_searchsorted(train_idx, test_idx, horizon, embargo)`
  replaces the v1.3 dense `(n_train, n_test)` bool mask with a
  `np.searchsorted` form. Memory bound goes from
  `O(n_train · n_test)` (~1 GB transient on n=2000 / n_blocks=8 / k=2)
  to `O(n_train + n_test)`. Bit-for-bit equivalent to the legacy path
  — pinned by a 50-seed parity test against the preserved
  `_legacy_purge_and_embargo` reference.
- **AF-11: `min_train_after_purge`** —
  `PurgedWalkForward.min_train_after_purge` makes the hard-coded
  `min_train // 2` skip threshold explicit. `None` (default) preserves
  v1.4 behaviour bit-for-bit; supplying an integer makes the rail
  tunable and emits an INFO log when a fold is skipped.
- **ASK-13: model-class factory** — `evaluate_walk_forward` accepts
  `model_class=` / `model_kwargs=` and instantiates a fresh estimator
  inside every fold via `_model_factory_default`. The closure-capture
  `predict_fn` path remains for back-compat; the docstring documents
  why the new class-based form prevents cross-fold state leakage.

### 5. Cross-cutting engine improvements

- **ASK-8 — Per-process pooled Warehouse**.
  `storage.get_pooled_warehouse(path)` returns the per-process
  `Warehouse` keyed by resolved absolute path. The FastAPI hot path no
  longer pays DuckDB catalog + WAL teardown per request. Writes
  serialise via `pooled_warehouse_write_lock` (re-entrant); reads run
  concurrently under DuckDB MVCC. The pool is drained on the FastAPI
  lifespan-shutdown event so a uvicorn reload cycle does not leak
  file handles.
- **AF-5 / ASK-10 — Lazy cache init + Redis JSON**.
  `_CACHE` is no longer constructed at module import time;
  `_get_cache()` lazily constructs on first use and `reset_cache()`
  lets operators pick up env-var changes between requests. The Redis
  cache defaults to JSON (`json.dumps(default=str)`); pickle is gated
  behind the opt-in env var `MRE_CACHE_ALLOW_PICKLE=1`. Closes the
  CVE-style attack surface where an attacker with write access to the
  shared Redis instance could land arbitrary-code execution via
  `pickle.loads` on the FastAPI worker.
- **AF-3 — Bounded observability histograms**.
  `observability.BoundedHistogram` (reservoir sampler, Algorithm R,
  4096-slot default) replaces the unbounded `list[float]` per
  metric key. Exact `count` and `sum` are preserved; approximate
  quantiles match the true sample quantile within a few percentage
  points on a stationary uniform distribution. Resident memory stays
  bounded at 32 KB per histogram on a 1M-insert hot path.
- **AF-10 — Cadence-strict horizon parsing**.
  `_parse_horizon_periods(horizon, cadence=...)` requires an explicit
  cadence (monthly / daily / intraday) and rejects mismatched
  suffixes — so `"12m"` with `cadence="daily"` now raises
  `ValueError` instead of silently mis-parsing as 12 months. The
  legacy `_parse_horizon_months` shim preserves v1.4 macro-backtest
  behaviour with a DeprecationWarning.

### 6. Validation primitives (Q-5 / deep research §"Validation & metrics")

Three Bailey–López de Prado primitives ship in
`market_regime_engine.validation`:

- **Deflated Sharpe Ratio** (`deflated_sharpe`). Adjusts the observed
  Sharpe for multiple-testing selection bias across `n_trials`
  candidate strategies and non-normality of returns (sample
  skew / excess kurtosis or operator-supplied values). Returns the
  probability that the true Sharpe exceeds `sharpe_target`.
- **Probability of Backtest Overfitting**
  (`probability_of_backtest_overfitting`). Combinatorial-purged-CV
  per BBLZ 2017: splits the time axis into `n_partitions` halves,
  ranks strategies in-sample, and measures how often the in-sample
  winner ranks below median out-of-sample.
- **Minimum Track Record Length**
  (`minimum_track_record_length`). Closed-form `n*` per BLP 2014
  eq. (8) so operators can plan how long a strategy must run before
  its Sharpe edge becomes defensible.

All three primitives are scipy-free (Beasley–Springer–Moro normal
inverse-CDF inlined) so the macro engine does not pick up a new hard
dependency.

**Release-gate integration.** The `production` profile gains:

- `min_dsr=0.5` — DSR ≥ 0.5 required to release.
- `max_pbo=0.05` — PBO ≤ 5% required to release.

Both columns are *optional* on the `confidence` frame; absence skips
the rail entirely so v1.4.1 callers without DSR / PBO emit the same
release decision. The `default` profile leaves both `None`.

### 7. RNG seed namespace contract (Q-3 / Q-11)

`ReproEnvelope.rng_seeds` carries the canonical namespace dict. The
class-level docstring documents the recommended keys
(`numpy`, `jax`, `torch`, `sklearn`) and the registered namespaces
per component (FI execution_confidence baseline → empty; v1.5.1
calibrated successor → `{numpy, sklearn}`; Bayesian MS-VAR →
`{numpy, jax}`; PatchTST → `{numpy, torch}`).

### Back-compat guarantees (PR-5)

- The pre-PR-5 `CombinatorialPurgedCV._purge_and_embargo` routes
  through the new `purge_and_embargo_searchsorted` and emits
  bit-for-bit identical outputs (pinned by a 50-seed parity test).
- `PurgedWalkForward(min_train_after_purge=None)` preserves the
  v1.4 fold count exactly.
- `_parse_horizon_months(...)` keeps working as a deprecation shim
  (DeprecationWarning fired, fold semantics unchanged).
- `Warehouse(path)` direct construction still works (pooled access
  is opt-in via `get_pooled_warehouse`).
- The macro release gate behaviour is unchanged when the
  `confidence` frame lacks `dsr` / `pbo` columns.
- `_CACHE` is now lazily constructed; callers that previously
  reached for `api_v1._CACHE` directly should switch to
  `_get_cache()`.

### Reference (PR-5)

- Plan: `.cursor/plans/fi_v1.5_implementation_plan_bcda9355.plan.md` §5 PR-5
- Markdown pack: `markdown-pack/MRE_FIXED_INCOME_AGENT.md` "PR 5 — execution confidence"
- Markdown pack: `markdown-pack/MRE_FIXED_INCOME_INSTRUCTIONS.md` §6.3
- Review thread: §3.1 AF-3 / AF-5 / AF-10 / AF-11; §3.2 ASK-1 / ASK-8 / ASK-10 / ASK-13; §3.4 Q-3 / Q-5 / Q-11; §3.6 PR-1 / PR-2 / PR-13
- Deep-research report: §3 (execution confidence); §"Validation & metrics" (DSR / PBO / MTRL)

## PR-6 surface (this release)

The PR-6 scope is **regime-aware TCA segmentation**: tag every trade
with the prevailing credit-regime / liquidity / execution-confidence
context, compute the 11 TCA metrics in `Decimal` precision, and
aggregate by the 9 spec-canonical segmentation dimensions to the
`tca_regime_segments` warehouse table.

### TCA Segmentation

#### Public surface

- `fixed_income.tag_trade_with_regime_context(trade, *, warehouse,
  use_hysteresis=True, tolerance=5min)` — attaches the regime label
  (hard + soft-weights), liquidity label, execution-confidence
  bucket, and the four trade-attribute buckets (sector / rating /
  maturity / notional) to a single `TradeRecord`. PIT-safe: every
  context read uses `asof <= trade.timestamp`; `assert_pit_safe`
  raises on a post-decision context row.
- `fixed_income.aggregate_tca_by_regime(trades, *, dimensions,
  metrics_names=TCA_METRICS, soft_weighting=False)` — long-form
  group-by. Returns one row per `(dim-combo, metric_name)` with
  `metric_value` and `sample_count`. Soft weighting (PR-6 §A.1)
  splits one trade across multiple regime labels by
  `TaggedTrade.regime_soft_weights`; the default (`False`) uses the
  hard label.
- `fixed_income.compute_tca_metrics_for_outcome(request, response,
  outcome, *, warehouse, asof_now=None)` — per-trade TCA metrics
  in `Decimal` precision. Strict-inequality outcome-lag rail (PR-10);
  markout windows use the SIFMA bond trading-day calendar (PR-3);
  unobservable / missing-data metrics return `None`.
- `fixed_income.compute_execution_success_label(request, outcome, *,
  success_threshold_bps=None)` — binary execution-success label per
  INSTRUCTIONS.md §6.4 (slippage strictly < threshold). The 25-bps
  default is overridable via `MRE_FI_TCA_SUCCESS_THRESHOLD_BPS`.
- `fixed_income.materialize_tca_segments_for_day(warehouse, *, date,
  soft_weighting=False, use_hysteresis=True, model_run_id=None)` —
  end-of-day driver. Reads `execution_outcomes` for the day, joins to
  `execution_confidence_predictions`, tags every trade, aggregates over
  the canonical dim-combos in `_dimension_combinations()`, and writes
  one row per `(combo, metric)`.
- `fixed_income.write_tca_regime_segment` /
  `fixed_income.latest_tca_regime_segments` — `TcaRegimeSegment`
  read/write via the warehouse. Missing dimensions persist as the
  `"__all__"` sentinel so the composite PK is stable across runs that
  aggregate over different dim subsets.

#### TCA metric catalogue

The 11 metrics in `TCA_METRICS` (AGENT.md §"PR 6" + INSTRUCTIONS.md
§6.4):

| Metric | Formula / source |
|---|---|
| `arrival_cost_bps` | `sign * (execution - arrival) / arrival * 10_000` |
| `vwap_slippage_bps` | `sign * (execution - vwap) / vwap * 10_000` |
| `price_improvement_bps` | buy: `(best_ask - execution) / mid * 10_000`; sell: symmetric |
| `market_impact_bps` | `sign * (execution - mid_at_arrival) / mid * 10_000` |
| `time_to_fill_seconds` | passthrough from the outcome row |
| `dealer_response_count` | passthrough from the outcome row |
| `quote_quality` | bounded ratio: `dealer_response_count / (dealer_response_count + expected_slippage_bps)` |
| `protocol_success` | `1.0` when `filled_quantity / notional >= 0.95`, else `0.0` |
| `post_trade_markout_1d_bps` | trading-day window; `None` when window not closed |
| `post_trade_markout_5d_bps` | trading-day window; `None` when window not closed |
| `execution_success` | binary; PR-10 strict-lag rail |

#### Segmentation dimensions

The 9 dimensions in `DIMENSION_COLUMNS` (INSTRUCTIONS.md §6.4):
`regime_label`, `liquidity_label`, `execution_confidence_bucket`,
`protocol`, `side`, `sector`, `rating`, `maturity_bucket`,
`notional_bucket`. Bucket boundaries:

- Maturity: `0-2y`, `2-5y`, `5-10y`, `10y+`.
- Notional: `<1M`, `1-5M`, `5-25M`, `25M+`.
- Execution confidence: `low` (`< 0.60`), `medium` (`< 0.80`),
  `high` (`<= 1.0`), `unavailable` (no prediction logged).
- Soft regime weights: triangular weighting between the two adjacent
  regime-score bucket centres at `{10, 30, 50, 70, 90}`; saturates at
  the endpoints. Sums to 1.0.

#### Decimal-precision arithmetic (Q-6)

`fixed_income/bps_precision.py` ships the Decimal helpers used
throughout the TCA pipeline:

- `TCA_PRECISION_CONTEXT` = 28 digits, `ROUND_HALF_EVEN` (banker's
  rounding to avoid cumulative bias).
- `to_bps(price_diff, reference)` — Decimal bps via
  `(diff / reference) * 10_000`; raises on zero reference.
- `bps_arithmetic_mean(values, weights=None)` — weighted Decimal mean.
- `bps_aggregate_sum(values)` — Decimal sum (for daily-volume aggregates).
- `decimal_to_float_for_report(d)` — the *only* place float conversion
  happens; called at the warehouse write / JSON response edge.

Acceptance gate ($1B daily × 0.5 bps over 100k synthetic trades):
Decimal aggregate error `< 1e-9 bps`; the informational regression
test contrasts the naive float64 aggregate at the same scale.

#### Outcome observation lag (Q-2 / PR-10)

`fixed_income/tca_outcome_lag.py` ships the canonical guard:

- `assert_outcome_after_decision(*, decision_timestamp, observed_at,
  label="outcome")` — strict inequality
  `observed_at > decision_timestamp` (a same-nanosecond report is a
  clock-drift artefact, not a real outcome). Returns the UTC-normalised
  timestamps so callers don't re-coerce.
- The storage writer (`Warehouse.write_execution_outcome` — PR-5) and
  `compute_tca_metrics_for_outcome` both route through the guard.
  Future labels (`post_trade_directional_pnl_1d`, etc.) inherit the
  rail by calling the helper.
- Markout windows use `fixed_income.next_trading_day` from PR-3 so a
  Friday trade's 1-day markout closes on Monday. When the window has
  not yet closed, the metric returns `None` and the segment row
  records `sample_count=0` for that metric.

#### NaN propagation (PR-11)

`aggregate_tca_by_regime` drops NaN rows at the aggregation boundary
(per-metric) so a single bad price never poisons the bucket mean. The
`fi_tca_dropped_rows_total` counter labelled by `metric` plus the
active grouping dimensions is emitted on every drop; the counter
family pre-registers at module load so a Prometheus scrape immediately
after import returns the family with a zero baseline rather than 404.

#### API

- `GET /v1/tca/regime-segments/latest?dimensions=&limit=` returns
  `{segments: [...], count: N}` from `tca_regime_segments`. 503 with
  `{"detail": "no_data"}` when empty; 400 on unknown dimension names.

#### CLI

- `mre fi-tca-segment --db <path> --date YYYY-MM-DD
  [--dimensions ...] [--soft-weighting] [--use-hysteresis true|false]
  [--model-run-id ...] [--output-json ...]` materialises segments for
  the target day. Defaults to the previous SIFMA bond trading day per
  PR-3 calendars. Exit codes: 0 on clean run; 2 on input validation /
  PIT failure.

#### Tests

- `tests/test_tca_decimal_precision.py` — Decimal helpers + $1B-scale
  acceptance.
- `tests/test_tca_outcome_observation_lag.py` — strict inequality at
  writer + helper boundaries.
- `tests/test_tca_label_construction_guard.py` — PR-10 guard semantics.
- `tests/test_tca_segmentation.py` — tag + aggregate + materialize
  acceptance.
- `tests/test_tca_segments_by_regime_and_liquidity.py` — 25-bucket
  AGENT.md catalog test.
- `tests/test_tca_nan_propagation.py` — drop + counter behaviour.
- `tests/test_tca_api_endpoint.py` — `GET /v1/tca/regime-segments/latest`.
- `tests/test_tca_segmentation_cli.py` — `mre fi-tca-segment`.

### Reference (PR-6)

- Plan: `.cursor/plans/fi_v1.5_implementation_plan_bcda9355.plan.md` §6 PR-6
- Markdown pack: `markdown-pack/MRE_FIXED_INCOME_AGENT.md` "PR 6 — TCA segmentation"
- Markdown pack: `markdown-pack/MRE_FIXED_INCOME_INSTRUCTIONS.md` §6.4
- Review thread: §3.4 Q-2 (outcome observation lag) / Q-6 (Decimal precision); §3.6 PR-10 / PR-11
- Deep-research report: §4 (Regime-Aware TCA Segmentation)
