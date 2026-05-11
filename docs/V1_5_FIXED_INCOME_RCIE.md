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
| PR-5 | Execution confidence | pending |
| PR-6 | TCA segmentation | pending |
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
