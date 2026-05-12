# v1.6.0 — Combined Engineering Plan

**Status:** Phase 1 cherry-pick branch landed on `pr-22-v1.6.0`; PR not yet opened.
**Base:** `main` @ `282212d` (post-v1.5.1)
**Scope:** MKTX integration (E1-MKTX ADX/Auto-X FIX + E4-MKTX Auto-X Reason payload + batch endpoint) **plus** PR #11 cherry-picks (adapters + online conformal + overfit control + production guard + validation packs).
**Effort estimate:** ~5–6 weeks engineering.
**Phasing:** Phase 1 (this branch) = cherry-pick + reconciliation; Phase 2 = MKTX integration; Phase 3 = PR-22 open / review / merge.

This document is the single point of reference for the v1.6.0 release. Sections A–F satisfy the planning spec in PR-22's task brief.

---

## Table of contents

- [A. Architecture diagrams](#a-architecture-diagrams)
  - [A.1 Data-flow — E1-MKTX regime/liquidity transition events](#a1-data-flow--e1-mktx-regimeliquidity-transition-events)
  - [A.2 Sequence — E4-MKTX Auto-X Reason payload mapping](#a2-sequence--e4-mktx-auto-x-reason-payload-mapping)
  - [A.3 Sequence — `/v1/execution_confidence/batch` fan-out](#a3-sequence--v1execution_confidencebatch-fan-out)
  - [A.4 Class diagram — absorbed `adapters/` package](#a4-class-diagram--absorbed-adapters-package)
  - [A.5 State machine — `production.py` fail-closed startup guards](#a5-state-machine--productionpy-fail-closed-startup-guards)
- [B. Failure-mode analysis](#b-failure-mode-analysis)
- [C. Test matrix](#c-test-matrix)
- [D. Acceptance criteria for v1.6.0 GA](#d-acceptance-criteria-for-v160-ga)
- [E. Sequencing plan](#e-sequencing-plan)
- [F. Dependencies on PR #11 disposition](#f-dependencies-on-pr-11-disposition)
- [Appendix — Files absorbed / skipped from PR #11](#appendix--files-absorbed--skipped-from-pr-11)

---

## A. Architecture diagrams

### A.1 Data-flow — E1-MKTX regime/liquidity transition events

**Goal:** Flow a regime or liquidity state-change from a warehouse write at decision tick through the Trade Engine SnapshotQueue to the Auto-X eligibility evaluator, with no engine-side enforcement (engine is publisher-only by design).

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │  Market Regime Engine (publisher)                                      │
 │                                                                        │
 │  ┌────────────────┐    ┌──────────────────────────────────────────┐    │
 │  │ Scorer tick    │    │ Warehouse                                │    │
 │  │  - credit       │   │   credit_regime_scores                   │    │
 │  │    regime       ├──►│   liquidity_stress_scores                │    │
 │  │  - liquidity    │   │   (DuckDB; PIT-safe)                     │    │
 │  │    stress       │   └─────────────┬────────────────────────────┘    │
 │  └────────────────┘                 │                                 │
 │                                     │ write-callback hook              │
 │                                     ▼                                 │
 │  ┌──────────────────────────────────────────────────────────────────┐ │
 │  │  fixed_income/event_publisher.py  (NEW in Phase 2)               │ │
 │  │   1. read previous row for (scope_id, score_kind)                │ │
 │  │   2. compute candidate label (5 buckets)                         │ │
 │  │   3. hysteresis check (N-tick threshold; reuses                  │ │
 │  │      fixed_income/hysteresis.py)                                 │ │
 │  │   4. if confirmed transition: build RegimeTransitionEvent        │ │
 │  │      (PIT-safe; HMAC-signed via evidence_common.hmac_sha256_hex) │ │
 │  └─────────────────────────────┬────────────────────────────────────┘ │
 │                                │                                       │
 │  ┌─────────────────────────────▼────────────────────────────────────┐ │
 │  │  pluggable transport                                             │ │
 │  │   - log + file       (Phase 2 default)                           │ │
 │  │   - SnapshotQueue    (Phase 2 ADX-FIX rider via simplefix)       │ │
 │  │   - retry queue → warehouse `pending_events` table on failure    │ │
 │  └─────────────────────────────┬────────────────────────────────────┘ │
 └────────────────────────────────┼───────────────────────────────────────┘
                                  │ regime / liquidity transitions
                                  │ (process boundary)
                                  ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │  MarketAxess Trade Engine                                              │
 │                                                                        │
 │  ┌────────────────────────────────────────────────────────────────┐    │
 │  │ SnapshotQueue (CG fanout)                                      │    │
 │  │   channel: regime_transitions                                  │    │
 │  │   channel: liquidity_transitions                               │    │
 │  └────────────────────────┬───────────────────────────────────────┘    │
 │                           │                                            │
 │  ┌────────────────────────▼───────────────────────────────────────┐    │
 │  │ Auto-X eligibility evaluator                                   │    │
 │  │   per affected scope_id (CUSIP):                               │    │
 │  │     re-evaluate autoXEligible / autoXReason                    │    │
 │  └────────────────────────┬───────────────────────────────────────┘    │
 │                           │                                            │
 │                           ▼                                            │
 │     Auto-X firings flip on/off according to new regime/liquidity       │
 │     state. ADX FIX session (Adaptive Auto-Ex pilot) consumes the same  │
 │     stream; the engine emits no orders.                                │
 └────────────────────────────────────────────────────────────────────────┘
```

**Failure isolation:** the warehouse write is the durable point-of-truth. If the transport publish fails, the event is persisted to `pending_events` (a new table in Phase 2) and replayed on the next tick. The Auto-X evaluator is therefore eventually consistent w.r.t. engine state; never out-of-sync silently.

---

### A.2 Sequence — E4-MKTX Auto-X Reason payload mapping

**Goal:** Map the `metadata.drivers[]` array on the `ExecutionConfidenceResponse` to the FMT-008 cell-renderer payload that the Auto-X UI consumes.

```
Trader      Trade Engine      ADX-FIX adapter    /v1/execution_confidence   Engine scorer
  │              │                  │                       │                      │
  │  RFQ create  │                  │                       │                      │
  ├─────────────►│                  │                       │                      │
  │              │ enrich w/ regime │                       │                      │
  │              │ snapshot         │                       │                      │
  │              ├─────────────────►│                       │                      │
  │              │                  │ POST {cusip, side,    │                      │
  │              │                  │       notional,       │                      │
  │              │                  │       protocol,       │                      │
  │              │                  │       request_id}     │                      │
  │              │                  ├──────────────────────►│                      │
  │              │                  │                       │ score_execution_     │
  │              │                  │                       │ confidence()         │
  │              │                  │                       ├─────────────────────►│
  │              │                  │                       │                      │ credit_regime
  │              │                  │                       │                      │ + liquidity
  │              │                  │                       │                      │ + logit blend
  │              │                  │                       │ ◄────────────────────┤
  │              │                  │                       │ ExecConfidence       │
  │              │                  │                       │ Response             │
  │              │                  │                       │   recommended_action │
  │              │                  │                       │   metadata.drivers[] │
  │              │                  │                       │   evidence_pack(HMAC)│
  │              │                  │ ◄─────────────────────┤                      │
  │              │                  │ ▼                                            │
  │              │                  │ fixed_income/auto_x_reason.py (NEW)          │
  │              │                  │   map drivers → FMT-008 payload:             │
  │              │                  │     {                                        │
  │              │                  │       reasonCode:    "REGIME_RISK_OFF",      │
  │              │                  │       reasonLabel:   "credit regime…",       │
  │              │                  │       reasonDetail:  {...},                  │
  │              │                  │       evidenceRef:   "<pack hash>",          │
  │              │                  │       cellRenderer:  "fmt-008"               │
  │              │                  │     }                                        │
  │              │                  │ ▼                                            │
  │              │ set autoXEligible│                                              │
  │              │ set autoXReason  │                                              │
  │              │ ◄────────────────┤                                              │
  │              │                  │                                              │
  │              │ fire / refuse    │                                              │
  │ ◄────────────┤ (per ADX rule)   │                                              │
  │              │                  │                                              │
  │              ├── fill / no-fill ──────────────────────────────────────────────►│
  │              │                                       outcomes table for E2 loop│
```

**Driver vocabulary (initial mapping):**

| Driver enum (engine-side)        | FMT-008 `reasonCode`      | UI cell tone |
|----------------------------------|---------------------------|--------------|
| `credit_regime_risk_off`         | `REGIME_RISK_OFF`         | red          |
| `credit_regime_risk_on`          | `REGIME_RISK_ON`          | green        |
| `liquidity_stress_high`          | `LIQUIDITY_STRESS_HIGH`   | red          |
| `liquidity_stress_low`           | `LIQUIDITY_STRESS_LOW`    | green        |
| `release_gate_hold`              | `RELEASE_GATE_HOLD`       | amber        |
| `evidence_unsigned`              | `EVIDENCE_UNSIGNED`       | grey         |

The full vocabulary is sealed once the XE team confirms the FMT-008 cell-renderer schema (decision D2 in §6 of the prior planning notes; blocks Phase 2 sub-PR).

---

### A.3 Sequence — `/v1/execution_confidence/batch` fan-out

**Goal:** Service a Lists / Portfolio-Trade (PT) request that arrives as one POST with N CUSIPs, fan out across CUSIPs, and return a single response that preserves request-order (clients re-key by `cusip`/`request_id` for display).

```
PT client       POST /v1/execution_confidence/batch     Engine scorer
   │                       │                                   │
   │  {requests: [ {cusip_1, side_1, notional_1, ...},          │
   │               {cusip_2, ...}, ..., {cusip_N, ...} ],      │
   │   batch_request_id: "..."}                                │
   ├──────────────────────►│                                   │
   │                       │ rate-limit check                  │
   │                       │   batch counter: +1               │
   │                       │   item counter:  +N               │
   │                       │   429 if either over limit        │
   │                       │                                   │
   │                       │ validate                          │
   │                       │   N ≤ MRE_FI_BATCH_MAX (default   │
   │                       │     500; env-overridable)         │
   │                       │   each item passes the same       │
   │                       │     schema as single endpoint     │
   │                       │                                   │
   │                       │ fan out across CUSIPs:            │
   │                       │   vectorized scorer call:         │
   │                       │     score_execution_confidence_   │
   │                       │     batch(requests) → DataFrame   │
   │                       │   single warehouse read (pooled)  │
   │                       │   single regime/liquidity         │
   │                       │     snapshot                      │
   │                       ├──────────────────────────────────►│
   │                       │                                   │ vectorized
   │                       │                                   │ logit blend
   │                       │                                   │ over N rows
   │                       │ ◄─────────────────────────────────┤
   │                       │ build N evidence packs            │
   │                       │   (single signing key load;       │
   │                       │    shared via evidence_common)    │
   │                       │ persist N pack rows in one INSERT │
   │                       │                                   │
   │                       │ assemble response:                │
   │                       │   {results: [                     │
   │                       │     {request_id, cusip,           │
   │                       │      recommended_action,          │
   │                       │      metadata: {drivers, ...},    │
   │                       │      evidence_pack: {...}},       │
   │                       │     ...                           │
   │                       │   ],                              │
   │                       │   batch_request_id, latencies}    │
   │ ◄─────────────────────┤                                   │
   │                       │                                   │
   │  optional streaming:  │                                   │
   │  Accept: application/jsonl                                │
   │  → one record per line, sent as engine produces them      │
```

**Performance contract:** p99 ≤ 200 ms at 100 CUSIPs (see acceptance criteria §D). Achieved by:
- single warehouse read per batch (vs N per-item reads)
- single regime/liquidity snapshot (vs N snapshots)
- vectorized logit (NumPy) over N rows
- bulk pack insert (one DuckDB transaction)
- shared HMAC signer (one key load per batch)

**Backpressure:** PT workflows can submit ~ once per minute per portfolio; the rate-limit defaults at 30 batches/min per API key + 5 000 items/min per key, env-overridable.

---

### A.4 Class diagram — absorbed `adapters/` package

```
                ┌──────────────────────────────────────────────────────┐
                │   adapters/core.py                                   │
                │                                                      │
                │   GOVERNED_SIGNAL_COLUMNS: tuple[str, ...]           │
                │     = 12 canonical columns                           │
                │                                                      │
                │   @dataclass(frozen=True)                            │
                │   class GovernedSignalExport:                        │
                │     path: str                                        │
                │     rows: int                                        │
                │     format: str                                      │
                │     columns: tuple[str, ...]                         │
                │                                                      │
                │   def normalize_governed_signals(frame) -> DataFrame │
                │   def assert_governed_signal_contract(frame) -> None │
                │   def parse_bool_series(values) -> Series            │
                │   def export_governed_signals(frame, path, *, fmt)   │
                │       -> GovernedSignalExport                        │
                └──────────────▲───────────▲──────────▲────────▲───────┘
                               │           │          │        │
                               │           │          │        │
        ┌──────────────────────┘           │          │        └──────────────────┐
        │                                  │          │                           │
┌───────┴───────────┐         ┌────────────┴───────┐  │     ┌────────────────────┴─┐
│ adapters/lean.py  │         │ adapters/vectorbt.py│  │     │ adapters/openbb.py    │
│                   │         │                    │  │     │                       │
│ to_lean_python    │         │ to_vectorbt_signal │  │     │ to_openbb_command_     │
│   _data(frame)    │         │  _series(frame)    │  │     │  payload(frame)       │
│ export_lean       │         │ export_vectorbt    │  │     │ export_openbb_signal  │
│   _python_data    │         │  _signal_series    │  │     │  (frame, out_path)    │
│   (frame, out)    │         │  (frame, out_path) │  │     │                       │
└───────────────────┘         └────────────────────┘  │     └───────────────────────┘
                                                       │
                                  ┌────────────────────┴────────────┐
                                  │ adapters/pyportfolioopt.py     │
                                  │                                │
                                  │ to_pyportfolioopt_views(frame) │
                                  │ export_pyportfolioopt_views    │
                                  │   (frame, out_path)            │
                                  └────────────────────────────────┘
```

**Contract:** every platform adapter MUST consume a 12-column governed-signal frame produced by `normalize_governed_signals(...)`, assert with `assert_governed_signal_contract(...)`, and produce a deterministic output for its target ecosystem. The 12 columns are the entire wire contract.

**Wire-format identifier:** every adapter stamps `metadata_json.adapter_contract = "governed_macro_regime_signal_v1"` so the consumer side can pin to a contract version. Bump to `_v2` only on a breaking column change.

---

### A.5 State machine — `production.py` fail-closed startup guards

`production.assert_production_ready()` is called at import time inside `api_v1.py`. It computes a `ProductionCheckResult` and raises `RuntimeError` if `ok=False`. The state machine for `MRE_ENV` and the consequent checks:

```
                            ┌─────────────────────────┐
   process imports api_v1   │                         │
   ─────────────────────────►│  env_name(MRE_ENV)     │
                            │                         │
                            └──────────┬──────────────┘
                                       │
                  ┌────────────────────┼──────────────────────┬───────────────────┐
                  │                    │                      │                   │
                  ▼                    ▼                      ▼                   ▼
        ┌─────────────────┐ ┌──────────────────┐ ┌────────────────────┐ ┌──────────────────────┐
        │ value in        │ │ value in         │ │ value not in       │ │ value = ""           │
        │ {"", "dev",     │ │ {"prod",         │ │ either set         │ │ (env unset)          │
        │  "development", │ │  "production"}   │ │                    │ │                      │
        │  "local",       │ │                  │ │                    │ │                      │
        │  "test",        │ │                  │ │                    │ │                      │
        │  "staging"}     │ │                  │ │                    │ │                      │
        └────────┬────────┘ └────────┬─────────┘ └─────────┬──────────┘ └──────────┬───────────┘
                 │                   │                     │                       │
                 ▼                   ▼                     ▼                       ▼
        ┌─────────────────┐ ┌──────────────────┐ ┌────────────────────┐ ┌──────────────────────┐
        │ production =    │ │ production =     │ │ warnings +=        │ │ production =         │
        │  False          │ │  True            │ │  unknown MRE_ENV   │ │  False               │
        │ no errors       │ │                  │ │  (treat non-prod)  │ │ no errors            │
        │ no warnings     │ │ check required   │ │                    │ │ no warnings          │
        │                 │ │ env vars ↓       │ │ production = False │ │                      │
        │ → return ok     │ │                  │ │ → return ok        │ │ → return ok          │
        └─────────────────┘ └────────┬─────────┘ └────────────────────┘ └──────────────────────┘
                                     │
                                     ▼
                            ┌───────────────────────────────────────────────────────────┐
                            │  Production guards (all must pass)                        │
                            │                                                           │
                            │   1. MRE_API_KEY set & non-empty                          │
                            │      ELSE error: "MRE_API_KEY is required in production"  │
                            │                                                           │
                            │   2. MRE_DB_PATH set & non-empty                          │
                            │      ELSE error: "MRE_DB_PATH is required in production"  │
                            │                                                           │
                            │   3. MRE_DB_PATH suffix == ".duckdb"                      │
                            │      ELSE warning: prefer .duckdb                         │
                            │                                                           │
                            │   4. MRE_LEGACY_API_ALLOW_UNAUTH != "1"                   │
                            │      OR MRE_ALLOW_LEGACY_API_IN_PRODUCTION == "1"        │
                            │      ELSE error: "legacy unauthenticated API is not       │
                            │      allowed in production"                               │
                            │                                                           │
                            │   5. MRE_CACHE_BACKEND=redis → MRE_REDIS_URL non-empty    │
                            │      ELSE error: "MRE_REDIS_URL is required when          │
                            │      MRE_CACHE_BACKEND=redis in production"               │
                            │                                                           │
                            └───────────────────┬───────────────────────────────────────┘
                                                │
                                ┌───────────────┴───────────────┐
                                │                               │
                                ▼                               ▼
                        ┌───────────────┐               ┌───────────────────┐
                        │  any errors?  │               │  no errors        │
                        │  → RAISE      │               │  → return         │
                        │    RuntimeError│              │    ProductionCheckResult
                        │    listing all │              │    (ok=True, errors=())
                        │    errors      │              │    warnings logged
                        └───────────────┘               └───────────────────┘
```

**Cache-backend rider (runtime):** `_build_cache_backend` re-queries `is_production_env()` per-call; if `MRE_ENV` is mutated post-import (e.g. by a misbehaving test fixture) and a Redis URL becomes unreachable, the production branch raises immediately rather than silently degrading to the local in-process cache (which would cause per-worker cache divergence under `uvicorn --workers`).

---

## B. Failure-mode analysis

Every component boundary, in the order traffic flows.

### B.1 Boundary: client → FastAPI app (`api_v1.app`)

| Failure | Detection | Mitigation |
|---|---|---|
| Missing `X-API-Key` in production | `require_api_key` → HTTP 401 | Already enforced; tests in `test_api_v1.py` |
| Production deploy boots without `MRE_API_KEY` env var | `assert_production_ready()` raises at import time | Module-level `_PRODUCTION_CHECK`; ensures the process exits 1 before serving traffic |
| Production deploy boots without `MRE_DB_PATH` env var | `select_api_db_path` raises | Same import-time guard |
| `MRE_CACHE_BACKEND=redis` but Redis unreachable | In prod: `_build_cache_backend` raises; in dev: warning + local fallback | Production-mode rider; tests in `test_v1_5_production.py` cover redis-required path |
| Body cap exceeded (32 KB) on `/v1/execution_confidence` | `MaxBodySizeMiddleware` → HTTP 413 | Already enforced (v1.5 PR-8 Tier-2 B-Ask-1) |
| Rate limit exceeded | `slowapi` → HTTP 429 with `Retry-After: 1` | Already enforced (v1.5 PR-5) |

### B.2 Boundary: FastAPI route → execution-confidence scorer

| Failure | Detection | Mitigation |
|---|---|---|
| Warehouse missing required tables (cold-start) | `Warehouse.read_*()` → empty DataFrame; scorer raises typed error | Single-CUSIP path returns 404; batch path returns per-item `error` |
| Concurrent DB write contention | DuckDB WAL serialization | Per-worker pooled `Warehouse` (v1.5 PR-5) |
| HMAC keys absent in production | `sign_pack` raises | Already enforced in `fixed_income/evidence_pack.py` (v1.5 PR-7) |
| `request_id` missing on production execution-confidence pack | `sign_pack` raises | Already enforced (v1.5.1 PR-9 FIX 3) |
| Evidence-pack envelope hash drift | `verify-run` raises | Already enforced (v1.5 PR-8 Tier-1 C-AUTO-1) |

### B.3 Boundary: `/v1/execution_confidence/batch` → vectorized scorer (Phase 2)

| Failure | Detection | Mitigation |
|---|---|---|
| Batch size > `MRE_FI_BATCH_MAX` | Request validation → HTTP 400 | Default cap 500; env-overridable |
| One CUSIP in the batch has malformed payload | Per-item error block in response | Batch never fails atomically; partial success is supported |
| Per-item rate-limit exhaustion | Per-batch counter trips first → HTTP 429 | Two counters (batch + item) so a batch of 500 small items doesn't starve a batch of 1 large item |
| Vectorized scorer numerical drift vs scalar | Property test in `test_execution_confidence_batch.py` asserts equality | Vectorized path falls back to scalar on assertion failure |
| Streaming response client disconnect | FastAPI `StreamingResponse` cancellation cleans up generator | Engine shuts down per-item evidence pack signer cleanly |

### B.4 Boundary: scorer → adapters

| Failure | Detection | Mitigation |
|---|---|---|
| Adapter consumes a non-canonical frame | `assert_governed_signal_contract` raises | All adapters call it before exporting |
| Adapter exports out-of-contract field (e.g. PII) | `normalize_governed_signals` drops fields not in the 12-column allow-list | Whitelist-by-construction |
| Boolean field is the string `"False"` | `parse_bool_series` raises on unknown literals | No silent `True` from `astype(bool)` |
| Date column unparseable | `assert_governed_signal_contract` raises | Caller fix-forward; engine never exports partial data |

### B.5 Boundary: adapters → external quant ecosystem

| Failure | Detection | Mitigation |
|---|---|---|
| Disk write fails (read-only volume) | `export_governed_signals` propagates `OSError` | Caller exit code; no partial files |
| Output format unsupported by adapter | `export_governed_signals` raises `ValueError` | Format inferred from extension; explicit `fmt=` argument overrides |
| Consumer pinned to wrong `adapter_contract` version | `metadata_json.adapter_contract` is stamped on every row | Consumer-side version check possible |

### B.6 Boundary: validation evidence pack → release-gate workflow

| Failure | Detection | Mitigation |
|---|---|---|
| Pack overwrite without `--force` | `build_evidence_pack` raises `FileExistsError` | Operator opts in to overwrite |
| Pack directory wipe of unsafe path (`/`, `$HOME`, repo root) | `_safe_rmtree_target` raises | Guarded list; tested in `test_v1_5_validation_pack.py` |
| Production pack built without HMAC key | `build_evidence_pack` raises when `require_signed=None` resolves to production | `MRE_EVIDENCE_HMAC_KEY` env var; fail-closed |
| Verify on a tampered pack | `verify_evidence_pack` differences include `manifest_hash`, per-file `sha256`, `size`, `extra:` (extra file), `hmac` | Exit code 2; CI hook |
| Verify on a signed pack with no key | `verify_evidence_pack` differences include `hmac: error` | Operator must provide key or pass `--require-signed` knowingly |

### B.7 Boundary: engine → MKTX Trade Engine (Phase 2)

| Failure | Detection | Mitigation |
|---|---|---|
| Warehouse INSERT OK but event publish fails | `event_publisher.publish()` returns failure; row persisted to `pending_events` | Replay on next tick; idempotency via `event_id` |
| SnapshotQueue back-pressure | Producer blocks or drops; engine sees timeout | Bounded buffer + circuit breaker → log + persist to `pending_events` |
| FIX session disconnect mid-publish | FIX adapter heartbeat detects | Reconnect + replay; engine emits ops alert |
| FIX reject (35=3) on malformed message | Adapter parses reject reason | Operator-visible alarm; engine continues publishing |
| TE consumer schema drift | TE-side schema-validator rejection visible in dashboards | Schema version stamped on every message |

### B.8 Boundary: Auto-X Reason payload (Phase 2)

| Failure | Detection | Mitigation |
|---|---|---|
| Unknown driver enum on response | `auto_x_reason.map_drivers()` raises | Engine never returns unknown drivers; CI lint over the enum |
| FMT-008 cell-renderer rejects payload | Contract test against MKTX-provided schema | Block PR-28 merge until schema is confirmed |
| Multiple drivers fire simultaneously | Mapping aggregates: highest-severity wins; tied → priority order in `auto_x_reason._DRIVER_PRIORITY` | Deterministic across runs |

---

## C. Test matrix

### C.1 Unit tests for each new module

| Module | Test file | Test count (Phase 1 absorbed; Phase 2 net new) | Critical assertions |
|---|---|---|---|
| `adapters/core.py` | `test_v1_5_adapters.py` (Phase 1) | 8 absorbed | Contract validation, boolean parsing, normalization happy + edge paths |
| `adapters/lean.py` | `test_v1_5_adapters.py` (Phase 1) | (subset) | LEAN PythonData stub parses round-trip |
| `adapters/vectorbt.py` | `test_v1_5_adapters.py` (Phase 1) | (subset) | Signal series shape + index preserved |
| `adapters/pyportfolioopt.py` | `test_v1_5_adapters.py` (Phase 1) | (subset) | Views payload schema |
| `adapters/openbb.py` | `test_v1_5_adapters.py` (Phase 1) | (subset) | Command payload schema |
| `frontier/online_conformal.py` | `test_v1_6_online_conformal.py` (Phase 1) | 5 absorbed | EnbPI quantile inflation, SAACI weight update, AgACI run |
| `frontier/overfit_control.py` | `test_v1_6_overfit_control.py` (Phase 1) | 5 absorbed | DSR + PBO numerics, MTRL formula, manifest hash |
| `production.py` | `test_v1_5_production.py` (Phase 1) | 4 absorbed | Env-var matrix; `select_api_db_path` raises in prod without var |
| `validation_pack.py` | `test_v1_5_validation_pack.py` (Phase 1) | 3 absorbed | Pack build/verify roundtrip; tamper detection; safe-rmtree guard |
| `evidence_common.py` | (covered by FI evidence-pack tests + validation-pack tests) | 0 new in Phase 1; planned `test_evidence_common.py` in Phase 2 | Canonical-JSON stability across modules; `hmac_sha256_hex` determinism |
| `fixed_income/event_publisher.py` | `test_event_publisher.py` (Phase 2) | ~20 new | Transition emission with/without hysteresis; idempotency; pending-events replay |
| `fixed_income/auto_x_reason.py` | `test_auto_x_reason.py` (Phase 2) | ~15 new | Driver → FMT-008 code mapping; tie-break ordering; unknown-driver guard |
| `fixed_income/adx_fix.py` | `test_adx_fix.py` (Phase 2) | ~25 new | FIX session lifecycle; reconnect; reject handling |

### C.2 Integration tests for E1, E4, batch endpoint (Phase 2)

| Suite | Target | Assertions |
|---|---|---|
| `test_e1_mktx_event_pipeline.py` | warehouse write → event publish → fake TE consumer | At-least-once delivery; replay after consumer outage; HMAC valid on every event |
| `test_e4_mktx_auto_x_reason.py` | `/v1/execution_confidence` → `/v1/auto-x/reason` round-trip | Drivers preserved; reason payload schema-valid; evidence-pack ref present |
| `test_batch_execution_confidence.py` | `POST /v1/execution_confidence/batch` happy + edge | Batch matches per-item scoring (golden parity); rate-limit semantics; chunked streaming; error in one item doesn't fail batch |

### C.3 Regression tests for cherry-picked modules

| Suite | Risk pinned |
|---|---|
| `test_fixed_income_evidence_pack.py` (existing) | FI pack canonical bytes unchanged after `evidence_common` refactor; HMAC v1/v2 unchanged |
| `test_fixed_income_hashing.py` (existing) | `canonical_json` / `canonical_sha256` exports still importable from `fixed_income.hashing` |
| `test_api_v1*.py` (existing) | No regression in cache, body cap, slowapi, correlation-id, lifespan |
| `test_v1_5_production.py` (Phase 1, new) | `validate_production_settings` env-var matrix; `select_api_db_path` raises in prod |

### C.4 Performance benchmarks

| Benchmark | Target | Measurement |
|---|---|---|
| Batch `/v1/execution_confidence/batch` p99 at 100 CUSIPs | ≤ 200 ms | `bench/bench_batch_endpoint.py` (Phase 2; uses `pytest-benchmark`); ≥ 1000 warm runs |
| Batch `/v1/execution_confidence/batch` p99 at 500 CUSIPs | ≤ 1 000 ms | Same harness; tests `MRE_FI_BATCH_MAX` upper limit |
| Validation pack build for ≤ 1 GB artifacts | ≤ 30 s | manual or scheduled CI bench |
| Adapter export (CSV, 10 000 rows) | ≤ 500 ms | property test asserts no quadratic scaling |

---

## D. Acceptance criteria for v1.6.0 GA

| Criterion | Target | How verified |
|---|---|---|
| **Test count** | ≥ 1100 from current 1035 baseline (Phase 1: 1035 → ~1060; Phase 2: ~1060 → ~1200) | `pytest --collect-only -q` |
| **mypy** | ≤ 14 errors (no regression from v1.5.1 baseline) | `mypy src/market_regime_engine` |
| **mypy (FI subpackage)** | clean | `mypy src/market_regime_engine/fixed_income` |
| **mypy (frontier subpackage)** | clean | `mypy src/market_regime_engine/frontier` |
| **ruff** | clean across `src/market_regime_engine` + `tests` | `ruff check src/market_regime_engine tests` |
| **Performance: batch p99 at 100 CUSIPs** | ≤ 200 ms | Bench in `bench/bench_batch_endpoint.py`, captured in PR-22 description |
| **HMAC v1 legacy compat** | a v1.5.0-signed pack still verifies | regression test in `test_fixed_income_evidence_pack.py` |
| **Validation pack roundtrip** | build + verify on a 100-file directory succeeds; tamper detection works | `test_v1_5_validation_pack.py` |
| **Adapter contract** | all 4 adapters consume a normalized frame + export deterministically | `test_v1_5_adapters.py` |
| **Production-mode fail-closed** | api_v1 import raises when `MRE_ENV=production` + `MRE_API_KEY` missing | `test_v1_5_production.py` + manual smoke test |
| **Docs** | this plan, `PRODUCT_IDENTITY.md`, `V1_5_GOVERNED_SIGNAL_LAYER.md`, `V1_6_FRONTIER_MATH_HARDENING.md` all present, dated, cross-referenced | review |
| **No regression** | every existing test still passes; legacy modules untouched | full `pytest -q` and `pytest -q -m slow` |

---

## E. Sequencing plan

### Phase 1 — Cherry-pick + reconciliation (this branch, `pr-22-v1.6.0`)

**Status: complete on this branch; PR not yet opened.**

Phase 1 absorbs the validated content from PR #11 onto a clean branch off `main`, reconciles the overlapping `api_v1.py` startup checks, and shares canonical-JSON + HMAC helpers between the two evidence-pack subpacks.

Deliverables:
1. `docs/V1_6_ENGINEERING_PLAN.md` (this file) + 3 absorbed docs.
2. `adapters/` package (core + 4 platform adapters), all 12-column-contract-compliant.
3. `frontier/online_conformal.py` (EnbPI / SAACI / AgACI).
4. `frontier/overfit_control.py` (DSR / PBO / MTRL / tournament manifest).
5. `production.py` fail-closed startup-guard module.
6. `validation_pack.py` + `validation_pack_cli.py` tamper-evident bundle builder.
7. `evidence_common.py` shared canonical-JSON + HMAC + git-revision helpers; `fixed_income/hashing.py` becomes a thin alias; `fixed_income/evidence_pack.py` + `validation_pack.py` import the shared primitives.
8. `api_v1.py` surgical edit: production-mode `MRE_API_KEY` / `MRE_DB_PATH` / Redis-URL guards added; PR-9 review fixes (body cap, lazy cache init, lifespan, correlation-id) preserved.
9. 25 new tests across the 5 absorbed modules.

Out-of-scope for Phase 1: any MKTX integration; no PR-22 yet.

### Phase 2 — MKTX integration (later, separate task)

**Status: scoped; not started in this task.**

Phase 2 lands the three MKTX integration capabilities on top of Phase 1's branch:

1. **E1-MKTX foundation** — `fixed_income/event_publisher.py` (new): regime/liquidity transition event publisher. Hooks `write_credit_regime_score` + `write_liquidity_stress_score`. Pluggable transport (log+file initial; SnapshotQueue producer next).
2. **Batch execution-confidence endpoint** — `fixed_income/api.py` extension + vectorized scorer in `execution_confidence.py`; `POST /v1/execution_confidence/batch` with rate-limit semantics (per-batch + per-item counters) and optional `application/jsonl` streaming response.
3. **E4-MKTX Auto-X Reason mapping** — `fixed_income/auto_x_reason.py` (new): map `metadata.drivers` → FMT-008 cell-renderer payload; `POST /v1/auto-x/reason` endpoint.
4. **E1-MKTX final** — `fixed_income/adx_fix.py` (new): ADX FIX adapter on top of `simplefix` for message encoding; hand-rolled session. SnapshotQueue producer for transition events. Integration test against a FIX-mock simulator.

Net new for Phase 2:
```
src/market_regime_engine/fixed_income/event_publisher.py          ~250 LoC
src/market_regime_engine/fixed_income/auto_x_reason.py            ~200 LoC
src/market_regime_engine/fixed_income/adx_fix.py                  ~500 LoC
src/market_regime_engine/fixed_income/api.py (extended)           ~150 LoC delta
src/market_regime_engine/fixed_income/execution_confidence.py     ~100 LoC delta
tests/test_event_publisher.py                                     ~20 tests
tests/test_auto_x_reason.py                                       ~15 tests
tests/test_adx_fix.py                                             ~25 tests
tests/test_execution_confidence_batch.py                          ~20 tests
tests/test_e1_mktx_event_pipeline.py    (integration)             ~10 tests
tests/test_e4_mktx_auto_x_reason.py     (integration)             ~8 tests
bench/bench_batch_endpoint.py                                     ~50 LoC
```

Effort estimate (Phase 2): 4–5 weeks engineering at 1 FTE.

Dependencies / unblockers needed before Phase 2 starts:
- **D1 — SnapshotQueue wire format.** Need a 30-min sync with the Trade Engine team to confirm whether existing TE messages are protobuf or JSON. Recommendation: match TE's existing format.
- **D2 — FMT-008 Auto-X Reason payload schema.** Need the XE team to share the cell-renderer schema doc; reverse-engineering is slower and more brittle. Blocks `auto_x_reason.py` final cut.
- **D4 — ADX FIX engine choice.** Recommendation: `simplefix` for encoding + hand-rolled session (~2 days), instead of `quickfix-py` (heavier dep) or fully hand-rolled (~3 days). User decision.

### Phase 3 — v1.6.0 PR opened, reviewed, merged

**Status: out of Phase 1 scope; ordered after Phase 2 completes.**

1. Bump `pyproject.toml` + `__init__.py` `__version__` to `1.6.0`.
2. Refresh `README.md` banner for v1.6.0.
3. Open PR-22 from `pr-22-v1.6.0` against `main`.
4. Code review + revisions.
5. Merge; tag `v1.6.0`; GitHub Release.
6. User-decision step: close PRs #10 and #11. Their content is fully absorbed by PR-22, so the close reasons are documented in PR-22's body.

---

## F. Dependencies on PR #11 disposition

This branch (`pr-22-v1.6.0`) is **independent** of PR #11. Specifically:

- The branch is created off `origin/main`, not off `origin/v1.6-frontier-hardening-rebased`.
- Files extracted from PR #11 are pulled via `git checkout origin/v1.6-frontier-hardening-rebased -- <path>` (file-level extraction), not via `git cherry-pick` (commit-level). The branch has no merge ancestry with PR #11.
- Every absorbed file is committed onto this branch as a fresh blob. If PR #11 is closed (with or without merge) the absorbed content remains intact on `pr-22-v1.6.0`.

**Therefore:**

- PR #10 (`v1.5-governed-signal-layer`) and PR #11 (`v1.6-frontier-hardening-rebased`) remain open and untouched on the remote during Phase 1.
- Phase 2 / Phase 3 do not need either PR to merge or close first.
- After PR-22 merges to `main` (Phase 3 outcome), the user can decide independently how to close PRs #10 and #11. Recommended: close as "absorbed into PR-22 (commit list in PR-22 body)" without merging, then delete the branches at user discretion.
- The two open PRs add no merge complexity to PR-22 because PR-22 targets `main`, not PR #11's branch.

If Phase 3 had to start while PR #11 was still being iterated, the only concern would be CI noise (PR #11's CI keeps churning); that has no functional impact on PR-22.

---

## Appendix — Files absorbed / skipped from PR #11

### Absorbed (10 commits on `pr-22-v1.6.0`):

| Commit | Files | Reason |
|---|---|---|
| 1 | `adapters/__init__.py`, `adapters/core.py` | New surface; 12-column governed-signal contract |
| 2 | `adapters/lean.py`, `vectorbt.py`, `pyportfolioopt.py`, `openbb.py` | Four platform adapters; depend on core |
| 3 | `frontier/online_conformal.py` | Complements existing batch `conformal_ts.py` |
| 4 | `frontier/overfit_control.py` | DSR / PBO / MTRL / tournament manifest |
| 5 | `production.py` | Fail-closed production-mode guard module |
| 6 | `validation_pack.py`, `validation_pack_cli.py` | Tamper-evident release-run bundle builder |
| 7 | `evidence_common.py` (new), updated `validation_pack.py`, `fixed_income/evidence_pack.py`, `fixed_income/hashing.py` | Refactor: share canonical-JSON + HMAC + git-revision helpers between the two evidence-pack subpacks. `fixed_income/hashing.py` becomes a thin re-export alias for back-compat. |
| 8 | `api_v1.py` (modified) | Production-mode `MRE_API_KEY` + `MRE_DB_PATH` startup guards. PR-9 review fixes (body cap, lazy cache init, OTel paths, DuckDB default path) preserved. |
| 9 | 5 absorbed test files (25 tests) | One per absorbed module |
| 10 | `docs/V1_6_ENGINEERING_PLAN.md` (this file) + 3 absorbed docs | `PRODUCT_IDENTITY.md`, `V1_5_GOVERNED_SIGNAL_LAYER.md`, `V1_6_FRONTIER_MATH_HARDENING.md` |

### Skipped (with reason):

| File | Skip reason |
|---|---|
| `.github/workflows/ci.yml` | PR #11's CI changes targeted a now-resolved billing/spending-limit issue; current `main` CI is correct. Phase 3 may layer its own bench-job on top. |
| `README.md` | Already refreshed for v1.5.1 in PR-9 FIX 6 (banner, test count, mypy baseline, commit table). v1.6.0 will append its own banner in Phase 3. |
| `pyproject.toml` version bump | Done as final commit of Phase 3, after MKTX work lands. |
| `__init__.py` `__version__ = "1.6.0"` | Same as above. |
| `cli_dispatch.py` | We did the CLI dispatch refactor differently in PR-7 review fixes; our version is canonical. |
| `models/__init__.py`, `models/legacy.py`, `models/linear_quantile.py` | We did the models-package reshuffle differently (top-level `models_legacy.py` + re-exporting `models/__init__.py`); ours is canonical. |
| `tests/test_package_metadata.py` | Tied to the obsolete package-metadata changes. |
| `frontier/__init__.py` line removing `data_cleaning` from `__all__` | This is a regression vs main (data_cleaning is real); we leave `__init__.py` as-is. New modules import via full path. |

---

*End of plan. Phase 1 deliverables landed on `pr-22-v1.6.0`; awaiting user review before Phase 2 kickoff.*
