# v1.5 X-Pro / Auto-X Adapter Contract

Versioned contract for X-Pro and Auto-X consumers integrating with
the v1.5 Fixed-Income Execution Confidence service. This document is
**the authoritative spec** for the request / response schemas, error
semantics, retry rules, idempotency, rate limits, stale-signal
handling, and fail-closed behaviour.

## 1. Endpoint

```
POST /v1/execution_confidence
```

Hosted on the FastAPI app at `market_regime_engine.api_v1:app`.
Authentication is gated by `MRE_API_KEY` (when set, every endpoint
except `/v1/health` requires `X-API-Key`).

### 1.1 Request body (`ExecutionConfidenceRequestModel`)

Pydantic v2 model; all fields validated at the FastAPI boundary.

| Field | Type | Required | Constraint |
|------|------|---------|------------|
| `timestamp` | string | yes | ISO-8601 with explicit tz info (`Z` or offset) |
| `cusip` | string | yes | alphanumeric, 8–12 chars |
| `side` | `"buy"` \| `"sell"` | yes | enum |
| `notional` | float | yes | `0 < notional ≤ 5e8` |
| `protocol` | `"Auto-X"` \| `"RFQ"` \| `"Manual"` | yes | enum |
| `limit_price` | float \| null | no | `> 0` when set |
| `urgency` | `"low"` \| `"normal"` \| `"high"` | no | default `"normal"` |
| `request_id` | string | yes | 1–128 chars; idempotency key |
| `sector`, `rating`, `maturity_bucket`, `client_request_id` | string \| null | no | informational |
| `metadata` | object \| null | no | passes through to the response |

Body cap: 32 KB (header `Content-Length` enforced; oversized requests
return 413).

### 1.2 Response body (`ExecutionConfidenceResponse`)

```json
{
  "timestamp": "2026-05-08T16:30:00Z",
  "cusip": "AAA111111",
  "side": "buy",
  "notional": 1000000.0,
  "protocol": "Auto-X",
  "confidence_score": 0.85,
  "expected_slippage_bps": 7.5,
  "confidence_interval_low": 0.75,
  "confidence_interval_high": 0.95,
  "recommended_action": "Auto-X allowed",
  "human_review_required": false,
  "model_run_id": "exec-conf-20260508T163000Z-...",
  "release_gate": true,
  "artifact_hash": "sha256:...",
  "metadata": {
    "signal_age_seconds_credit_regime": 45.0,
    "signal_age_seconds_liquidity": 60.0,
    "max_signal_age_seconds": 60.0,
    "max_signal_staleness_threshold_seconds": 900.0,
    "release_gate_input": true
  }
}
```

`recommended_action` is one of:

- `"Auto-X allowed"` (`confidence_score >= 0.80` AND liquidity not in
  `{"Severe Stress", "Crisis Liquidity"}`)
- `"Auto-X caution / trader confirm"` (`confidence_score >= 0.60`)
- `"Manual review required"` (any of: release gate failed, stale
  signal, low confidence, severe liquidity)

## 2. Error semantics

| HTTP | Body | Meaning | Retry? |
|------|------|---------|--------|
| 200 | full response | success (including fail-closed payloads where `release_gate=false`) | n/a |
| 400 | `{"detail":"pit_violation",...}` | client supplied a future-dated timestamp | NO |
| 401 | `{"detail":"invalid or missing X-API-Key"}` | auth gate (when `MRE_API_KEY` is set) | NO |
| 413 | `{"detail":"request body exceeds 32 KB cap","limit_bytes":32768}` | body too large | NO |
| 422 | Pydantic validation envelope | bad request shape | NO |
| 429 | `{"detail":"rate limit exceeded: ..."}` (header `Retry-After: 1`) | per-API-key rate limit (default 100 req/s) | YES — back off per `Retry-After` |
| 503 | `{"detail":"no_data","release_gate":false}` | upstream FI signals not yet computed | YES — exponential backoff |

## 3. Retry behaviour

- **429**: honour `Retry-After` exactly; exponential backoff is
  unnecessary because the rate limit window resets per second.
- **503**: exponential backoff starting at 5 seconds, capped at 60
  seconds. Continue indefinitely; the engine will return data once
  the FI signal pipeline has produced its first row.
- **Network / 5xx**: 3 retries with jittered exponential backoff
  (1, 2, 4 seconds). Beyond that, surface the failure to the human
  operator — there is no silent fallback; consumers MUST fail closed.

## 4. Idempotency

`request_id` is the **idempotency key**. The
`execution_confidence_predictions` table has a composite primary key
on `(request_id, timestamp)`; submitting the same `request_id` twice
returns the same response without re-scoring.

## 5. Rate limit

Default 100 req/s per `X-API-Key` (anonymous callers share one
bucket). Override via `MRE_FI_EXEC_CONF_RATE_LIMIT` (slowapi spec
string, e.g. `"500/second"`). 429 carries `Retry-After: 1`.

## 6. Stale signal handling

Each request computes `signal_age_seconds_credit_regime` and
`signal_age_seconds_liquidity` against the request `timestamp`. If
either exceeds `MRE_FI_MAX_SIGNAL_STALENESS_SEC` (default 900s),
the response soft-fails:

```json
{
  ...
  "release_gate": false,
  "recommended_action": "Manual review required",
  "human_review_required": true,
  "metadata": {
    "max_signal_age_seconds": 1500.0,
    "max_signal_staleness_threshold_seconds": 900.0,
    "stale_signal": true,
    ...
  }
}
```

Consumers MUST check `release_gate` and refuse Auto-X when it is
`false`. Stale-signal soft-fails do not raise; the consumer treats
them as fail-closed.

PR-7 §N also exposes `metadata.signal_age_seconds` on every other
FI endpoint (`/v1/regime_index/latest`,
`/v1/liquidity_index/*`) so a consumer can pre-flight the SLA
without invoking the execution-confidence endpoint.

## 7. Fail-closed contract

Per AGENT.md non-negotiable 8 + INSTRUCTIONS.md §10 governance rule
3: when `release_gate=false`, `recommended_action` is **always**
`"Manual review required"` and `human_review_required` is `true`.
Auto-X consumers MUST treat this as a hard stop and route to a
human trader. There is no soft-fallback, and no "best-effort" Auto-X
path.

## 8. Versioning + breaking-change contract

- The `/v1/execution_confidence` path is stable for the v1.x line.
- Field-level additions (e.g. new `metadata.*` keys) are
  non-breaking; consumers MUST tolerate unknown keys.
- Field-level removals or type changes will land on `/v2/...` and
  be announced via `docs/V1_5_BREAKING_CHANGES.md` (or its v2
  successor).
- The Pydantic model uses `extra="forbid"` — undefined request
  fields trigger 422. Client libraries must align before any field
  rename.

## 9. Reference

- Plan: `.cursor/plans/fi_v1.5_implementation_plan_bcda9355.plan.md`
  §5 PR-5 + §7 PR-7 §N.
- Markdown pack: `markdown-pack/MRE_FIXED_INCOME_INSTRUCTIONS.md`
  §6.3.
- Schema: `src/market_regime_engine/fixed_income/schemas.py`
  (`ExecutionConfidenceRequest`, `ExecutionConfidenceResponse`).
- Pydantic model: `src/market_regime_engine/fixed_income/api.py`
  (`ExecutionConfidenceRequestModel`).
- Tests:
  `tests/test_execution_confidence.py`,
  `tests/test_execution_confidence_release_gate_false_fails_closed.py`,
  `tests/test_api_v1_pydantic_validation.py`,
  `tests/test_signal_age_seconds_in_all_fi_responses.py`.
