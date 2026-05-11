# `execution_confidence.py` — Execution Confidence Baseline

## Purpose

Deterministic logistic baseline for the
`POST /v1/execution_confidence` endpoint. Returns confidence score,
expected slippage, recommended action, and the fail-closed
governance triple. Powered by `score_execution_confidence(...)`.

## Inputs

`ExecutionConfidenceRequest` (frozen dataclass) — see `schemas.py`.
The Pydantic-validated `ExecutionConfidenceRequestModel` in
`api.py` is the API boundary.

## Outputs

`ExecutionConfidenceResponse` carrying `confidence_score` (0–1),
`expected_slippage_bps`, `confidence_interval_low` / `_high`,
`recommended_action` (one of: `"Auto-X allowed"`, `"Auto-X
caution / trader confirm"`, `"Manual review required"`,
`"Unavailable — governance gate failed"`, `"Unavailable — stale
signal"`), `human_review_required` bool, plus the governance
triple. `metadata` exposes
`signal_age_seconds_credit_regime`,
`signal_age_seconds_liquidity`,
`max_signal_age_seconds`,
`max_signal_staleness_threshold_seconds`,
`release_gate_input`.

## Validation rules

1. **Fail-closed (non-negotiable 8)**: `release_gate=False` ⇒
   `recommended_action="Manual review required"` AND
   `human_review_required=True`.
2. **Stale signal**: when either credit-regime or liquidity feed is
   older than `MRE_FI_MAX_SIGNAL_STALENESS_SEC` (default 900s), the
   scorer soft-fails with
   `recommended_action="Unavailable — stale signal"` +
   `release_gate=False`.
3. **PIT**: every signal read uses `asof <= request.timestamp`.

## Decision rule

```
if release_gate is False:
    "Manual review required"
elif confidence_score >= 0.80 and liquidity_label NOT IN
        {"Severe Stress", "Crisis Liquidity"}:
    "Auto-X allowed"
elif confidence_score >= 0.60:
    "Auto-X caution / trader confirm"
else:
    "Manual review required"
```

## References

- AGENT.md PR-5 + INSTRUCTIONS.md §6.3.
- `docs/V1_5_AUTOX_CONTRACT.md`.
