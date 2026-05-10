# Product identity: Governed Macro Regime Signal Layer v1.5.0

## Narrowed identity

Market Regime Engine v1.5.0 is not a broker, backtester, execution engine, or portfolio optimizer.

It is a **governed macro regime signal layer**:

```text
point-in-time macro / market data
        ↓
regime + change-point + risk probability models
        ↓
calibration, coverage, drift, invalidation, release-gate checks
        ↓
governed signal contract
        ↓
LEAN / vectorbt / PyPortfolioOpt / OpenBB / dashboards / internal APIs
```

The engine's job is to answer:

1. What regime does the evidence support?
2. How uncertain is that regime estimate?
3. Is the model output still valid under PIT, drift, coverage, and release-gate controls?
4. What auditable signal record should downstream systems consume?

The engine should **not** decide orders by itself. Strategy systems consume the governed signal and remain responsible for portfolio construction, sizing, routing, and execution.

## Canonical signal contract

Adapters export this column set:

| Column | Meaning |
|---|---|
| `date` | As-of date for the governed signal |
| `regime_state` | Decoded or posterior-modal regime |
| `regime_confidence` | Model confidence in the regime state, 0..1 |
| `change_point_prob` | Change-point probability, 0..1 |
| `drawdown_prob` | Drawdown risk probability, 0..1 |
| `recession_prob` | Recession probability, 0..1 |
| `confidence_score` | Governance confidence score, 0..1 |
| `release_gate_decision` | `release`, `hold`, `unknown`, etc. |
| `release_gate_approved` | Boolean gate state |
| `model_run_id` | Immutable model run ID when available |
| `artifact_hash` | Reproducibility envelope hash when available |
| `metadata_json` | Adapter/run metadata |

## Adapter posture

Adapters are intentionally conservative:

- LEAN adapter emits custom-data CSV + a BaseData stub, not orders.
- vectorbt adapter emits entry/exit boolean series derived from governed states, not a full strategy.
- PyPortfolioOpt adapter adjusts expected-return inputs transparently and refuses risk-on tilt when release gates fail.
- OpenBB adapter emits JSON/OBBject-like records for provider-extension or dashboard consumption.

## Production-mode contract

When `MRE_ENV=production`:

- `MRE_API_KEY` is required.
- `MRE_DB_PATH` is required.
- unauthenticated legacy API is blocked unless explicitly overridden.
- Redis cache misconfiguration fails closed instead of falling back silently.

Local and dev mode remain ergonomic. Production mode is deliberately irritating because markets are already doing enough damage without software helping.

## Evidence-pack contract

Empirical validation packs are built as immutable-ish evidence bundles:

- selected validation artifacts are copied into `artifacts/`
- every file receives SHA-256 hash + size metadata
- canonical `manifest.json` is written
- `manifest.sha256` hashes the manifest
- optional `manifest.hmac.sha256` signs the manifest using `MRE_EVIDENCE_HMAC_KEY`

This does not prove a model is good. It proves the evidence pack was not quietly edited after the fact.
