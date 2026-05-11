# `market_regime_engine.fixed_income`

The Fixed-Income RCIE / X-Pro Auto-X adapter. v1.5 introduces seven
deterministic baseline modules, the warehouse schema for 13 FI
tables, the FI v1 API router, the seven `mre fi-*` CLI commands, the
HMAC-signed evidence pack, and the FI report writer.

## Module map

| Module | Scope |
|--------|-------|
| `schemas.py` | Frozen data contracts + label enums |
| `schema.py` | Warehouse table specs (13 tables) |
| `hashing.py` | Canonical JSON + SHA-256 helper |
| `pit_guard.py` | Point-in-time validation + trading-day assert |
| `posterior_mode.py` | `Filtered` / `Smoothed` posterior wrappers |
| `timestamps.py` | UTC enforcement (`to_utc`, `iso8601_z`) |
| `calendars.py` | SIFMA bond + NYSE calendar helpers |
| `feature_builders.py` | Credit / liquidity / execution-confidence feature pipelines |
| `credit_spread_regime.py` | Deterministic credit-regime composite scorer (PR-3) — see `docs/components/credit_spread_regime.md` |
| `liquidity_stress.py` | Per-scope liquidity-stress scorer (PR-4) — see `docs/components/liquidity_stress.md` |
| `execution_confidence.py` | Logistic execution-confidence baseline (PR-5) — see `docs/components/execution_confidence.md` |
| `tca_segmentation.py` | Regime-aware TCA tag + aggregate (PR-6) — see `docs/components/tca_segmentation.md` |
| `evidence_pack.py` | Build / sign / verify FI evidence packs (PR-7) — see `docs/components/evidence_pack.md` |
| `report.py` | FI RCIE Markdown / HTML report writer (PR-7) |
| `api.py` | FastAPI router (6 endpoints) |
| `cli.py` | `mre fi-*` CLI dispatcher (7 commands) |
| `correlation.py` | Correlation-ID middleware + log filter (PR-7) |
| `dashboard_tab.py` | Streamlit FI tab helpers (PR-7) |
| `observability_ext.py` | Pre-registered FI counters / histograms (PR-7) |
| `ingest/` | Per-vendor `IngestContract` (TRACE + MarketAxess RFQ) (PR-7) |

Per-component documentation lives at `docs/components/*.md`. Top-level
release notes: [`docs/V1_5_FIXED_INCOME_RCIE.md`](../../../docs/V1_5_FIXED_INCOME_RCIE.md).

## Governance contract

Every output dataclass carries the v1.5 governance triple:

- `model_run_id` — UUID-style identifier; PK on the row.
- `release_gate` — bool; `False` ⇒ consumers fail closed.
- `artifact_hash` — `"sha256:<hex>"` over the canonical row payload.

Evidence packs add HMAC signing: `hmac_signature = "v<ver>:<hex>"`
over the canonical pack JSON excluding the signature field. See
`docs/V1_5_HMAC_OPERATIONS.md` for the operating procedure.

## Testing

The FI test catalog lives at `tests/test_*credit*`,
`tests/test_*liquidity*`, `tests/test_*execution_confidence*`,
`tests/test_*tca*`, `tests/test_fixed_income_*`,
`tests/test_fi_*`, `tests/test_ingest_contract_*`,
`tests/test_streamlit_fi_tab.py`,
`tests/test_correlation_id_middleware.py`,
`tests/test_otel_metrics.py`,
`tests/test_models_registry_fi_components.py`,
`tests/test_signal_age_seconds_in_all_fi_responses.py`,
`tests/test_versioned_cache_key.py`,
`tests/test_check_duckdb_writers.py`,
`tests/test_verify_run_fi_extension.py`.

Acceptance command:

```
python -m pytest tests/test_fixed_income_*.py
```
