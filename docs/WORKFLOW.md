# Workflow

This document turns the README into executable workflows. It is the operator
map for running the macro/regime engine, the fixed-income XPro layer, and the
release gates without reading the full release-history narrative.

## End-to-end flow

```text
external data or sample data
        |
        v
warehouse ingestion
        |
        v
point-in-time materialization and audit
        |
        v
feature build and regime scoring
        |
        v
training, validation, calibration, stacking
        |
        v
governance: drift, confidence, release gate, promotion
        |
        v
model run envelope, evidence packs, reports, APIs, exports
```

The hard invariant is that every training, validation, inference, and
decisioning path must prove the information set available at decision time.
Rows that cannot prove point-in-time lineage should be held out of production
flows.

## Macro/regime daily workflow

Use this sequence for a local deterministic smoke run. DuckDB is the default
warehouse for current builds.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev,dashboard,analytics]"

mre bootstrap-sample --db data/mre.duckdb
mre audit-release-calendar --db data/mre.duckdb --enforce
mre build-exact-release-calendar --db data/mre.duckdb --enforce
mre pit-check --db data/mre.duckdb

mre seed-vintage-from-observations --db data/mre.duckdb
mre materialize-asof-features --db data/mre.duckdb --write-features
mre audit-vintage --db data/mre.duckdb --enforce

mre build-features --db data/mre.duckdb
mre label-recessions --db data/mre.duckdb --max-stale-months 12
mre score-regime --db data/mre.duckdb

mre train-baseline --db data/mre.duckdb
mre train-survival --db data/mre.duckdb
mre train-fitted-hazard --db data/mre.duckdb --oos
mre validate --db data/mre.duckdb --out data/validation --min-train 120 --step 6
mre calibrate-probabilities --db data/mre.duckdb --validation-dir data/validation
mre optimize-stacking --db data/mre.duckdb --out data/stacking
mre optimize-regime-stacking --db data/mre.duckdb --validation-dir data/validation

mre invalidation-triggers --db data/mre.duckdb
mre monitor-drift --db data/mre.duckdb
mre score-confidence --db data/mre.duckdb --validation-dir data/validation
mre release-gate --db data/mre.duckdb --validation-dir data/validation

mre model-run --db data/mre.duckdb --purpose "governed runtime run"
mre verify-run --db data/mre.duckdb
mre institutional-report --db data/mre.duckdb --out data/reports/institutional_report.md
mre export-warehouse --db data/mre.duckdb --out data/lake --duckdb data/mre.duckdb
```

Expected release condition: `mre verify-run` exits `0`, and
`mre release-gate` returns `decision == release`.

## Live vintage workflow

Use real ALFRED/FRED vintage dates for production data review. The synthetic
vintage seed is only for local smoke tests.

```powershell
$env:FRED_API_KEY = "<key>"

mre alfred-real-plan `
  --series UNRATE CPIAUCSL PAYEMS `
  --vintage-start 2000-01-01 `
  --vintage-end 2001-01-01 `
  --max-vintages-per-series 10

mre ingest-alfred-real `
  --db data/mre.duckdb `
  --series UNRATE CPIAUCSL PAYEMS `
  --observation-start 1960-01-01 `
  --vintage-start 2000-01-01 `
  --max-vintages-per-series 20

mre materialize-asof-features --db data/mre.duckdb --write-features
mre audit-vintage --db data/mre.duckdb --enforce
```

If `audit-vintage --enforce` fails, stop. Do not train or score from the
warehouse until the lineage issue is corrected.

## Frontier workflow

The frontier layer is optional and separated from stable-core promotion.
Install extras only when the target workflow requires them.

```powershell
pip install -e ".[frontier]"
pip install -e ".[bayesian]"
pip install -e ".[scraping]"

mre refresh-release-calendars
mre nowcast --db data/mre.duckdb
mre conformal-conditional --db data/mre.duckdb --validation-dir data/validation
mre e-value-test --db data/mre.duckdb --challenger candidate_logistic
mre bayesian-msvar-fit --db data/mre.duckdb
mre deep-kernel-train --db data/mre.duckdb
```

Stable-core release gates should not silently depend on experimental frontier
models. Use `MRE_ENABLE_EXPERIMENTAL_FRONTIER=1` only when intentionally
reviewing the frontier boundary.

## Fixed-income XPro workflow

The fixed-income lane adds governed execution intelligence for Auto-X, RFQ,
and Manual protocol decisions.

```text
FI market data and order context
        |
        v
credit regime score + liquidity stress score
        |
        v
execution-confidence score
        |
        v
counterfactual protocol recommendation
        |
        v
fixed-point XPro decision artifact
        |
        v
HMAC/evidence validation, realized-outcome certification, API/CLI output
```

Core CLI commands:

```powershell
mre fi-build-features --db data/mre.duckdb
mre fi-score-credit-regime --db data/mre.duckdb
mre fi-score-liquidity --db data/mre.duckdb
mre fi-score-execution-confidence --db data/mre.duckdb --input order.json --output-json score.json
mre fi-recommend-execution-protocol --db data/mre.duckdb --input order.json --output-json decision.json
mre fi-verify-xpro-decision --db data/mre.duckdb --decision-id <decision_id>
mre fi-verify-xpro-decision --db data/mre.duckdb --decision-id <decision_id> --require-hmac
mre fi-validate-execution-confidence --db data/mre.duckdb --asof 2026-01-02T00:00:00Z --out-json xpro_certification_report.json
```

The XPro path keeps legacy float responses available for backward
compatibility, but new XPro artifacts use scaled integers and timestamp
strings per `docs/NUMERIC_CONTRACT.md`.

## API workflow

Use the hardened `/v1` API surface for production-facing deployments.

```powershell
$env:MRE_API_KEY = "rotate-me"
uvicorn market_regime_engine.api_v1:app --reload
```

Use the legacy API only with an explicit acknowledgement:

```powershell
$env:MRE_LEGACY_API_ALLOW_UNAUTH = "1"
uvicorn market_regime_engine.api:app --reload
```

Relevant XPro API endpoints:

```text
POST /v1/xpro/decision
GET  /v1/xpro/decision/{decision_id}
POST /v1/xpro/decision/verify
```

## Release validation workflow

Run these before treating a branch as release-ready:

```powershell
python -m compileall -q src tests
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src/market_regime_engine --show-error-codes --pretty
python -m pytest --collect-only -q
python -m pytest -q --durations=25
```

Track B focused validation:

```powershell
python -m pytest -q tests/test_numeric_contracts.py tests/test_protocol_recommendation.py tests/test_xpro_decision_artifact.py
python -m pytest -q tests/test_xpro_decision_api_endpoint.py tests/test_xpro_decision_cli.py
python -m pytest -q tests/test_execution_validation_certification_cli.py tests/test_storage_xpro_decision_artifacts.py
python -m pytest -q tests/test_execution_confidence.py tests/test_execution_confidence_api_endpoint.py tests/test_execution_confidence_cli.py
python -m pytest -q tests/test_canonical_json_rfc8785.py tests/test_fixed_income_evidence_pack_hmac.py tests/test_certification_release_and_execution_validation.py
python -m pytest -q tests/test_method_cards_docs_audit.py
```

## Recovery workflow

Use these rules when a run fails:

- PIT audit failure: stop, inspect `feature_asof_values`, release-calendar
  metadata, and vintage timestamps; do not train.
- Release gate hold: inspect confidence, drift, invalidation, MCS, coverage,
  DSR/PBO, Brier/ECE, and TCA-lift fields before changing thresholds.
- HMAC verification failure: treat as tamper or key-rotation mismatch until
  proven otherwise; verify active key version and payload hash.
- XPro decision failure: route to Manual with human review; do not permit
  Auto-X from an unverifiable artifact.
- CI performance timeout: use `python -m pytest -q --durations=25` and split
  around the slowest tests listed in the duration tail.
