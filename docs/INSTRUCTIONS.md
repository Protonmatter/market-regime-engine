# Instructions

This document is the short operational instruction set for a new engineer or
operator. Use `README.md` for the release narrative and `docs/WORKFLOW.md` for
full command sequences.

## Prerequisites

- Python 3.11 or 3.12 for CI parity.
- Git.
- Optional Rust toolchain for `rust_ext` hot-path kernels.
- Optional `FRED_API_KEY` for live FRED/ALFRED ingestion.
- Optional Redis for shared API cache.
- Optional HMAC keys for production FI/XPro evidence signing.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev,dashboard,analytics]"
```

Optional extras:

```powershell
pip install -e ".[frontier]"
pip install -e ".[bayesian]"
pip install -e ".[scraping]"
pip install -e ".[security]"
pip install -e ".[redis]"
```

Optional Rust extension:

```powershell
pip install maturin
Push-Location rust_ext
maturin develop --release
Pop-Location
```

## Validate the checkout

```powershell
python -m compileall -q src tests
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src/market_regime_engine --show-error-codes --pretty
python -m pytest -q --durations=25
```

If the full suite is too slow for an inner loop, run focused tests for the area
you changed, then run the full suite before publishing.

## Run the macro smoke pipeline

```powershell
mre bootstrap-sample --db data/mre.duckdb
mre seed-vintage-from-observations --db data/mre.duckdb
mre materialize-asof-features --db data/mre.duckdb --write-features
mre audit-vintage --db data/mre.duckdb --enforce
mre build-features --db data/mre.duckdb
mre score-regime --db data/mre.duckdb
mre label-recessions --db data/mre.duckdb --max-stale-months 12
mre train-baseline --db data/mre.duckdb
mre validate --db data/mre.duckdb --out data/validation --min-train 120 --step 6
mre calibrate-probabilities --db data/mre.duckdb --validation-dir data/validation
mre score-confidence --db data/mre.duckdb --validation-dir data/validation
mre release-gate --db data/mre.duckdb --validation-dir data/validation
mre model-run --db data/mre.duckdb --purpose "local smoke"
mre verify-run --db data/mre.duckdb
```

## Run the API

Preferred production-style API:

```powershell
$env:MRE_API_KEY = "rotate-me"
uvicorn market_regime_engine.api_v1:app --reload
```

Legacy API for compatibility testing only:

```powershell
$env:MRE_LEGACY_API_ALLOW_UNAUTH = "1"
uvicorn market_regime_engine.api:app --reload
```

Dashboard:

```powershell
streamlit run src/market_regime_engine/dashboard.py
```

## Configure production FI/XPro HMAC

Use versioned keys. Rotate by adding a new version and changing the active
version; keep prior versions available for verification.

```powershell
$env:MRE_ENV = "production"
$env:MRE_FI_REQUIRE_HMAC = "1"
$env:MRE_FI_HMAC_KEY_VERSIONS = "v1=<old-secret>,v2=<new-secret>"
$env:MRE_FI_HMAC_ACTIVE_VERSION = "v2"
```

Strict verification:

```powershell
mre fi-verify-xpro-decision --db data/mre.duckdb --decision-id <decision_id> --require-hmac
```

## Create an XPro decision

Prepare `order.json` with the execution-confidence/XPro request payload, then
run:

```powershell
mre fi-recommend-execution-protocol `
  --db data/mre.duckdb `
  --input order.json `
  --output-json decision.json
```

Verify the persisted artifact:

```powershell
mre fi-verify-xpro-decision --db data/mre.duckdb --decision-id <decision_id>
```

## Produce realized-outcome certification evidence

After outcomes are available:

```powershell
mre fi-validate-execution-confidence `
  --db data/mre.duckdb `
  --asof 2026-01-02T00:00:00Z `
  --out-json xpro_certification_report.json `
  --dsr 0.75 `
  --pbo 0.01 `
  --evidence-pack-hmac v1:<hmac>
```

Then evaluate the certification release profile:

```powershell
mre release-gate --db data/mre.duckdb --validation-dir data/validation --profile certification
```

## Publish workflow

Use a branch and PR. Do not push directly to `main`.

```powershell
git fetch origin
git switch -c codex/<short-topic> origin/main

# edit files

python -m compileall -q src tests
python -m pytest -q <focused tests>
git status --short
git diff --check
git add <intended files>
git commit -m "<short imperative message>"
git push -u origin codex/<short-topic>
```

Open a draft PR first. Mark ready only after CI is green and the PR body lists
the validation commands that were actually run.

## Troubleshooting

| Symptom | Action |
|---|---|
| `audit-vintage --enforce` fails | Inspect vintage and source timestamps; do not train |
| `release-gate` holds | Read the `reasons` column; fix evidence, not just thresholds |
| `verify-run` fails | Inspect code SHA, lockfile hash, feature/output/vintage payload hashes, RNG seeds, and `extra` |
| XPro artifact fails verification | Compare canonical hash, HMAC key version, and persisted payload JSON |
| API refuses legacy import | Set `MRE_LEGACY_API_ALLOW_UNAUTH=1` only for compatibility testing |
| Frontier import fails | Install the relevant optional extra or keep the path disabled |
