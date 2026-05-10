# Market Regime Engine v1.6.0

Governed macro regime signal layer with point-in-time lineage, production guardrails, adapter exports, tamper-evident validation evidence packs, anti-overfit controls, online conformal calibration, and a baseline model zoo.

Market Regime Engine is not a broker, execution engine, or portfolio optimizer. It is a governed signal layer that produces auditable macro/market regime intelligence for downstream systems such as LEAN, vectorbt, PyPortfolioOpt, OpenBB, dashboards, and internal APIs.

```text
PIT macro / market data
  -> feature and label contracts
  -> regime + change-point + risk probability models
  -> calibration, drift, release-gate, and anti-overfit checks
  -> governed signal contract
  -> external consumers and decision systems
```

## v1.6.0 focus

This build preserves the current model-zoo work on `main` and adds the v1.6 hardening layer:

- fail-closed production API posture
- canonical governed signal exports
- strict boolean parsing for release-gate fields
- LEAN, vectorbt, PyPortfolioOpt, and OpenBB adapter surfaces
- tamper-evident validation evidence packs
- Deflated Sharpe Ratio, PBO, MTRL, and model-tournament manifests
- EnbPI-style intervals, strongly adaptive ACI, and AgACI-style online conformal primitives

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,analytics]"
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,analytics]"
```

Optional extras:

```bash
pip install -e ".[dashboard]"
pip install -e ".[bayesian]"
pip install -e ".[frontier]"
pip install -e ".[security]"
pip install -e ".[adapter-vectorbt]"
pip install -e ".[adapter-pyportfolioopt]"
pip install -e ".[adapter-openbb]"
```

## CLI quick checks

```bash
mre --version
mre pit-audit --features data/features.csv --labels data/labels.csv --fail-on-leakage
mre snapshot-build --input data/validation --out data/snapshots/validation_manifest.json
mre snapshot-verify --manifest data/snapshots/validation_manifest.json --fail-on-mismatch
```

Legacy CLI commands are still delegated through `market_regime_engine.cli_dispatch:main`.

## Point-in-time invariant

The engine must never train on information unavailable at forecast time:

```text
observation_date <= as_of_date
vintage_date     <= as_of_date
feature.as_of    <= forecast_origin
label_available  <= join/as_of time
```

Use `mre pit-audit` and the schema/leakage checks before treating any validation result as decision-grade.

## Production API

Local/dev:

```bash
uvicorn market_regime_engine.api_v1:app --reload
```

Production:

```bash
export MRE_ENV=production
export MRE_API_KEY="rotate-this"
export MRE_DB_PATH="data/mre.duckdb"
uvicorn market_regime_engine.api_v1:app
```

Production mode requires explicit `MRE_API_KEY` and `MRE_DB_PATH`. Redis cache fallback is refused in production if `MRE_CACHE_BACKEND=redis` is set without a valid `MRE_REDIS_URL`.

API routes:

```text
/health
/v1/health
/v1/metrics
/v1/regime/latest
/v1/model-outputs/latest
/v1/calibrated-outputs/latest
/v1/release-gate/latest
/v1/analogs/latest
```

Protected `/v1/*` routes require `X-API-Key` when `MRE_API_KEY` is set.

## Governed signal contract

External adapters normalize outputs to this schema:

| Column | Meaning |
|---|---|
| `date` | As-of date for the signal |
| `regime_state` | Decoded or posterior-modal regime |
| `regime_confidence` | Regime confidence in `[0, 1]` |
| `change_point_prob` | Change-point probability in `[0, 1]` |
| `drawdown_prob` | Drawdown-risk probability in `[0, 1]` |
| `recession_prob` | Recession probability in `[0, 1]` |
| `confidence_score` | Governance confidence score in `[0, 1]` |
| `release_gate_decision` | `release`, `hold`, `unknown`, etc. |
| `release_gate_approved` | Strictly parsed boolean gate state |
| `model_run_id` | Immutable run identifier where available |
| `artifact_hash` | Reproducibility/evidence artifact hash |
| `metadata_json` | Adapter/run metadata |

Adapter modules:

```text
market_regime_engine.adapters.core
market_regime_engine.adapters.lean
market_regime_engine.adapters.vectorbt
market_regime_engine.adapters.pyportfolioopt
market_regime_engine.adapters.openbb
```

Adapters emit governed signals and decision inputs. They do not place orders.

## Validation evidence packs

Build a signed evidence pack:

```bash
export MRE_EVIDENCE_HMAC_KEY="rotate-this"

mre-validation-pack build \
  --out data/evidence/run_001 \
  --include data/validation \
  --include data/reports/institutional_report.md \
  --include data/mre.duckdb \
  --require-signed \
  --hmac-key "$MRE_EVIDENCE_HMAC_KEY"
```

Verify:

```bash
mre-validation-pack verify data/evidence/run_001 \
  --require-signed \
  --hmac-key "$MRE_EVIDENCE_HMAC_KEY"
```

Evidence packs include:

```text
manifest.json
manifest.sha256
manifest.hmac.sha256   # optional / required in signed workflows
artifacts/
```

The manifest records engine version, git SHA, dirty state, platform, Python version, redacted command line, lockfile hashes, redacted source map, file hashes, and file sizes. Verification detects changed files, missing files, extra payload files, manifest hash drift, and signature failure.

## Anti-overfit controls

`market_regime_engine.frontier.overfit_control` provides:

- Deflated Sharpe Ratio approximation
- Probability of Backtest Overfitting via CSCV-style tournaments
- Minimum Track Record Length approximation
- pre-registered model tournament manifest writer and verifier

Typical flow:

```python
from market_regime_engine.frontier.overfit_control import (
    deflated_sharpe_ratio,
    freeze_model_tournament,
    probability_of_backtest_overfitting,
)
```

Freeze the model tournament before validation, run all candidates, then evidence-pack the manifest and results. Do not promote post-hoc winners that were not declared in the tournament manifest.

## Online conformal primitives

`market_regime_engine.frontier.online_conformal` provides:

- `EnbPIInterval`
- `StronglyAdaptiveACI`
- `AgACI`

These are intended to complement the existing conformal stack with online, nonstationary calibration controls. They are primitives, not a full claim that the engine is frontier-complete.

## Baseline model zoo

The current model-zoo layer remains available through:

```text
market_regime_engine.models
market_regime_engine.prediction_evidence
market_regime_engine.prediction_evidence_cli
```

The baseline model zoo emits evidence-compatible prediction frames consumed by `mre-prediction-evidence`.

## Recommended local validation

```bash
pytest tests/test_v1_5_production.py \
  tests/test_v1_5_adapters.py \
  tests/test_v1_5_validation_pack.py \
  tests/test_v1_6_overfit_control.py \
  tests/test_v1_6_online_conformal.py -q

ruff check src tests
ruff format --check src tests
python -m build
```

Full CI also runs package sanity, golden trace, mypy, security scans, lockfile checks, SBOM generation, license audit, benchmark checks, and Rust extension wheel jobs.

## Release posture

v1.6.0 is intentionally classified as **Beta** until production release gates consume signed evidence packs, synthetic regime-recovery benchmarks are added, and online conformal / anti-overfit controls are integrated into promotion decisions.

## Documentation

- `docs/PRODUCT_IDENTITY.md`
- `docs/V1_5_GOVERNED_SIGNAL_LAYER.md`
- `docs/V1_6_FRONTIER_MATH_HARDENING.md`
- `docs/V1_4_1_FIXES.md`
- `docs/V1_4_RELEASE.md`
- `docs/V1_3_RELEASE.md`
- `docs/V1_2_1_FIXES.md`
- `docs/V1_2_FRONTIER.md`
