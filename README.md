# Market Regime Engine v1.6.0

Governed macro regime signal layer with point-in-time lineage, production guardrails, adapter exports, tamper-evident validation evidence packs, anti-overfit controls, and online conformal calibration primitives.

## Product boundary

Market Regime Engine is not a broker, backtester, execution engine, or optimizer. It is a governed macro regime signal layer:

```text
PIT macro / market data
  -> regime + change-point + risk probability models
  -> calibration, drift, release-gate, and anti-overfit controls
  -> governed signal contract
  -> LEAN / vectorbt / PyPortfolioOpt / OpenBB / dashboards / APIs
```

## v1.6.0 frontier math hardening

This build implements the highest-ROI deep-research findings:

- strict boolean parsing for governed release-gate signals
- safer LEAN CSV parsing for quoted metadata fields
- probabilistic vectorbt signal scores instead of only hard labels
- PyPortfolioOpt allocation permission metadata instead of hiding blocks inside `mu * 0`
- provenance-rich validation evidence packs with git dirty state, platform, command line, lockfile hashes, and optional HMAC requirements
- anti-overfit controls: CSCV/PBO, Deflated Sharpe Ratio, minimum track record length, and pre-registered model tournament manifests
- online conformal primitives: EnbPI-style intervals, strongly adaptive ACI, and AgACI-style expert aggregation

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,analytics]"
```

Optional adapter extras:

```bash
pip install -e ".[adapter-vectorbt]"
pip install -e ".[adapter-pyportfolioopt]"
pip install -e ".[adapter-openbb]"
```

## Production API

```bash
export MRE_ENV=production
export MRE_API_KEY="rotate-this"
export MRE_DB_PATH="data/mre.duckdb"
uvicorn market_regime_engine.api_v1:app
```

## Validation evidence pack

```bash
mre-validation-pack build --out data/evidence/run_001 \
  --include data/validation \
  --include data/reports/institutional_report.md \
  --include data/mre.duckdb \
  --require-signed \
  --hmac-key "$MRE_EVIDENCE_HMAC_KEY"

mre-validation-pack verify data/evidence/run_001 --hmac-key "$MRE_EVIDENCE_HMAC_KEY"
```

## Targeted tests

```bash
pytest tests/test_v1_5_production.py \
  tests/test_v1_5_adapters.py \
  tests/test_v1_5_validation_pack.py \
  tests/test_v1_6_overfit_control.py \
  tests/test_v1_6_online_conformal.py -q
```

## Docs

- `docs/PRODUCT_IDENTITY.md`
- `docs/V1_5_GOVERNED_SIGNAL_LAYER.md`
- `docs/V1_6_FRONTIER_MATH_HARDENING.md`
