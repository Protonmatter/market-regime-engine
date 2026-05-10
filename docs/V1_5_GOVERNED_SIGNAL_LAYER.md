# v1.5.0 governed signal layer build notes

This branch narrows Market Regime Engine from a broad "market prediction app" into a governed macro regime signal layer and bumps the build identity to **1.5.0**.

## Version identity

| File | Version |
|---|---:|
| `pyproject.toml` | `1.5.0` |
| `src/market_regime_engine/__init__.py` | `__version__ = "1.5.0"` |
| `README.md` | `Market Regime Engine v1.5.0` |

Historical release docs such as `V1_4_1_FIXES.md`, `V1_4_RELEASE.md`, and earlier are intentionally left as historical records rather than rewritten into fake history. Even software deserves an honest paper trail, depressing as that is.

## Implemented changes

### 1. Product identity narrowed

Added `docs/PRODUCT_IDENTITY.md` to define the product boundary:

- not a broker
- not a backtester
- not an optimizer
- not an execution engine
- yes: governed PIT macro regime signal layer

### 2. Production mode hardened

Added `market_regime_engine.production` and wired it into `api_v1`.

Production checks:

```bash
export MRE_ENV=production
export MRE_API_KEY="..."
export MRE_DB_PATH="data/mre.duckdb"
```

If `MRE_ENV=production` and either required value is missing, `api_v1` fails at import/startup instead of serving with unsafe defaults.

The API default DB path outside production now aligns with the DuckDB-first posture: `data/mre.duckdb`.

### 3. External adapters added

New package:

```text
src/market_regime_engine/adapters/
```

Adapters:

- `core.py` — canonical governed signal schema and export validation
- `lean.py` — LEAN custom-data CSV + Python BaseData stub
- `vectorbt.py` — entry/exit/risk-off boolean series for `Portfolio.from_signals`
- `pyportfolioopt.py` — regime-conditioned expected return vector + EfficientFrontier factory
- `openbb.py` — OpenBB-style records and OBBject-like JSON export

### 4. Evidence pack added

New module:

```text
src/market_regime_engine/validation_pack.py
src/market_regime_engine/validation_pack_cli.py
```

New console script:

```bash
mre-validation-pack build --out data/evidence/run_001 \
  --include data/validation \
  --include data/reports/institutional_report.md \
  --include data/mre.duckdb

mre-validation-pack verify data/evidence/run_001
```

Optional HMAC signing:

```bash
export MRE_EVIDENCE_HMAC_KEY="rotate-this"
mre-validation-pack build --out data/evidence/run_001 --include data/validation --force
mre-validation-pack verify data/evidence/run_001
```

### 5. Package posture corrected

`pyproject.toml` now describes the package as a governed macro regime signal layer and downgrades the classifier from `Production/Stable` to `Beta` until type-checking, production-deployment, and empirical-evidence gates are fully hardened.

## Remaining recommended next patch

1. Wire `mre-validation-pack` into the main `mre` CLI as a subcommand.
2. Add production-mode strict DuckDB enforcement inside `storage.py` for every caller, not just API startup.
3. Make mypy fail CI once the current error budget is burned down.
4. Add a CI job that builds and verifies a sample evidence pack.
5. Add a formal model card generator for each promoted run.
