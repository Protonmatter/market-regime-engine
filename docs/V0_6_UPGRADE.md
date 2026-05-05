# v0.6 Upgrade: Governed Forecast Runtime

v0.6 moves the engine from research scaffold toward a governed forecast runtime.

## Added

- DuckDB/Parquet/CSV lake export layer with dependency-safe fallback.
- Conservative exact release timestamp calendar and audit/enforcement path.
- Survival-style recession timing model.
- Constrained binary stacking optimizer for ensemble weights.
- Feature drift monitor using PSI and standardized mean shift.
- Confidence-aware release gate.
- v0.6 institutional report sections.
- New API/dashboard surfaces for drift, gates, and ensemble weights.

## New CLI

```bash
mre export-warehouse --db data/mre.db --out data/lake --duckdb data/mre.duckdb
mre warehouse-health --lake data/lake
mre build-exact-release-calendar --db data/mre.db --enforce
mre train-survival --db data/mre.db
mre optimize-stacking --db data/mre.db --out data/stacking
mre monitor-drift --db data/mre.db
mre release-gate --db data/mre.db --validation-dir data/validation
```

## Intended use

v0.6 should be run after v0.5 calibration/confidence/invalidation artifacts exist. The release gate then decides whether the forecast should be published or held for review.

## Design principle

No model output should be promoted only because it exists. It must pass point-in-time checks, calibration checks, drift checks, invalidation checks, and release gates. Apparently models also need adult supervision.
