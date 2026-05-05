# v0.4 Upgrade Notes

v0.4 upgrades the market-regime engine from a modeling scaffold into a more institutional research platform.

## Added

- Built-in NBER recession month labels for offline target generation.
- FRED vintage ingestion plan wrapper for repeated point-in-time pulls.
- Historical analog engine using rolling standardized feature similarity.
- Analog summary with weighted forward returns, drawdowns, and regime mix.
- Domain and feature driver-attribution tables.
- Institutional Markdown report writer.
- Dynamic ensemble-weight utility with change-point, calibration, correlation, and staleness terms.
- API endpoints for analogs and attribution.
- Dashboard panes for analogs and attribution.
- Rust/PyO3 extension scaffold for future acceleration.

## New CLI flow

```bash
mre bootstrap-sample --db data/mre.db
mre pit-check --db data/mre.db
mre build-features --db data/mre.db
mre label-recessions --db data/mre.db
mre score-regime --db data/mre.db
mre train-baseline --db data/mre.db
mre validate --db data/mre.db --out data/validation
mre analogs --db data/mre.db --out data/analogs.csv
mre attribute --db data/mre.db --out data/attribution
mre model-card --db data/mre.db
mre institutional-report --db data/mre.db --out data/reports/institutional_report.md
mre report --db data/mre.db
```

## Live vintage ingestion

```bash
export FRED_API_KEY="..."
mre ingest-fred-vintages \
  --db data/mre.db \
  --series FEDFUNDS DGS10 T10Y3M UNRATE CPIAUCSL BAA10Y PERMIT HOUST DCOILWTICO DTWEXBGS GFDEGDQ188S \
  --observation-start 1960-01-01 \
  --vintage-start 1990-01-01 \
  --vintage-frequency QS
```

Monthly vintage pulls across many series can be slow and API-heavy. Start quarterly, prove the pipeline, then densify.

## What is still not done

- Exact economic release calendars by source.
- True full ALFRED bulk vintage reconstruction.
- Real options/credit/earnings private-sector ingestion.
- Rust compiled BOCPD/WFST kernels.
- Bayesian hierarchical model and calibrated stacking estimator.
- Production authentication, scheduler, and monitoring.
