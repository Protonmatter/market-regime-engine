# v0.7 Upgrade

v0.7 moves the engine from a governed MVP into a stronger champion/challenger research platform.

## Added

- ALFRED/FRED observation-by-vintage ingestion planning and live ingestion hook.
- Fitted discrete-time recession hazard model.
- Out-of-sample hazard backtest matrix support.
- Regime-conditioned stacking from validation prediction files.
- Routed model-risk alerts.
- Champion/challenger promotion workflow.
- API/dashboard hooks for alerts, promotion workflow, and fitted hazard diagnostics.
- Rust/PyO3 kernel scaffold expanded for BOCPD and WFST functions.

## New commands

```bash
mre alfred-plan --series UNRATE CPIAUCSL FEDFUNDS --vintage-start 2000-01-01
mre ingest-alfred --dry-run --series UNRATE CPIAUCSL
mre train-fitted-hazard --db data/mre.db --oos
mre optimize-regime-stacking --db data/mre.db --validation-dir data/validation
mre route-alerts --db data/mre.db --validation-dir data/validation
mre promotion-workflow --db data/mre.db --validation-dir data/validation
```

## Recommended v0.7 flow

```bash
mre bootstrap-sample --db data/mre.db
mre audit-release-calendar --db data/mre.db --enforce
mre build-exact-release-calendar --db data/mre.db --enforce
mre pit-check --db data/mre.db
mre build-features --db data/mre.db
mre label-recessions --db data/mre.db
mre score-regime --db data/mre.db
mre train-baseline --db data/mre.db
mre train-survival --db data/mre.db
mre train-fitted-hazard --db data/mre.db --oos
mre validate --db data/mre.db --out data/validation --min-train 120 --step 6
mre calibrate-probabilities --db data/mre.db --validation-dir data/validation
mre optimize-stacking --db data/mre.db --out data/stacking
mre optimize-regime-stacking --db data/mre.db --validation-dir data/validation
mre analogs --db data/mre.db --regime-weighted --out data/analogs.csv
mre attribute --db data/mre.db --out data/attribution
mre invalidation-triggers --db data/mre.db
mre monitor-drift --db data/mre.db
mre score-confidence --db data/mre.db --validation-dir data/validation
mre release-gate --db data/mre.db --validation-dir data/validation
mre route-alerts --db data/mre.db --validation-dir data/validation
mre promotion-workflow --db data/mre.db --validation-dir data/validation
mre model-run --db data/mre.db --purpose "v0.7 governed runtime run"
mre institutional-report --db data/mre.db --out data/reports/institutional_report.md
```

## Notes

The ALFRED live ingestion command requires `FRED_API_KEY`. Use `alfred-plan` or `ingest-alfred --dry-run` to validate request sizes before running live pulls, because blindly hammering an API is how dashboards become incident tickets.
