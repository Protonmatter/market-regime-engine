# v0.5 Upgrade

v0.5 adds the first institutional governance layer on top of the v0.4 quant scaffold.

## Added

- Exact/conservative release-calendar metadata via `config/release_calendar.yaml`
- Release-calendar audit and enforcement command
- FRED recession indicator ingestion command for `USREC`
- Calibration model storage and calibrated probability outputs
- Immutable model-run IDs and artifact hashes
- Regime-weighted historical analogs
- Forecast invalidation trigger generation
- Model confidence score
- API/dashboard support for confidence, invalidation, calibrated outputs, and model runs

## New recommended v0.5 workflow

```bash
mre bootstrap-sample --db data/mre.db
mre audit-release-calendar --db data/mre.db --enforce
mre pit-check --db data/mre.db
mre build-features --db data/mre.db
mre label-recessions --db data/mre.db
mre score-regime --db data/mre.db
mre train-baseline --db data/mre.db
mre validate --db data/mre.db --out data/validation --min-train 120 --step 6
mre calibrate-probabilities --db data/mre.db --validation-dir data/validation
mre analogs --db data/mre.db --regime-weighted --out data/analogs.csv
mre attribute --db data/mre.db --out data/attribution
mre invalidation-triggers --db data/mre.db
mre score-confidence --db data/mre.db --validation-dir data/validation
mre model-run --db data/mre.db --purpose "v0.5 baseline governed run"
mre model-card --db data/mre.db --out data/model_cards
mre institutional-report --db data/mre.db --out data/reports/institutional_report.md
```

## Promotion logic

v0.5 still uses the v0.3/v0.4 benchmark validation scaffolds. Calibrated outputs do not automatically promote a model. Promotion requires:

1. Walk-forward validation artifacts.
2. Candidate comparison against naive baselines.
3. Calibration evidence.
4. Point-in-time release-calendar audit.
5. Confidence score above local policy threshold.
6. No severe breached invalidation triggers.

## Limitations

- The release-calendar file is conservative metadata, not exact official timestamp history.
- Calibration uses a Platt/logit layer over existing binary predictions.
- The analog reweighting is posterior-aware when HMM posterior metadata exists, but it is still a similarity method, not causal proof.
- FRED `USREC` ingestion requires `FRED_API_KEY`.
