# v0.8 Upgrade: Real Point-in-Time ALFRED Layer

v0.8 upgrades the prior real-ish vintage ingestion scaffolding into a point-in-time enforcement layer.

## Added modules

- `alfred_real.py`: real FRED/ALFRED vintage-date retrieval and observation-by-vintage ingestion.
- `asof.py`: as-of observation selection, feature snapshot materialization, and lineage audits.
- `report_writer_v5.py`: institutional report sections for vintage coverage and as-of lineage.

## New tables

- `series_vintages`
- `vintage_observations`
- `feature_asof_values`
- `vintage_audits`

## New commands

```bash
mre alfred-real-plan
mre ingest-alfred-real
mre seed-vintage-from-observations
mre materialize-asof-features
mre audit-vintage
```

## Production invariant

```text
observation_date <= as_of_date
vintage_date     <= as_of_date
```

`audit-vintage --enforce` fails closed when that invariant is violated.

## Local smoke-test path

```bash
mre bootstrap-sample --db data/mre.db
mre seed-vintage-from-observations --db data/mre.db
mre materialize-asof-features --db data/mre.db --write-features
mre audit-vintage --db data/mre.db --enforce
```

The seeded vintage path is not official ALFRED data. It exists only so the as-of pipeline can be tested without an API key.
