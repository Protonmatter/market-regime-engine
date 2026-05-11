# v1.5.0 Breaking Changes

The v1.5.0 release is delivered as the Fixed-Income RCIE / X-Pro
Auto-X **adapter** — it sits alongside the existing macro/regime
engine. The macro behaviour is preserved, but the v1.4 → v1.5 jump
introduces a small set of intentional behavioural changes that
operators must review before deploying.

## 1. `api_v1._db_path()` default flipped (PR-1 AF-1, P0)

**v1.4:**

```python
def _db_path() -> str:
    return os.environ.get("MRE_DB_PATH") or "data/mre.db"  # SQLite
```

**v1.5:**

```python
def _db_path() -> str:
    explicit = os.environ.get("MRE_DB_PATH")
    path = explicit or "data/mre.duckdb"
    if explicit and not os.path.exists(path):
        raise RuntimeError(f"MRE_DB_PATH={path} but file does not exist")
    return path
```

**Impact**: pre-v1.5 deployments that left `MRE_DB_PATH` unset were
serving the API from a stale SQLite warehouse while the CLI wrote
to the DuckDB default. v1.5 unifies the default at
`data/mre.duckdb`.

**Action**:

- Workers that intentionally read SQLite must set
  `MRE_DB_PATH=data/mre.db`.
- An explicitly-set `MRE_DB_PATH` that does not exist now raises
  on first read (was: silently auto-created an empty SQLite). Set
  the path to a real warehouse or unset it to use the default.

## 2. `release_gates.evaluate_release_gate` output frame adds `resolved_profile` (PR-1 ASK-7)

The output frame now carries a `resolved_profile` column reporting
which threshold profile was applied (`"production"`, `"default"`).
The change is **additive**: existing consumers that select specific
columns are unaffected; consumers that materialise the full frame
must accept the new column.

## 3. `release_gates.evaluate_release_gate` fails closed on empty coverage (PR-1 AF-6)

**v1.4:** an empty `coverage_report` argument silently produced a
permissive gate (`worst_coverage=0.0`).

**v1.5:** an empty / all-NaN `coverage_report` adds
`"coverage_data_missing"` to `reasons`, sets `worst_coverage=NaN`,
and the gate fails. This closes a P0 governance hole where a
coverage-report build failure produced a passing release gate.

**Action**: ensure the coverage-report pipeline runs to completion
before invoking the release gate. The auditor for this change is
`tests/test_release_gates_empty_coverage.py`.

## 4. `release_gates.evaluate_release_gate` raises on missing e_value `decision` (PR-1 AF-14)

**v1.4:** when `promotion_method="e_values"` and the e-value log
lacked a `decision` column, the gate defaulted to `"promote"`.

**v1.5:** raises `ValueError("e_value_log missing 'decision' column")`.

**Action**: ensure your e-value log carries the `decision` column
(emitted by `mre e-value-test`). If you have legacy logs, add the
column or migrate to the explicit-promotion path.

## 5. `WalkForward.__init__` accepts `min_train_after_purge` (PR-5 AF-11)

**v1.4:**

```python
WalkForward(min_train=120)
# purge gating: hard-coded `min_train // 2`
```

**v1.5:**

```python
WalkForward(min_train=120, min_train_after_purge=None)
# default None preserves the v1.4 `min_train // 2` semantics; set an
# explicit int to override.
```

The default is back-compat. Pass `min_train_after_purge=int` to
override.

## 6. New schema columns

PR-7 evidence-pack table:

```sql
CREATE TABLE fixed_income_evidence_packs (
    model_run_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    component_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    code_sha TEXT,
    model_hash TEXT NOT NULL,
    input_features_hash TEXT NOT NULL,
    output_hash TEXT NOT NULL,
    data_vintages_json JSON,
    validation_results_json JSON,
    release_gate INTEGER NOT NULL,
    random_seeds_json JSON,
    python_version TEXT,
    lockfile_hash TEXT,
    hmac_signature TEXT,
    metadata_json JSON,
    PRIMARY KEY(model_run_id, request_id)
);
```

PR-1 added `release_gates.resolved_profile` (TEXT).

PR-5 added the request-id PK on
`execution_confidence_predictions(request_id, timestamp)`.

These are additive and don't conflict with v1.4 schemas.

## 7. New optional dependency: `[observability]` extra adds OTel

The `[observability]` extra now pulls in:

- `opentelemetry-api>=1.30`
- `opentelemetry-sdk>=1.30`
- `opentelemetry-exporter-otlp>=1.30`
- `opentelemetry-instrumentation-fastapi>=0.51b0`

These are **optional** — the legacy in-process `MetricsRegistry`
remains the default backend. Operators wanting the OTLP path must
install `[observability]` and call
`observability.configure_otel(...)` once at process start.

## 8. Production-mode HMAC requirement (PR-7 §A)

When `MRE_ENV=production` (or `MRE_FI_REQUIRE_HMAC=1`) and no HMAC
key is configured via `MRE_FI_HMAC_KEY_VERSIONS` /
`MRE_FI_HMAC_KEY`, the FI evidence-pack write path raises rather
than silently emitting unsigned packs.

**Action**: see `docs/V1_5_HMAC_OPERATIONS.md` for the key-rollout
playbook. Dev / single-key deployments are unaffected — the
fail-closed gate only triggers when production mode is set without
keys.

## 9. Rename of `_other_fi_endpoints_still_return_501` test

`tests/test_credit_regime_api_endpoint.py` had an assertion that
`GET /v1/evidence-pack/{model_run_id}` returns 501. PR-7 makes the
endpoint live (200 / 404), so the test was renamed to
`test_all_fi_endpoints_live_in_pr7` and now asserts the new
behaviour. Downstream test suites that referenced the old name
should follow.

## 10. None — most surfaces are unchanged

All PR-1..PR-6 features ship as additive: 13 new warehouse tables,
6 new API endpoints, 7 new CLI commands, and the `fixed_income/`
subpackage. Macro callers, dashboards, alerts, release-calendar
audits, ALFRED ingestion, validation, calibration, hazard, and
stacking pipelines are unchanged.
