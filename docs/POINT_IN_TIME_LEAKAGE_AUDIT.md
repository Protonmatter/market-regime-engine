# Point-in-Time Leakage Audit

This document defines the point-in-time data contract and CLI workflow for preventing market-data leakage in training, validation, and benchmark runs.

The blunt version: a model that uses future features, unreleased vintages, or labels before they were available is not predictive. It is cheating with timestamps. Very sophisticated cheating, naturally, because finance likes to put a tie on bad ideas.

## Goals

- Enforce a feature schema that records observation, availability, and model-use time.
- Enforce a label schema that records forecast origin, label horizon, label time, and label availability.
- Detect feature rows whose `as_of` timestamp occurs after the prediction origin.
- Detect labels joined before the label was available.
- Detect data vintages or source revisions used before they were available.
- Produce deterministic raw-input snapshot manifests with SHA-256 hashes.

## Feature contract

Feature tables must include these columns:

| Column | Purpose |
|---|---|
| `series_id` | Feature or source-series identifier. |
| `entity_id` | Asset, market, country, instrument, or entity key. |
| `forecast_origin` | Timestamp when the prediction is made. |
| `observation_date` | Economic or market observation period. |
| `observed_at` | Timestamp when the source value was observed. |
| `available_at` | Timestamp when the value became available to the system. |
| `as_of` | Timestamp of the value/vintage actually used by the model. |
| `value` | Feature value. |
| `source` | Source system or dataset name. |
| `source_revision_id` | Revision or vintage identifier. |
| `snapshot_id` | Raw-input snapshot identifier. |

Optional revision availability columns:

```text
source_revision_available_at
revision_available_at
```

If either optional column is present, it is checked against `as_of`.

## Label contract

Label tables must include these columns:

| Column | Purpose |
|---|---|
| `entity_id` | Asset, market, country, instrument, or entity key. |
| `forecast_origin` | Timestamp when the prediction is made. |
| `label_time` | Timestamp the outcome is measured. |
| `horizon` | Forecast horizon such as `1m`, `3m`, `6m`. |
| `target` | Target definition, for example `drawdown_gt_10pct`. |
| `label_value` | Realized outcome value. |
| `label_available_at` | Timestamp when the label became available for training/evaluation. |

Optional label join-time columns:

```text
joined_at
label_joined_at
as_of
```

If present, the audit checks that `label_available_at <= joined_at`, `label_available_at <= label_joined_at`, or `label_available_at <= as_of`.

## Enforced invariants

### Feature invariants

```text
observed_at <= as_of
available_at <= as_of
as_of <= forecast_origin
source_revision_available_at <= as_of   # when present
revision_available_at <= as_of          # when present
```

### Label invariants

```text
forecast_origin <= label_time
label_time <= label_available_at
label_available_at <= joined_at         # when present
label_available_at <= label_joined_at   # when present
label_available_at <= as_of             # when present
```

### Cross-table invariants

Feature and label tables are joined on:

```text
entity_id, forecast_origin
```

The joined audit enforces:

```text
feature.as_of <= label.forecast_origin
label_available_at <= label join timestamp, when a join timestamp exists
revision availability timestamp <= feature.as_of
```

## CLI usage

### Audit PIT feature and label tables

```bash
mre pit-audit \
  --features artifacts/features.csv \
  --labels artifacts/labels.csv \
  --out-json artifacts/pit_audit.json \
  --out-md artifacts/PIT_AUDIT.md \
  --fail-on-leakage
```

Supported table formats:

```text
.csv
.json
.jsonl
.parquet
.pq
```

Exit behavior:

| Condition | Exit code |
|---|---:|
| Audit passes | `0` |
| Audit fails and `--fail-on-leakage` is set | `2` |
| Audit fails and `--fail-on-leakage` is not set | `0` |

The non-failing mode is useful for exploratory reporting. CI and release gates should use `--fail-on-leakage`, because otherwise the gate is just a polite suggestion wearing a badge.

## Snapshot manifest usage

### Build a snapshot manifest

```bash
mre snapshot-build \
  --input data/raw \
  --out artifacts/snapshot_manifest.json
```

With an explicit snapshot ID:

```bash
mre snapshot-build \
  --input data/raw \
  --out artifacts/snapshot_manifest.json \
  --snapshot-id fred_2020_01_vintage
```

The manifest includes:

- `schema_version`
- `snapshot_id`
- `input_root`
- sorted file entries
- `size_bytes`
- `sha256`
- `manifest_sha256`

### Verify a snapshot manifest

```bash
mre snapshot-verify \
  --manifest artifacts/snapshot_manifest.json \
  --out-json artifacts/snapshot_verify.json \
  --out-md artifacts/SNAPSHOT_VERIFY.md \
  --fail-on-mismatch
```

Exit behavior:

| Condition | Exit code |
|---|---:|
| Snapshot matches | `0` |
| Snapshot mismatches and `--fail-on-mismatch` is set | `2` |
| Snapshot mismatches and `--fail-on-mismatch` is not set | `0` |

## Recommended benchmark workflow

```text
1. Ingest raw source files into an immutable raw-data directory.
2. Build a snapshot manifest for the raw-data directory.
3. Generate feature and label tables with explicit PIT timestamps.
4. Run `mre pit-audit --fail-on-leakage`.
5. Run prediction evidence benchmarks only after PIT audit passes.
6. Store PIT audit reports and snapshot manifests beside benchmark outputs.
```

## Acceptance criteria

- Missing required feature or label columns produce blocker issues.
- Unparseable required PIT timestamps produce blocker issues.
- Feature rows with `as_of > forecast_origin` fail.
- Feature rows using unavailable source revisions fail.
- Labels joined before `label_available_at` fail.
- Snapshot verification detects changed file hashes or sizes.
- CLI failure flags return non-zero exit codes suitable for CI.

## Adversarial tests

| Claim | Adversarial test |
|---|---|
| Features are point-in-time safe. | Set `as_of` after `forecast_origin`; audit must fail. |
| Labels are not leaked into training. | Set `joined_at` before `label_available_at`; audit must fail. |
| Vintage data is controlled. | Set `source_revision_available_at` after `as_of`; audit must fail. |
| Raw data snapshot is reproducible. | Modify a file after manifest creation; verification must fail. |
| CLI is CI-safe. | Run with `--fail-on-leakage` or `--fail-on-mismatch`; bad inputs must exit non-zero. |
