# Validation Report Template

Use this template for every candidate model release. Fill it with generated
artifacts from `scripts/run_prediction_benchmark.py`, validation CSVs, CI links,
and release-gate output.

## 1. Executive decision

| Field | Value |
|---|---|
| Candidate model | TBD |
| Engine version | TBD |
| Validation window | TBD |
| Training window | TBD |
| Decision | RELEASE / HOLD |
| Primary blocker | TBD |
| Reviewer | TBD |

## 2. Data lineage

| Check | Result | Evidence |
|---|---|---|
| Point-in-time features used | TBD | `mre audit-vintage --enforce` |
| Vintage dates <= as-of dates | TBD | audit output |
| Observation dates <= as-of dates | TBD | audit output |
| Feature payload hash | TBD | `mre model-run` |
| Output payload hash | TBD | `mre model-run` |
| Vintage payload hash | TBD | `mre model-run` |
| Lockfile hash | TBD | `mre verify-run` |

## 3. Binary probability performance

Paste generated `binary_metrics` from `prediction_evidence.json`.

| Target | Horizon | Model | N | Event rate | Brier | Log loss | ECE | Benchmark DM p-value |
|---|---|---|---:|---:|---:|---:|---:|---:|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 4. Quantile / interval performance

| Target | Horizon | Model | N | Coverage | Mean width | Width vs benchmark | Tail miss rate |
|---|---|---|---:|---:|---:|---:|---:|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 5. Regime-sliced calibration

| Target | Horizon | Model | Regime | N | Event rate | Brier | ECE |
|---|---|---|---|---:|---:|---:|---:|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 6. Tail-risk diagnostics

| Target | Horizon | Model | Slice | N | Lower-tail miss | Upper-tail miss | False alarm rate |
|---|---|---|---|---:|---:|---:|---:|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 7. Forecast comparison

| Candidate | Benchmark | Metric | Direction | p-value | Decision |
|---|---|---|---|---:|---|
| TBD | TBD | DM / GW / MCS / CRPS-DM | TBD | TBD | TBD |

## 8. Crisis-period slices

| Period | Dates | Binary ECE | Interval coverage | Tail miss rate | Notes |
|---|---|---:|---:|---:|---|
| Dotcom | TBD | TBD | TBD | TBD | TBD |
| GFC | TBD | TBD | TBD | TBD | TBD |
| COVID | TBD | TBD | TBD | TBD | TBD |
| 2022 rates shock | TBD | TBD | TBD | TBD | TBD |

## 9. Model-risk notes

### Known failure modes

- TBD

### Conditions where the model should not be trusted

- TBD

### Required operator warnings

- TBD

## 10. Release-gate output

Paste `mre release-gate` output here.

```json
{}
```

## 11. Reproducibility verification

Paste `mre verify-run` output here.

```json
{}
```

## 12. Final reviewer call

```text
RELEASE / HOLD
```

Reason:

```text
TBD
```
