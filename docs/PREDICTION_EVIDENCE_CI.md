# Prediction Evidence CI Gate

This document describes the deterministic CI contract for the prediction-evidence harness.

The point is deliberately simple: known-good prediction fixtures must pass release rails, and known-bad prediction fixtures must fail. If the harness cannot distinguish those cases, it has no business judging real forecasts. Humanity has already produced enough dashboards that smile while being wrong.

## Fixtures

```text
tests/fixtures/prediction_evidence/
  binary_oos_good.csv
  binary_oos_bad_calibration.csv
  quantile_oos_good.csv
  quantile_oos_bad_coverage.csv
```

### Good binary fixture

`binary_oos_good.csv` contains 60 deterministic binary predictions:

- `y = 0` rows use `p = 0.05`
- `y = 1` rows use `p = 0.95`
- rows include simple `expansion` / `stress` regime labels

Expected result: **release**.

### Bad binary calibration fixture

`binary_oos_bad_calibration.csv` intentionally inverts the probabilities:

- `y = 0` rows use `p = 0.95`
- `y = 1` rows use `p = 0.05`

Expected result: **hold** due failed probability rails such as Brier and log loss.

### Good quantile fixture

`quantile_oos_good.csv` contains 60 deterministic interval predictions:

- `y = 0.0`
- `q_lo = -0.10`
- `q_hi = 0.10`
- `q50 = 0.0`

Expected result: **release**.

### Bad quantile coverage fixture

`quantile_oos_bad_coverage.csv` intentionally puts every outcome outside the interval:

- `y = 0.0`
- `q_lo = 0.10`
- `q_hi = 0.20`
- `q50 = 0.15`

Expected result: **hold** due interval coverage failure.

## Local commands

### Good fixtures must pass

```bash
mre-prediction-evidence \
  --binary tests/fixtures/prediction_evidence/binary_oos_good.csv \
  --quantile tests/fixtures/prediction_evidence/quantile_oos_good.csv \
  --out-json .ci-artifacts/prediction-evidence/good/prediction_evidence.json \
  --out-md .ci-artifacts/prediction-evidence/good/PREDICTION_EVIDENCE.md \
  --fail-on-hold
```

### Bad binary fixture must fail

```bash
mre-prediction-evidence \
  --binary tests/fixtures/prediction_evidence/binary_oos_bad_calibration.csv \
  --out-json .ci-artifacts/prediction-evidence/bad-binary/prediction_evidence.json \
  --out-md .ci-artifacts/prediction-evidence/bad-binary/PREDICTION_EVIDENCE.md \
  --fail-on-hold
```

Expected exit: non-zero.

### Bad quantile fixture must fail

```bash
mre-prediction-evidence \
  --quantile tests/fixtures/prediction_evidence/quantile_oos_bad_coverage.csv \
  --out-json .ci-artifacts/prediction-evidence/bad-quantile/prediction_evidence.json \
  --out-md .ci-artifacts/prediction-evidence/bad-quantile/PREDICTION_EVIDENCE.md \
  --fail-on-hold
```

Expected exit: non-zero.

## CI workflow

The workflow lives at:

```text
.github/workflows/prediction-evidence.yml
```

It performs three checks:

1. Good binary + good quantile fixtures must pass.
2. Bad binary calibration fixture must fail.
3. Bad quantile coverage fixture must fail.

The workflow uploads generated JSON and Markdown reports from all three paths as `prediction-evidence-fixtures` artifacts.

## Acceptance criteria

- The good fixture path exits `0`.
- The bad binary calibration path exits non-zero.
- The bad quantile coverage path exits non-zero.
- Artifacts are uploaded even when a negative-control fixture fails as expected.
- Fixture-level pytest tests also assert the same contract.

## Why this exists

The evidence harness is a release gate. Release gates need known-good and known-bad sentinels. Without both, a gate is just a decorative turnstile for bad forecasts wearing a lab coat.
