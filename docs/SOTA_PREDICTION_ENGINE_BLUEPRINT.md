# SOTA Prediction Engine Blueprint

This document defines the path from “probabilistic research engine” to a
defensible institutional prediction application.

The core rule is simple: no model is SOTA because it has an impressive acronym.
A model is SOTA when it survives leakage checks, calibration checks, regime
slices, tail-risk tests, benchmark comparisons, and operational release gates.

## Target architecture

```text
raw macro/market data
    |
    v
point-in-time vintage store
    |
    v
as-of feature materialization  --->  audit-vintage --enforce
    |
    v
purged / embargoed walk-forward validation
    |
    +--> binary probability heads
    +--> quantile / distributional heads
    +--> regime / change-point heads
    +--> benchmark baselines
    |
    v
calibration + conformal wrappers
    |
    v
prediction evidence harness
    |
    v
release-gate + model registry + immutable model run
    |
    v
API / dashboard / institutional report
```

## What “SOTA” means here

The engine must prove all six claims before being treated as production-grade:

| Claim | Required evidence |
|---|---|
| No leakage | Point-in-time lineage, purged/embargoed OOS validation, synthetic leakage traps |
| Calibrated probabilities | Brier, log-loss, ECE, reliability tables by horizon and regime |
| Useful tails | Interval coverage, lower-tail miss rate, crisis-period slices |
| Regime value-add | Regime-aware model beats non-regime benchmark after proper OOS testing |
| Stable model selection | Hansen MCS / e-value promotion evidence, not leaderboard luck |
| Reproducible release | Lockfile hash, code SHA, feature/output/vintage payload hashes, rng_seeds, full extra envelope |

## New v1.4.2 evidence harness

This branch adds `market_regime_engine.prediction_evidence`, which consumes
out-of-sample prediction tables and emits:

- hard pass/fail gate checks
- binary probability metrics
- quantile / interval coverage metrics
- regime-sliced calibration
- tail-risk diagnostics
- JSON and Markdown reports for CI/change-management review

Run it with:

```bash
python scripts/run_prediction_benchmark.py \
  --binary data/validation/binary_oos.csv \
  --quantile data/validation/quantile_oos.csv \
  --out-json data/validation/prediction_evidence.json \
  --out-md data/validation/PREDICTION_EVIDENCE.md \
  --fail-on-hold
```

## Required input contracts

### Binary OOS predictions

Required columns:

```text
date,target,horizon,model_name,y,p
```

Optional columns:

```text
regime,benchmark_p,change_point_prob
```

### Quantile / interval OOS predictions

Required columns:

```text
date,target,horizon,model_name,y
```

Supported interval forms:

```text
q_lo,q_hi
q05,q95
q10,q90
```

Optional columns:

```text
q50,regime,benchmark_q_lo,benchmark_q_hi
```

## SOTA scorecard

| Dimension | Minimum release rail | Stretch SOTA rail |
|---|---:|---:|
| Binary Brier score | <= 0.25 | better than benchmark by DM test |
| Binary log loss | <= 0.75 | regime-stable improvement |
| ECE | <= 0.08 | <= 0.04 by regime |
| Regime ECE | <= 0.12 | <= 0.06 |
| Interval coverage | >= 0.85 | >= nominal coverage across crisis slices |
| Tail miss rate | <= 0.20 | lower than benchmark without over-widening |
| Evidence sample size | >= 60 | >= 120 per horizon/regime where possible |

## What still has to be built

1. **Baseline library**
   - naive base rate
   - rolling base rate
   - logistic baseline
   - random forest / gradient boosting baseline
   - simple macro-factor benchmark
   - buy-and-hold / no-warning benchmark for economic impact

2. **Crisis-period slices**
   - dotcom
   - GFC
   - COVID crash
   - 2022 inflation / rates shock
   - user-defined custom windows

3. **Regime-stability testing**
   - label identity drift across retrains
   - transition matrix stability
   - posterior entropy and confidence collapse checks

4. **Change-point evidence**
   - event-window precision/recall around known breaks
   - false positives under smooth drift
   - hazard-rate sensitivity sweep

5. **Economic usefulness**
   - warning hit rate
   - false alarm cost
   - avoided drawdown estimate
   - opportunity cost of defensive positioning
   - turnover / transaction assumptions

6. **Model cards**
   - training window
   - validation window
   - target definitions
   - feature lineage
   - known failure modes
   - release-gate state
   - reason not to trust the model

## Non-negotiable release rule

A model cannot be promoted because it is newer, fancier, more Bayesian, more
neural, more Rust-shaped, or more expensive to explain.

Promotion requires evidence from the prediction harness plus release-gate
approval. Anything else is just a spreadsheet learning to lie with confidence.
