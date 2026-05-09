# Implementation Plan from Deep Research Findings

This plan converts the research findings and the merged prediction-evidence harness into a repo-native execution roadmap for turning `market-regime-engine` into a production-grade, state-of-the-art market prediction and probabilistic forecasting platform.

The immediate strategic point: do **not** add another fancy model first. The repo now has an evidence harness. The next work is to make every future model pass point-in-time data lineage, leakage controls, calibrated probability tests, regime-sliced validation, tail-risk diagnostics, and production release gates. Otherwise the project becomes an acronym aquarium. Impressive to stare at, useless when it leaks.

## Current repo baseline

PR #1 already merged the first productionization layer:

- `src/market_regime_engine/prediction_evidence.py`
- `src/market_regime_engine/prediction_evidence_cli.py`
- `scripts/run_prediction_benchmark.py`
- `tests/test_prediction_evidence.py`
- `docs/SOTA_PREDICTION_ENGINE_BLUEPRINT.md`
- `docs/VALIDATION_REPORT_TEMPLATE.md`

This plan assumes that merged evidence harness is the new foundation.

## Target architecture

```text
raw data snapshots
  -> point-in-time normalized observations
  -> as-of feature materialization
  -> labels with explicit forecast_origin and label_time
  -> leakage/adversarial validation
  -> baseline model zoo
  -> regime/change-point posterior
  -> calibrated probability + interval forecasts
  -> dynamic ensemble and champion/challenger promotion
  -> prediction evidence harness
  -> release gate + immutable model run
  -> API/dashboard/reporting + monitoring
```

## Design rule

A model cannot be promoted because it is newer, Bayesian, neural, sparse, foundation-scale, or painful to explain at parties.

Promotion requires:

1. Point-in-time feature lineage.
2. Purged or embargoed out-of-sample validation where labels overlap.
3. Proper scoring-rule improvement.
4. Calibration and conformal coverage checks.
5. Regime and crisis-slice survival.
6. Statistical comparison against a benchmark/champion.
7. Reproducible model-run envelope.
8. Operational monitoring plan.

## Phase 0: Stabilize the evidence harness already merged

### Goal

Make the merged prediction-evidence harness unavoidable in local dev, CI, and release gating.

### Repo changes

```text
src/market_regime_engine/prediction_evidence.py        # already added
src/market_regime_engine/prediction_evidence_cli.py    # already added
scripts/run_prediction_benchmark.py                    # already added
tests/test_prediction_evidence.py                      # already added
docs/VALIDATION_REPORT_TEMPLATE.md                     # already added
```

### Add next

```text
tests/fixtures/prediction_evidence/
  binary_oos_good.csv
  binary_oos_bad_calibration.csv
  quantile_oos_good.csv
  quantile_oos_bad_coverage.csv

.github/workflows/prediction-evidence.yml
```

### CI acceptance criteria

- `mre-prediction-evidence --fail-on-hold` passes on the good fixture.
- `mre-prediction-evidence --fail-on-hold` fails on intentionally bad calibration.
- The JSON output is archived as a CI artifact.
- The Markdown output is archived as a CI artifact.
- The harness is invoked by release jobs before model promotion.

### Implementation checklist

- [ ] Add deterministic OOS fixture files.
- [ ] Add GitHub Actions job for `mre-prediction-evidence`.
- [ ] Add CI artifact upload for `prediction_evidence.json` and `PREDICTION_EVIDENCE.md`.
- [ ] Add README quickstart for the evidence harness.
- [ ] Add release-gate integration so an evidence hold blocks release.

## Phase 1: Point-in-time data contracts

### Goal

Eliminate silent look-ahead bias and revision leakage before expanding model capacity.

### New modules

```text
src/market_regime_engine/data_contracts.py
src/market_regime_engine/pit_schema.py
src/market_regime_engine/leakage_checks.py
src/market_regime_engine/snapshot_manifest.py
```

### Required canonical columns

Every training row must carry:

```text
series_id
entity_id
forecast_origin
observation_date
observed_at
available_at
as_of
value
source
source_revision_id
snapshot_id
```

Every label row must carry:

```text
entity_id
forecast_origin
label_time
horizon
target
label_value
label_available_at
```

### Core invariants

```text
observed_at <= as_of
available_at <= as_of
forecast_origin <= label_time
feature.as_of <= label.forecast_origin
label_available_at >= label_time
```

### Tests

```text
tests/test_pit_schema.py
tests/test_leakage_checks.py
tests/test_snapshot_manifest.py
tests/property/test_no_future_features.py
```

### Acceptance criteria

- Any feature row with `as_of > forecast_origin` fails.
- Any label joined before `label_available_at` fails.
- Any vintage revision used before it was available fails.
- The training panel can be rebuilt from `snapshot_id` and hash-matched.
- Negative-control tests with intentionally injected future features are caught.

### CLI

```bash
mre audit-pit --features data/features.parquet --labels data/labels.parquet --enforce
mre build-snapshot-manifest --input data/raw --out data/manifests/latest.json
mre verify-snapshot --manifest data/manifests/latest.json
```

## Phase 2: Baseline model zoo

### Goal

Build hard-to-beat baselines before deep/foundation models. Deep models should earn their food, like everyone else.

### New package layout

```text
src/market_regime_engine/models/base.py
src/market_regime_engine/models/baselines.py
src/market_regime_engine/models/tree_quantile.py
src/market_regime_engine/models/probability_heads.py
src/market_regime_engine/models/registry.py
```

### Baseline families

| Family | Purpose | Output |
|---|---|---|
| Naive historical base rate | sanity floor | probability |
| Rolling base rate | adaptive probability floor | probability |
| Logistic / elastic net | interpretable linear baseline | probability |
| Quantile regression | interpretable interval baseline | quantiles |
| HistGradientBoosting | current practical baseline | probability / quantile |
| LightGBM | fast tabular benchmark | probability / quantile |
| XGBoost | robust tabular benchmark | probability / quantile |
| CatBoost | categorical-aware benchmark | probability / uncertainty |

### Common model interface

```python
class ForecastModel:
    model_name: str
    output_type: str

    def fit(self, X_train, y_train, *, metadata=None): ...
    def predict(self, X_test) -> pd.DataFrame: ...
    def get_params(self) -> dict: ...
    def model_card(self) -> dict: ...
```

### Acceptance criteria

- Every model emits `date`, `target`, `horizon`, `model_name`, `value`, `metadata_json`.
- Probability models emit calibrated-ready `p` columns in validation mode.
- Quantile models emit one of `q_lo/q_hi`, `q05/q95`, or full quantile grids.
- Each model has a deterministic seed path.
- Every model has a smoke test and an evidence-harness integration test.

## Phase 3: Walk-forward benchmark runner

### Goal

Create the main proving ground: nested walk-forward validation with purging, embargo, crisis slices, and benchmark comparison.

### New modules

```text
src/market_regime_engine/benchmark_runner.py
src/market_regime_engine/validation_slices.py
src/market_regime_engine/adversarial_validation.py
src/market_regime_engine/model_selection.py
```

### Runner design

```text
BenchmarkConfig
  data_snapshot_id
  feature_set_id
  target_set_id
  horizons
  models
  outer_splitter
  inner_splitter
  calibration_window
  crisis_slices
  benchmark_models
  output_dir
```

### Required outputs

```text
data/validation/<run_id>/
  binary_predictions_3m.csv
  binary_predictions_6m.csv
  binary_predictions_12m.csv
  quantile_predictions_3m.csv
  quantile_predictions_6m.csv
  quantile_predictions_12m.csv
  prediction_evidence.json
  PREDICTION_EVIDENCE.md
  model_comparison.csv
  crisis_slices.csv
  manifest.json
```

### Acceptance criteria

- Runner creates OOS predictions compatible with `mre-prediction-evidence`.
- The same config + snapshot + seed produces identical hashes.
- Inner-loop tuning never sees outer-loop labels.
- Purge/embargo is mandatory for overlapping horizons.
- Negative controls fail.
- Champion/challenger table is generated automatically.

### CLI

```bash
mre benchmark \
  --config configs/benchmark/default.yaml \
  --snapshot data/manifests/latest.json \
  --out data/validation
```

## Phase 4: Calibration and conformal control plane

### Goal

Make raw model outputs trustworthy enough to drive downstream decisions.

### Existing pieces

The repo already has calibration and conformal foundations. The next step is to standardize them as a control plane instead of scattered utilities.

### New modules

```text
src/market_regime_engine/calibration_registry.py
src/market_regime_engine/conformal_registry.py
src/market_regime_engine/uncertainty_report.py
```

### Calibrators

| Calibrator | Use case |
|---|---|
| Platt / temperature scaling | small binary probability correction |
| Isotonic regression | flexible binary calibration with enough data |
| Venn-Abers | high-stakes probability intervals |
| CQR | interval calibration for quantile forecasts |
| ACI / AgACI | online adaptation under drift |
| EnbPI | sequential prediction intervals |

### Acceptance criteria

- Calibration improves or preserves Brier/log-loss/ECE on OOS validation.
- Any calibration method that degrades ECE beyond threshold is rejected.
- Coverage reports are produced by horizon and regime.
- Calibrator artifacts include fit window, sample count, method, and hash.
- Release gate blocks uncalibrated models unless explicitly allowed.

## Phase 5: Regime and change-point layer

### Goal

Model non-stationarity explicitly and use regime state as both a forecast signal and a model-risk signal.

### New modules

```text
src/market_regime_engine/regime/base.py
src/market_regime_engine/regime/sticky_hmm.py
src/market_regime_engine/regime/switching_state_space.py
src/market_regime_engine/regime/change_points.py
src/market_regime_engine/regime/regime_report.py
```

### Required outputs

```text
date
regime_state
regime_probability_<state>
regime_entropy
change_point_probability
run_length_mean
regime_confidence
metadata_json
```

### Model families

| Model | Phase | Purpose |
|---|---:|---|
| Sticky HMM | P1 | fast daily regime posterior |
| MS-VAR | P1 | regime-dependent macro covariance and dynamics |
| Bayesian MS-VAR | P1/P2 | posterior uncertainty and credible bands |
| BOCPD | P1 | online change-point probability |
| KernelCPD / PELT | P1 | offline regime segmentation benchmark |
| HSMM | P2 | duration-aware macro regimes |

### Acceptance criteria

- Regime labels are stable across retrains within tolerance.
- Regime entropy spikes around known breaks.
- Change-point detector has bounded false positives on smooth-drift synthetic data.
- Regime posterior improves forecast selection or calibration in at least one horizon/slice.
- Regime state is logged into every prediction evidence report.

## Phase 6: Dynamic ensemble and champion/challenger promotion

### Goal

Move from static model ranking to evidence-driven ensemble weighting.

### New modules

```text
src/market_regime_engine/ensemble/static_stacking.py
src/market_regime_engine/ensemble/online_bma.py
src/market_regime_engine/ensemble/regime_gated.py
src/market_regime_engine/promotion_policy.py
```

### Ensemble layers

1. Static stacking optimized on proper scoring rules.
2. Online Bayesian model averaging with forgetting factor.
3. Regime-gated mixture of experts.
4. Tail-specialist ensemble for drawdown and volatility-expansion targets.

### Promotion rules

A challenger can replace or join champion only if:

```text
DM/GW/MCS evidence supports improvement OR evidence-neutral improvement exists on a required risk slice;
calibration does not degrade beyond threshold;
tail coverage does not degrade beyond threshold;
runtime and reproducibility gates pass;
model card is generated;
release-gate approves.
```

### Acceptance criteria

- Champion/challenger report is generated from benchmark runs.
- Promotion policy is executable, not just prose.
- Release gate reads promotion evidence.
- Rejected challengers include machine-readable reasons.

## Phase 7: Deep sequence and foundation-model experts

### Goal

Add advanced models only after the baseline/evidence platform is hard to fool.

### New modules

```text
src/market_regime_engine/models/deep/tide.py
src/market_regime_engine/models/deep/timemixer.py
src/market_regime_engine/models/deep/patchtst.py
src/market_regime_engine/models/deep/itransformer.py
src/market_regime_engine/models/foundation/chronos.py
src/market_regime_engine/models/foundation/timesfm.py
src/market_regime_engine/models/foundation/lag_llama.py
src/market_regime_engine/models/foundation/tabpfn_ts.py
```

### Integration posture

These should enter as **experts**, not as replacements for the whole engine.

```text
foundation/deep expert -> common ForecastModel interface -> OOS benchmark -> calibration/conformal layer -> evidence harness -> ensemble eligibility
```

### Acceptance criteria

- Each expert can be disabled when optional dependencies are missing.
- Each expert runs through the same benchmark runner.
- No expert can bypass PIT feature contracts.
- Any deep/foundation expert must beat baseline in at least one target/horizon/slice without calibration collapse.
- Runtime and memory budgets are recorded.

## Phase 8: Economic utility and decision layer

### Goal

Convert forecasts into decision-quality signals without pretending the model is a money printer. Markets dislike arrogance. They invoice it.

### New modules

```text
src/market_regime_engine/decision/utility.py
src/market_regime_engine/decision/risk_policy.py
src/market_regime_engine/decision/position_sizing.py
src/market_regime_engine/decision/economic_backtest.py
```

### Outputs

```text
expected_return_distribution
probability_drawdown_gt_threshold
probability_volatility_expansion
recommended_action
recommended_size
abstain_flag
risk_reason
utility_score
metadata_json
```

### Decision policies

| Policy | Purpose |
|---|---|
| Abstention under uncertainty | avoid false precision |
| Kelly-fraction cap | risk-aware sizing ceiling |
| CVaR-aware utility | tail-risk-sensitive decisioning |
| Turnover penalty | reduce overtrading fantasyland |
| Regime throttle | reduce exposure in unstable regimes |

### Acceptance criteria

- Economic backtest includes transaction cost and slippage assumptions.
- Forecast improvement must survive cost stress.
- Action policy can abstain when uncertainty is too high.
- Decision output includes explanation and uncertainty.
- No decision layer can use uncalibrated raw probabilities by default.

## Phase 9: Production platform and governance

### Goal

Make the engine deployable, observable, auditable, and reversible.

### New modules

```text
src/market_regime_engine/ops/telemetry.py
src/market_regime_engine/ops/drift_monitor.py
src/market_regime_engine/ops/model_registry.py
src/market_regime_engine/ops/model_card.py
src/market_regime_engine/ops/release_bundle.py
```

### CI/CD additions

```text
.github/workflows/prediction-evidence.yml
.github/workflows/release-bundle.yml
.github/workflows/model-card.yml
```

### Release bundle

```text
dist/release_bundle/<version>/
  wheel
  source.zip
  sbom.json
  model_card.md
  prediction_evidence.json
  prediction_evidence.md
  validation_manifest.json
  release_gate.json
  verify_run.json
  artifact_hashes.json
```

### Acceptance criteria

- Every release has SBOM, artifact hashes, and validation report.
- Every production model has a model card.
- Every prediction has model version, feature snapshot, calibration version, and regime posterior logged.
- Drift monitor emits daily reports.
- Rollback path is documented and tested.

## Recommended issue breakdown

Create these GitHub issues or project cards:

1. **P0: Add PIT data contracts and leakage audit CLI**
2. **P0: Add deterministic prediction-evidence CI fixtures**
3. **P0: Build baseline model zoo behind common ForecastModel interface**
4. **P0: Build nested walk-forward benchmark runner**
5. **P1: Standardize calibration/conformal control plane**
6. **P1: Add regime posterior and change-point service**
7. **P1: Add champion/challenger promotion policy**
8. **P2: Add deep/foundation expert adapters**
9. **P2: Add economic utility and risk-policy layer**
10. **P2: Add production release bundle and monitoring dashboard**

## Implementation order

### Sprint 1: Evidence hardening

```text
prediction evidence fixtures
prediction-evidence CI job
README quickstart
release-gate integration stub
```

### Sprint 2: PIT contracts

```text
PIT schema dataclasses
snapshot manifest
leakage checker
negative-control tests
```

### Sprint 3: Baseline model zoo

```text
ForecastModel interface
rolling base-rate model
logistic baseline
quantile baseline
tree model adapters
```

### Sprint 4: Benchmark runner

```text
benchmark config schema
OOS prediction writer
crisis slices
benchmark report generator
prediction evidence invocation
```

### Sprint 5: Calibration and conformal registry

```text
calibrator registry
CQR registry
coverage report output
calibration artifact metadata
```

### Sprint 6: Regime/change-point layer

```text
regime posterior schema
BOCPD / KernelCPD wrapper
regime report
regime-sliced evidence integration
```

### Sprint 7: Promotion and release bundle

```text
champion/challenger policy
model card generator
release bundle builder
verify-run integration
```

### Sprint 8+: Deep/foundation experts

```text
TiDE / TimeMixer / PatchTST / iTransformer adapters
Chronos / TimesFM / Lag-Llama / TabPFN-TS adapters
runtime budget tests
evidence-gated ensemble eligibility
```

## Definition of done for production-grade v1

The repo can be called production-grade only when this command chain works end-to-end:

```bash
mre build-snapshot-manifest --input data/raw --out data/manifests/latest.json
mre audit-pit --features data/features.parquet --labels data/labels.parquet --enforce
mre benchmark --config configs/benchmark/default.yaml --snapshot data/manifests/latest.json --out data/validation
mre-prediction-evidence --binary data/validation/binary_oos.csv --quantile data/validation/quantile_oos.csv --fail-on-hold
mre release-gate --profile production --validation-dir data/validation
mre model-run --purpose "production candidate"
mre verify-run
mre build-release-bundle --run-id <run_id>
```

And all of these must be true:

- No PIT audit failures.
- No leakage sentinel failures.
- Prediction evidence does not hold.
- Release gate approves.
- Model run verifies.
- Model card exists.
- Release bundle contains hashes, SBOM, evidence, and run manifest.
- CI reproduces the smoke benchmark.

## Practical priority call

The next pull request should not add foundation models. The next pull request should implement **Phase 0 and Phase 1**:

```text
prediction-evidence CI fixtures
PIT schema
snapshot manifest
leakage checker
negative-control tests
```

That gives every later SOTA model a legal courtroom to stand in. Without it, every backtest is just a nicely formatted rumor.
