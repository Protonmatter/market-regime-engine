# Baseline Forecast Model Zoo

This document describes the baseline model zoo added for forecast and prediction-evidence benchmarking.

The blunt version: every serious forecasting stack needs boring baselines. If a complicated model cannot beat a base rate, a linear model, or a small tree ensemble, the complicated model is not advanced. It is expensive theater with a progress bar.

## Goals

- Provide a common `ForecastModel` interface for baseline forecast models.
- Emit prediction frames compatible with `mre-prediction-evidence`.
- Support binary probability and quantile interval outputs.
- Keep sklearn-backed baselines dependency-light and deterministic enough for CI.
- Provide registry lookup by canonical model name and short aliases.

## Interface

All baseline models implement the same practical interface:

```python
model.fit(X_train, y_train)
predictions = model.predict(X_test)
params = model.get_params()
card = model.model_card()
```

The shared protocol and frame helpers live in:

```text
src/market_regime_engine/models/base.py
```

## Registry usage

```python
from market_regime_engine.models import available_models, make_model, model_cards

print(available_models())
model = make_model("lr", max_iter=1000)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

Useful aliases:

| Alias | Canonical model |
|---|---|
| `prior` | `persistence` |
| `base_rate` | `rolling_base_rate` |
| `rolling_prior` | `rolling_base_rate` |
| `lr` | `logistic_regression` |
| `elastic_logistic` | `elastic_net_logistic` |
| `enet_logistic` | `elastic_net_logistic` |
| `rf` | `random_forest` |
| `quantile_regression` | `linear_quantile` |
| `linear_qr` | `linear_quantile` |
| `rf_quantile` | `random_forest_quantile` |
| `hgb_probability` | `hist_gradient_boosting_probability` |
| `hgb_classifier` | `hist_gradient_boosting_probability` |
| `hgb_quantile` | `hist_gradient_boosting_quantile` |

## Binary probability models

| Model | Description |
|---|---|
| `persistence` | Smoothed historical positive-rate model. |
| `rolling_base_rate` | Smoothed base-rate model over the most recent fitted window. |
| `logistic_regression` | Median-imputed, standardized logistic regression. |
| `elastic_net_logistic` | Median-imputed, standardized elastic-net logistic regression. |
| `random_forest` | Median-imputed random forest probability model. |
| `hist_gradient_boosting_probability` | Histogram gradient boosting probability model. |

Binary models emit:

```text
date, target, horizon, regime?, model_name, y, p
```

Where:

- `p` is clipped to `[1e-6, 1 - 1e-6]` for stable log-loss evaluation.
- `date`, `target`, `horizon`, `regime`, and `y` are copied from `X` when present.
- Missing metadata defaults to simple placeholders.

## Quantile interval models

| Model | Description |
|---|---|
| `historical_quantile` | Constant historical quantile interval model. |
| `linear_quantile` | Linear quantile-regression interval model. |
| `random_forest_quantile` | Random forest interval model using empirical per-tree quantiles. |
| `hist_gradient_boosting_quantile` | Histogram gradient boosting quantile interval model. |

Quantile models emit:

```text
date, target, horizon, regime?, model_name, y, q_lo, q50?, q_hi
```

The helper normalizes interval bounds so `q_lo <= q_hi`, because apparently even quantile models deserve guardrails.

## Metadata handling

The model zoo treats these columns as metadata, not numeric features:

```text
date, target, horizon, regime, y
```

All other numeric columns are treated as features. At least one numeric feature column is required for non-constant models.

## Model cards

Every model returns a compact model card:

```python
from market_regime_engine.models import make_model

card = make_model("random_forest").model_card()
```

Model cards include:

- `model_name`
- `output_type`
- `family`
- `description`
- `params`
- `dependencies`

## Optional third-party adapters

Issue #5 requested soft-degrade adapters for LightGBM, XGBoost, and CatBoost. Those adapters are not included in this PR.

I attempted to add them, but this environment repeatedly blocked the write operation for those adapter files. Rather than pretend they shipped, this PR focuses on the core sklearn-backed model zoo and leaves optional third-party adapters as a follow-up.

Recommended follow-up design:

```text
src/market_regime_engine/models/optional_adapters.py
```

Each adapter should:

- be registered by name only when importable, or raise a clear optional-dependency error on `fit`;
- emit the same prediction-evidence-compatible frames;
- include model cards with dependency names;
- have tests that skip cleanly when the optional package is unavailable.

## Adversarial checks

| Claim | Test |
|---|---|
| Registry lookup is usable. | Instantiate each model through `make_model`. |
| Binary outputs are evidence-compatible. | Assert `date,target,horizon,model_name,y,p` are present. |
| Binary probabilities are bounded. | Assert `p` is between 0 and 1. |
| Quantile outputs are evidence-compatible. | Assert `q_lo` and `q_hi` are present. |
| Quantile intervals are valid. | Assert `q_lo <= q_hi`. |
| Single-class binary training is safe. | Assert logistic-style models fall back to constant probability instead of crashing. |
