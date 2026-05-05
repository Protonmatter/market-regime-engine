# SPDX-License-Identifier: Apache-2.0
"""Counterfactual driver attribution.

The legacy ``attribution.py`` reports z-scores; that's diagnostic but doesn't
quantify *what would change if a feature were different*. This module
implements two complementary techniques:

- :func:`counterfactual_delta` — flip a single feature to its 12-month-ago
  value (or any user-supplied baseline), re-run the model, and report the
  change in the predicted probability or score. Honest, model-agnostic, and
  immediately interpretable.
- :func:`linear_shap_decomposition` — linear / kernel SHAP-style additive
  decomposition for a regression / classification head. Optional dependency
  on the ``shap`` package; fall back to a deterministic permutation
  approximation when ``shap`` isn't installed.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def counterfactual_delta(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    *,
    baseline: pd.DataFrame | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Per-feature counterfactual deltas.

    For each column ``c`` in ``columns`` (or every column when ``None``),
    replace ``X[c]`` with ``baseline[c]`` and report ``predict_fn(X')`` minus
    ``predict_fn(X)``.

    ``baseline`` defaults to a 12-month-shifted version of ``X``: i.e. "what
    if this feature were where it was a year ago?". This is the practitioner-
    friendly counterfactual the engine's docs promise.
    """
    if X is None or X.empty:
        return pd.DataFrame()
    base = baseline if baseline is not None else X.shift(12)
    base = base.reindex(X.index).ffill().bfill()
    base_pred = np.asarray(predict_fn(X), dtype=float)
    cols = list(columns) if columns else list(X.columns)
    rows = []
    for col in cols:
        Xc = X.copy()
        Xc[col] = base[col].values
        delta = np.asarray(predict_fn(Xc), dtype=float) - base_pred
        for date, d in zip(X.index, delta, strict=False):
            rows.append(
                {
                    "date": date,
                    "feature": col,
                    "delta": float(d),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_delta"] = out["delta"].abs()
    return out.sort_values(["date", "abs_delta"], ascending=[True, False]).reset_index(drop=True)


def permutation_attribution(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    *,
    n_samples: int = 32,
    seed: int = 0,
) -> pd.DataFrame:
    """Owen-style permutation feature attribution.

    Approximates each feature's marginal contribution by averaging over
    ``n_samples`` random permutations of column-shuffles of the *baseline*
    distribution. This is a SHAP-flavoured approximation that does not require
    the ``shap`` dependency.
    """
    rng = np.random.default_rng(seed)
    if X is None or X.empty:
        return pd.DataFrame()
    base_pred = np.asarray(predict_fn(X), dtype=float)
    cols = list(X.columns)
    rows: list[dict] = []
    for col in cols:
        contributions = np.zeros(len(X), dtype=float)
        for _ in range(n_samples):
            perm = rng.permutation(len(X))
            Xc = X.copy()
            Xc[col] = X[col].values[perm]
            contributions += base_pred - np.asarray(predict_fn(Xc), dtype=float)
        contributions /= n_samples
        for date, c in zip(X.index, contributions, strict=False):
            rows.append({"date": date, "feature": col, "contribution": float(c)})
    return pd.DataFrame(rows)


def shap_attribution_if_available(
    estimator,
    X: pd.DataFrame,
    *,
    explainer: str = "auto",
) -> pd.DataFrame:
    """Wrapper around shap.Explainer that returns a tidy frame.

    If the ``shap`` package isn't installed, returns an empty frame. Callers
    should treat the returned frame as best-effort and fall back to
    :func:`permutation_attribution` when empty.
    """
    try:  # pragma: no cover - optional dependency
        import shap  # type: ignore[import-not-found]
    except Exception:
        return pd.DataFrame()
    if X is None or X.empty:
        return pd.DataFrame()
    if explainer == "auto":
        explainer_obj = shap.Explainer(estimator, X)
    elif explainer == "linear":
        explainer_obj = shap.LinearExplainer(estimator, X)
    elif explainer == "tree":
        explainer_obj = shap.TreeExplainer(estimator)
    else:
        explainer_obj = shap.Explainer(estimator)
    values = explainer_obj(X)
    rows: list[dict] = []
    for j, col in enumerate(X.columns):
        for i, idx in enumerate(X.index):
            rows.append(
                {
                    "date": idx,
                    "feature": col,
                    "shap_value": float(values.values[i, j]),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "counterfactual_delta",
    "permutation_attribution",
    "shap_attribution_if_available",
]
