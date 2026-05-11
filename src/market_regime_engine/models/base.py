# SPDX-License-Identifier: Apache-2.0
"""Shared forecast-model protocol and prediction-frame helpers.

The model zoo intentionally emits the same columns consumed by
``mre-prediction-evidence``. The alternative is another bespoke prediction
format, because apparently every project needs one avoidable translation layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

METADATA_COLUMNS: tuple[str, ...] = ("date", "target", "horizon", "regime", "y")


class OptionalDependencyError(ImportError):
    """Raised when an optional model backend is not installed."""

    def __init__(self, package: str, extra: str, model_name: str) -> None:
        super().__init__(
            f"{model_name} requires optional dependency {package!r}. "
            f"Install with: pip install 'market-regime-engine[{extra}]'"
        )
        self.package = package
        self.extra = extra
        self.model_name = model_name


@runtime_checkable
class ForecastModel(Protocol):
    """Common interface for probability and quantile forecasting models."""

    model_name: str
    output_type: str

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, **kwargs: Any) -> ForecastModel:
        """Fit the model and return ``self``."""

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> pd.DataFrame:
        """Return an evidence-compatible prediction frame."""

    def get_params(self) -> dict[str, Any]:
        """Return serializable model parameters."""

    def model_card(self) -> dict[str, Any]:
        """Return a compact model-card dictionary."""


@dataclass(frozen=True)
class ModelCard:
    """Serializable summary of a baseline model."""

    model_name: str
    output_type: str
    family: str
    description: str
    params: dict[str, Any]
    dependencies: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "output_type": self.output_type,
            "family": self.family,
            "description": self.description,
            "params": dict(self.params),
            "dependencies": list(self.dependencies),
        }


def check_is_fitted(model: object, attrs: tuple[str, ...] = ("is_fitted_",)) -> None:
    """Raise a clear error if a model has not been fitted."""

    if not all(hasattr(model, attr) for attr in attrs):
        raise RuntimeError(f"{model.__class__.__name__} must be fitted before prediction")


def as_frame(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Normalize array-like input to a pandas DataFrame."""

    if isinstance(X, pd.DataFrame):
        return X.copy()
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])])


def numeric_feature_frame(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Return numeric feature columns, excluding evidence metadata columns."""

    frame = as_frame(X)
    features = frame.drop(columns=[c for c in METADATA_COLUMNS if c in frame.columns], errors="ignore")
    numeric = features.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError("ForecastModel inputs must contain at least one numeric feature column")
    return numeric.astype(float)


def metadata_frame(
    X: pd.DataFrame | np.ndarray,
    *,
    n: int | None = None,
    default_target: str = "target",
    default_horizon: str = "1m",
) -> pd.DataFrame:
    """Build the metadata columns expected by prediction-evidence reporting."""

    frame = as_frame(X)
    row_count = int(n if n is not None else len(frame))
    out = pd.DataFrame(index=np.arange(row_count))
    out["date"] = frame["date"].to_numpy() if "date" in frame.columns else np.arange(row_count)
    out["target"] = frame["target"].to_numpy() if "target" in frame.columns else default_target
    out["horizon"] = frame["horizon"].to_numpy() if "horizon" in frame.columns else default_horizon
    if "regime" in frame.columns:
        out["regime"] = frame["regime"].to_numpy()
    return out


def observed_y(
    X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None, *, n: int | None = None
) -> np.ndarray:
    """Extract observed outcomes for evidence frames, or return NaN placeholders."""

    if y is not None:
        return np.asarray(y, dtype=float)
    frame = as_frame(X)
    if "y" in frame.columns:
        return frame["y"].to_numpy(dtype=float)
    row_count = int(n if n is not None else len(frame))
    return np.full(row_count, np.nan, dtype=float)


def binary_prediction_frame(
    X: pd.DataFrame | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    *,
    model_name: str,
    y: pd.Series | np.ndarray | None = None,
    default_target: str = "target",
    default_horizon: str = "1m",
) -> pd.DataFrame:
    """Build a binary probability frame compatible with ``mre-prediction-evidence``."""

    p = np.asarray(probabilities, dtype=float).reshape(-1)
    out = metadata_frame(X, n=len(p), default_target=default_target, default_horizon=default_horizon)
    out["model_name"] = model_name
    out["y"] = observed_y(X, y, n=len(p))
    out["p"] = np.clip(p, 1e-6, 1.0 - 1e-6)
    return out[[c for c in ("date", "target", "horizon", "regime", "model_name", "y", "p") if c in out.columns]]


def quantile_prediction_frame(
    X: pd.DataFrame | np.ndarray,
    *,
    q_lo: pd.Series | np.ndarray,
    q_hi: pd.Series | np.ndarray,
    model_name: str,
    y: pd.Series | np.ndarray | None = None,
    q50: pd.Series | np.ndarray | None = None,
    default_target: str = "target",
    default_horizon: str = "1m",
) -> pd.DataFrame:
    """Build a quantile interval frame compatible with ``mre-prediction-evidence``."""

    lo = np.asarray(q_lo, dtype=float).reshape(-1)
    hi = np.asarray(q_hi, dtype=float).reshape(-1)
    lower = np.minimum(lo, hi)
    upper = np.maximum(lo, hi)
    out = metadata_frame(X, n=len(lower), default_target=default_target, default_horizon=default_horizon)
    out["model_name"] = model_name
    out["y"] = observed_y(X, y, n=len(lower))
    out["q_lo"] = lower
    out["q_hi"] = upper
    if q50 is not None:
        median = np.asarray(q50, dtype=float).reshape(-1)
        out["q50"] = np.clip(median, lower, upper)
    ordered = ("date", "target", "horizon", "regime", "model_name", "y", "q_lo", "q50", "q_hi")
    return out[[c for c in ordered if c in out.columns]]


__all__ = [
    "METADATA_COLUMNS",
    "ForecastModel",
    "ModelCard",
    "OptionalDependencyError",
    "as_frame",
    "binary_prediction_frame",
    "check_is_fitted",
    "metadata_frame",
    "numeric_feature_frame",
    "observed_y",
    "quantile_prediction_frame",
]
