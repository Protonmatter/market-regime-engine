"""Single source of truth for training/validation data access.

The pre-v1.0 pipeline read whichever of ``observations`` / ``features`` /
``feature_asof_values`` the call site happened to know about, which made it
trivial to silently train on revised macro data. This module centralises the
choice:

- :func:`load_training_panel` returns either the legacy non-vintage feature
  matrix (with an explicit ``warn_legacy=True`` flag) or the vintage-correct
  matrix derived from ``feature_asof_values``.
- :func:`load_targets` builds drawdown/return/recession targets from the
  observation panel using ``forward_fill_limit=0`` so the target column never
  bleeds future information into the training window.

The :data:`TrainingMode` enum makes the contract explicit at every call site.
"""

from __future__ import annotations

import logging
import warnings
from enum import Enum

import pandas as pd

from market_regime_engine.asof import feature_asof_to_features
from market_regime_engine.features import build_features, feature_matrix, monthly_panel  # noqa: F401
from market_regime_engine.targets import make_targets

log = logging.getLogger(__name__)


_LEGACY_DEPRECATION_MESSAGE = (
    "Legacy mode is deprecated for production training. "
    "Set TrainingMode.POINT_IN_TIME and run mre materialize-asof-features "
    "--write-features first."
)

PIT_FAIL_CLOSED_MESSAGE = (
    "POINT_IN_TIME mode requires non-empty feature_asof_values. "
    "Run `mre materialize-asof-features --write-features` first, or pass "
    "--allow-legacy-fallback to opt into the deprecated legacy path."
)


class TrainingMode(str, Enum):
    """Which feature source feeds the model."""

    POINT_IN_TIME = "point_in_time"  # feature_asof_values (default in v1.0+)
    LEGACY = "legacy"  # features table built from latest observations


def load_training_panel(
    *,
    mode: TrainingMode,
    observations: pd.DataFrame,
    features: pd.DataFrame,
    feature_asof_values: pd.DataFrame,
    catalog: list[dict] | None = None,
    allow_legacy_fallback: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return ``(X, panel, audit)`` for downstream training.

    ``X`` is the wide feature matrix used as the model input. ``panel`` is the
    monthly observation panel used to build targets. ``audit`` summarises the
    PIT lineage so the caller (and ``model_runs``) can record exactly which
    source was used and how many rows were available.

    Parameters
    ----------
    allow_legacy_fallback:
        v1.2.1 fail-closed switch. In ``POINT_IN_TIME`` mode the historical
        behavior was to silently fall back to LEGACY when
        ``feature_asof_values`` was empty (logging a warning but proceeding).
        That made the most sensitive part of the pipeline fail open, which
        defeats the entire point of the PIT router. The default is now to
        raise ``RuntimeError`` so an operator who forgot to run
        ``materialize-asof-features --write-features`` cannot silently train
        on revised macro data. Pass ``allow_legacy_fallback=True`` to keep
        the old behavior; the audit dict then records ``mode_used =
        "legacy_fallback_explicit"`` and ``fallback_authorized = True`` so
        ``mre verify-run`` can surface the conscious downgrade as a warning.
    """
    audit: dict[str, object] = {"mode": mode.value, "rows": 0, "as_of_dates": 0}
    if mode == TrainingMode.POINT_IN_TIME:
        if feature_asof_values is None or feature_asof_values.empty:
            if not allow_legacy_fallback:
                # Fail closed by default (v1.2.1). The previous fail-open
                # path silently swapped LEGACY data into a model the
                # operator believed was being trained on PIT features —
                # exactly the leakage scenario the router was supposed to
                # eliminate. Record the audit fields a verify-run can pick
                # up before raising so a downstream caller wrapping this
                # in a try/except still has structured context.
                audit["mode_used"] = "fail_closed"
                audit["fallback_reason"] = "feature_asof_values empty"
                audit["fallback_authorized"] = False
                log.error(
                    "PIT training failed closed because feature_asof_values is empty.",
                    extra={"audit": audit},
                )
                raise RuntimeError(PIT_FAIL_CLOSED_MESSAGE)
            # Operator explicitly opted in. Mark the audit clearly so
            # downstream verify-run surfaces a non-fatal warning ("legacy
            # fallback authorized as a safety net") rather than treating
            # the run as if it had been a real PIT run.
            log.warning(
                "POINT_IN_TIME mode active but legacy fallback explicitly "
                "authorized via allow_legacy_fallback=True. Training will "
                "proceed against the legacy features table.",
            )
            audit["mode_used"] = "legacy_fallback_explicit"
            audit["fallback_reason"] = "feature_asof_values empty"
            audit["fallback_authorized"] = True
            X = feature_matrix(features)
            audit["rows"] = len(features) if features is not None else 0
        else:
            converted = feature_asof_to_features(feature_asof_values)
            X = feature_matrix(converted)
            audit["rows"] = len(feature_asof_values)
            audit["as_of_dates"] = int(pd.to_datetime(feature_asof_values["as_of_date"]).nunique())
            audit["mode_used"] = TrainingMode.POINT_IN_TIME.value
            audit["fallback_authorized"] = False
    elif mode == TrainingMode.LEGACY:
        X = feature_matrix(features)
        audit["rows"] = len(features) if features is not None else 0
        audit["mode_used"] = TrainingMode.LEGACY.value
        audit["fallback_authorized"] = False
        # v1.1: escalate the prior log.warning to an actual DeprecationWarning
        # so the project-level filterwarnings rule trips during tests and
        # callers cannot ignore the legacy path. The log line stays for
        # operators reading the JSON log stream.
        warnings.warn(_LEGACY_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        log.warning(_LEGACY_DEPRECATION_MESSAGE)
    else:
        raise ValueError(f"Unknown TrainingMode: {mode}")
    panel = (
        monthly_panel(observations, forward_fill_limit=0)
        if observations is not None and not observations.empty
        else pd.DataFrame()
    )
    return X, panel, audit


def load_targets(
    panel: pd.DataFrame, *, price_col: str = "SPX", horizons: tuple[int, ...] = (3, 6, 12)
) -> pd.DataFrame:
    """Convenience wrapper so callers don't import :mod:`targets` directly."""
    if panel is None or panel.empty:
        return pd.DataFrame()
    return make_targets(panel, price_col=price_col, horizons=horizons)


def join_X_y(X: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-join feature matrix and targets on date and return aligned slices."""
    if X is None or X.empty or targets is None or targets.empty:
        return pd.DataFrame(), pd.DataFrame()
    joined = X.join(targets, how="inner")
    return joined[X.columns], joined[targets.columns]


__all__ = [
    "TrainingMode",
    "join_X_y",
    "load_targets",
    "load_training_panel",
]
