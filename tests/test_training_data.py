"""Direct unit tests for the PIT routing core in ``training_data``.

Per the v1.1 second-opinion review, ``training_data.py`` originally carried
0% direct test coverage even though every claim about "PIT-by-default"
rests on its routing semantics.

v1.2.1 hardens the contract — PIT mode now fails closed by default when
``feature_asof_values`` is empty. The previous fail-open behavior silently
swapped LEGACY data into a model the operator believed was being trained
on PIT features. Operators who genuinely want the fallback (smoke tests,
bootstrap pipelines) opt in with ``allow_legacy_fallback=True``; the
audit dict then records the conscious downgrade.

Cases covered:

(a) POINT_IN_TIME with non-empty ``feature_asof_values`` returns a
    non-empty matrix.
(b) POINT_IN_TIME with empty ``feature_asof_values`` raises
    ``RuntimeError`` by default (v1.2.1 fail-closed contract).
(c) POINT_IN_TIME + ``allow_legacy_fallback=True`` keeps the previous
    fallback behavior, but stamps the audit with
    ``mode_used == "legacy_fallback_explicit"`` and
    ``fallback_authorized == True``.
(d) LEGACY mode emits a ``DeprecationWarning``.
(e) ``join_X_y`` handles disjoint indices.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.training_data import (
    TrainingMode,
    join_X_y,
    load_training_panel,
)


def _observations(n: int = 24) -> pd.DataFrame:
    """Synthetic monthly observations with two series (SPX + a macro panel)."""
    dates = pd.date_range("2018-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(0)
    rows = []
    for s in ("SPX", "UNRATE", "CPIAUCSL"):
        for d in dates:
            base = 100.0 + rng.normal()
            rows.append(
                {
                    "series_id": s,
                    "date": d,
                    "value": float(base),
                    "vintage_date": d,
                    "source": "synthetic",
                    "metadata_json": "{}",
                }
            )
    return pd.DataFrame(rows)


def _features(n: int = 24) -> pd.DataFrame:
    dates = pd.date_range("2018-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(1)
    rows = []
    for feat in ("growth_pca1", "inflation_pca1", "liquidity_pca1"):
        for d in dates:
            rows.append(
                {
                    "feature_name": feat,
                    "date": d,
                    "value": float(rng.normal()),
                    "domain": "macro",
                    "metadata_json": "{}",
                }
            )
    return pd.DataFrame(rows)


def _feature_asof_values(n: int = 24) -> pd.DataFrame:
    dates = pd.date_range("2018-01-01", periods=n, freq="MS").strftime("%Y-%m-%d")
    rng = np.random.default_rng(2)
    rows = []
    for feat in ("growth_pca1", "inflation_pca1"):
        for d in dates:
            rows.append(
                {
                    "as_of_date": str(d),
                    "feature_name": feat,
                    "source_series_id": "macro",
                    "observation_date": str(d),
                    "vintage_date": str(d),
                    "value": float(rng.normal()),
                    "transform_name": "level",
                    "created_at_utc": "2024-01-01T00:00:00+00:00",
                    "metadata_json": "{}",
                }
            )
    return pd.DataFrame(rows)


def test_pit_mode_returns_non_empty_matrix_when_feature_asof_values_populated() -> None:
    X, panel, audit = load_training_panel(
        mode=TrainingMode.POINT_IN_TIME,
        observations=_observations(),
        features=_features(),
        feature_asof_values=_feature_asof_values(),
    )
    assert audit["mode_used"] == TrainingMode.POINT_IN_TIME.value
    assert "fallback_reason" not in audit
    assert audit["fallback_authorized"] is False
    assert audit["rows"] > 0
    assert audit["as_of_dates"] > 0
    assert not X.empty
    assert not panel.empty


def test_pit_mode_fails_closed_by_default_when_asof_empty() -> None:
    """v1.2.1: PIT must fail closed when ``feature_asof_values`` is empty.

    The previous fail-open behavior silently substituted LEGACY data into a
    model the operator believed was being trained on PIT features —
    exactly the leakage the router was supposed to eliminate.
    """
    with pytest.raises(RuntimeError, match="POINT_IN_TIME mode requires non-empty feature_asof_values"):
        load_training_panel(
            mode=TrainingMode.POINT_IN_TIME,
            observations=_observations(),
            features=_features(),
            feature_asof_values=pd.DataFrame(),
        )


def test_pit_mode_legacy_fallback_explicit_path_records_audit() -> None:
    """``allow_legacy_fallback=True`` keeps the legacy fallback alive but
    stamps the audit so verify-run can surface the conscious downgrade."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        X, _panel, audit = load_training_panel(
            mode=TrainingMode.POINT_IN_TIME,
            observations=_observations(),
            features=_features(),
            feature_asof_values=pd.DataFrame(),
            allow_legacy_fallback=True,
        )
    assert audit["mode_used"] == "legacy_fallback_explicit"
    assert audit["fallback_authorized"] is True
    assert audit["fallback_reason"] == "feature_asof_values empty"
    # The original requested mode must still be recorded so an auditor can
    # see the operator asked for PIT and was given a legacy fallback.
    assert audit["mode"] == TrainingMode.POINT_IN_TIME.value
    assert not X.empty


def test_legacy_mode_emits_deprecation_warning() -> None:
    """LEGACY mode must surface a ``DeprecationWarning`` so the project-level
    ``filterwarnings = ["error::DeprecationWarning:market_regime_engine"]``
    rule trips on accidental legacy usage in production CI."""
    with pytest.warns(DeprecationWarning, match="Legacy mode is deprecated"):
        X, _panel, audit = load_training_panel(
            mode=TrainingMode.LEGACY,
            observations=_observations(),
            features=_features(),
            feature_asof_values=pd.DataFrame(),
        )
    assert audit["mode"] == TrainingMode.LEGACY.value
    assert audit["mode_used"] == TrainingMode.LEGACY.value
    assert audit["fallback_authorized"] is False
    assert not X.empty


def test_join_X_y_handles_disjoint_indices() -> None:
    """``join_X_y`` must inner-join on date and return a pair of aligned
    frames, even when the calendars overlap only partially or not at all."""
    idx_a = pd.date_range("2020-01-01", periods=12, freq="MS")
    idx_b = pd.date_range("2020-07-01", periods=12, freq="MS")  # 6-month overlap
    X = pd.DataFrame({"f": np.arange(12, dtype=float)}, index=idx_a)
    targets = pd.DataFrame({"y": np.arange(12, dtype=float)}, index=idx_b)
    Xj, yj = join_X_y(X, targets)
    assert len(Xj) == 6
    assert len(yj) == 6
    assert (Xj.index == yj.index).all()

    # Fully disjoint -> empty frames, not exception.
    idx_c = pd.date_range("1995-01-01", periods=6, freq="MS")
    targets_disjoint = pd.DataFrame({"y": np.zeros(6)}, index=idx_c)
    Xj2, yj2 = join_X_y(X, targets_disjoint)
    assert Xj2.empty and yj2.empty


def test_join_X_y_handles_either_input_empty() -> None:
    Xj, yj = join_X_y(pd.DataFrame(), pd.DataFrame({"y": [1.0]}))
    assert Xj.empty and yj.empty
    Xj, yj = join_X_y(pd.DataFrame({"f": [1.0]}), pd.DataFrame())
    assert Xj.empty and yj.empty


def test_create_model_run_records_training_audit_in_metadata() -> None:
    """End-to-end: the new ``training_audit`` parameter on
    ``create_model_run`` lands in metadata + envelope.extra."""
    import json as _json

    from market_regime_engine.model_runs import create_model_run

    features = pd.DataFrame([{"feature_name": "f1", "date": "2024-01-01", "value": 1.0}])
    outputs = pd.DataFrame([{"model_name": "m", "date": "2024-01-01", "horizon": "3m", "target": "t", "value": 0.5}])
    audit = {
        "mode": "point_in_time",
        "mode_used": "legacy",
        "fallback_reason": "feature_asof_values empty",
        "rows": 100,
    }
    run = create_model_run(
        engine_version="1.0.0-test",
        purpose="unit test",
        features=features,
        model_outputs=outputs,
        training_audit=audit,
    )
    meta = _json.loads(run.metadata_json)
    assert "training_audit" in meta
    assert meta["training_audit"]["fallback_reason"] == "feature_asof_values empty"
    # Envelope should also carry it via ``extra`` so verify-run can detect it.
    assert "training_audit" in meta["repro_envelope"]["extra"]
