# SPDX-License-Identifier: Apache-2.0
"""PR-3 deterministic credit-regime scorer acceptance suite."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401  - register FI schema
from market_regime_engine.fixed_income.credit_spread_regime import (
    DEFAULT_WEIGHTS,
    latest_credit_regime_score,
    score_credit_regime,
    write_credit_regime_score,
)
from market_regime_engine.fixed_income.pit_guard import PitViolationError
from market_regime_engine.fixed_income.schemas import (
    CreditRegimeOutput,
    RegimeLabel,
)
from market_regime_engine.frontier.data_cleaning import NanPolicy
from market_regime_engine.storage import Warehouse

# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------


_ASOF = pd.Timestamp("2026-05-08T16:00:00+00:00")  # Friday — open trading day


def _row(
    date: pd.Timestamp,
    feature_name: str,
    value: float,
    *,
    vintage: pd.Timestamp | None = None,
) -> dict:
    return {
        "date": date,
        "feature_name": feature_name,
        "value": float(value),
        "source_timestamp": date,
        "vintage_date": vintage,
    }


def _synthetic_features(
    *,
    asof: pd.Timestamp = _ASOF,
    n_days: int = 60,
    spread_pct: float = 0.5,
    cds_pct: float = 0.5,
    vol_pct: float = 0.5,
    slope_pct: float = 0.5,
    include: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build a long-form feature frame whose latest values land at the given percentiles.

    ``spread_pct`` / ``cds_pct`` / ``vol_pct`` / ``slope_pct`` are the
    percentile (0..1) of the *latest* value within the rolling window
    of that feature. ``0.5`` is the neutral midline (score == 50).
    """
    if include is None:
        include = (
            "ust_level",
            "ust_slope",
            "ust_curvature",
            "cdx_ig_5y",
            "cdx_hy_5y",
            "vix",
            "move",
            "etf_prem_disc",
        )
    include_set = set(include)
    # Build a deterministic monotonic series; the percentile of the
    # latest value in a monotonic series is exactly its rank fraction.
    dates = pd.date_range(end=asof, periods=n_days, freq="D", tz="UTC")
    rows: list[dict] = []

    def _emit(feature: str, values: list[float], *, vintage_offset_days: int = 0) -> None:
        if feature not in include_set:
            return
        for ts, val in zip(dates, values, strict=False):
            v = pd.Timestamp(ts) + pd.Timedelta(days=vintage_offset_days) if vintage_offset_days else None
            rows.append(_row(ts, feature, val, vintage=v))

    def _ramp(target_pct: float, *, scale: float = 1.0, sign: int = 1) -> list[float]:
        # Monotonic increasing series whose latest value sits at target_pct.
        base = np.linspace(0.0, 1.0, n_days)
        # Adjust last value so cumulative percentile equals target_pct.
        # Replace the final n_days*(1-target_pct) tail with a saturated
        # value so the rank of the latest matches target_pct.
        idx_target = round(target_pct * (n_days - 1))
        if idx_target < 0:
            idx_target = 0
        if idx_target > n_days - 1:
            idx_target = n_days - 1
        # Rotate so the latest is at percentile target_pct.
        rot = np.concatenate([base[idx_target:], base[:idx_target]])
        return [sign * float(v * scale) for v in rot]

    _emit("ust_level", _ramp(0.5, scale=4.5))
    _emit("ust_slope", _ramp(1.0 - slope_pct, scale=2.0, sign=+1))
    _emit("ust_curvature", _ramp(0.5, scale=0.5))
    _emit("cdx_ig_5y", _ramp(spread_pct, scale=120.0))
    _emit("cdx_hy_5y", _ramp(cds_pct, scale=500.0))
    _emit("vix", _ramp(vol_pct, scale=30.0))
    _emit("move", _ramp(vol_pct, scale=100.0))
    _emit("etf_prem_disc", _ramp(vol_pct, scale=0.5))

    frame = pd.DataFrame(rows)
    frame.attrs["nan_policy"] = NanPolicy.NAN_FAILS_PIT_AUDIT.value
    return frame


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_score_credit_regime_returns_output_with_required_fields() -> None:
    features = _synthetic_features()
    out = score_credit_regime(features, asof=_ASOF, model_run_id="run-1")
    assert isinstance(out, CreditRegimeOutput)
    assert out.timestamp.endswith("Z")
    assert 0.0 <= out.regime_score <= 100.0
    assert out.regime_label
    assert 0.0 <= out.confidence <= 1.0
    assert out.model_run_id == "run-1"
    assert out.release_gate is True
    assert out.artifact_hash.startswith("sha256:")
    assert "weights_used" in out.metadata
    assert out.metadata["feature_count"] == len(features)
    assert "score_components" in out.metadata


def test_score_credit_regime_score_bounded_0_100() -> None:
    """Even pathologically skewed inputs stay in [0, 100]."""
    for pct in (0.0, 0.25, 0.5, 0.75, 1.0):
        features = _synthetic_features(spread_pct=pct, cds_pct=pct, vol_pct=pct, slope_pct=pct)
        out = score_credit_regime(features, asof=_ASOF, model_run_id="run-x")
        assert 0.0 <= out.regime_score <= 100.0, f"pct={pct} score={out.regime_score}"


@pytest.mark.parametrize(
    "pct,expected_label",
    [
        (0.05, RegimeLabel.RISK_ON_COMPRESSION),
        (0.25, RegimeLabel.NORMAL_LIQUIDITY),
        (0.50, RegimeLabel.WATCH_TRANSITION),
        (0.75, RegimeLabel.RISK_OFF_HIGH_RISK_AVERSION),
        (0.95, RegimeLabel.CRISIS_SEVERE_DISLOCATION),
    ],
)
def test_score_credit_regime_label_bucket_mapping(pct: float, expected_label: RegimeLabel) -> None:
    """Synthetic features producing scores in each bucket map to the right label."""
    features = _synthetic_features(spread_pct=pct, cds_pct=pct, vol_pct=pct, slope_pct=pct)
    out = score_credit_regime(features, asof=_ASOF, model_run_id="run-bucket")
    assert out.regime_label == expected_label.label


def test_score_credit_regime_release_gate_false_caps_confidence() -> None:
    features = _synthetic_features()
    out_true = score_credit_regime(features, asof=_ASOF, model_run_id="run-rg-t", release_gate=True)
    out_false = score_credit_regime(features, asof=_ASOF, model_run_id="run-rg-f", release_gate=False)
    assert out_false.release_gate is False
    assert out_false.confidence <= 0.5
    assert out_false.confidence < out_true.confidence + 1e-9


def test_score_credit_regime_missing_features_triggers_pit_audit_failure() -> None:
    """A column with zero non-NaN observations trips the audit.

    Under ``NAN_FAILS_PIT_AUDIT`` (the FI default), a feature that has
    no usable observation anywhere in the lookback window means an
    input is genuinely missing; the gate flips and confidence is
    capped at 0.5 per AGENT.md non-negotiable 8.
    """
    # Include VIX rows but null every one of them so the column exists
    # in the pivot but has zero non-NaN entries.
    features = _synthetic_features(include=("ust_level", "ust_slope", "ust_curvature", "cdx_ig_5y", "cdx_hy_5y", "vix"))
    features.attrs["nan_policy"] = NanPolicy.NAN_FAILS_PIT_AUDIT.value
    features.loc[features["feature_name"] == "vix", "value"] = float("nan")

    out = score_credit_regime(features, asof=_ASOF, model_run_id="run-missing")
    assert out.release_gate is False
    assert out.confidence <= 0.5
    assert out.metadata.get("pit_audit_failed") is True


def test_score_credit_regime_drivers_are_top_2_components() -> None:
    """Drivers are the two components most-deviated from 50.0."""
    # Push spreads & CDS to 0.95 (very risk-off), volatility neutral 0.5,
    # slope mildly stressed 0.6.
    features = _synthetic_features(spread_pct=0.95, cds_pct=0.95, vol_pct=0.5, slope_pct=0.6)
    out = score_credit_regime(features, asof=_ASOF, model_run_id="run-drivers")
    assert set(out.drivers) == {"spreads", "cds"}
    assert len(out.drivers) == 2


def test_score_credit_regime_artifact_hash_stable() -> None:
    """Same input → same artifact_hash; ``model_run_id`` is *not* part of the hash."""
    features = _synthetic_features()
    out_a = score_credit_regime(features, asof=_ASOF, model_run_id="run-A")
    out_b = score_credit_regime(features, asof=_ASOF, model_run_id="run-B")
    assert out_a.artifact_hash == out_b.artifact_hash


def test_score_credit_regime_artifact_hash_changes_when_output_changes() -> None:
    base = _synthetic_features(spread_pct=0.5)
    shifted = _synthetic_features(spread_pct=0.95)
    out_base = score_credit_regime(base, asof=_ASOF, model_run_id="run-1")
    out_shift = score_credit_regime(shifted, asof=_ASOF, model_run_id="run-1")
    assert out_base.artifact_hash != out_shift.artifact_hash


def test_write_and_latest_credit_regime_score_round_trip(tmp_path: Path) -> None:
    wh = Warehouse(str(tmp_path / "fi.duckdb"))
    try:
        features = _synthetic_features()
        out = score_credit_regime(features, asof=_ASOF, model_run_id="run-rt")
        rows = write_credit_regime_score(wh, out)
        assert rows == 1
        readback = latest_credit_regime_score(wh)
        assert readback is not None
        assert readback.model_run_id == out.model_run_id
        assert readback.timestamp == out.timestamp
        assert math.isclose(readback.regime_score, out.regime_score, rel_tol=1e-6)
        assert readback.regime_label == out.regime_label
        assert readback.drivers == out.drivers
        assert readback.release_gate is out.release_gate
        assert readback.artifact_hash == out.artifact_hash
    finally:
        wh.close()


def test_score_credit_regime_rejects_post_asof_features() -> None:
    features = _synthetic_features()
    # Plant a row whose source_timestamp is AFTER asof — PIT violation.
    future_row = _row(
        _ASOF + pd.Timedelta(hours=2),
        "ust_slope",
        2.5,
    )
    features = pd.concat([features, pd.DataFrame([future_row])], ignore_index=True)
    with pytest.raises(PitViolationError):
        score_credit_regime(features, asof=_ASOF, model_run_id="run-pit")


def test_score_credit_regime_default_weights_sum_to_one() -> None:
    assert math.isclose(sum(DEFAULT_WEIGHTS.values()), 1.0, abs_tol=1e-9)


def test_score_credit_regime_custom_weights_normalised() -> None:
    features = _synthetic_features()
    # Weights that don't sum to 1.0 must be normalised internally.
    out = score_credit_regime(
        features,
        asof=_ASOF,
        model_run_id="run-w",
        weights={"spreads": 2.0, "cds": 2.0, "treasury_curve": 1.0, "volatility": 0.0, "etf_dislocation": 0.0},
    )
    weights_used = out.metadata["weights_used"]
    assert math.isclose(sum(weights_used.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(weights_used["spreads"], 2.0 / 5.0, abs_tol=1e-9)


def test_score_credit_regime_empty_features_fails_closed() -> None:
    out = score_credit_regime(pd.DataFrame(), asof=_ASOF, model_run_id="run-empty")
    assert out.release_gate is False
    assert out.confidence == 0.0
    assert out.regime_score == 50.0
    assert out.metadata["feature_count"] == 0


def test_score_credit_regime_serialisable_metadata() -> None:
    """The whole output round-trips through ``json.dumps`` with default=str."""
    features = _synthetic_features()
    out = score_credit_regime(features, asof=_ASOF, model_run_id="run-json")
    data = {
        "drivers_json": json.dumps(list(out.drivers)),
        "metadata_json": json.dumps(out.metadata, default=str, sort_keys=True),
        "component_scores_json": json.dumps(out.component_scores, sort_keys=True),
    }
    for v in data.values():
        assert isinstance(v, str)
