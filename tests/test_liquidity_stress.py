# SPDX-License-Identifier: Apache-2.0
"""PR-4 deterministic liquidity-stress scorer acceptance suite (task H.1)."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401  - register FI schema
from market_regime_engine.fixed_income.liquidity_stress import (
    DEFAULT_WEIGHTS,
    score_liquidity_stress,
)
from market_regime_engine.fixed_income.pit_guard import PitViolationError
from market_regime_engine.fixed_income.schemas import (
    LiquidityLabel,
    LiquidityStressOutput,
)
from market_regime_engine.frontier.data_cleaning import NanPolicy

_ASOF = pd.Timestamp("2026-05-08T16:00:00+00:00")


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------


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


def _ramp(target_pct: float, n: int) -> list[float]:
    """Monotonic 0-1 ramp where the *latest* value sits at ``target_pct``."""
    base = np.linspace(0.0, 1.0, n)
    idx = max(0, min(n - 1, round(float(target_pct) * (n - 1))))
    return [float(v) for v in np.concatenate([base[idx:], base[:idx]])]


def _liquidity_features(
    *,
    asof: pd.Timestamp = _ASOF,
    n_days: int = 30,
    target_score: float = 50.0,
    include: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build features whose composite ~equals ``target_score``.

    Each component is designed so that the latest value in the trailing
    window produces a component score of ``target_score`` after the
    deterministic mapping in :mod:`liquidity_stress`. The result is a
    composite ~``target_score`` (since the weighted average of equal
    components equals the equal value).
    """
    if include is None:
        include = (
            "quote_dispersion",
            "bid_ask_width",
            "trade_count_velocity",
            "dealers_requested",
            "quotes_received",
            "amihud_illiquidity",
            "time_since_last_trade",
        )
    include_set = set(include)
    dates = pd.date_range(end=asof, periods=n_days, freq="D", tz="UTC")
    rows: list[dict] = []
    target_pct = float(target_score) / 100.0

    def _emit(feature: str, values: list[float]) -> None:
        if feature not in include_set:
            return
        for ts, val in zip(dates, values, strict=False):
            rows.append(_row(ts, feature, val))

    # quote_dispersion: latest sits at z-score that the sigmoid maps to
    # target_score. The trailing distribution is a deterministic linear
    # ramp 0..1, mean ≈ 0.5, std ≈ 0.29 (uniform), so we compute the
    # latest explicitly.
    base = np.linspace(0.0, 1.0, n_days)
    mean = float(base.mean())
    std = float(base.std(ddof=0))
    if 0.0 < target_pct < 1.0:
        target_z = -math.log(max(1.0 / target_pct - 1.0, 1e-12))
    elif target_pct <= 0.0:
        target_z = -10.0
    else:
        target_z = 10.0
    qd_latest = mean + target_z * std
    qd_values = list(base.tolist())
    qd_values[-1] = qd_latest
    _emit("quote_dispersion", qd_values)

    # bid_ask / amihud: percentile rank, direction +1.
    _emit("bid_ask_width", _ramp(target_pct, n_days))
    _emit("amihud_illiquidity", _ramp(target_pct, n_days))

    # trade_count_velocity: inverse percentile (direction -1). To get a
    # component score of ``target_score`` we need the latest to sit at
    # rank ``1 - target_pct`` of the trailing distribution.
    _emit("trade_count_velocity", _ramp(1.0 - target_pct, n_days))

    # rfq_fill_rate: stress = (1 - received/requested) * 100. To get
    # stress = target_score we need received = requested * (1 - target_pct).
    _emit("dealers_requested", [10.0] * n_days)
    _emit("quotes_received", [10.0 * (1.0 - target_pct)] * n_days)

    # time_gap: stress = min(100, latest_minutes * 2). To get stress =
    # target_score, set latest_minutes = target_score / 2.
    minutes = target_score / 2.0
    _emit("time_since_last_trade", [minutes] * n_days)

    frame = pd.DataFrame(rows)
    frame.attrs["nan_policy"] = NanPolicy.NAN_FAILS_PIT_AUDIT.value
    return frame


# ---------------------------------------------------------------------------
# Required-fields contract
# ---------------------------------------------------------------------------


def test_score_liquidity_stress_returns_output_with_required_fields() -> None:
    features = _liquidity_features(target_score=50.0)
    out = score_liquidity_stress(
        features,
        scope_type="market",
        scope_id="ALL",
        asof=_ASOF,
        model_run_id="run-fields",
    )
    assert isinstance(out, LiquidityStressOutput)
    assert out.timestamp.endswith("Z")
    assert out.scope_type == "market"
    assert out.scope_id == "ALL"
    assert 0.0 <= out.liquidity_index <= 100.0
    assert out.liquidity_label
    assert 0.0 <= out.confidence <= 1.0
    assert out.model_run_id == "run-fields"
    assert out.release_gate is True
    assert out.artifact_hash.startswith("sha256:")
    assert "weights_used" in out.metadata
    assert out.metadata["feature_count"] == len(features)
    assert "score_components" in out.metadata
    assert out.metadata["hysteresis_applied"] is False
    assert out.metadata["prev_label"] is None


# ---------------------------------------------------------------------------
# Boundedness + bucket mapping
# ---------------------------------------------------------------------------


def test_score_liquidity_stress_score_bounded_0_100() -> None:
    for target in (0.0, 25.0, 50.0, 75.0, 100.0):
        out = score_liquidity_stress(
            _liquidity_features(target_score=target),
            scope_type="market",
            scope_id="ALL",
            asof=_ASOF,
            model_run_id="run-bounded",
        )
        assert 0.0 <= out.liquidity_index <= 100.0, f"target={target}"


@pytest.mark.parametrize(
    "target,expected_label",
    [
        (5.0, LiquidityLabel.NORMAL),
        (30.0, LiquidityLabel.MILD_STRESS),
        (50.0, LiquidityLabel.ELEVATED_STRESS),
        (70.0, LiquidityLabel.SEVERE_STRESS),
        (90.0, LiquidityLabel.CRISIS_LIQUIDITY),
    ],
)
def test_score_liquidity_stress_label_bucket_mapping_no_prev_label(
    target: float, expected_label: LiquidityLabel
) -> None:
    """Without ``prev_label``, the score maps directly to the sharp bucket."""
    out = score_liquidity_stress(
        _liquidity_features(target_score=target),
        scope_type="market",
        scope_id="ALL",
        asof=_ASOF,
        model_run_id="run-bucket",
    )
    assert out.liquidity_label == expected_label.label, (
        f"target={target} got score={out.liquidity_index:.2f} -> {out.liquidity_label}"
    )


# ---------------------------------------------------------------------------
# Release-gate semantics + confidence capping
# ---------------------------------------------------------------------------


def test_score_liquidity_stress_release_gate_false_caps_confidence() -> None:
    features = _liquidity_features(target_score=50.0)
    out_t = score_liquidity_stress(
        features,
        scope_type="market",
        scope_id="ALL",
        asof=_ASOF,
        model_run_id="run-rg-t",
        release_gate=True,
    )
    out_f = score_liquidity_stress(
        features,
        scope_type="market",
        scope_id="ALL",
        asof=_ASOF,
        model_run_id="run-rg-f",
        release_gate=False,
    )
    assert out_f.release_gate is False
    assert out_f.confidence <= 0.5 + 1e-9
    assert out_f.confidence <= out_t.confidence + 1e-9


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def test_score_liquidity_stress_drivers_top_2_components() -> None:
    """Push two components to extreme and verify drivers pick those."""
    # Build a mostly-neutral fixture, then force bid_ask and amihud to
    # their max stress percentile by setting the latest value above the
    # entire trailing window.
    features = _liquidity_features(target_score=50.0)
    n = features["date"].nunique()
    asof_ts = features["date"].max()
    # Replace latest bid_ask + amihud with a value above the ramp max.
    features.loc[
        (features["feature_name"] == "bid_ask_width") & (features["date"] == asof_ts),
        "value",
    ] = 99.0
    features.loc[
        (features["feature_name"] == "amihud_illiquidity") & (features["date"] == asof_ts),
        "value",
    ] = 99.0
    out = score_liquidity_stress(
        features,
        scope_type="market",
        scope_id="ALL",
        asof=_ASOF,
        model_run_id="run-drivers",
    )
    assert len(out.drivers) == 2
    assert "bid_ask" in out.drivers
    assert "amihud" in out.drivers
    assert n  # silence unused-var lint


# ---------------------------------------------------------------------------
# Artifact-hash stability
# ---------------------------------------------------------------------------


def test_score_liquidity_stress_artifact_hash_stable_and_changes() -> None:
    """Identical inputs hash identically; the hash excludes ``model_run_id``."""
    features = _liquidity_features(target_score=50.0)
    out_a = score_liquidity_stress(
        features,
        scope_type="market",
        scope_id="ALL",
        asof=_ASOF,
        model_run_id="run-A",
    )
    out_b = score_liquidity_stress(
        features,
        scope_type="market",
        scope_id="ALL",
        asof=_ASOF,
        model_run_id="run-B",
    )
    assert out_a.artifact_hash == out_b.artifact_hash

    shifted = _liquidity_features(target_score=80.0)
    out_shift = score_liquidity_stress(
        shifted,
        scope_type="market",
        scope_id="ALL",
        asof=_ASOF,
        model_run_id="run-shift",
    )
    assert out_a.artifact_hash != out_shift.artifact_hash


def test_score_liquidity_stress_default_weights_sum_to_one() -> None:
    assert math.isclose(sum(DEFAULT_WEIGHTS.values()), 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# PIT enforcement
# ---------------------------------------------------------------------------


def test_score_liquidity_stress_rejects_post_asof_features() -> None:
    features = _liquidity_features(target_score=50.0)
    future_row = _row(
        _ASOF + pd.Timedelta(hours=2),
        "bid_ask_width",
        1.5,
    )
    features = pd.concat([features, pd.DataFrame([future_row])], ignore_index=True)
    with pytest.raises(PitViolationError):
        score_liquidity_stress(
            features,
            scope_type="market",
            scope_id="ALL",
            asof=_ASOF,
            model_run_id="run-pit",
        )


def test_score_liquidity_stress_empty_features_fails_closed() -> None:
    out = score_liquidity_stress(
        pd.DataFrame(),
        scope_type="market",
        scope_id="ALL",
        asof=_ASOF,
        model_run_id="run-empty",
    )
    assert out.release_gate is False
    assert out.confidence == 0.0
    assert out.liquidity_index == 50.0
    assert out.metadata["feature_count"] == 0


# ---------------------------------------------------------------------------
# Hysteresis flowthrough into the scorer output
# ---------------------------------------------------------------------------


def test_score_liquidity_stress_prev_label_sticky_inside_band() -> None:
    """A sticky previous label is preserved when the new score is inside its band."""
    # Build features at target=22 so the sharp bucket is MILD_STRESS but
    # prev_label=NORMAL keeps us in NORMAL (band exit is 25).
    out = score_liquidity_stress(
        _liquidity_features(target_score=22.0),
        scope_type="market",
        scope_id="ALL",
        asof=_ASOF,
        model_run_id="run-sticky",
        prev_label=LiquidityLabel.NORMAL,
    )
    assert out.liquidity_label == LiquidityLabel.NORMAL.label
    assert out.metadata["hysteresis_applied"] is True
    assert out.metadata["prev_label"] == LiquidityLabel.NORMAL.value


# ---------------------------------------------------------------------------
# Scope validation
# ---------------------------------------------------------------------------


def test_score_liquidity_stress_rejects_invalid_scope_type() -> None:
    with pytest.raises(ValueError, match="scope_type"):
        score_liquidity_stress(
            _liquidity_features(),
            scope_type="garbage",  # type: ignore[arg-type]
            scope_id="ALL",
            asof=_ASOF,
            model_run_id="run-scope",
        )


def test_score_liquidity_stress_rejects_empty_scope_id() -> None:
    with pytest.raises(ValueError, match="scope_id"):
        score_liquidity_stress(
            _liquidity_features(),
            scope_type="market",
            scope_id="",
            asof=_ASOF,
            model_run_id="run-scope",
        )
