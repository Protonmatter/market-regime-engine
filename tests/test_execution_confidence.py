# SPDX-License-Identifier: Apache-2.0
"""PR-5 §A acceptance tests for the execution-confidence scorer.

Pinned semantics:

- Decision rule per INSTRUCTIONS.md §6.3
  (release_gate=False → MANUAL_REVIEW_REQUIRED + human_review_required=True;
   ≥0.80 AND not severe/crisis → AUTO_X_ALLOWED;
   ≥0.60 → AUTO_X_CAUTION;
   else → MANUAL_REVIEW_REQUIRED).
- Drivers metadata holds the top-3 logit components by |magnitude|.
- ``signal_age_seconds_*`` keys are always present.
- PIT enforcement raises on post-decision signal timestamps.
- Stale-signal staleness threshold (``MRE_FI_MAX_SIGNAL_STALENESS_SEC``)
  yields ``recommended_action="Unavailable — stale signal"`` and
  ``release_gate=False`` without raising.
- Hash stability: identical inputs produce identical artifact_hash.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# Importing the FI package registers the 13 FI tables (required for
# ``write_credit_regime_score`` / ``write_liquidity_stress_score`` below).
import market_regime_engine.fixed_income  # noqa: F401
from market_regime_engine.fixed_income import (
    ExecutionConfidenceRequest,
    LiquidityLabel,
    score_credit_regime,
    score_execution_confidence,
    score_liquidity_stress,
    write_credit_regime_score,
    write_liquidity_stress_score,
)
from market_regime_engine.fixed_income.pit_guard import PitViolationError
from market_regime_engine.storage import Warehouse


@pytest.fixture
def wh(tmp_path: Path) -> Warehouse:
    return Warehouse(tmp_path / "ec.duckdb")


def _seed_signals(
    wh: Warehouse,
    *,
    asof: pd.Timestamp,
    regime_score: float = 30.0,
    liquidity_index: float = 30.0,
    cusip: str = "00206RGB6",
    liquidity_label_enum: LiquidityLabel | None = None,
) -> None:
    """Seed the warehouse with one credit-regime + one liquidity-stress row."""
    credit = _make_credit_regime_output(asof=asof, regime_score=regime_score)
    write_credit_regime_score(wh, credit)
    liq = _make_liquidity_output(
        asof=asof,
        cusip=cusip,
        liquidity_index=liquidity_index,
        label=liquidity_label_enum,
    )
    write_liquidity_stress_score(wh, liq)


def _coerce_utc(asof: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(asof)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _make_credit_regime_output(*, asof: pd.Timestamp, regime_score: float):
    """Build a CreditRegimeOutput by routing a hand-crafted feature frame
    through the deterministic scorer.

    Easier than constructing the frozen dataclass by hand because the
    scorer mints model_run_id / artifact_hash deterministically."""
    rows = []
    # ``spreads`` component drives the regime score via OAS percentile of
    # cdx_ig_5y. Seed a window where the latest value sits at the
    # requested target percentile.
    base = _coerce_utc(asof)
    # 100 days of CDX history; the latest is the *target* percentile.
    for i in range(100):
        ts = base - pd.Timedelta(days=100 - i)
        rows.append(
            {
                "date": ts,
                "feature_name": "cdx_ig_5y",
                "value": float(i),  # 0..99; pct rank of latest = 100%
                "source_timestamp": ts,
                "vintage_date": None,
            }
        )
    # Override the last value to target the requested regime_score.
    rows[-1]["value"] = float(regime_score - 1)  # roughly that percentile
    features = pd.DataFrame(rows)
    features.attrs["nan_policy"] = "NAN_TO_LAST_VALID"
    return score_credit_regime(features, asof=base, release_gate=True, profile="test")


def _make_liquidity_output(
    *,
    asof: pd.Timestamp,
    cusip: str,
    liquidity_index: float,
    label: LiquidityLabel | None,
):
    """Build a LiquidityStressOutput targeting the requested index."""
    base = _coerce_utc(asof)
    rows = []
    for i in range(100):
        ts = base - pd.Timedelta(days=100 - i)
        rows.append(
            {
                "date": ts,
                "feature_name": "bid_ask_width",
                "value": float(i),
                "source_timestamp": ts,
                "vintage_date": None,
            }
        )
    # Push the latest bid_ask_width to land near the requested percentile.
    rows[-1]["value"] = float(liquidity_index)
    features = pd.DataFrame(rows)
    features.attrs["nan_policy"] = "NAN_TO_LAST_VALID"
    out = score_liquidity_stress(
        features,
        scope_type="cusip",
        scope_id=cusip,
        asof=base,
        release_gate=True,
        profile="test",
    )
    if label is not None:
        # Re-build with the label override so we can pin the decision-rule
        # path against severe / crisis labels.
        out = type(out)(
            timestamp=out.timestamp,
            scope_type=out.scope_type,
            scope_id=out.scope_id,
            liquidity_index=out.liquidity_index,
            liquidity_label=label.label,
            confidence=out.confidence,
            drivers=out.drivers,
            model_run_id=out.model_run_id,
            release_gate=out.release_gate,
            artifact_hash=out.artifact_hash,
            metadata=out.metadata,
        )
    return out


def _request(
    *,
    cusip: str = "00206RGB6",
    side: str = "buy",
    notional: float = 1_000_000.0,
    protocol: str = "Auto-X",
    urgency: str = "normal",
    rating: str | None = "BBB+",
    limit_price: float | None = None,
    timestamp: pd.Timestamp | None = None,
    metadata: dict | None = None,
) -> ExecutionConfidenceRequest:
    ts = timestamp or pd.Timestamp("2026-05-01T16:00:00Z")
    return ExecutionConfidenceRequest(
        timestamp=ts.isoformat(),
        cusip=cusip,
        side=side,
        notional=notional,
        protocol=protocol,
        urgency=urgency,
        rating=rating,
        limit_price=limit_price,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# core decision rule
# ---------------------------------------------------------------------------


def test_scorer_emits_governance_triple_and_metadata(wh: Warehouse) -> None:
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signals(wh, asof=asof, regime_score=20.0, liquidity_index=15.0)
    request = _request(timestamp=asof + pd.Timedelta(seconds=30))
    out = score_execution_confidence(request, warehouse=wh, release_gate=True)
    assert out.model_run_id.startswith("execution_confidence-production-")
    assert out.release_gate is True
    assert out.artifact_hash and isinstance(out.artifact_hash, str)
    assert "drivers" in out.metadata
    assert out.metadata["signal_age_seconds_credit_regime"] >= 0.0
    assert out.metadata["signal_age_seconds_liquidity"] >= 0.0
    assert "max_signal_age_seconds" in out.metadata


def test_scorer_release_gate_false_fails_closed(wh: Warehouse) -> None:
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signals(wh, asof=asof, regime_score=10.0, liquidity_index=10.0)
    request = _request(timestamp=asof + pd.Timedelta(seconds=30))
    out = score_execution_confidence(request, warehouse=wh, release_gate=False)
    assert out.recommended_action == "Manual review required"
    assert out.human_review_required is True
    # release_gate flag should propagate to the response.
    assert out.release_gate is False


def test_scorer_auto_x_allowed_on_strong_signal(wh: Warehouse) -> None:
    """The deterministic baseline weights are intentionally conservative
    (the AGENT.md "explainable baselines first" non-negotiable). Per the
    plan spec, AUTO_X_ALLOWED requires score ≥ 0.80; this test pushes the
    intercept up via the ``weights`` override so we exercise the decision-
    rule top branch end-to-end. v1.5.1 swaps in a calibrated logistic
    that can reach 0.80 without the override."""
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signals(wh, asof=asof, regime_score=10.0, liquidity_index=10.0)
    request = _request(
        timestamp=asof + pd.Timedelta(seconds=30),
        notional=100_000,
        urgency="low",
        rating="AAA",
        protocol="Auto-X",
    )
    out = score_execution_confidence(
        request,
        warehouse=wh,
        release_gate=True,
        weights={"base_intercept": 2.0},  # pushes sigmoid into the 0.80+ band
    )
    assert out.confidence_score >= 0.80, out.confidence_score
    assert out.recommended_action == "Auto-X allowed"
    assert out.human_review_required is False


def test_scorer_default_weights_produce_caution_on_clean_signal(wh: Warehouse) -> None:
    """Sanity rail: with the spec's conservative baseline weights, even a
    clean IG / low-urgency / 100k-notional setup tops out in the
    ``AUTO_X_CAUTION`` band. The next-tier upgrade is the v1.5.1
    calibrated model, not a weight tweak."""
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signals(wh, asof=asof, regime_score=10.0, liquidity_index=10.0)
    request = _request(
        timestamp=asof + pd.Timedelta(seconds=30),
        notional=100_000,
        urgency="low",
        rating="AAA",
        protocol="Auto-X",
    )
    out = score_execution_confidence(request, warehouse=wh, release_gate=True)
    assert 0.60 <= out.confidence_score < 0.80, out.confidence_score
    assert out.recommended_action == "Auto-X caution / trader confirm"


def test_scorer_auto_x_blocked_under_severe_liquidity_even_with_high_confidence(
    wh: Warehouse,
) -> None:
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signals(
        wh,
        asof=asof,
        regime_score=10.0,
        liquidity_index=10.0,
        liquidity_label_enum=LiquidityLabel.SEVERE_STRESS,
    )
    request = _request(
        timestamp=asof + pd.Timedelta(seconds=30),
        notional=100_000,
        urgency="low",
        rating="AAA",
        protocol="Auto-X",
    )
    out = score_execution_confidence(request, warehouse=wh, release_gate=True)
    # Severe Stress label downgrades Auto-X to caution OR manual review,
    # never AUTO_X_ALLOWED.
    assert out.recommended_action != "Auto-X allowed"


def test_scorer_manual_review_on_weak_signal(wh: Warehouse) -> None:
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signals(wh, asof=asof, regime_score=80.0, liquidity_index=80.0)
    request = _request(
        timestamp=asof + pd.Timedelta(seconds=30),
        notional=50_000_000,
        urgency="high",
        rating="CCC",
        protocol="Manual",
    )
    out = score_execution_confidence(request, warehouse=wh, release_gate=True)
    assert out.confidence_score < 0.60
    assert out.recommended_action == "Manual review required"
    assert out.human_review_required is True


# ---------------------------------------------------------------------------
# drivers + hash stability
# ---------------------------------------------------------------------------


def test_drivers_top_three_by_magnitude(wh: Warehouse) -> None:
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signals(wh, asof=asof, regime_score=10.0, liquidity_index=10.0)
    request = _request(timestamp=asof + pd.Timedelta(seconds=30), notional=100_000)
    out = score_execution_confidence(request, warehouse=wh, release_gate=True)
    drivers = out.metadata["drivers"]
    assert len(drivers) == 3
    assert all(isinstance(d, str) for d in drivers)
    # All listed drivers must exist in logit_components.
    components = out.metadata["logit_components"]
    for name in drivers:
        assert name in components


def test_artifact_hash_is_stable_across_runs(wh: Warehouse) -> None:
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signals(wh, asof=asof, regime_score=20.0, liquidity_index=15.0)
    request = _request(timestamp=asof + pd.Timedelta(seconds=30))
    a = score_execution_confidence(request, warehouse=wh, release_gate=True, model_run_id="fixed-run-id")
    b = score_execution_confidence(request, warehouse=wh, release_gate=True, model_run_id="fixed-run-id")
    assert a.artifact_hash == b.artifact_hash


# ---------------------------------------------------------------------------
# PIT + stale signal
# ---------------------------------------------------------------------------


def test_scorer_raises_on_post_decision_signal(wh: Warehouse) -> None:
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    # Seed at *later* timestamp than the request → PIT violation.
    _seed_signals(wh, asof=asof + pd.Timedelta(hours=1))
    request = _request(timestamp=asof)
    with pytest.raises(PitViolationError):
        score_execution_confidence(request, warehouse=wh, release_gate=True)


def test_scorer_soft_fails_on_stale_signal(wh: Warehouse, monkeypatch) -> None:
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signals(wh, asof=asof - pd.Timedelta(hours=1))
    request = _request(timestamp=asof)
    monkeypatch.setenv("MRE_FI_MAX_SIGNAL_STALENESS_SEC", "60")
    out = score_execution_confidence(request, warehouse=wh, release_gate=True)
    assert out.recommended_action == "Unavailable — stale signal"
    assert out.release_gate is False
    assert out.human_review_required is True
    assert out.metadata["reason"] == "stale_signal"


def test_scorer_soft_fails_on_missing_signal(tmp_path: Path) -> None:
    wh = Warehouse(tmp_path / "empty.duckdb")
    request = _request(timestamp=pd.Timestamp("2026-05-01T16:00:00Z"))
    out = score_execution_confidence(request, warehouse=wh, release_gate=True)
    assert out.recommended_action == "Unavailable — stale signal"
    assert out.release_gate is False
    assert out.metadata["reason"] == "missing_signal"


def test_signal_age_seconds_embedded_in_metadata(wh: Warehouse) -> None:
    """PR-5 review §3.6 PR-13: every response carries the signal_age_seconds_*
    metadata keys."""
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signals(wh, asof=asof - pd.Timedelta(seconds=120))
    request = _request(timestamp=asof)
    out = score_execution_confidence(request, warehouse=wh, release_gate=True)
    assert out.metadata["signal_age_seconds_credit_regime"] >= 100
    assert out.metadata["signal_age_seconds_liquidity"] >= 100
    assert out.metadata["max_signal_age_seconds"] >= 100



# ---------------------------------------------------------------------------
# v1.6.0 PIT regression tests on build_execution_features
# (REVIEW_DEEP_V1_5_2.md A6 / Finding #15)
# ---------------------------------------------------------------------------


def test_build_execution_features_raises_on_post_decision_regime(wh: Warehouse) -> None:
    """REVIEW_DEEP_V1_5_2.md A6: the CLI / batch ``build_execution_features``
    path must mirror the hot-path PIT enforcement so future-dated regime
    rows cannot leak into offline training data."""
    from market_regime_engine.fixed_income.execution_confidence import build_execution_features

    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    # Seed signals at asof + 1 hour (FUTURE relative to the request).
    _seed_signals(wh, asof=asof + pd.Timedelta(hours=1))
    request = _request(timestamp=asof)
    with pytest.raises(PitViolationError):
        build_execution_features(wh, request)


def test_build_execution_features_passes_when_signals_are_pit_safe(wh: Warehouse) -> None:
    """Sanity rail: when signals are at or before the decision timestamp
    the builder produces a single-row feature frame without raising."""
    from market_regime_engine.fixed_income.execution_confidence import build_execution_features

    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signals(wh, asof=asof - pd.Timedelta(seconds=30))
    request = _request(timestamp=asof)
    frame = build_execution_features(wh, request)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert "regime_score" in row.index
    assert "liquidity_index" in row.index
    assert row["signal_age_seconds_credit_regime"] >= 0.0
    assert row["signal_age_seconds_liquidity"] >= 0.0
