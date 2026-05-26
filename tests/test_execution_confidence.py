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

    Seeds BOTH critical credit features (``cdx_ig_5y`` and ``cdx_hy_5y``)
    so the critical-feature audit passes — otherwise the v1.6.0 A11 fix
    will reset the regime_score to neutral 50 regardless of the requested
    target."""
    rows = []
    base = _coerce_utc(asof)
    # 100 days of complete component history; the latest is the *target* percentile.
    credit_feature_names = ("cdx_ig_5y", "cdx_hy_5y")
    for i in range(100):
        ts = base - pd.Timedelta(days=100 - i)
        for fname in credit_feature_names:
            rows.append(
                {
                    "date": ts,
                    "feature_name": fname,
                    "value": float(i),  # 0..99; pct rank of latest = 100%
                    "source_timestamp": ts,
                    "vintage_date": None,
                }
            )
    # Override the last values to target the requested regime_score.
    for r in rows[-len(credit_feature_names) :]:
        r["value"] = float(regime_score - 1)  # roughly that percentile
    features = pd.DataFrame(rows)
    features.attrs["nan_policy"] = "NAN_TO_LAST_VALID"
    out = score_credit_regime(features, asof=base, release_gate=True, profile="test")
    # These tests exercise the execution-confidence scorer, not the upstream
    # FI model completeness rails. Force the fixture signal through the
    # governance gate; dedicated tests below verify upstream gate propagation.
    return type(out)(
        timestamp=out.timestamp,
        regime_score=out.regime_score,
        regime_label=out.regime_label,
        confidence=out.confidence,
        drivers=out.drivers,
        component_scores=out.component_scores,
        model_run_id=out.model_run_id,
        release_gate=True,
        artifact_hash=out.artifact_hash,
        metadata=dict(out.metadata),
    )


def _make_liquidity_output(
    *,
    asof: pd.Timestamp,
    cusip: str,
    liquidity_index: float,
    label: LiquidityLabel | None,
):
    """Build a LiquidityStressOutput targeting the requested index.

    Seeds BOTH critical liquidity features (``bid_ask_width`` and
    ``quotes_received``) so the critical-feature audit passes — otherwise
    the v1.6.0 A11 fix will reset the liquidity_index to neutral 50."""
    base = _coerce_utc(asof)
    rows = []
    liquidity_feature_names = ("bid_ask_width", "quotes_received")
    for i in range(100):
        ts = base - pd.Timedelta(days=100 - i)
        for fname in liquidity_feature_names:
            rows.append(
                {
                    "date": ts,
                    "feature_name": fname,
                    "value": float(i),
                    "source_timestamp": ts,
                    "vintage_date": None,
                }
            )
    for r in rows[-len(liquidity_feature_names) :]:
        r["value"] = float(liquidity_index)
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
    return type(out)(
        timestamp=out.timestamp,
        scope_type=out.scope_type,
        scope_id=out.scope_id,
        liquidity_index=out.liquidity_index,
        liquidity_label=out.liquidity_label,
        confidence=out.confidence,
        drivers=out.drivers,
        model_run_id=out.model_run_id,
        release_gate=True,
        artifact_hash=out.artifact_hash,
        metadata=dict(out.metadata),
    )


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


def test_scorer_does_not_select_post_decision_signal(wh: Warehouse) -> None:
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    # Seed at *later* timestamp than the request. The scorer must not
    # select this row and then raise/null post hoc; it should perform an
    # as-of read and fail closed as missing context.
    _seed_signals(wh, asof=asof + pd.Timedelta(hours=1))
    request = _request(timestamp=asof)
    out = score_execution_confidence(request, warehouse=wh, release_gate=True)
    assert out.recommended_action == "Unavailable — stale signal"
    assert out.release_gate is False
    assert out.metadata["reason"] == "missing_signal"


def test_scorer_selects_prior_row_when_future_row_exists(wh: Warehouse) -> None:
    """P0 adversarial PIT regression: a future latest row must not block a
    valid historical row that existed at decision time."""
    t1 = pd.Timestamp("2026-05-01T15:59:30Z")
    t2 = pd.Timestamp("2026-05-01T16:00:00Z")
    t3 = pd.Timestamp("2026-05-01T17:00:00Z")
    _seed_signals(wh, asof=t1, regime_score=20.0, liquidity_index=15.0)
    _seed_signals(wh, asof=t3, regime_score=95.0, liquidity_index=95.0)

    request = _request(timestamp=t2)
    out = score_execution_confidence(request, warehouse=wh, release_gate=True)

    assert out.release_gate is True
    assert out.metadata["reason"] == "scored"
    assert out.metadata["signal_age_seconds_credit_regime"] == pytest.approx(30.0)
    assert out.metadata["signal_age_seconds_liquidity"] == pytest.approx(30.0)


def test_scorer_blocks_when_upstream_release_gate_false(wh: Warehouse) -> None:
    """P1 governance regression: caller-level release_gate=True cannot
    override an unreleased upstream regime/liquidity signal."""
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    credit = _make_credit_regime_output(asof=asof, regime_score=10.0)
    credit = type(credit)(
        timestamp=credit.timestamp,
        regime_score=credit.regime_score,
        regime_label=credit.regime_label,
        confidence=credit.confidence,
        drivers=credit.drivers,
        component_scores=credit.component_scores,
        model_run_id=credit.model_run_id,
        release_gate=False,
        artifact_hash=credit.artifact_hash,
        metadata=dict(credit.metadata),
    )
    write_credit_regime_score(wh, credit)
    write_liquidity_stress_score(
        wh,
        _make_liquidity_output(
            asof=asof,
            cusip="00206RGB6",
            liquidity_index=10.0,
            label=None,
        ),
    )

    request = _request(timestamp=asof + pd.Timedelta(seconds=30))
    out = score_execution_confidence(request, warehouse=wh, release_gate=True)

    assert out.release_gate is False
    assert out.recommended_action == "Manual review required"
    assert out.human_review_required is True
    assert out.metadata["reason"] == "upstream_release_gate_false"
    assert out.metadata["blocked_by_upstream_release_gate"] is True


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


def test_build_execution_features_does_not_select_post_decision_regime(wh: Warehouse) -> None:
    """REVIEW_DEEP_V1_5_2.md A6: the CLI / batch ``build_execution_features``
    path must mirror the hot-path PIT enforcement so future-dated regime
    rows cannot leak into offline training data."""
    from market_regime_engine.fixed_income.execution_confidence import build_execution_features

    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    # Seed signals at asof + 1 hour (FUTURE relative to the request).
    _seed_signals(wh, asof=asof + pd.Timedelta(hours=1))
    request = _request(timestamp=asof)
    frame = build_execution_features(wh, request)
    assert len(frame) == 1
    assert "regime_score" not in frame.columns
    assert "liquidity_index" not in frame.columns


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


# ---------------------------------------------------------------------------
# v1.6.0 A5 — limit_distance_bps Decimal arithmetic parity
# (REVIEW_DEEP_V1_5_2.md A5 / Finding §3.1)
# ---------------------------------------------------------------------------


def test_limit_distance_bps_decimal_arithmetic_matches_golden_fixture(
    wh: Warehouse,
) -> None:
    """Golden-fixture parity test: limit_distance_bps must be computed via
    bps_precision.to_bps / decimal_to_float_for_report instead of raw
    float multiplication.

    The pinned input ``mid=99.875, limit=100.125`` has an exact bps
    answer (25.0312890... bps) that the prior raw-float math could
    drift on at the 1e-13 level — Decimal arithmetic eliminates that
    drift. We assert the value matches the Decimal-computed reference
    exactly (no ULP slop) so any future regression to raw floats
    would fail this test.
    """
    from decimal import Decimal

    from market_regime_engine.fixed_income.bps_precision import (
        decimal_to_float_for_report,
        to_bps,
        to_decimal,
    )

    mid_price = Decimal("99.875")
    limit_price = Decimal("100.125")
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signals(wh, asof=asof, regime_score=20.0, liquidity_index=15.0)
    request = _request(
        timestamp=asof + pd.Timedelta(seconds=30),
        limit_price=float(limit_price),
        metadata={"mid_price": float(mid_price)},
    )
    out = score_execution_confidence(request, warehouse=wh, release_gate=True)

    expected_bps = decimal_to_float_for_report(
        to_bps(to_decimal(limit_price) - to_decimal(mid_price), to_decimal(mid_price))
    )
    actual_bps = out.metadata["limit_distance_bps"]
    assert actual_bps is not None
    assert actual_bps == expected_bps, f"limit_distance_bps drifted: actual={actual_bps!r} vs expected={expected_bps!r}"


def test_limit_distance_bps_returns_none_on_zero_mid(wh: Warehouse) -> None:
    """Defensive rail: a zero mid_price triggers ZeroDivisionError inside
    to_bps; the v1.6.0 contract catches it and returns ``None`` rather
    than propagating."""
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signals(wh, asof=asof, regime_score=20.0, liquidity_index=15.0)
    request = _request(
        timestamp=asof + pd.Timedelta(seconds=30),
        limit_price=100.0,
        metadata={"mid_price": 0.0},
    )
    out = score_execution_confidence(request, warehouse=wh, release_gate=True)
    assert out.metadata["limit_distance_bps"] is None


# ---------------------------------------------------------------------------
# v1.6.0 F4 — _signal_age_seconds raises PitViolationError on future delta
# (REVIEW_DEEP_V1_5_2.md F4 / Finding §3.10)
# ---------------------------------------------------------------------------


def test_signal_age_seconds_raises_on_future_signal_timestamp() -> None:
    """A signal timestamp AFTER the decision timestamp is a PIT
    violation. The helper must raise rather than silently clamping
    the delta to 0 (the v1.5.x behaviour)."""
    from market_regime_engine.fixed_income.execution_confidence import (
        _signal_age_seconds,
    )

    decision_ts = pd.Timestamp("2026-05-01T16:00:00Z")
    future_signal = "2026-05-01T17:00:00Z"
    with pytest.raises(PitViolationError, match="PIT violation"):
        _signal_age_seconds(future_signal, decision_ts)


def test_signal_age_seconds_returns_positive_delta_for_past_signal() -> None:
    from market_regime_engine.fixed_income.execution_confidence import (
        _signal_age_seconds,
    )

    decision_ts = pd.Timestamp("2026-05-01T16:00:00Z")
    past_signal = "2026-05-01T15:00:00Z"
    delta = _signal_age_seconds(past_signal, decision_ts)
    assert delta == 3600.0


def test_signal_age_seconds_returns_inf_for_none_signal() -> None:
    """Cold-start: a missing signal returns +inf — by convention this
    exceeds any sane staleness threshold."""
    import math

    from market_regime_engine.fixed_income.execution_confidence import (
        _signal_age_seconds,
    )

    decision_ts = pd.Timestamp("2026-05-01T16:00:00Z")
    assert math.isinf(_signal_age_seconds(None, decision_ts))
