# SPDX-License-Identifier: Apache-2.0
"""Label-hysteresis acceptance tests (PR-4 task C.3).

The hysteresis machinery is shared between :mod:`fixed_income.credit_spread_regime`
and :mod:`fixed_income.liquidity_stress`. Both modules ship their own
``HYSTERESIS_BANDS_*`` table and a ``classify_with_hysteresis`` helper
that delegates to :func:`fixed_income.hysteresis.apply_hysteresis`. The
contracts under test:

1. No ``prev_label`` → sharp-bucket fallback (back-compat with PR-3
   callers that never pass ``prev_label=``).
2. ``prev_label`` is sticky inside its band (Schmitt trigger).
3. Outside the band, the score re-classifies via the sharp buckets.
4. The hysteresis state propagates to the credit module too (retro).
5. The output ``metadata`` records ``hysteresis_applied`` and
   ``prev_label`` so downstream telemetry can audit transitions.
"""

from __future__ import annotations

import pandas as pd

import market_regime_engine.fixed_income  # noqa: F401  - register FI schema
from market_regime_engine.fixed_income.credit_spread_regime import (
    HYSTERESIS_BANDS_CREDIT,
    score_credit_regime,
)
from market_regime_engine.fixed_income.credit_spread_regime import (
    classify_with_hysteresis as classify_credit_with_hysteresis,
)
from market_regime_engine.fixed_income.liquidity_stress import (
    HYSTERESIS_BANDS_LIQUIDITY,
)
from market_regime_engine.fixed_income.liquidity_stress import (
    classify_with_hysteresis as classify_liquidity_with_hysteresis,
)
from market_regime_engine.fixed_income.schemas import (
    LiquidityLabel,
    RegimeLabel,
    liquidity_label_from_score,
    regime_label_from_score,
)

_ASOF = pd.Timestamp("2026-05-08T16:00:00+00:00")


def _credit_features(asof: pd.Timestamp = _ASOF, n: int = 30) -> pd.DataFrame:
    """Minimal feature frame so :func:`score_credit_regime` can exit
    cleanly without tripping the PIT audit."""
    dates = pd.date_range(end=asof, periods=n, freq="D", tz="UTC")
    rows: list[dict] = []
    for ts in dates:
        rows.append(
            {"date": ts, "feature_name": "ust_slope", "value": 0.5, "source_timestamp": ts, "vintage_date": None}
        )
        rows.append(
            {"date": ts, "feature_name": "ust_curvature", "value": 0.1, "source_timestamp": ts, "vintage_date": None}
        )
        rows.append(
            {"date": ts, "feature_name": "cdx_ig_5y", "value": 65.0, "source_timestamp": ts, "vintage_date": None}
        )
        rows.append(
            {"date": ts, "feature_name": "cdx_hy_5y", "value": 350.0, "source_timestamp": ts, "vintage_date": None}
        )
        rows.append({"date": ts, "feature_name": "vix", "value": 18.0, "source_timestamp": ts, "vintage_date": None})
        rows.append({"date": ts, "feature_name": "move", "value": 100.0, "source_timestamp": ts, "vintage_date": None})
        rows.append(
            {"date": ts, "feature_name": "etf_prem_disc", "value": 0.10, "source_timestamp": ts, "vintage_date": None}
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §C.3 generic hysteresis contract — liquidity reference implementation
# ---------------------------------------------------------------------------


def test_hysteresis_no_prev_label_falls_back_to_sharp_buckets() -> None:
    """Cold-start path: ``prev_label is None`` → sharp bucket mapping for every score."""
    for score in (0.0, 19.99, 20.0, 39.99, 40.0, 59.99, 60.0, 79.99, 80.0, 100.0):
        lab = classify_liquidity_with_hysteresis(score, None)
        assert lab == liquidity_label_from_score(score), f"liquidity score={score}"
        reg = classify_credit_with_hysteresis(score, None)
        assert reg == regime_label_from_score(score), f"credit score={score}"


def test_hysteresis_stays_in_label_until_exit_threshold() -> None:
    """Inside the band, ``prev_label`` is sticky (Schmitt trigger)."""
    # MILD_STRESS band is (20, 45); a score of 30 sits squarely inside, so
    # a previous label of MILD_STRESS must persist, ignoring the sharp
    # bucket which would put 30 in MILD_STRESS too. The interesting case
    # is a previous NORMAL with score 22 — sharp bucket says MILD_STRESS,
    # but hysteresis keeps us in NORMAL until the score crosses 25.
    assert classify_liquidity_with_hysteresis(22.0, LiquidityLabel.NORMAL) is LiquidityLabel.NORMAL
    # ELEVATED_STRESS band is (40, 65); inside the band a previous
    # ELEVATED stays ELEVATED even at a sharp-bucket WATCH_TRANSITION-ish
    # score of 44.
    assert classify_liquidity_with_hysteresis(44.0, LiquidityLabel.ELEVATED_STRESS) is LiquidityLabel.ELEVATED_STRESS
    # CRISIS_LIQUIDITY band is (80, None) — terminal upper edge.
    assert classify_liquidity_with_hysteresis(95.0, LiquidityLabel.CRISIS_LIQUIDITY) is LiquidityLabel.CRISIS_LIQUIDITY


def test_hysteresis_transitions_when_score_clears_exit() -> None:
    """Outside the band the helper re-classifies by sharp buckets."""
    # NORMAL → MILD_STRESS only after the score clears the NORMAL exit
    # threshold of 25 (not the sharp bucket boundary of 20).
    assert classify_liquidity_with_hysteresis(24.99, LiquidityLabel.NORMAL) is LiquidityLabel.NORMAL
    assert classify_liquidity_with_hysteresis(25.0, LiquidityLabel.NORMAL) is LiquidityLabel.MILD_STRESS
    # MILD_STRESS → ELEVATED_STRESS at 45+ (the band's exit threshold).
    assert classify_liquidity_with_hysteresis(45.0, LiquidityLabel.MILD_STRESS) is LiquidityLabel.ELEVATED_STRESS
    # Downward transition: ELEVATED_STRESS → NORMAL when the score drops
    # below ELEVATED's enter threshold (40). The intermediate MILD_STRESS
    # is skipped because the sharp-bucket re-classify maps 18 → NORMAL.
    assert classify_liquidity_with_hysteresis(18.0, LiquidityLabel.ELEVATED_STRESS) is LiquidityLabel.NORMAL
    # CRISIS_LIQUIDITY → SEVERE_STRESS when score drops below 80.
    assert classify_liquidity_with_hysteresis(75.0, LiquidityLabel.CRISIS_LIQUIDITY) is LiquidityLabel.SEVERE_STRESS


def test_hysteresis_applied_to_credit_regime_too() -> None:
    """The credit module mirrors the liquidity implementation (retro)."""
    # RISK_ON_COMPRESSION band is (None, 25); sticky up to 25.
    assert classify_credit_with_hysteresis(22.0, RegimeLabel.RISK_ON_COMPRESSION) is RegimeLabel.RISK_ON_COMPRESSION
    # WATCH_TRANSITION band is (40, 65); a previous WATCH stays WATCH
    # at a sharp-bucket NORMAL_LIQUIDITY-ish 44.
    assert classify_credit_with_hysteresis(44.0, RegimeLabel.WATCH_TRANSITION) is RegimeLabel.WATCH_TRANSITION
    assert (
        classify_credit_with_hysteresis(65.0, RegimeLabel.WATCH_TRANSITION) is RegimeLabel.RISK_OFF_HIGH_RISK_AVERSION
    )
    # CRISIS_SEVERE_DISLOCATION → drop below 80 transitions out.
    assert (
        classify_credit_with_hysteresis(70.0, RegimeLabel.CRISIS_SEVERE_DISLOCATION)
        is RegimeLabel.RISK_OFF_HIGH_RISK_AVERSION
    )


def test_hysteresis_metadata_tracks_prev_label_and_applied_flag() -> None:
    """``score_credit_regime`` writes ``hysteresis_applied`` and ``prev_label`` to metadata."""
    features = _credit_features()
    # Cold start.
    out_cold = score_credit_regime(features, asof=_ASOF, model_run_id="run-cold")
    assert out_cold.metadata["hysteresis_applied"] is False
    assert out_cold.metadata["prev_label"] is None
    # With prev_label supplied — even when the new sharp bucket is the
    # same, the metadata reflects the applied policy.
    out_warm = score_credit_regime(
        features,
        asof=_ASOF,
        model_run_id="run-warm",
        prev_label=RegimeLabel.WATCH_TRANSITION,
    )
    assert out_warm.metadata["hysteresis_applied"] is True
    assert out_warm.metadata["prev_label"] == RegimeLabel.WATCH_TRANSITION.value


# ---------------------------------------------------------------------------
# §H.4 spec-named tests so the acceptance harness picks them up
# ---------------------------------------------------------------------------


def test_hysteresis_bands_credit_cover_all_labels() -> None:
    """Defence-in-depth: every credit label has an entry in the band table."""
    assert set(HYSTERESIS_BANDS_CREDIT) == set(RegimeLabel)


def test_hysteresis_bands_liquidity_cover_all_labels() -> None:
    """Defence-in-depth: every liquidity label has an entry in the band table."""
    assert set(HYSTERESIS_BANDS_LIQUIDITY) == set(LiquidityLabel)
