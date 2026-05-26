# SPDX-License-Identifier: Apache-2.0
"""v1.5.1 (PR-9 FIX 8): strict missing-data fail-closed contract tests.

A missing :class:`CriticalFeature` value must force:

- ``release_gate=False``
- ``confidence <= 0.5``
- a fail-closed label (``"UNCERTAIN"`` for credit, ``"NO_DECISION"``
  for liquidity)

REGARDLESS of the active :class:`NanPolicy`. Optional (non-critical)
features fall back to the legacy re-weighting behaviour and remain
unchanged.

Tests:

- credit: bond-spread proxy missing under each NanPolicy
- credit: CDS-basis proxy missing
- liquidity: bid-ask missing
- liquidity: RFQ-response missing
- credit: an optional feature missing keeps the legacy re-weighting
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from market_regime_engine.fixed_income.credit_spread_regime import (
    score_credit_regime,
)
from market_regime_engine.fixed_income.liquidity_stress import (
    score_liquidity_stress,
)
from market_regime_engine.fixed_income.schemas import CriticalFeature
from market_regime_engine.frontier.data_cleaning import NanPolicy

# -- Credit scorer -------------------------------------------------------------


def _credit_feature_row(
    *,
    feature_name: str,
    value: float,
    date: pd.Timestamp,
) -> dict[str, object]:
    return {
        "date": date,
        "feature_name": feature_name,
        "value": value,
        "source_timestamp": date,
        "vintage_date": date,
    }


def _credit_features_full() -> pd.DataFrame:
    """A complete credit feature frame for every component."""
    asof = pd.Timestamp("2026-01-02 18:00", tz="UTC")
    rows = []
    feature_names = (
        "ust_slope",
        "ust_curvature",
        "cdx_ig_5y",
        "cdx_hy_5y",
        "move",
        "vix",
        "etf_prem_disc",
    )
    for fname in feature_names:
        for offset in range(5):
            rows.append(
                _credit_feature_row(
                    feature_name=fname,
                    value=10.0 + offset,
                    date=asof - pd.Timedelta(days=offset),
                )
            )
    return pd.DataFrame(rows)


def _credit_features_missing_spread() -> pd.DataFrame:
    """All features present except ``cdx_ig_5y`` (the credit-bond-spread proxy)."""
    frame = _credit_features_full()
    return frame[frame["feature_name"] != "cdx_ig_5y"].reset_index(drop=True)


def _credit_features_missing_cds() -> pd.DataFrame:
    frame = _credit_features_full()
    return frame[frame["feature_name"] != "cdx_hy_5y"].reset_index(drop=True)


def _credit_features_missing_optional() -> pd.DataFrame:
    """Only the optional ``etf_prem_disc`` feature is missing."""
    frame = _credit_features_full()
    return frame[frame["feature_name"] != "etf_prem_disc"].reset_index(drop=True)


@pytest.mark.parametrize(
    "policy",
    [
        NanPolicy.NAN_FAILS_PIT_AUDIT,
        NanPolicy.NAN_TO_ZERO,
        NanPolicy.NAN_TO_LAST_VALID,
        NanPolicy.NAN_DROPS_ROW,
    ],
)
def test_credit_fail_closed_when_bond_spread_missing(policy: NanPolicy) -> None:
    asof = pd.Timestamp("2026-01-02 18:00", tz="UTC")
    features = _credit_features_missing_spread()
    features.attrs["nan_policy"] = policy.value
    out = score_credit_regime(features, asof=asof)
    assert out.release_gate is False
    assert out.confidence <= 0.5
    assert out.regime_label == "UNCERTAIN"
    assert CriticalFeature.CREDIT_BOND_SPREAD.value in out.metadata["critical_features_missing"]
    assert out.metadata["critical_features_fail_closed"] is True


@pytest.mark.parametrize(
    "policy",
    [
        NanPolicy.NAN_FAILS_PIT_AUDIT,
        NanPolicy.NAN_TO_ZERO,
        NanPolicy.NAN_TO_LAST_VALID,
        NanPolicy.NAN_DROPS_ROW,
    ],
)
def test_credit_fail_closed_when_cds_basis_missing(policy: NanPolicy) -> None:
    asof = pd.Timestamp("2026-01-02 18:00", tz="UTC")
    features = _credit_features_missing_cds()
    features.attrs["nan_policy"] = policy.value
    out = score_credit_regime(features, asof=asof)
    assert out.release_gate is False
    assert out.confidence <= 0.5
    assert out.regime_label == "UNCERTAIN"
    assert CriticalFeature.CREDIT_CDS_BASIS.value in out.metadata["critical_features_missing"]


def test_credit_optional_feature_missing_does_not_trigger_critical_gate() -> None:
    """Missing the ETF-dislocation proxy is fine — it is optional."""
    asof = pd.Timestamp("2026-01-02 18:00", tz="UTC")
    features = _credit_features_missing_optional()
    features.attrs["nan_policy"] = NanPolicy.NAN_TO_ZERO.value
    out = score_credit_regime(features, asof=asof)
    assert out.metadata["critical_features_fail_closed"] is False
    # The optional-feature path keeps the regular label.
    assert out.regime_label != "UNCERTAIN"


def test_credit_full_input_passes_critical_audit() -> None:
    asof = pd.Timestamp("2026-01-02 18:00", tz="UTC")
    features = _credit_features_full()
    features.attrs["nan_policy"] = NanPolicy.NAN_FAILS_PIT_AUDIT.value
    out = score_credit_regime(features, asof=asof)
    assert out.metadata["critical_features_fail_closed"] is False
    assert out.metadata["critical_features_missing"] == []


# -- Liquidity scorer ----------------------------------------------------------


def _liquidity_feature_row(
    *,
    feature_name: str,
    value: float,
    date: pd.Timestamp,
) -> dict[str, object]:
    return {
        "date": date,
        "feature_name": feature_name,
        "value": value,
        "source_timestamp": date,
        "vintage_date": date,
    }


def _liquidity_features_full() -> pd.DataFrame:
    asof = pd.Timestamp("2026-01-02 18:00", tz="UTC")
    rows = []
    feature_names = (
        "bid_ask_width",
        "trade_count_velocity",
        "volume_over_adv",
        "quotes_received",
        "dealer_response_count",
    )
    for fname in feature_names:
        for offset in range(5):
            rows.append(
                _liquidity_feature_row(
                    feature_name=fname,
                    value=0.5 + offset * 0.1,
                    date=asof - pd.Timedelta(days=offset),
                )
            )
    return pd.DataFrame(rows)


def _liquidity_features_missing_bidask() -> pd.DataFrame:
    frame = _liquidity_features_full()
    return frame[frame["feature_name"] != "bid_ask_width"].reset_index(drop=True)


def _liquidity_features_missing_rfq() -> pd.DataFrame:
    frame = _liquidity_features_full()
    return frame[frame["feature_name"] != "quotes_received"].reset_index(drop=True)


@pytest.mark.parametrize(
    "policy",
    [
        NanPolicy.NAN_FAILS_PIT_AUDIT,
        NanPolicy.NAN_TO_ZERO,
        NanPolicy.NAN_TO_LAST_VALID,
        NanPolicy.NAN_DROPS_ROW,
    ],
)
def test_liquidity_fail_closed_when_bidask_missing(policy: NanPolicy) -> None:
    asof = pd.Timestamp("2026-01-02 18:00", tz="UTC")
    features = _liquidity_features_missing_bidask()
    features.attrs["nan_policy"] = policy.value
    out = score_liquidity_stress(features, scope_type="cusip", scope_id="037833100", asof=asof)
    assert out.release_gate is False
    assert out.confidence <= 0.5
    assert out.liquidity_label == "NO_DECISION"
    assert CriticalFeature.LIQUIDITY_BIDASK.value in out.metadata["critical_features_missing"]


@pytest.mark.parametrize(
    "policy",
    [
        NanPolicy.NAN_FAILS_PIT_AUDIT,
        NanPolicy.NAN_TO_ZERO,
        NanPolicy.NAN_TO_LAST_VALID,
        NanPolicy.NAN_DROPS_ROW,
    ],
)
def test_liquidity_fail_closed_when_rfq_response_missing(policy: NanPolicy) -> None:
    asof = pd.Timestamp("2026-01-02 18:00", tz="UTC")
    features = _liquidity_features_missing_rfq()
    features.attrs["nan_policy"] = policy.value
    out = score_liquidity_stress(features, scope_type="cusip", scope_id="037833100", asof=asof)
    assert out.release_gate is False
    assert out.confidence <= 0.5
    assert out.liquidity_label == "NO_DECISION"
    assert CriticalFeature.LIQUIDITY_RFQ_RESPONSE.value in out.metadata["critical_features_missing"]


def test_liquidity_full_input_passes_critical_audit() -> None:
    asof = pd.Timestamp("2026-01-02 18:00", tz="UTC")
    features = _liquidity_features_full()
    features.attrs["nan_policy"] = NanPolicy.NAN_FAILS_PIT_AUDIT.value
    out = score_liquidity_stress(features, scope_type="cusip", scope_id="037833100", asof=asof)
    assert out.metadata["critical_features_fail_closed"] is False
    assert out.metadata["critical_features_missing"] == []


# -- Helpers -------------------------------------------------------------------


def test_critical_feature_enum_has_expected_members() -> None:
    """Pin the enum membership so the contract is auditable from one place."""
    values = {member.value for member in CriticalFeature}
    assert values == {
        "credit_bond_spread",
        "credit_cds_basis",
        "liquidity_bidask",
        "liquidity_rfq_response",
    }


def test_audit_helper_handles_empty_frame() -> None:
    from market_regime_engine.fixed_income.critical_features import (
        CREDIT_CRITICAL_COLUMNS,
        evaluate_critical_features,
    )

    audit = evaluate_critical_features(pd.DataFrame(), contract=CREDIT_CRITICAL_COLUMNS)
    assert audit.fail_closed is True
    assert set(audit.missing) == set(CREDIT_CRITICAL_COLUMNS.values())


def test_audit_helper_handles_all_nan_column() -> None:
    """A column present but entirely NaN is still treated as missing."""
    from market_regime_engine.fixed_income.critical_features import (
        CREDIT_CRITICAL_COLUMNS,
        evaluate_critical_features,
    )

    frame = pd.DataFrame(
        {
            "cdx_ig_5y": [math.nan, math.nan],
            "cdx_hy_5y": [50.0, 55.0],
        }
    )
    audit = evaluate_critical_features(frame, contract=CREDIT_CRITICAL_COLUMNS)
    assert CriticalFeature.CREDIT_BOND_SPREAD in audit.missing
    assert CriticalFeature.CREDIT_CDS_BASIS not in audit.missing


# ---------------------------------------------------------------------------
# v1.6.0 regime_score / liquidity_index reset on fail-closed override
# (REVIEW_DEEP_V1_5_2.md A11 / Finding #11)
# ---------------------------------------------------------------------------


def test_credit_fail_closed_resets_regime_score_to_neutral() -> None:
    """REVIEW_DEEP_V1_5_2.md A11: when ``critical_audit.fail_closed`` flips
    the gate and label, the numeric ``regime_score`` must also be reset to
    50.0 (neutral midpoint). Otherwise downstream consumers see an
    inconsistent state of e.g. score=85 paired with label='UNCERTAIN'."""
    asof = pd.Timestamp("2026-01-02 18:00", tz="UTC")
    # Construct a credit panel where the bond-spread proxy is missing
    # (forcing fail-closed) but the OTHER components would compose to a
    # high score (so the un-fixed code would emit score >> 50 with the
    # UNCERTAIN label).
    rows = []
    feature_names_keep = ("ust_slope", "ust_curvature", "cdx_hy_5y", "move", "vix")
    for fname in feature_names_keep:
        for offset in range(5):
            rows.append(
                _credit_feature_row(
                    feature_name=fname,
                    value=100.0 + offset,
                    date=asof - pd.Timedelta(days=offset),
                )
            )
    features = pd.DataFrame(rows)
    features.attrs["nan_policy"] = NanPolicy.NAN_TO_LAST_VALID.value
    out = score_credit_regime(features, asof=asof)
    assert out.regime_label == "UNCERTAIN"
    assert out.release_gate is False
    # Critical fix: score must equal the neutral midpoint, not the
    # composite of partially-imputed components.
    assert out.regime_score == 50.0, f"regime_score={out.regime_score} should be reset to neutral 50.0 on fail_closed"


def test_liquidity_fail_closed_resets_index_to_neutral() -> None:
    """REVIEW_DEEP_V1_5_2.md A11: same reset must apply to the liquidity
    scorer's ``liquidity_index``."""
    asof = pd.Timestamp("2026-01-02 18:00", tz="UTC")
    rows = []
    feature_names_keep = ("trade_count_velocity", "volume_over_adv", "dealer_response_count")
    for fname in feature_names_keep:
        for offset in range(5):
            rows.append(
                _liquidity_feature_row(
                    feature_name=fname,
                    value=0.95,
                    date=asof - pd.Timedelta(days=offset),
                )
            )
    features = pd.DataFrame(rows)
    features.attrs["nan_policy"] = NanPolicy.NAN_TO_LAST_VALID.value
    out = score_liquidity_stress(features, scope_type="cusip", scope_id="037833100", asof=asof)
    assert out.liquidity_label == "NO_DECISION"
    assert out.release_gate is False
    assert out.liquidity_index == 50.0, (
        f"liquidity_index={out.liquidity_index} should be reset to neutral 50.0 on fail_closed"
    )
