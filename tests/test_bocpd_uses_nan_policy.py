# SPDX-License-Identifier: Apache-2.0
"""bocpd / MSVAR / GP-CPD accept ``nan_policy`` (PR-3 ASK-5 / AF-8).

Back-compat: the default ``NanPolicy.NAN_TO_ZERO`` reproduces the
legacy ``ffill().fillna(0.0)`` cleaner bit-for-bit, so the existing
fixtures keep passing. New: callers may override the policy on the
public ``score`` API for FI use cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.bocpd import DiagonalStudentTBOCPD, MultivariateNIWBOCPD
from market_regime_engine.frontier.data_cleaning import NanPolicy, PitAuditFailure
from market_regime_engine.frontier.gp_cpd import GPBOCPD


def _toy_panel(seed: int = 7) -> pd.DataFrame:
    """Small synthetic panel suitable for all three scorers."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=40, freq="MS")
    pre = rng.normal(0.0, 0.3, size=(20, 2))
    post = rng.normal(1.0, 0.3, size=(20, 2))
    arr = np.concatenate([pre, post], axis=0)
    return pd.DataFrame(arr, index=dates, columns=["labor", "credit"])


def test_bocpd_default_behavior_unchanged_with_policy_default() -> None:
    """Default ``NanPolicy.NAN_TO_ZERO`` matches the legacy cleaner output."""
    panel = _toy_panel()
    # Inject a NaN and an inf so the policy actually fires.
    panel.iloc[0, 0] = np.nan
    panel.iloc[5, 1] = np.inf

    diag = DiagonalStudentTBOCPD(hazard=0.05, max_run=12).score(panel)
    diag_explicit = DiagonalStudentTBOCPD(hazard=0.05, max_run=12).score(panel, nan_policy=NanPolicy.NAN_TO_ZERO)
    pd.testing.assert_frame_equal(diag, diag_explicit)

    niw = MultivariateNIWBOCPD(hazard=0.05, max_run=12).score(panel)
    niw_explicit = MultivariateNIWBOCPD(hazard=0.05, max_run=12).score(panel, nan_policy=NanPolicy.NAN_TO_ZERO)
    pd.testing.assert_frame_equal(niw, niw_explicit)

    gp = GPBOCPD(hazard=0.05, max_run=12).score(panel)
    gp_explicit = GPBOCPD(hazard=0.05, max_run=12).score(panel, nan_policy=NanPolicy.NAN_TO_ZERO)
    pd.testing.assert_frame_equal(gp, gp_explicit)


def test_bocpd_accepts_nan_policy_override() -> None:
    """``NAN_FAILS_PIT_AUDIT`` raises on NaN inputs for all three scorers."""
    panel = _toy_panel()
    panel.iloc[3, 0] = np.nan  # plant a missing observation

    for scorer in (
        DiagonalStudentTBOCPD(hazard=0.05, max_run=12),
        MultivariateNIWBOCPD(hazard=0.05, max_run=12),
        GPBOCPD(hazard=0.05, max_run=12),
    ):
        with pytest.raises(PitAuditFailure):
            scorer.score(panel, nan_policy=NanPolicy.NAN_FAILS_PIT_AUDIT)


def test_bocpd_per_column_override() -> None:
    """A per-column override lets callers fail only on the FI columns."""
    panel = _toy_panel()
    panel.iloc[2, 1] = np.nan

    # Default ``NAN_TO_ZERO`` succeeds; per-column flip on ``credit`` raises.
    DiagonalStudentTBOCPD(hazard=0.05, max_run=12).score(panel)
    with pytest.raises(PitAuditFailure):
        DiagonalStudentTBOCPD(hazard=0.05, max_run=12).score(
            panel,
            column_policies={"credit": NanPolicy.NAN_FAILS_PIT_AUDIT},
        )
