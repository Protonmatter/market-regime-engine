# SPDX-License-Identifier: Apache-2.0
"""Baseline tests for :mod:`market_regime_engine.frontier.gp_cpd`.

Phase 5.2 of v1.6.0 (REVIEW_DEEP_V1_5_2.md §4 / A16 / Finding #18). Pin
the smoke surface plus the Phase-2 fixes:

- ``reset_kernel_per_panel=True`` clears an auto-trained kernel between
  panels (no silent reuse across heterogeneous panels).
- The deep-kernel transform path no longer silently swallows exceptions
  via :func:`contextlib.suppress` — failures log a warning and re-raise.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.frontier.gp_cpd import GPBOCPD


def _changepoint_panel(n_per_segment: int = 60, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=0.0, scale=1.0, size=(n_per_segment, 2))
    b = rng.normal(loc=3.0, scale=1.0, size=(n_per_segment, 2))
    arr = np.vstack([a, b])
    idx = pd.date_range("2020-01-01", periods=arr.shape[0], freq="D")
    return pd.DataFrame(arr, index=idx, columns=["x1", "x2"])


def test_gp_bocpd_score_smoke_emits_changepoint_columns():
    panel = _changepoint_panel(n_per_segment=40)
    out = GPBOCPD(max_run=24).score(panel)
    assert {
        "date",
        "change_point_prob",
        "bocpd_run_length_mean",
        "bocpd_map_run_length",
        "predictive_log_likelihood",
    }.issubset(out.columns)
    assert len(out) == len(panel)


def test_gp_bocpd_detects_synthetic_changepoint():
    panel = _changepoint_panel(n_per_segment=50, seed=1)
    out = GPBOCPD(max_run=32).score(panel)
    # The change-point at index 50 should yield an elevated cp_prob in
    # the post-change segment.
    pre_max = float(out["change_point_prob"].iloc[20:45].max())
    post_max = float(out["change_point_prob"].iloc[55:80].max())
    assert post_max > pre_max, (
        f"post-change max cp_prob {post_max:.3f} should exceed pre-change "
        f"max cp_prob {pre_max:.3f}"
    )


def test_gp_bocpd_reset_kernel_per_panel_clears_auto_trained_kernel():
    """A16 / Finding #18: panel-A's auto-trained kernel must NOT silently
    apply to panel B. With ``reset_kernel_per_panel=True`` (default) the
    kernel is dropped at the top of every ``.score`` call, so two
    consecutive ``.score`` calls each re-build (or re-fit) the kernel.
    """

    def fake_kernel(arr: np.ndarray) -> np.ndarray:
        # Identity transform — exercising the kernel-reset machinery, not
        # an actual deep-kernel fit.
        return arr

    detector = GPBOCPD(max_run=16, reset_kernel_per_panel=True, auto_train_deep_kernel=True)
    detector.deep_kernel = fake_kernel
    panel = _changepoint_panel(n_per_segment=10)
    detector.score(panel)
    # After .score, the reset-per-panel machinery must have nulled the
    # kernel reference (so the NEXT .score will re-train rather than
    # silently reuse).
    assert detector.deep_kernel is None or detector.deep_kernel is not fake_kernel


def test_gp_bocpd_deep_kernel_failure_reraises_with_warning(caplog):
    """A16 / Finding #18: a deep-kernel transform that raises must
    surface as a warning + re-raise (NOT silently swallowed via
    contextlib.suppress).
    """

    def angry_kernel(arr: np.ndarray) -> np.ndarray:
        raise RuntimeError("kernel boom")

    detector = GPBOCPD(max_run=16, deep_kernel=angry_kernel)
    panel = _changepoint_panel(n_per_segment=10)
    with (
        caplog.at_level(logging.WARNING, logger="market_regime_engine.frontier.gp_cpd"),
        pytest.raises(RuntimeError, match="kernel boom"),
    ):
        detector.score(panel)
    assert any("deep_kernel transform failed" in rec.message for rec in caplog.records)


def test_gp_bocpd_empty_panel_returns_empty_frame():
    out = GPBOCPD().score(pd.DataFrame())
    assert out.empty
    assert "change_point_prob" in out.columns
