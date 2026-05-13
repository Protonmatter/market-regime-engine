# SPDX-License-Identifier: Apache-2.0
"""Baseline tests for :mod:`market_regime_engine.frontier.bayesian_msvar`.

Phase 5.2 of v1.6.0 (REVIEW_DEEP_V1_5_2.md §4). Complements the existing
``test_bayesian_msvar.py`` (which covers the heavy NUTS-converges /
SVI-completes / EM-parity contracts) with a thin baseline-tests file
that pins:

- import-time soft-degrade contract,
- ``_coerce_panel`` domain-alignment + NaN policy,
- ``_logsumexp`` helper basic shape contract,
- ``BayesianMSVAR`` dataclass default surface,
- F14 short-panel branch resets fitted state on a synthetic-posterior
  fixture (no numpyro required).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.frontier.bayesian_msvar import (
    BayesianMSVAR,
    _coerce_panel,
    _logsumexp,
)
from market_regime_engine.frontier.data_cleaning import NanPolicy

# ---------------------------------------------------------------------------
# Smoke instantiation + dataclass defaults
# ---------------------------------------------------------------------------


def test_bayesian_msvar_default_surface():
    model = BayesianMSVAR()
    assert model.fitted is False
    # The 9-state x 8-domain default mirrors the production HMM contract.
    assert len(model.states) == 9
    assert len(model.domains) == 8
    assert model.p == 1
    assert model.last_diagnostics == {}


def test_bayesian_msvar_score_before_fit_returns_empty_frame():
    model = BayesianMSVAR(states=["a", "b"], domains=["x1", "x2"])
    panel = pd.DataFrame(
        {"x1": [0.0, 1.0], "x2": [0.5, 0.5]},
        index=pd.date_range("2020-01-01", periods=2, freq="MS"),
    )
    out = model.score(panel)
    assert out.empty
    assert "msvar_regime" in out.columns


# ---------------------------------------------------------------------------
# _coerce_panel — domain alignment + NaN policy
# ---------------------------------------------------------------------------


def test_coerce_panel_aligns_domains_and_handles_missing_columns():
    """Missing domain columns are filled with zeros so the model never
    sees a KeyError at fit time on a partial panel.
    """
    frame = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
    out = _coerce_panel(frame, ["x1", "x2", "x3"], nan_policy=NanPolicy.NAN_TO_ZERO)
    assert out.shape == (3, 3)
    assert np.allclose(out[:, 0], [1.0, 2.0, 3.0])
    assert np.allclose(out[:, 1:], 0.0)


def test_coerce_panel_empty_returns_zero_rows():
    out = _coerce_panel(pd.DataFrame(), ["x1", "x2"])
    assert out.shape == (0, 2)


# ---------------------------------------------------------------------------
# _logsumexp helper
# ---------------------------------------------------------------------------


def test_logsumexp_axis_none_handles_minus_inf():
    """``_logsumexp`` with all ``-inf`` inputs must return ``-inf`` rather
    than NaN (numerical-stability contract).
    """
    arr = np.array([-np.inf, -np.inf, -np.inf])
    out = _logsumexp(arr)
    assert float(out) == float("-inf")


def test_logsumexp_axis_int_returns_array_of_correct_shape():
    arr = np.array([[0.0, 1.0], [2.0, 3.0]])
    out = _logsumexp(arr, axis=0)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2,)


# ---------------------------------------------------------------------------
# F14 short-panel branch resets fitted state — no numpyro required
# ---------------------------------------------------------------------------


def test_short_panel_resets_fitted_state():
    """REVIEW_DEEP_V1_5_2.md F14 / Finding #19: refit on insufficient
    data clears any stale ``fitted=True`` so callers cannot accidentally
    serve a previous panel's posterior.

    The short-panel branch sits inside ``fit`` AFTER ``_require_numpyro``,
    so this test skips when numpyro isn't installed.
    """
    pytest.importorskip("numpyro")
    pytest.importorskip("jax")
    model = BayesianMSVAR(states=["a", "b"], domains=["x1", "x2"])
    model.fitted = True  # Pretend a previous fit succeeded.
    short = pd.DataFrame(
        {"x1": [0.0], "x2": [0.0]},
        index=pd.date_range("2020-01-01", periods=1, freq="MS"),
    )
    # Insufficient rows (need >= p + len(states) * 4 = 9): the early-return
    # branch must clear ``fitted`` rather than silently keeping the
    # previous fit's posterior.
    model.fit(short, method="nuts", num_warmup=10, num_samples=10, num_chains=1)
    assert model.fitted is False


def test_optional_extras_soft_degrade_install_hint(monkeypatch):
    """When numpyro isn't importable AND the panel is large enough to hit
    the model-build path, ``fit`` raises ImportError with the [bayesian]
    hint.
    """
    for mod in ("numpyro", "jax", "jax.numpy"):
        monkeypatch.setitem(sys.modules, mod, None)
    model = BayesianMSVAR(states=["a", "b"], domains=["x1", "x2"])
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        rng.normal(size=(60, 2)),
        columns=["x1", "x2"],
        index=pd.date_range("2020-01-01", periods=60, freq="MS"),
    )
    with pytest.raises(ImportError, match=r"\[bayesian\] extra"):
        model.fit(panel, method="nuts", num_warmup=10, num_samples=10, num_chains=1)
