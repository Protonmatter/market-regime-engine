# SPDX-License-Identifier: Apache-2.0
"""Baseline tests for
:mod:`market_regime_engine.frontier.hierarchical_liquidity`.

Phase 5.2 of v1.6.0 (REVIEW_DEEP_V1_5_2.md §4). Tests the
constructor / index-mapping path and the predict back-off
machinery using a hand-crafted synthetic posterior (so the tests run
without numpyro on the dev box).
"""

from __future__ import annotations

import numpy as np
import pytest

from market_regime_engine.frontier.hierarchical_liquidity import (
    HierarchicalLiquidityModel,
)


def test_hierarchical_liquidity_model_default_surface():
    model = HierarchicalLiquidityModel(sectors=["fin"], ratings=["IG"])
    assert model.fitted is False
    assert model._sector_to_idx == {"fin": 0}
    assert model._rating_to_idx == {"IG": 0}


def test_predict_before_fit_raises_runtime_error():
    model = HierarchicalLiquidityModel(sectors=["fin"], ratings=["IG"])
    with pytest.raises(RuntimeError, match=r"called before fit"):
        model.predict(cusip="123")


def test_diagnostics_before_fit_raises_runtime_error():
    model = HierarchicalLiquidityModel(sectors=["fin"], ratings=["IG"])
    with pytest.raises(RuntimeError, match=r"called before fit"):
        model.diagnostics()


def test_predict_market_path_with_synthetic_posterior():
    """Inject a synthetic posterior so we exercise the back-off path
    without a NumPyro NUTS run (which would require the optional dep).
    """
    model = HierarchicalLiquidityModel(sectors=["fin", "ind"], ratings=["IG", "HY"])
    model.fitted = True
    rng = np.random.default_rng(0)
    n_samples = 500
    n_sectors = 2
    n_ratings = 2
    n_cusips = 4
    model._posterior_samples = {
        "mu_global": rng.normal(loc=2.0, scale=0.5, size=n_samples),
        "group_effect": rng.normal(scale=0.2, size=(n_samples, n_sectors, n_ratings)),
        "cusip_effect": rng.normal(scale=0.3, size=(n_samples, n_cusips)),
    }
    model._cusip_to_idx = {"a": 0, "b": 1, "c": 2, "d": 3}
    model._cusip_metadata = {
        "a": ("fin", "IG"),
        "b": ("fin", "HY"),
        "c": ("ind", "IG"),
        "d": ("ind", "HY"),
    }

    out_market = model.predict()
    assert out_market["hierarchy_level"] == "market"
    assert np.isfinite(out_market["posterior_mean"])
    assert out_market["ci_low_5"] <= out_market["ci_high_95"]
    assert out_market["n_obs"] == 4

    out_cusip = model.predict(cusip="a")
    assert out_cusip["hierarchy_level"] == "cusip"
    assert out_cusip["n_obs"] == 1


def test_predict_sector_rating_back_off():
    model = HierarchicalLiquidityModel(sectors=["fin"], ratings=["IG", "HY"])
    model.fitted = True
    rng = np.random.default_rng(1)
    n_samples = 200
    model._posterior_samples = {
        "mu_global": rng.normal(loc=1.0, scale=0.4, size=n_samples),
        "group_effect": rng.normal(scale=0.2, size=(n_samples, 1, 2)),
        "cusip_effect": rng.normal(scale=0.3, size=(n_samples, 1)),
    }
    model._cusip_to_idx = {"a": 0}
    model._cusip_metadata = {"a": ("fin", "IG")}
    out = model.predict(sector="fin", rating="HY")
    assert out["hierarchy_level"] == "sector_rating"
    # n_obs counts cusips that match the (sector, rating) — none with "HY" here.
    assert out["n_obs"] == 0
