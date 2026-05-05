"""Tests for the Diebold-Mariano / Giacomini-White / MCS / PIT / Christoffersen / Murphy stack."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_regime_engine.forecast_compare import (
    christoffersen_coverage,
    diebold_mariano,
    giacomini_white,
    hansen_mcs,
    murphy_decomposition,
    pit_uniformity,
)


def test_dm_detects_clearly_better_model():
    rng = np.random.default_rng(0)
    n = 400
    a = rng.normal(loc=1.0, scale=0.5, size=n)  # higher loss
    b = rng.normal(loc=0.6, scale=0.5, size=n)  # lower loss
    res = diebold_mariano(a, b, h=1)
    assert res.pvalue < 0.05
    assert res.direction == "model_b_better"


def test_dm_returns_tie_for_equal_losses():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.5, scale=0.4, size=300)
    b = rng.normal(loc=0.5, scale=0.4, size=300)
    res = diebold_mariano(a, b, h=1)
    # large variance, equal means -> p-value should not strongly reject
    assert res.pvalue > 0.01


def test_gw_unconditional_recovers_dm_direction():
    rng = np.random.default_rng(1)
    n = 300
    a = rng.normal(loc=0.8, scale=0.4, size=n)
    b = rng.normal(loc=0.5, scale=0.4, size=n)
    res = giacomini_white(a, b)
    assert res["pvalue"] < 0.05


def test_mcs_keeps_only_best_when_one_dominates():
    rng = np.random.default_rng(0)
    n = 200
    losses = pd.DataFrame(
        {
            "best": rng.normal(loc=0.20, scale=0.05, size=n),
            "mid": rng.normal(loc=0.50, scale=0.05, size=n),
            "worst": rng.normal(loc=0.80, scale=0.05, size=n),
        }
    )
    out = hansen_mcs(losses, confidence=0.10, bootstrap=300, block_size=10, seed=0)
    # Worst should always be eliminated; best should always survive.
    assert "worst" not in out["mcs"]
    assert "best" in out["mcs"]


def test_pit_uniformity_passes_for_uniform_data():
    rng = np.random.default_rng(0)
    u = rng.uniform(size=2000)
    res = pit_uniformity(u, bins=10)
    assert res["chi2_pvalue"] > 0.01


def test_pit_uniformity_rejects_skewed_data():
    rng = np.random.default_rng(0)
    skewed = rng.beta(2.0, 5.0, size=2000)  # very non-uniform
    res = pit_uniformity(skewed, bins=10)
    assert res["chi2_pvalue"] < 0.05


def test_christoffersen_passes_for_correctly_calibrated_hits():
    rng = np.random.default_rng(0)
    alpha = 0.05
    hits = (rng.uniform(size=600) < alpha).astype(int)
    res = christoffersen_coverage(hits, alpha)
    assert res["uc_pvalue"] > 0.05


def test_christoffersen_rejects_when_too_many_hits():
    rng = np.random.default_rng(0)
    hits = (rng.uniform(size=600) < 0.30).astype(int)
    res = christoffersen_coverage(hits, alpha=0.05)
    assert res["uc_pvalue"] < 0.05


def test_murphy_decomposition_brier_matches_components():
    rng = np.random.default_rng(0)
    n = 1000
    y = rng.binomial(1, 0.3, size=n).astype(float)
    p = np.clip(0.3 + rng.normal(scale=0.1, size=n), 0.01, 0.99)
    res = murphy_decomposition(y, p, bins=10)
    # MCB - DSC + UNC ~= Brier within 1e-2 (binning round-off)
    assert abs(res["reliability"] - res["resolution"] + res["uncertainty"] - res["brier"]) < 5e-2
