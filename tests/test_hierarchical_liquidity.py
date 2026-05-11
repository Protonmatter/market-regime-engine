# SPDX-License-Identifier: Apache-2.0
"""Hierarchical Bayesian liquidity model acceptance tests (task H.5).

The model lives behind the optional ``[bayesian]`` extra so the bulk
of these tests are skipped when ``numpyro`` isn't installed. The one
non-skipped test pins the import-time soft-degrade contract.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest


def _numpyro_available() -> bool:
    try:
        import numpyro  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Soft-degrade contract — runs on every install
# ---------------------------------------------------------------------------


def test_hierarchical_liquidity_imports_only_when_bayesian_extra(monkeypatch) -> None:
    """``fit`` raises a clean ``ImportError`` with install hint when numpyro is missing."""
    # Stub ``numpyro`` and ``jax`` so the internal ``_require_numpyro`` raises.
    monkeypatch.setitem(sys.modules, "numpyro", None)
    monkeypatch.setitem(sys.modules, "numpyro.distributions", None)
    monkeypatch.setitem(sys.modules, "numpyro.infer", None)
    monkeypatch.setitem(sys.modules, "jax", None)
    monkeypatch.setitem(sys.modules, "jax.numpy", None)

    from market_regime_engine.frontier.hierarchical_liquidity import (
        HierarchicalLiquidityModel,
    )

    model = HierarchicalLiquidityModel(sectors=["financials"], ratings=["AA"])
    panel = pd.DataFrame(
        {
            "cusip": ["A"],
            "sector": ["financials"],
            "rating": ["AA"],
            "liquidity_value": [1.0],
        }
    )
    with pytest.raises(ImportError, match=r"\[bayesian\] extra"):
        model.fit(panel)


# ---------------------------------------------------------------------------
# Numpyro-gated smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _numpyro_available(), reason="numpyro not installed")
def test_hierarchical_liquidity_fits_on_synthetic_panel() -> None:
    """The model fits a small synthetic panel and exposes posterior samples."""
    from market_regime_engine.frontier.hierarchical_liquidity import (
        HierarchicalLiquidityModel,
    )

    rng = np.random.default_rng(0)
    sectors = ["financials", "energy"]
    ratings = ["AA", "BBB"]
    cusips = [f"CUS{i:02d}" for i in range(8)]
    rows = []
    for i, cusip in enumerate(cusips):
        sector = sectors[i % 2]
        rating = ratings[(i // 2) % 2]
        for _ in range(5):
            value = float(rng.normal(loc=0.5 if sector == "financials" else 0.0, scale=0.3))
            rows.append({"cusip": cusip, "sector": sector, "rating": rating, "liquidity_value": value})
    panel = pd.DataFrame(rows)
    model = HierarchicalLiquidityModel(
        sectors=sectors,
        ratings=ratings,
        num_warmup=50,
        num_samples=100,
        num_chains=1,
        seed=0,
    )
    model.fit(panel)
    assert model.fitted is True


@pytest.mark.skipif(not _numpyro_available(), reason="numpyro not installed")
def test_hierarchical_liquidity_predicts_at_cusip_sector_rating_levels() -> None:
    from market_regime_engine.frontier.hierarchical_liquidity import (
        HierarchicalLiquidityModel,
    )

    rng = np.random.default_rng(0)
    sectors = ["financials", "energy"]
    ratings = ["AA", "BBB"]
    cusips = [f"CUS{i:02d}" for i in range(6)]
    rows = []
    for i, cusip in enumerate(cusips):
        sector = sectors[i % 2]
        rating = ratings[(i // 2) % 2]
        for _ in range(4):
            rows.append(
                {
                    "cusip": cusip,
                    "sector": sector,
                    "rating": rating,
                    "liquidity_value": float(rng.normal(0.0, 0.3)),
                }
            )
    panel = pd.DataFrame(rows)
    model = HierarchicalLiquidityModel(
        sectors=sectors,
        ratings=ratings,
        num_warmup=50,
        num_samples=100,
        num_chains=1,
        seed=1,
    )
    model.fit(panel)

    cusip_pred = model.predict(cusip="CUS00")
    assert cusip_pred["hierarchy_level"] == "cusip"
    assert "posterior_mean" in cusip_pred
    assert cusip_pred["ci_low_5"] <= cusip_pred["posterior_mean"] <= cusip_pred["ci_high_95"]

    sr_pred = model.predict(sector="financials", rating="AA")
    assert sr_pred["hierarchy_level"] == "sector_rating"
    assert sr_pred["n_obs"] >= 1

    rating_pred = model.predict(rating="AA")
    assert rating_pred["hierarchy_level"] == "rating"

    market_pred = model.predict()
    assert market_pred["hierarchy_level"] == "market"

    # Unseen cusip falls back to market.
    unseen = model.predict(cusip="DOES-NOT-EXIST")
    assert unseen["hierarchy_level"] == "market"


@pytest.mark.skipif(not _numpyro_available(), reason="numpyro not installed")
def test_hierarchical_liquidity_diagnostics_returns_rhat() -> None:
    from market_regime_engine.frontier.hierarchical_liquidity import (
        HierarchicalLiquidityModel,
    )

    rng = np.random.default_rng(0)
    rows = []
    for i in range(4):
        for _ in range(4):
            rows.append(
                {
                    "cusip": f"CUS{i:02d}",
                    "sector": "financials",
                    "rating": "AA",
                    "liquidity_value": float(rng.normal(0.0, 0.3)),
                }
            )
    panel = pd.DataFrame(rows)
    model = HierarchicalLiquidityModel(
        sectors=["financials"],
        ratings=["AA"],
        num_warmup=50,
        num_samples=100,
        num_chains=1,
        seed=2,
    )
    model.fit(panel)

    diag = model.diagnostics()
    assert "r_hat" in diag
    assert "n_eff" in diag
    # Sanity: at least one parameter has a finite r_hat.
    finite_r_hats = [v for v in diag["r_hat"].values() if v is not None and np.isfinite(v)]
    assert finite_r_hats, "expected at least one finite r_hat in diagnostics"
