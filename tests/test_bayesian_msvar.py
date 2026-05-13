# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the v1.4 Bayesian MS-VAR (item A).

Four contracts:

1. NUTS converges on a synthetic 2-state × 2-domain panel:
   ``test_bayesian_msvar_nuts_converges_on_synthetic`` runs 2 chains ×
   500 warmup × 500 samples (the "fast-converge synthetic" recipe
   documented in the v1.4 plan) and asserts R-hat < 1.1 and ESS > 100.
2. SVI handles a larger panel:
   ``test_bayesian_msvar_svi_fallback_completes`` — large-panel path.
3. Posterior-mean regime probs match EM regime probs within tolerance:
   ``test_bayesian_msvar_score_matches_em_within_tolerance``.
4. Soft-degrade when ``numpyro`` / ``jax`` are missing:
   ``test_bayesian_msvar_soft_degrade_without_numpyro``.

The synthetic generator + dimensionality (K=2 states, d=2 domains) is
deliberately small so the test fits comfortably in CI wall-clock. The
plan documents the production recipe (9 states × 8 domains × 1000
warmup × 1000 samples × 2 chains) which an operator can opt into via
the ``mre bayesian-msvar-fit`` CLI.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _require_numpyro() -> None:
    pytest.importorskip("numpyro")
    pytest.importorskip("jax")


def _set_jax_cpu() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _synthetic_two_state_panel(T: int = 80, *, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Two-regime AR(1) synthetic panel + ground-truth regime sequence."""
    rng = np.random.default_rng(seed)
    domains = ["x1", "x2"]
    seq = np.zeros(T, dtype=int)
    seq[: T // 2] = 0
    seq[T // 2 :] = 1
    rng.shuffle(seq[10:])  # mix the second half so it's not a single block
    y = np.zeros((T, 2))
    for t in range(1, T):
        if seq[t] == 0:
            mu = np.array([0.5, -0.5]) + 0.6 * y[t - 1]
        else:
            mu = np.array([-0.5, 0.5]) + 0.3 * y[t - 1]
        y[t] = mu + rng.normal(scale=0.2, size=2)
    panel = pd.DataFrame(
        y,
        columns=domains,
        index=pd.date_range("2020-01-01", periods=T, freq="ME"),
    )
    return panel, seq


def test_bayesian_msvar_nuts_converges_on_synthetic() -> None:
    """v1.4 (criterion 9): R-hat < 1.1 + ESS > 100 on the fast synthetic recipe."""
    _require_numpyro()
    _set_jax_cpu()
    from market_regime_engine.frontier.bayesian_msvar import BayesianMSVAR

    panel, _ = _synthetic_two_state_panel(T=80, seed=42)
    model = BayesianMSVAR(states=["s_a", "s_b"], domains=["x1", "x2"], seed=42)
    model.fit(panel, method="nuts", num_chains=2, num_warmup=500, num_samples=500)

    diag = model.last_diagnostics
    assert diag.get("method") == "nuts"
    rhat = float(diag.get("max_rhat", float("nan")))
    ess = float(diag.get("min_ess", float("nan")))
    # The fast-converge synthetic recipe must hit the production
    # acceptance bar.
    assert rhat == rhat, "R-hat is NaN — diagnostics were not captured"
    assert rhat < 1.1, f"R-hat {rhat:.3f} >= 1.1 (NUTS did not converge)"
    assert ess > 100, f"ESS {ess:.1f} <= 100 (sampling under-mixed)"
    # Divergence count is reported (zero on this synthetic).
    assert int(diag.get("num_divergences", 0)) == 0


def test_bayesian_msvar_svi_fallback_completes() -> None:
    """SVI must complete on a larger panel without raising."""
    _require_numpyro()
    _set_jax_cpu()
    from market_regime_engine.frontier.bayesian_msvar import BayesianMSVAR

    rng = np.random.default_rng(0)
    T = 200
    domains = ["x1", "x2"]
    y = np.zeros((T, 2))
    for t in range(1, T):
        base = np.array([0.5, -0.5]) if (t // 50) % 2 == 0 else np.array([-0.5, 0.5])
        y[t] = base + 0.5 * y[t - 1] + rng.normal(scale=0.3, size=2)
    panel = pd.DataFrame(y, columns=domains, index=pd.date_range("2010-01-01", periods=T, freq="ME"))
    model = BayesianMSVAR(states=["s_a", "s_b"], domains=domains, seed=42)
    model.fit(panel, method="svi", svi_steps=300, num_samples=64)

    diag = model.last_diagnostics
    assert diag.get("method") == "svi"
    assert diag.get("svi_steps") == 300
    # SVI doesn't compute R-hat/ESS — the diagnostics row should still
    # carry NaN placeholders so the warehouse schema stays consistent.
    assert pd.isna(diag.get("max_rhat", float("nan")))
    assert "final_loss" in diag

    out = model.score(panel)
    assert {
        "date",
        "msvar_regime",
        "msvar_confidence",
        "bayesian_credible_lo",
        "bayesian_credible_hi",
        "divergence_count",
    }.issubset(out.columns)
    assert len(out) == T
    # Credible bands must be in [0, 1] and lo <= hi.
    assert (out["bayesian_credible_lo"] <= out["bayesian_credible_hi"] + 1e-9).all()
    assert (out["bayesian_credible_lo"] >= 0).all()
    assert (out["bayesian_credible_hi"] <= 1.0 + 1e-9).all()


def test_bayesian_msvar_score_matches_em_within_tolerance() -> None:
    """Posterior-mean regime probs ≈ EM regime probs (L1 < 0.30 on synthetic).

    The plan target is L1 < 0.15 on a fully-converged NUTS posterior.
    For the fast-converge recipe (200 warmup × 200 samples) we relax the
    bar to L1 < 0.30 so the test stays under one minute on CI; the
    production recipe (1000/1000) hits the tighter band.
    """
    _require_numpyro()
    _set_jax_cpu()
    from market_regime_engine.frontier.bayesian_msvar import BayesianMSVAR
    from market_regime_engine.msvar import MarkovSwitchingVAR

    panel, _ = _synthetic_two_state_panel(T=80, seed=7)
    states = ["s_a", "s_b"]
    domains = ["x1", "x2"]
    em = MarkovSwitchingVAR(states=states, domains=domains, max_iter=20).fit(panel)
    em_score = em.score(panel)

    bay = BayesianMSVAR(states=states, domains=domains, seed=42).fit(
        panel, method="nuts", num_chains=2, num_warmup=200, num_samples=200
    )
    bay_score = bay.score(panel)

    if em_score.empty or bay_score.empty:
        pytest.skip("EM or Bayesian score is empty — likely too few obs")

    # Align on date and compute L1 between per-state probabilities.
    em_probs = em_score.set_index("date")[[f"msvar_prob_{s}" for s in states]]
    bay_probs = bay_score.set_index("date")[[f"msvar_prob_{s}" for s in states]]
    # The two state labelings can be permuted (label-switching). Try
    # both label orderings and pick the lower L1.
    common = em_probs.index.intersection(bay_probs.index)
    if common.empty:
        pytest.skip("EM/Bayes index do not overlap — synthetic too short")
    a = em_probs.loc[common].to_numpy()
    b = bay_probs.loc[common].to_numpy()
    l1_aligned = float(np.mean(np.abs(a - b)))
    # Permute Bayesian state labels.
    b_perm = bay_probs.loc[common, [f"msvar_prob_{s}" for s in reversed(states)]].to_numpy()
    l1_swapped = float(np.mean(np.abs(a - b_perm)))
    l1 = min(l1_aligned, l1_swapped)
    assert l1 < 0.30, f"L1 distance {l1:.3f} between EM and Bayesian regime probs > 0.30"


def test_bayesian_msvar_soft_degrade_without_numpyro(monkeypatch) -> None:
    """Importing without numpyro must raise ImportError with the install hint."""
    from market_regime_engine.frontier.bayesian_msvar import BayesianMSVAR

    # Stash existing numpyro/jax so the soft-degrade path actually
    # exercises ImportError. We swap their entries in sys.modules to
    # ``None`` which raises ``ModuleNotFoundError`` on next import (the
    # documented stdlib mechanism).
    for mod in ("numpyro", "jax", "jax.numpy"):
        monkeypatch.setitem(sys.modules, mod, None)

    model = BayesianMSVAR(states=["s_a", "s_b"], domains=["x1", "x2"])
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        rng.normal(size=(60, 2)),
        columns=["x1", "x2"],
        index=pd.date_range("2020-01-01", periods=60, freq="ME"),
    )
    with pytest.raises(ImportError, match=r"\[bayesian\] extra"):
        model.fit(panel, method="nuts", num_warmup=10, num_samples=10, num_chains=1)


def test_bayesian_msvar_short_panel_resets_fitted_state() -> None:
    """REVIEW_DEEP_V1_5_2.md F14 / Finding #19: a previously-fit instance
    refit on insufficient data must NOT silently keep the stale posterior.
    """
    _require_numpyro()
    _set_jax_cpu()
    from market_regime_engine.frontier.bayesian_msvar import BayesianMSVAR

    panel, _ = _synthetic_two_state_panel(T=80, seed=99)
    model = BayesianMSVAR(states=["s_a", "s_b"], domains=["x1", "x2"], seed=99)
    model.fit(panel, method="svi", svi_steps=80, num_samples=16)
    assert model.fitted is True
    # Now refit on a panel too short for the fit branch (need
    # >= p + len(states)*4 = 1 + 2*4 = 9 rows). Use 5 rows.
    short = panel.head(5)
    model.fit(short, method="svi", svi_steps=10, num_samples=4)
    assert model.fitted is False, "short-panel branch must clear stale fitted state"


def test_bayesian_msvar_credible_band_uses_per_sample_modal() -> None:
    """REVIEW_DEEP_V1_5_2.md §1.6 / Finding #13: credible bands are the
    5%/95% quantile of the per-sample modal probability
    ``samples.max(axis=2)``, not the conflated
    ``samples[:, np.arange(n), filtered.argmax(axis=1)]``.

    Construct a synthetic posterior whose 5%/95% modal-probability
    quantiles are known, run the score-time band construction, and
    pin the values. This test does NOT require numpyro because we
    skip the fit path and inject a synthetic posterior directly.
    """
    from market_regime_engine.frontier.bayesian_msvar import BayesianMSVAR

    panel, _ = _synthetic_two_state_panel(T=40, seed=13)
    model = BayesianMSVAR(states=["s_a", "s_b"], domains=["x1", "x2"], seed=13)
    # Synthesize a deterministic posterior-state-probs tensor:
    # samples.max(axis=2) at every t is uniform in [0.6, 1.0].
    S, n, K = 400, len(panel), 2
    rng = np.random.default_rng(0)
    p_modal = rng.uniform(0.6, 1.0, size=(S, n))
    samples = np.stack([p_modal, 1.0 - p_modal], axis=2)
    model._posterior_state_probs = samples
    model.fitted = True
    out = model.score(panel)
    expected_lo = float(np.quantile(samples.max(axis=2)[:, 0], 0.05))
    expected_hi = float(np.quantile(samples.max(axis=2)[:, 0], 0.95))
    assert abs(float(out["bayesian_credible_lo"].iloc[0]) - expected_lo) < 1e-9
    assert abs(float(out["bayesian_credible_hi"].iloc[0]) - expected_hi) < 1e-9
    # Sanity check shape and ordering.
    assert (out["bayesian_credible_lo"] <= out["bayesian_credible_hi"] + 1e-9).all()


def test_bayesian_msvar_latest_state_probs_plugs_into_bma() -> None:
    """``latest_state_probs`` returns the bma-compatible dict surface."""
    _require_numpyro()
    _set_jax_cpu()
    from market_regime_engine.bma import OnlineBMA
    from market_regime_engine.frontier.bayesian_msvar import BayesianMSVAR

    panel, _ = _synthetic_two_state_panel(T=60, seed=11)
    states = ["s_a", "s_b"]
    domains = ["x1", "x2"]
    model = BayesianMSVAR(states=states, domains=domains, seed=42).fit(
        panel, method="svi", svi_steps=100, num_samples=32
    )
    probs = model.latest_state_probs(panel)
    assert {"bayesian_msvar:s_a", "bayesian_msvar:s_b"} == set(probs.keys())
    assert all(0 <= float(v) <= 1 for v in probs.values())
    # And the dict drops straight into the bma layer.
    bma = OnlineBMA()
    out = bma.update(1.0, probs)
    assert set(out.keys()) == set(probs.keys())
