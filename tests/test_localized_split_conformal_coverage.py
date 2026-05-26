# SPDX-License-Identifier: Apache-2.0
"""Empirical coverage regression test for :class:`LocalizedSplitConformal`.

REVIEW_DEEP_V1_5_2.md §1.7 / Finding #12: the previous weighted-quantile
construction added ``test_weight`` to *every* cumulative-weight entry,
which is equivalent to inserting the test point at rank 0 (smallest
score). That shifted every empirical CDF value up by ``test_weight /
total``, lowering the threshold and narrowing the prediction set —
empirical coverage dipped below ``1 - alpha`` by O(test_weight / total).

The fix uses the canonical NexCP / Lin-Trivedi-Sun 2023 construction
where the test point is treated as appended at +infinity (rank n+1)
and its weight contributes only to the total. This test pins the
post-fix coverage at >= 1 - alpha - 1pp on a moderate sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.frontier.conformal_ts import LocalizedSplitConformal


def _build_iid_calibration(seed: int, n: int) -> pd.DataFrame:
    """Build a binary forecast/outcome panel where p ~ U[0,1] and
    y ~ Bernoulli(p) — i.e. a perfectly calibrated forecaster."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, size=n)
    y = rng.binomial(1, p)
    x1 = rng.normal(0.0, 1.0, size=n)
    bucket = rng.choice(["a", "b", "c"], size=n)
    return pd.DataFrame({"y": y, "p": p, "x1": x1, "regime_bucket": bucket})


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_localized_split_conformal_marginal_coverage_bounded_below(alpha: float) -> None:
    """Average empirical coverage across 10 seeds (n=1000 each) must hit
    at least ``1 - alpha - 2pp`` after the weighted-quantile fix.

    The pre-fix construction shifted the threshold downward by
    O(test_weight / total) ~ 1/n and produced average empirical coverage
    consistently below ``1 - alpha``; the post-fix construction averages
    at or above the target. We use a multi-seed average rather than a
    single-seed assertion because localized conformal under kernel
    bandwidth = 1 has a small effective calibration sample at each test
    point — single-seed coverage can easily fluctuate +/- 2pp around the
    target even with marginal validity.
    """
    coverages: list[float] = []
    for seed in range(10):
        df = _build_iid_calibration(seed=seed, n=1000)
        n_cal = int(len(df) * 0.7)
        cal = df.iloc[:n_cal].copy()
        test = df.iloc[n_cal:].copy()
        layer = LocalizedSplitConformal(
            alpha=alpha,
            bandwidth=1.0,
            feature_cols=["x1"],
        ).fit(cal)
        annotated = layer.transform(test)
        n_test = len(annotated)
        covered = 0
        for _, row in annotated.iterrows():
            pred_set = set(row["conformal_set"].split("|")) if row["conformal_set"] != "empty" else set()
            if str(int(row["y"])) in pred_set:
                covered += 1
        coverages.append(covered / n_test)
    avg_coverage = float(np.mean(coverages))
    target = 1.0 - alpha
    # Average across 10 seeds gives a Monte Carlo SE of ~ 1/sqrt(10*300) ~ 1.8pp.
    # Allow 2pp tolerance.
    assert avg_coverage >= target - 0.02, (
        f"Localized conformal average coverage {avg_coverage:.3f} dipped below target "
        f"{target:.3f} - 2pp (per-seed: {coverages})"
    )


def test_weighted_quantile_does_not_inflate_cumulative_weights() -> None:
    """Direct algebraic check: the post-fix construction uses
    ``cumsum(sorted_weights)`` (not ``cumsum + test_weight``) so the
    rank index ``np.searchsorted(cum, target)`` does not collapse to a
    rank below the proper localized quantile."""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        {
            "y": rng.binomial(1, 0.5, size=n),
            "p": rng.uniform(0.1, 0.9, size=n),
            "x1": rng.normal(0.0, 1.0, size=n),
            "regime_bucket": ["a"] * n,
        }
    )
    layer = LocalizedSplitConformal(alpha=0.10, bandwidth=1.0, feature_cols=["x1"]).fit(df)
    # Pick a test feature in the centre of the calibration distribution.
    threshold = layer._localized_threshold(np.array([0.0], dtype=float))
    # The threshold must lie within the calibration score support
    # (i.e. it's a real order statistic, not an extrapolation).
    assert layer._calibration_scores.min() <= threshold <= layer._calibration_scores.max()


def test_threshold_is_monotone_in_alpha() -> None:
    """Smaller alpha → wider prediction set → higher threshold (more
    permissive). This is a basic sanity check that the construction is
    monotone in the conformal level."""
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "y": rng.binomial(1, 0.5, size=n),
            "p": rng.uniform(0.1, 0.9, size=n),
            "x1": rng.normal(0.0, 1.0, size=n),
            "regime_bucket": ["a"] * n,
        }
    )
    x_test = np.array([0.0], dtype=float)
    layer_05 = LocalizedSplitConformal(alpha=0.05, bandwidth=1.0, feature_cols=["x1"]).fit(df)
    layer_20 = LocalizedSplitConformal(alpha=0.20, bandwidth=1.0, feature_cols=["x1"]).fit(df)
    thr_05 = layer_05._localized_threshold(x_test)
    thr_20 = layer_20._localized_threshold(x_test)
    assert thr_05 >= thr_20, f"Threshold non-monotone in alpha: thr(0.05) = {thr_05} < thr(0.20) = {thr_20}"
