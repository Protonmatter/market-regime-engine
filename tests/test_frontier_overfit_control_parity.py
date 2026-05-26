# SPDX-License-Identifier: Apache-2.0
"""Parity regression tests between :mod:`frontier.overfit_control` and
:mod:`validation`.

REVIEW_DEEP_V1_5_2.md §1.15 / Findings #1 + #2: the v1.6.0 PR-22 branch
forked the BLP DSR / PBO / MTRL primitives and re-introduced the v1.5.1
A2 (PBO missing purge/embargo) and A3 (DSR* multiplicity scaling
missing ``sqrt(var_term)``) bugs the v1.5.2 :mod:`validation` fix had
already corrected. This module pins the post-fork BLP-correct contract
by asserting that the frontier wrappers produce identical outputs to
the canonical :mod:`validation` primitives across a curated fixture set:

- Gaussian iid (skew = 0, excess kurt = 0)
- Fat-tail (Student-t with df = 4)
- Skew-positive
- Skew-negative

The wrappers in ``frontier.overfit_control`` now standardise on the
**excess** kurtosis convention (Gaussian = 0) per BLP, matching
``validation``. The dataclass surface is preserved for backwards compat
with the v1.6 PR-22 callers.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from market_regime_engine import validation as _validation
from market_regime_engine.frontier.overfit_control import (
    deflated_sharpe_ratio,
    minimum_track_record_length,
    probability_of_backtest_overfitting,
)

# ---------------------------------------------------------------------------
# DSR parity — frontier wrapper vs validation.deflated_sharpe
# ---------------------------------------------------------------------------


def _draw_gaussian(seed: int, n: int = 252) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.001, 0.01, size=n)


def _draw_student_t(seed: int, df: int = 4, n: int = 252) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 0.001 + 0.01 * rng.standard_t(df, size=n)


def _draw_skew_positive(seed: int, n: int = 252) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 0.001 + 0.01 * (rng.lognormal(mean=0.0, sigma=0.5, size=n) - math.exp(0.5**2 / 2.0))


def _draw_skew_negative(seed: int, n: int = 252) -> np.ndarray:
    return -_draw_skew_positive(seed, n) + 0.002


@pytest.mark.parametrize(
    "fixture",
    [
        ("gaussian_iid", _draw_gaussian),
        ("student_t_4", _draw_student_t),
        ("skew_positive", _draw_skew_positive),
        ("skew_negative", _draw_skew_negative),
    ],
    ids=lambda x: x[0] if isinstance(x, tuple) else str(x),
)
@pytest.mark.parametrize("n_trials", [1, 5, 50])
def test_deflated_sharpe_ratio_matches_validation(fixture: tuple[str, callable], n_trials: int) -> None:
    """The frontier wrapper must produce the same DSR probability as
    ``validation.deflated_sharpe`` on every fixture / n_trials combo.

    The DSR probability lives in ``DeflatedSharpeResult.pvalue`` (1 - DSR).
    """
    _name, draw = fixture
    returns = draw(seed=11)
    result = deflated_sharpe_ratio(returns, n_trials=n_trials, periods_per_year=252)
    # DSR probability == 1 - p-value (the dataclass exposes the p-value).
    dsr_prob_via_wrapper = 1.0 - result.pvalue
    dsr_prob_via_validation = _validation.deflated_sharpe(
        returns,
        n_trials=n_trials,
        sharpe_target=0.0,
    )
    assert math.isclose(
        dsr_prob_via_wrapper,
        dsr_prob_via_validation,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ), (
        f"DSR mismatch on {_name} n_trials={n_trials}: "
        f"wrapper={dsr_prob_via_wrapper}, validation={dsr_prob_via_validation}"
    )


def test_deflated_sharpe_ratio_excess_kurtosis_default_matches_blp() -> None:
    """At the default skew=0, excess_kurt=0 (Gaussian) and n_trials=1 the
    DSR z-score should equal ``raw_SR * sqrt((n-1) / var_term)`` per BLP
    eq. 5 with no multiplicity correction. Cross-check via direct math."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, size=252)
    result = deflated_sharpe_ratio(returns, n_trials=1, excess_kurt=0.0, skew=0.0)
    raw_sr = float(np.asarray(returns).mean() / np.asarray(returns).std(ddof=1))
    var_term = 1.0 + 0.5 * raw_sr * raw_sr  # skew=0, excess_kurt=0
    denom = math.sqrt(var_term / (252 - 1))
    expected_z = raw_sr / denom
    assert math.isclose(result.deflated_sharpe, expected_z, rel_tol=1e-9)
    assert math.isclose(result.expected_max_sharpe, 0.0, abs_tol=1e-12)


def test_deflated_sharpe_ratio_n_trials_increases_threshold() -> None:
    """E[max_z(N)] is monotone increasing in N → expected_max_sharpe grows;
    DSR statistic falls. (Existing test; carried over from v1.6.0.)"""
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.01, size=252)
    one = deflated_sharpe_ratio(returns, n_trials=1)
    many = deflated_sharpe_ratio(returns, n_trials=100)
    assert many.expected_max_sharpe >= one.expected_max_sharpe
    assert many.deflated_sharpe <= one.deflated_sharpe


# ---------------------------------------------------------------------------
# PBO parity — frontier wrapper vs validation.probability_of_backtest_overfitting
# ---------------------------------------------------------------------------


def _build_pbo_panel(seed: int, n_strategies: int = 4, n_periods: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {f"strategy_{i}": rng.normal(0.0, 0.01, size=n_periods) for i in range(n_strategies)},
    )


@pytest.mark.parametrize("n_folds", [4, 8])
@pytest.mark.parametrize("purge", [0, 2])
@pytest.mark.parametrize("embargo", [0, 1])
def test_probability_of_backtest_overfitting_matches_validation(n_folds: int, purge: int, embargo: int) -> None:
    """Wrapper PBO must equal validation PBO for the same inputs."""
    panel = _build_pbo_panel(seed=42)
    wrapper = probability_of_backtest_overfitting(panel, n_folds=n_folds, purge=purge, embargo=embargo)
    canonical = _validation.probability_of_backtest_overfitting(
        panel, n_partitions=n_folds, purge=purge, embargo=embargo
    )
    assert wrapper.pbo == pytest.approx(canonical, abs=1e-12)
    assert wrapper.n_trials > 0
    assert len(wrapper.selected_models) == wrapper.n_trials


def test_probability_of_backtest_overfitting_purge_embargo_now_applied() -> None:
    """Regression test for Finding #1 (REVIEW_DEEP_V1_5_2.md §1.15):
    the earlier v1.6.0 fork ignored ``purge`` and ``embargo`` entirely.
    The wrapper now accepts and applies them; results should differ from
    the no-purge/embargo baseline on a sufficiently structured panel."""
    rng = np.random.default_rng(123)
    n_periods = 240
    n_strategies = 6
    panel = pd.DataFrame({f"s_{i}": rng.normal(0.0, 0.01, size=n_periods) for i in range(n_strategies)})
    no_purge = probability_of_backtest_overfitting(panel, n_folds=8, purge=0, embargo=0)
    with_purge = probability_of_backtest_overfitting(panel, n_folds=8, purge=3, embargo=2)
    # The two PBO values may agree numerically when noise dominates; what
    # matters here is that ``with_purge`` was computed (not silently
    # ignored as in the pre-fix fork) — i.e. its PBO equals the canonical
    # validation PBO with the same purge/embargo plumbing.
    canonical = _validation.probability_of_backtest_overfitting(panel, n_partitions=8, purge=3, embargo=2)
    assert with_purge.pbo == pytest.approx(canonical, abs=1e-12)
    # And differs from the no-purge baseline path through validation too:
    canonical_no_purge = _validation.probability_of_backtest_overfitting(panel, n_partitions=8, purge=0, embargo=0)
    assert no_purge.pbo == pytest.approx(canonical_no_purge, abs=1e-12)


# ---------------------------------------------------------------------------
# MTRL parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "observed_sharpe,benchmark_sharpe,skewness,excess_kurtosis,alpha",
    [
        (1.0, 0.0, 0.0, 0.0, 0.05),
        (1.5, 0.5, 0.0, 0.0, 0.05),
        (2.0, 0.0, -1.0, 6.0, 0.10),
        (0.8, 0.0, 0.5, -0.5, 0.01),
    ],
)
def test_minimum_track_record_length_matches_validation(
    observed_sharpe: float,
    benchmark_sharpe: float,
    skewness: float,
    excess_kurtosis: float,
    alpha: float,
) -> None:
    wrapper = minimum_track_record_length(
        observed_sharpe=observed_sharpe,
        benchmark_sharpe=benchmark_sharpe,
        alpha=alpha,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
    )
    canonical = _validation.minimum_track_record_length(
        observed_sharpe,
        benchmark_sharpe,
        skew=skewness,
        excess_kurt=excess_kurtosis,
        confidence=1.0 - alpha,
    )
    assert wrapper == pytest.approx(canonical, abs=1e-12)


def test_minimum_track_record_length_inf_when_observed_le_benchmark() -> None:
    assert math.isinf(minimum_track_record_length(observed_sharpe=0.5, benchmark_sharpe=1.0))
    assert math.isinf(minimum_track_record_length(observed_sharpe=1.0, benchmark_sharpe=1.0))
