"""Golden-master regression on the synthetic sample pipeline.

The first time this test runs against a new code revision it bootstraps a
deterministic CSV under ``tests/golden/regime_trace.csv``. Subsequent runs
compare the freshly produced trace against that snapshot row-for-row, with a
small numerical tolerance.

Set ``MRE_REFRESH_GOLDEN=1`` to overwrite the snapshot from the current
behavior. Refresh deliberately in a dedicated commit so the diff is reviewable.

The trace columns are intentionally narrow: ``date``, ``regime``,
``decoded_regime``, ``change_point_prob``, and ``score``. Adding columns to
the regime score frame should not break this test as long as the existing
columns remain stable.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.config import load_catalog
from market_regime_engine.features import build_features, monthly_panel
from market_regime_engine.regimes import score_regimes
from market_regime_engine.sample import generate_sample_observations

GOLDEN = Path(__file__).parent / "golden" / "regime_trace.csv"
COLUMNS = ["date", "regime", "decoded_regime", "change_point_prob", "score"]


def _produce_trace() -> pd.DataFrame:
    obs = generate_sample_observations()
    panel = monthly_panel(obs)
    feats = build_features(panel, load_catalog())
    regimes = score_regimes(feats, use_bocpd=True, bocpd_core="diagonal", fit_hmm=False)
    out = regimes[COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["change_point_prob"] = out["change_point_prob"].astype(float).round(6)
    out["score"] = out["score"].astype(float).round(6)
    return out.reset_index(drop=True)


@pytest.mark.golden
def test_golden_regime_trace():
    trace = _produce_trace()
    if (
        os.getenv("MRE_REFRESH_GOLDEN") == "1"
        or not GOLDEN.exists()
        or GOLDEN.read_text(encoding="utf-8").strip() == "placeholder"
    ):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        trace.to_csv(GOLDEN, index=False)
        if os.getenv("MRE_REFRESH_GOLDEN") != "1":
            pytest.skip("Bootstrapped golden trace; run again to enforce.")
        return
    expected = pd.read_csv(GOLDEN, dtype={"date": str, "regime": str, "decoded_regime": str})
    assert list(trace.columns) == list(expected.columns)
    assert len(trace) == len(expected), f"Trace length changed: got {len(trace)} rows, expected {len(expected)}"
    pd.testing.assert_series_equal(
        trace["date"].astype(str),
        expected["date"].astype(str),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        trace["regime"].astype(str),
        expected["regime"].astype(str),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        trace["decoded_regime"].astype(str),
        expected["decoded_regime"].astype(str),
        check_names=False,
    )
    np.testing.assert_allclose(
        trace["change_point_prob"].astype(float).to_numpy(),
        expected["change_point_prob"].astype(float).to_numpy(),
        atol=1e-5,
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        trace["score"].astype(float).to_numpy(),
        expected["score"].astype(float).to_numpy(),
        atol=1e-5,
        rtol=1e-4,
    )
