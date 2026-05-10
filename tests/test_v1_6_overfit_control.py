from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.frontier.overfit_control import (
    deflated_sharpe_ratio,
    freeze_model_tournament,
    minimum_track_record_length,
    probability_of_backtest_overfitting,
    verify_model_tournament_manifest,
)


def test_deflated_sharpe_ratio_penalizes_multiple_trials() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.01, size=252)
    one = deflated_sharpe_ratio(returns, n_trials=1)
    many = deflated_sharpe_ratio(returns, n_trials=100)
    assert many.expected_max_sharpe >= one.expected_max_sharpe
    assert many.deflated_sharpe <= one.deflated_sharpe


def test_probability_of_backtest_overfitting_shape() -> None:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "a": rng.normal(0.001, 0.01, 160),
            "b": rng.normal(0.0005, 0.01, 160),
            "c": rng.normal(0.0, 0.01, 160),
        }
    )
    out = probability_of_backtest_overfitting(frame, n_folds=8)
    assert 0.0 <= out.pbo <= 1.0
    assert out.n_trials > 0
    assert out.selected_models


def test_probability_of_backtest_overfitting_requires_even_folds() -> None:
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(pd.DataFrame({"a": range(20), "b": range(20)}), n_folds=3)


def test_minimum_track_record_length() -> None:
    assert minimum_track_record_length(observed_sharpe=1.0, benchmark_sharpe=0.0) > 1
    assert minimum_track_record_length(observed_sharpe=0.0, benchmark_sharpe=1.0) == float("inf")


def test_freeze_and_verify_model_tournament_manifest(tmp_path) -> None:
    result = freeze_model_tournament(
        out_path=tmp_path / "manifest.json",
        candidates=["candidate_a"],
        benchmarks=["benchmark"],
        metrics=["crps", "brier"],
        primary_metric="crps",
        validation_windows=[{"start": "2020-01-01", "end": "2021-01-01"}],
        random_seeds={"candidate_a": 42},
    )
    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert payload["manifest_hash"] == result.manifest_hash
    assert payload["primary_metric"] == "crps"
    assert verify_model_tournament_manifest(tmp_path / "manifest.json")["approved"] is True

    payload["primary_metric"] = "brier"
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    assert verify_model_tournament_manifest(tmp_path / "manifest.json")["approved"] is False
