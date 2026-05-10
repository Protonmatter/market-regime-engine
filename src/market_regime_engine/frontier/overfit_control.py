# SPDX-License-Identifier: Apache-2.0
"""Anti-overfit controls for model tournaments and strategy-like outputs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DeflatedSharpeResult:
    sharpe: float
    deflated_sharpe: float
    pvalue: float
    expected_max_sharpe: float
    n_observations: int
    n_trials: int
    skewness: float
    kurtosis: float


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    n_trials: int
    logits: list[float]
    selected_models: list[str]


@dataclass(frozen=True)
class TournamentManifest:
    path: str
    manifest_hash: str
    candidates: tuple[str, ...]
    benchmarks: tuple[str, ...]
    primary_metric: str


def _sharpe(x: Sequence[float], *, periods_per_year: int = 252) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    sd = float(arr.std(ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float(arr.mean() / sd * math.sqrt(periods_per_year))


def deflated_sharpe_ratio(
    returns: Sequence[float],
    *,
    n_trials: int = 1,
    periods_per_year: int = 252,
) -> DeflatedSharpeResult:
    """Approximate Bailey-Lopez de Prado deflated Sharpe ratio.

    ``sharpe`` and ``expected_max_sharpe`` are reported annualized for operator
    readability. The deflated Sharpe statistic is computed on the per-period
    Sharpe scale so units stay consistent inside the hypothesis test.
    """

    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 4:
        return DeflatedSharpeResult(float("nan"), float("nan"), float("nan"), float("nan"), n, n_trials, float("nan"), float("nan"))
    sd = float(arr.std(ddof=1))
    raw_sr = 0.0 if sd <= 1e-12 else float(arr.mean() / sd)
    annualized_sr = raw_sr * math.sqrt(periods_per_year)
    demeaned = arr - arr.mean()
    z = demeaned / max(sd, 1e-12)
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    n_trials = max(int(n_trials), 1)
    normal = NormalDist()
    if n_trials <= 1:
        expected_max_raw_sr = 0.0
    else:
        gamma = 0.5772156649015329
        expected_max_raw_sr = math.sqrt(1.0 / max(n - 1, 1)) * (
            (1 - gamma) * normal.inv_cdf(1 - 1 / n_trials)
            + gamma * normal.inv_cdf(1 - 1 / (math.e * n_trials))
        )
    denom = math.sqrt(max(1e-12, 1 - skew * raw_sr + ((kurt - 1) / 4.0) * raw_sr * raw_sr))
    dsr_stat = (raw_sr - expected_max_raw_sr) * math.sqrt(n - 1) / denom
    pvalue = 1.0 - normal.cdf(dsr_stat)
    return DeflatedSharpeResult(
        float(annualized_sr),
        float(dsr_stat),
        float(pvalue),
        float(expected_max_raw_sr * math.sqrt(periods_per_year)),
        n,
        n_trials,
        skew,
        kurt,
    )


def probability_of_backtest_overfitting(
    returns_by_model: pd.DataFrame,
    *,
    n_folds: int = 8,
    periods_per_year: int = 252,
) -> PBOResult:
    """Estimate PBO with a CSCV-style train/test fold tournament."""

    if returns_by_model is None or returns_by_model.empty:
        return PBOResult(float("nan"), 0, [], [])
    if n_folds < 2 or n_folds % 2 != 0:
        raise ValueError("n_folds must be an even integer >= 2")
    frame = returns_by_model.dropna(how="any")
    if frame.shape[0] < n_folds * 2 or frame.shape[1] < 2:
        return PBOResult(float("nan"), 0, [], [])
    folds = np.array_split(np.arange(len(frame)), n_folds)
    k = n_folds // 2
    logits: list[float] = []
    selected_models: list[str] = []
    for train_fold_ids in combinations(range(n_folds), k):
        train_idx = np.concatenate([folds[i] for i in train_fold_ids])
        test_idx = np.concatenate([folds[i] for i in range(n_folds) if i not in train_fold_ids])
        train_scores = frame.iloc[train_idx].apply(_sharpe, periods_per_year=periods_per_year)
        test_scores = frame.iloc[test_idx].apply(_sharpe, periods_per_year=periods_per_year)
        winner = str(train_scores.idxmax())
        selected_models.append(winner)
        ranks = test_scores.rank(method="average", ascending=True)
        pct = float((ranks[winner] - 0.5) / len(ranks))
        pct = min(max(pct, 1e-6), 1 - 1e-6)
        logits.append(float(math.log(pct / (1 - pct))))
    pbo = float(np.mean(np.asarray(logits) <= 0.0)) if logits else float("nan")
    return PBOResult(pbo=pbo, n_trials=len(logits), logits=logits, selected_models=selected_models)


def minimum_track_record_length(
    *,
    observed_sharpe: float,
    benchmark_sharpe: float = 0.0,
    alpha: float = 0.05,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Approximate minimum observations needed to reject benchmark Sharpe."""

    normal = NormalDist()
    z = normal.inv_cdf(1.0 - alpha)
    delta = float(observed_sharpe - benchmark_sharpe)
    if delta <= 0:
        return float("inf")
    denom = max(1e-12, delta * delta)
    non_normal = max(1e-12, 1 - skewness * observed_sharpe + ((kurtosis - 1) / 4.0) * observed_sharpe**2)
    return float(1.0 + (z * z * non_normal) / denom)


def _canonical_json(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def freeze_model_tournament(
    *,
    out_path: str | Path,
    candidates: Sequence[str],
    benchmarks: Sequence[str],
    metrics: Sequence[str],
    primary_metric: str,
    validation_windows: Sequence[Mapping[str, object]],
    promotion_thresholds: Mapping[str, object] | None = None,
    random_seeds: Mapping[str, int] | None = None,
) -> TournamentManifest:
    """Write a pre-registered model tournament manifest and its hash."""

    if primary_metric not in metrics:
        raise ValueError("primary_metric must be included in metrics")
    manifest = {
        "schema": "mre.model_tournament.v1",
        "candidates": list(candidates),
        "benchmarks": list(benchmarks),
        "metrics": list(metrics),
        "primary_metric": primary_metric,
        "validation_windows": list(validation_windows),
        "promotion_thresholds": dict(promotion_thresholds or {}),
        "random_seeds": dict(random_seeds or {}),
    }
    payload = _canonical_json(manifest)
    digest = hashlib.sha256(payload).hexdigest()
    manifest["manifest_hash"] = digest
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return TournamentManifest(str(path), digest, tuple(candidates), tuple(benchmarks), primary_metric)


def verify_model_tournament_manifest(path: str | Path) -> dict:
    """Verify a frozen model tournament manifest hash."""

    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("manifest_hash")
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    actual = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return {"approved": expected == actual, "expected": expected, "actual": actual, "path": str(manifest_path)}


__all__ = [
    "DeflatedSharpeResult",
    "PBOResult",
    "TournamentManifest",
    "deflated_sharpe_ratio",
    "freeze_model_tournament",
    "minimum_track_record_length",
    "probability_of_backtest_overfitting",
    "verify_model_tournament_manifest",
]
