# SPDX-License-Identifier: Apache-2.0
"""Anti-overfit controls for model tournaments and strategy-like outputs.

v1.6.0 (REVIEW_DEEP_V1_5_2.md §1.15 / Findings #1 + #2): this module
previously shipped a *forked duplicate* of the BLP DSR / PBO / MTRL
primitives. That fork re-introduced the v1.5.1 A2 (PBO missing
purge/embargo) and A3 (DSR* multiplicity scaling missing
``sqrt(var_term)``) bugs that the v1.5.2 :mod:`market_regime_engine.validation`
fix already landed.

To eliminate the fork (and the structural failure mode of "duplicate the
primitive, get the math slightly wrong"), the math now delegates to
:mod:`market_regime_engine.validation`. The dataclass return shapes
``DeflatedSharpeResult`` / ``PBOResult`` are preserved for backwards
compatibility with the v1.6 PR-22 callers; the kurtosis convention is
standardised on **excess** kurtosis (Gaussian = 0) per BLP convention,
matching the validation primitives.

References:

- Bailey & López de Prado (2014), "The Deflated Sharpe Ratio".
- Bailey, Borwein, López de Prado, Zhu (2017), "The probability of
  backtest overfitting".
- López de Prado (2018), *Advances in Financial Machine Learning*
  §7.4-7.5 (combinatorial purged cross-validation).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from market_regime_engine import validation as _validation


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """BLP 2014 DSR result.

    v1.6.0: the ``kurtosis`` field now reports **excess** kurtosis
    (Gaussian = 0) per BLP convention, matching :func:`validation.deflated_sharpe`.
    """

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
    sharpe_target: float = 0.0,
    skew: float | None = None,
    excess_kurt: float | None = None,
) -> DeflatedSharpeResult:
    """BLP 2014 deflated Sharpe ratio (delegates to :mod:`validation`).

    Returns the structured :class:`DeflatedSharpeResult` while the
    underlying DSR probability is computed by
    :func:`validation.deflated_sharpe`, so the math is BLP-correct
    (``var_term`` scaling, multiplicity threshold, excess-kurtosis
    convention).

    The ``excess_kurt`` argument is **excess** kurtosis (Gaussian = 0).
    The earlier v1.6.0 forked implementation accepted Pearson kurtosis
    (Gaussian = 3); the convention is now standardised on excess to
    match BLP / :mod:`validation`. Callers that pre-computed Pearson
    kurtosis should subtract 3 before passing.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    n_trials_int = max(int(n_trials), 1)
    if n < 4:
        return DeflatedSharpeResult(
            sharpe=float("nan"),
            deflated_sharpe=float("nan"),
            pvalue=float("nan"),
            expected_max_sharpe=float("nan"),
            n_observations=n,
            n_trials=n_trials_int,
            skewness=float("nan"),
            kurtosis=float("nan"),
        )

    sd = float(arr.std(ddof=1))
    raw_sr = 0.0 if sd <= 1e-12 else float(arr.mean() / sd)
    annualized_sr = raw_sr * math.sqrt(periods_per_year)

    if skew is None or excess_kurt is None:
        sample_skew, sample_excess = _validation._sample_skew_kurt(arr)
        if skew is None:
            skew = sample_skew
        if excess_kurt is None:
            excess_kurt = sample_excess
    skew = float(skew)
    excess_kurt = float(excess_kurt)

    # Delegate the DSR probability to validation (BLP-correct math).
    dsr_prob = _validation.deflated_sharpe(
        arr,
        n_trials=n_trials_int,
        skew=skew,
        kurt=excess_kurt,
        sharpe_target=sharpe_target,
    )

    # Reconstruct the threshold-relative z-score and the annualized SR*
    # offset for the dataclass surface. The arithmetic here is the same
    # closed form ``validation.deflated_sharpe`` runs internally; we
    # recompute (rather than invert ``dsr_prob``) so the dataclass
    # z-score is exact even when the probability numerically saturates.
    var_term = max(
        1.0 - skew * raw_sr + (excess_kurt + 2.0) / 4.0 * raw_sr * raw_sr,
        1e-12,
    )
    denom = math.sqrt(var_term / max(n - 1, 1))
    e_max_z = _validation._expected_max_z(n_trials_int)
    sr_star_threshold = float(sharpe_target) + e_max_z * denom
    if denom > 0:
        dsr_stat = (raw_sr - sr_star_threshold) / denom
    else:
        dsr_stat = float("nan")
    expected_max_sharpe = (sr_star_threshold - float(sharpe_target)) * math.sqrt(periods_per_year)
    pvalue = 1.0 - dsr_prob

    return DeflatedSharpeResult(
        sharpe=float(annualized_sr),
        deflated_sharpe=float(dsr_stat),
        pvalue=float(pvalue),
        expected_max_sharpe=float(expected_max_sharpe),
        n_observations=n,
        n_trials=n_trials_int,
        skewness=float(skew),
        kurtosis=float(excess_kurt),
    )


def probability_of_backtest_overfitting(
    returns_by_model: pd.DataFrame,
    *,
    n_folds: int = 8,
    periods_per_year: int = 252,
    embargo: int = 0,
    purge: int = 0,
) -> PBOResult:
    """BBLZ 2017 probability of backtest overfitting (delegates to :mod:`validation`).

    The canonical PBO statistic is computed by
    :func:`validation.probability_of_backtest_overfitting`, which applies
    the purge + embargo gaps per BLP §7.4 (the v1.5.1 PR-9 FIX 4a fix).
    The dataclass ``selected_models`` and ``logits`` diagnostics are
    populated by walking the same C(N, k) splits and applying matching
    purge/embargo to the IS aggregation so the per-split logits stay
    consistent with the canonical PBO.

    v1.6.0 (REVIEW_DEEP_V1_5_2.md Finding #1): now correctly accepts and
    applies ``purge`` and ``embargo`` (defaults to 0 for backwards compat
    with the earlier v1.6.0 PR-22 fork; production callers should set
    ``purge >= 1`` for serially dependent panels).
    """
    if returns_by_model is None or returns_by_model.empty:
        return PBOResult(float("nan"), 0, [], [])
    if n_folds < 2 or n_folds % 2 != 0:
        raise ValueError("n_folds must be an even integer >= 2")
    if embargo < 0 or purge < 0:
        raise ValueError("embargo and purge must be >= 0")
    frame = returns_by_model.dropna(how="any")
    if frame.shape[0] < n_folds * 2 or frame.shape[1] < 2:
        return PBOResult(float("nan"), 0, [], [])

    pbo_value = _validation.probability_of_backtest_overfitting(
        frame, n_partitions=n_folds, embargo=embargo, purge=purge
    )

    # Diagnostic walk: extract per-split Sharpe winner + BBLZ logit so the
    # dataclass surface stays informative. Mirror validation's purge /
    # embargo boundary semantics (purge both sides of every OOS block;
    # embargo the post-OOS region) so the diagnostic logits stay
    # consistent with the canonical PBO statistic.
    n = len(frame)
    folds = np.array_split(np.arange(n), n_folds)
    k = n_folds // 2
    logits: list[float] = []
    selected_models: list[str] = []

    for train_fold_ids in combinations(range(n_folds), k):
        train_idx = np.concatenate([folds[i] for i in train_fold_ids])
        test_fold_ids = tuple(i for i in range(n_folds) if i not in train_fold_ids)
        test_idx = np.concatenate([folds[i] for i in test_fold_ids])
        if purge > 0 or embargo > 0:
            mask_drop = np.zeros(n, dtype=bool)
            for oos_idx in test_fold_ids:
                block = folds[oos_idx]
                if block.size == 0:
                    continue
                start = int(block[0])
                end = int(block[-1])
                if purge > 0:
                    purge_lo = max(0, start - purge)
                    purge_hi = min(n, end + 1 + purge)
                    mask_drop[purge_lo:purge_hi] = True
                if embargo > 0:
                    embargo_hi = min(n, end + 1 + embargo)
                    mask_drop[end + 1 : embargo_hi] = True
            keep = ~mask_drop[train_idx]
            train_idx = train_idx[keep]
        if train_idx.size == 0:
            continue
        train_scores = frame.iloc[train_idx].apply(_sharpe, periods_per_year=periods_per_year)
        test_scores = frame.iloc[test_idx].apply(_sharpe, periods_per_year=periods_per_year)
        winner = str(train_scores.idxmax())
        selected_models.append(winner)
        ranks = test_scores.rank(method="average", ascending=True)
        pct = float((ranks[winner] - 0.5) / len(ranks))
        pct = min(max(pct, 1e-6), 1 - 1e-6)
        logits.append(float(math.log(pct / (1 - pct))))

    return PBOResult(
        pbo=float(pbo_value),
        n_trials=len(logits),
        logits=logits,
        selected_models=selected_models,
    )


def minimum_track_record_length(
    *,
    observed_sharpe: float,
    benchmark_sharpe: float = 0.0,
    alpha: float = 0.05,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """BLP 2014 minimum track record length (delegates to :mod:`validation`).

    v1.6.0: kurtosis convention standardised on **excess** (Gaussian = 0)
    via the ``excess_kurtosis`` keyword (default 0). The earlier v1.6.0
    fork accepted Pearson kurtosis with default 3.0; callers that
    previously passed Pearson should subtract 3 before passing the
    excess form.
    """
    return _validation.minimum_track_record_length(
        float(observed_sharpe),
        float(benchmark_sharpe),
        skew=float(skewness),
        excess_kurt=float(excess_kurtosis),
        confidence=1.0 - float(alpha),
    )


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
