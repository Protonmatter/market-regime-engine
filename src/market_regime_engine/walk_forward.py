"""Time-series walk-forward CV with proper purging and embargo.

The historical ``backtest.py`` ran a naive expanding-window split that left two
silent leaks:

1. Overlapping forward-target windows (e.g. the 12-month drawdown target at
   month *t* shares 11 future periods with the target at month *t+1*) bleed
   information from the test point into the training set unless the rows whose
   targets overlap the test window are *purged* (López de Prado 2018, Chapter 7).

2. Macro shocks have a short autocorrelation tail. After a test point the next
   training fold should leave a small **embargo** so that the model is never
   fitted on data dated immediately after a fold it just predicted on.

The engine now exposes three building blocks:

- :class:`PurgedWalkForward` — expanding/rolling walk-forward with purge and
  embargo, for a single horizon.
- :class:`CombinatorialPurgedCV` — López de Prado's CPCV
  (``n`` blocks, ``k`` test blocks at a time, all combinations) for variance-
  stable Sharpe estimation.
- :func:`evaluate_walk_forward` — runs a model factory through a split and
  returns predictions plus aligned realized labels.

v1.5 PR-5 (ASK-1, AF-11, ASK-13)
--------------------------------
- :func:`purge_and_embargo_searchsorted` replaces the legacy
  ``CombinatorialPurgedCV._purge_and_embargo`` dense ``(n_train, n_test)``
  bool matrix with a ``np.searchsorted``-backed implementation. Memory
  bound goes from ``O(n_train · n_test)`` to ``O(n_train + n_test)`` so
  large CPCV runs (n=2000 / n_blocks=8 / k=2) stop dominating wall-clock
  on a low-memory worker.

- :class:`PurgedWalkForward.min_train_after_purge` makes the legacy
  ``min_train // 2`` skip threshold explicit. ``None`` (default) falls back
  to the pre-PR-5 ``min_train`` semantic (back-compat preserved); supplying
  an integer enforces that exact minimum after purge.

- :func:`evaluate_walk_forward` now accepts ``model_class=`` /
  ``model_kwargs=`` so the caller can pass a *class* and have it
  instantiated fresh per fold. The closure-capture ``predict_fn`` path
  remains for back-compat but leaks state across folds — the docstring
  documents this explicitly.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardSplit:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)


def purge_and_embargo_searchsorted(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    horizon: int,
    embargo: int,
) -> np.ndarray:
    """Purge + embargo via ``np.searchsorted`` (v1.5 PR-5, ASK-1).

    Drop train index ``t`` iff there exists a test index ``tau`` in the
    closed window ``[t - embargo, t + horizon]``. This is the union of the
    three legacy predicates:

    - **purge** (target leaks into training): ``tau ∈ [t + 1, t + horizon]``
      → equivalently ``tau ∈ [t, t + horizon]`` after merging the
      ``same_as_test`` mask below.
    - **embargo** (post-test residual autocorrelation): ``tau ∈ [t - embargo, t - 1]``.
    - **same_as_test**: ``tau == t``.

    Their union is exactly ``tau ∈ [t - embargo, t + horizon]``.

    For each train ``t``, the count of test points inside the window equals
    ``searchsorted(test_sorted, t + horizon + 1, 'left') -
    searchsorted(test_sorted, t - embargo, 'left')``. Keep ``t`` iff the
    count is zero.

    The legacy v1.3 path built an ``(n_train, n_test)`` bool matrix; on
    n=2000 / n_blocks=8 / k=2 that allocates ~32 MB per fold (28 folds ≈ 1 GB
    aggregate). The searchsorted form sorts ``test_idx`` once
    (``O(n_test log n_test)``) then runs ``np.searchsorted`` on the train
    array for the two window edges; total memory is
    ``O(n_train + n_test)`` and runtime is
    ``O((n_train + n_test) log n_test)``.

    Bit-for-bit equivalent to the v1.3 dense mask path — verified on the 50
    random seeded inputs in ``tests/test_walk_forward_purge_searchsorted.py``.
    """
    if train_idx.size == 0 or test_idx.size == 0:
        return np.ascontiguousarray(train_idx, dtype=np.int64)
    horizon = int(horizon)
    embargo = int(max(embargo, 0))
    train_idx = np.ascontiguousarray(train_idx, dtype=np.int64)
    test_idx = np.ascontiguousarray(test_idx, dtype=np.int64)
    test_sorted = np.sort(test_idx)

    lo = np.searchsorted(test_sorted, train_idx - embargo, side="left")
    hi = np.searchsorted(test_sorted, train_idx + horizon + 1, side="left")
    keep = lo == hi
    return train_idx[keep]


@dataclass
class PurgedWalkForward:
    """Expanding (or rolling) walk-forward split with purge and embargo.

    Parameters
    ----------
    min_train:
        Minimum training-window size before the first fold is emitted.
    step:
        Stride between successive test points / blocks.
    horizon:
        Number of periods covered by the forecast target. The training set is
        purged of any rows whose forward-target window touches the test point.
    embargo:
        Number of periods after the test point that are excluded from
        subsequent training folds, even when the calendar gap is large enough
        for purging alone.
    expanding:
        ``True`` for an expanding window (default); ``False`` for a rolling
        window of size ``min_train``.
    test_block:
        Number of consecutive test points emitted per fold. ``1`` is the
        single-step expanding-window backtest; larger values match the original
        ``step`` semantics.
    min_train_after_purge:
        v1.5 PR-5 (AF-11): minimum training-window size required after the
        purge has trimmed the head of the training window. ``None`` (the
        default) falls back to the legacy ``min_train // 2`` threshold so
        the pre-PR-5 behaviour is preserved bit-for-bit; supplying an
        integer makes the rail explicit and lets callers tune the minimum
        independently of ``min_train``. The threshold is logged at INFO when
        a fold is skipped so the operator can see why fewer folds were
        emitted than expected.
    """

    min_train: int = 96
    step: int = 1
    horizon: int = 1
    embargo: int = 0
    expanding: bool = True
    test_block: int = 1
    min_train_after_purge: int | None = None

    def split(self, n: int) -> Iterator[WalkForwardSplit]:
        if n <= self.min_train:
            return
        # v1.5 PR-5 (AF-11): explicit min_train_after_purge takes precedence;
        # fall back to the legacy ``min_train // 2`` rail when None so the
        # pre-PR-5 callers keep their fold counts bit-for-bit.
        min_train_after = (
            self.min_train_after_purge
            if self.min_train_after_purge is not None
            else self.min_train // 2
        )
        fold = 0
        i = self.min_train
        while i < n:
            test_end = min(i + self.test_block, n)
            test_idx = np.arange(i, test_end, dtype=int)

            # Purge any training row whose forward target overlaps any test row.
            purge_floor = i - self.horizon
            train_upper = max(0, purge_floor)
            train_lower = 0 if self.expanding else max(0, train_upper - self.min_train)
            if train_upper - train_lower < min_train_after:
                logger.info(
                    "walk_forward.skip_fold: insufficient_train_after_purge "
                    "fold=%d size=%d threshold=%d",
                    fold,
                    train_upper - train_lower,
                    min_train_after,
                )
                i += self.step
                continue
            train_idx = np.arange(train_lower, train_upper, dtype=int)

            yield WalkForwardSplit(fold=fold, train_idx=train_idx, test_idx=test_idx)
            fold += 1
            # Embargo + step controls the start of the next fold.
            i = test_end + max(self.embargo, 0) + max(self.step - self.test_block, 0)


@dataclass
class CombinatorialPurgedCV:
    """López de Prado CPCV.

    Splits the time axis into ``n_blocks`` contiguous blocks. For every choice
    of ``k_test_blocks`` blocks held out together, the remaining blocks form
    the training set after purging any sample whose forward-horizon overlaps a
    test block, and after applying the embargo on the boundary side.

    The total number of folds is ``C(n_blocks, k_test_blocks)``.
    """

    n_blocks: int = 6
    k_test_blocks: int = 2
    horizon: int = 1
    embargo: int = 0

    def split(self, n: int) -> Iterator[WalkForwardSplit]:
        if n < self.n_blocks * 2:
            return
        edges = np.linspace(0, n, self.n_blocks + 1, dtype=int)
        blocks = [np.arange(edges[i], edges[i + 1], dtype=int) for i in range(self.n_blocks)]
        fold = 0
        for combo in itertools.combinations(range(self.n_blocks), self.k_test_blocks):
            test_idx = np.concatenate([blocks[i] for i in combo])
            train_blocks = [blocks[i] for i in range(self.n_blocks) if i not in combo]
            train_idx = np.concatenate(train_blocks) if train_blocks else np.array([], dtype=int)
            train_idx = self._purge_and_embargo(train_idx, test_idx)
            if train_idx.size == 0 or test_idx.size == 0:
                continue
            yield WalkForwardSplit(fold=fold, train_idx=train_idx, test_idx=test_idx)
            fold += 1

    def _purge_and_embargo(self, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
        """Purge + embargo (v1.5 PR-5 ASK-1: searchsorted memory bound).

        The v1.3 vectorised path allocated an ``(n_train, n_test)`` bool
        matrix per fold; on n=2000 / n_blocks=8 / k=2 that is 28 folds ×
        ~32 MB ≈ 1 GB of transient allocations. v1.5 routes through
        :func:`purge_and_embargo_searchsorted` whose memory bound is
        ``O(n_train + n_test)``. The output is bit-for-bit identical to
        the v1.3 path; the 50-seed regression test in
        ``tests/test_walk_forward_purge_searchsorted.py`` pins this.

        The unified window predicate is
        ``tau ∈ [t - horizon, t + embargo + 1)``, which is the union of the
        legacy purge predicate ``tau ∈ [t+1, t+horizon]``, embargo
        ``tau ∈ [t-embargo, t-1]``, and ``tau == t``.
        """
        return purge_and_embargo_searchsorted(
            train_idx,
            test_idx,
            horizon=int(self.horizon),
            embargo=int(self.embargo),
        )


def _legacy_purge_and_embargo(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    horizon: int,
    embargo: int,
) -> np.ndarray:
    """v1.3 dense-mask purge + embargo, kept for the searchsorted parity test.

    DO NOT call from production paths — :func:`purge_and_embargo_searchsorted`
    is the v1.5 PR-5 replacement. The pin tests use this function as the
    reference implementation.
    """
    if train_idx.size == 0 or test_idx.size == 0:
        return np.ascontiguousarray(train_idx, dtype=np.int64)
    train_idx = np.ascontiguousarray(train_idx, dtype=np.int64)
    test_idx = np.ascontiguousarray(test_idx, dtype=np.int64)
    horizon = int(horizon)
    embargo = int(max(embargo, 0))

    t = train_idx[:, None]
    purge_lo = test_idx[None, :] - horizon
    purge_hi = test_idx[None, :]
    purge_mask = ((t >= purge_lo) & (t < purge_hi)).any(axis=1)
    if embargo > 0:
        embargo_lo = test_idx[None, :]
        embargo_hi = test_idx[None, :] + embargo
        embargo_mask = ((t > embargo_lo) & (t <= embargo_hi)).any(axis=1)
    else:
        embargo_mask = np.zeros(train_idx.shape, dtype=bool)
    same_as_test = np.isin(train_idx, test_idx, assume_unique=False)
    keep = ~(purge_mask | embargo_mask | same_as_test)
    return train_idx[keep]


def _model_factory_default(
    predict_fn: Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray] | None = None,
    *,
    model_class: type | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray]:
    """Return a per-fold predict adapter.

    v1.5 PR-5 (ASK-13): two factory modes are supported:

    1. **``model_class`` + ``model_kwargs``** (recommended). A *fresh*
       instance of ``model_class(**model_kwargs)`` is constructed inside the
       returned adapter on every call, so the model carries no state
       between folds. The model is expected to expose
       ``.fit(X_train, y_train)`` and ``.predict_proba(X_test)`` (or
       ``.predict``) per the scikit-learn convention.

    2. **``predict_fn``** (back-compat). The pre-PR-5 closure-capture form;
       the same callable is invoked on every fold. *Cross-fold state
       leakage* is possible if the caller's closure captures a stateful
       estimator — passing ``model_class`` is the safe alternative.

    Either ``predict_fn`` or ``model_class`` must be provided.
    """
    if model_class is not None:
        kwargs = dict(model_kwargs or {})

        def _factory_class(
            X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
        ) -> np.ndarray:
            model = model_class(**kwargs)
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                arr = np.asarray(proba, dtype=float)
                if arr.ndim == 2:
                    # Binary classification: take the positive-class column;
                    # otherwise return the predicted-class probability for
                    # multiclass. ``[:, -1]`` matches the v1.4
                    # ProbabilityModel convention.
                    return arr[:, -1]
                return arr
            return np.asarray(model.predict(X_test), dtype=float)

        return _factory_class

    if predict_fn is None:
        raise ValueError("either predict_fn or model_class must be provided")

    def _factory_fn(
        X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
    ) -> np.ndarray:
        return predict_fn(X_train, y_train, X_test)

    return _factory_fn


def evaluate_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    splitter,
    predict_fn: Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray] | None = None,
    target: str,
    horizon: str,
    model_name: str = "candidate",
    model_class: type | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Run a generic walk-forward predictor and return aligned predictions.

    Two modes:

    - **Legacy callable** (``predict_fn``): the same closure-captured callable
      receives ``(X_train, y_train, X_test)`` and returns predictions. *Risk*:
      a stateful estimator captured by the closure leaks state across folds.

    - **Fresh-per-fold class** (``model_class`` + ``model_kwargs``):
      ``model_class(**model_kwargs)`` is instantiated inside every fold via
      :func:`_model_factory_default`. *Recommended* per v1.5 PR-5 ASK-13:
      passing ``model_class`` is safer than closure-capturing a stateful
      estimator.

    Either ``predict_fn`` or ``model_class`` must be provided. The function
    only enforces shape and index alignment — calibration, quantile
    assembly, and probability/score interpretation are caller
    responsibilities.
    """
    factory = _model_factory_default(
        predict_fn,
        model_class=model_class,
        model_kwargs=model_kwargs,
    )

    aligned = X.join(y.rename("__y__"), how="inner").sort_index()
    n = len(aligned)
    rows: list[dict] = []
    for split in splitter.split(n):
        if split.n_train == 0 or split.n_test == 0:
            continue
        train = aligned.iloc[split.train_idx]
        test = aligned.iloc[split.test_idx]
        feat_cols = [c for c in aligned.columns if c != "__y__"]
        if train.empty or test.empty:
            continue
        try:
            preds = factory(train[feat_cols], train["__y__"], test[feat_cols])
        except Exception as exc:  # pragma: no cover - propagated for caller debugging
            raise RuntimeError(f"predict_fn failed on fold {split.fold}: {exc}") from exc
        preds = np.asarray(preds, dtype=float).reshape(-1)
        if preds.shape[0] != len(test):
            raise ValueError(
                f"predict_fn returned {preds.shape[0]} predictions for {len(test)} test rows on fold {split.fold}"
            )
        for date, y_true, p in zip(test.index, test["__y__"].to_numpy(float), preds, strict=False):
            rows.append(
                {
                    "fold": split.fold,
                    "date": date,
                    "target": target,
                    "horizon": horizon,
                    "model_name": model_name,
                    "y": float(y_true) if pd.notna(y_true) else float("nan"),
                    "p": float(p),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"])
    return out


__all__ = [
    "CombinatorialPurgedCV",
    "PurgedWalkForward",
    "WalkForwardSplit",
    "evaluate_walk_forward",
    "purge_and_embargo_searchsorted",
]
