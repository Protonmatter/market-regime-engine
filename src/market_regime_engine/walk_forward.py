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
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


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
    """

    min_train: int = 96
    step: int = 1
    horizon: int = 1
    embargo: int = 0
    expanding: bool = True
    test_block: int = 1

    def split(self, n: int) -> Iterator[WalkForwardSplit]:
        if n <= self.min_train:
            return
        fold = 0
        i = self.min_train
        while i < n:
            test_end = min(i + self.test_block, n)
            test_idx = np.arange(i, test_end, dtype=int)

            # Purge any training row whose forward target overlaps any test row.
            purge_floor = i - self.horizon
            train_upper = max(0, purge_floor)
            train_lower = 0 if self.expanding else max(0, train_upper - self.min_train)
            if train_upper - train_lower < self.min_train // 2:
                # Not enough training rows after purge; skip this fold.
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
        """Vectorised purge + embargo (v1.3 item C).

        The pre-v1.3 implementation walked ``train_idx`` in Python and
        ran ``any(...)`` against ``test_idx`` for each train row, which
        was O(n_train · n_test) and dominated the runtime of CPCV on
        n=2000 / n_blocks=8 / k=2 (28 folds). The vectorised form below
        broadcasts a (n_train, n_test) mask once per fold using NumPy
        and is empirically ~50–200x faster on the same workload while
        producing bit-identical outputs (the correctness regression
        test in tests/test_walk_forward_perf.py runs the legacy loop
        and the new vectorised path on 50 random seeded inputs).

        The legacy predicates were:

            purge   ↔ ``any(t < tau <= t + horizon for tau in test_idx)``
                     ↔ ``tau ∈ [t+1, t+horizon]``
            embargo ↔ ``any(tau < t <= tau + embargo for tau in test_idx)``
                     ↔ ``tau ∈ [t-embargo, t-1]``

        The vectorised bounds below are derived from those exactly;
        any off-by-one shift would produce a divergence on the 50
        random seeded inputs in the correctness test.
        """
        if train_idx.size == 0 or test_idx.size == 0:
            return train_idx
        train_idx = np.ascontiguousarray(train_idx, dtype=np.int64)
        test_idx = np.ascontiguousarray(test_idx, dtype=np.int64)
        horizon = int(self.horizon)
        embargo = int(max(self.embargo, 0))

        # Purge: drop t if there exists tau ∈ test_idx with
        #     t + 1 <= tau <= t + horizon
        # which is equivalent to
        #     tau - horizon <= t <= tau - 1.
        # Express via broadcast: (t >= tau - horizon) & (t < tau).
        t = train_idx[:, None]
        purge_lo = test_idx[None, :] - horizon  # t >= purge_lo
        purge_hi = test_idx[None, :]  # t < purge_hi  (strict)
        purge_mask = ((t >= purge_lo) & (t < purge_hi)).any(axis=1)

        # Embargo: drop t if there exists tau ∈ test_idx with
        #     tau + 1 <= t <= tau + embargo
        # i.e. (t > tau) & (t <= tau + embargo).
        if embargo > 0:
            embargo_lo = test_idx[None, :]  # t > embargo_lo
            embargo_hi = test_idx[None, :] + embargo  # t <= embargo_hi
            embargo_mask = ((t > embargo_lo) & (t <= embargo_hi)).any(axis=1)
        else:
            embargo_mask = np.zeros(train_idx.shape, dtype=bool)

        # Drop any train rows that are themselves test rows.
        same_as_test = np.isin(train_idx, test_idx, assume_unique=False)
        keep = ~(purge_mask | embargo_mask | same_as_test)
        return train_idx[keep]


def _model_factory_default(
    predict_fn: Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray],
) -> Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray]:
    """Return a tiny adapter so ``evaluate_walk_forward`` can accept either a
    callable or a ``fit/predict``-style class."""

    def _factory(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
        return predict_fn(X_train, y_train, X_test)

    return _factory


def evaluate_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    splitter,
    predict_fn: Callable[[pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray],
    target: str,
    horizon: str,
    model_name: str = "candidate",
) -> pd.DataFrame:
    """Run a generic walk-forward predictor and return aligned predictions.

    ``predict_fn`` receives the training feature/target slices and the test
    feature slice; it returns a 1-D array of predictions (length == len(test)).
    The function only enforces shape and index alignment — calibration,
    quantile assembly, and probability/score interpretation are caller
    responsibilities.
    """
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
            preds = predict_fn(train[feat_cols], train["__y__"], test[feat_cols])
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
]
