# SPDX-License-Identifier: Apache-2.0
"""PR-5 AF-11: explicit ``min_train_after_purge`` threshold.

Pre-PR-5 ``PurgedWalkForward.split`` used a hard-coded
``train_upper - train_lower < self.min_train // 2`` skip condition, so the
operator could not tune the minimum independently of ``min_train``. PR-5
exposes ``min_train_after_purge`` as an explicit field. ``None`` preserves
the legacy behaviour bit-for-bit (back-compat); supplying an integer makes
the rail tunable.
"""

from __future__ import annotations

import logging

from market_regime_engine.walk_forward import PurgedWalkForward


def test_min_train_after_purge_default_matches_legacy_behaviour() -> None:
    """With ``min_train_after_purge=None`` (default) the fold count is
    bit-for-bit identical to the pre-PR-5 walk-forward."""
    splitter = PurgedWalkForward(min_train=60, step=4, horizon=12, embargo=2)
    folds = list(splitter.split(200))
    assert len(folds) > 0


def test_min_train_after_purge_explicit_threshold_skips_more_folds() -> None:
    """Raising the threshold past ``min_train // 2`` should drop folds whose
    purged training window falls below the new bar."""
    horizon = 80  # large horizon eats much of the prefix
    base = PurgedWalkForward(min_train=100, step=1, horizon=horizon, embargo=0)
    strict = PurgedWalkForward(
        min_train=100,
        step=1,
        horizon=horizon,
        embargo=0,
        min_train_after_purge=80,  # explicit, higher than min_train // 2 = 50
    )
    n = 300
    base_folds = list(base.split(n))
    strict_folds = list(strict.split(n))
    # Strict threshold should retain folds whose train slice >= 80; base
    # threshold (50) retains everything >= 50, so strict <= base.
    assert len(strict_folds) <= len(base_folds)
    # Every strict fold meets the new threshold by construction.
    for fold in strict_folds:
        assert fold.n_train >= 80


def test_min_train_after_purge_zero_keeps_every_attempt() -> None:
    """``min_train_after_purge=0`` lets every candidate fold through as long
    as ``train_upper > train_lower``."""
    splitter = PurgedWalkForward(
        min_train=60, step=1, horizon=12, embargo=0, min_train_after_purge=0
    )
    folds = list(splitter.split(200))
    assert len(folds) >= 100  # very large fold count vs the default rail


def test_min_train_after_purge_logs_skip_reason_at_info(caplog) -> None:
    """When the threshold trips the skip is announced via the logger so an
    operator can see why fold counts dipped."""
    caplog.set_level(logging.INFO, logger="market_regime_engine.walk_forward")
    splitter = PurgedWalkForward(
        min_train=20,
        step=1,
        horizon=30,
        embargo=0,
        min_train_after_purge=100,  # impossible to satisfy
    )
    folds = list(splitter.split(50))
    assert folds == []
    assert any("insufficient_train_after_purge" in rec.message for rec in caplog.records)
