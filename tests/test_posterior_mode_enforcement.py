# SPDX-License-Identifier: Apache-2.0
"""PR-1 acceptance tests for filtered-vs-smoothed posterior enforcement
(AGENT.md non-negotiable constraint 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.fixed_income.posterior_mode import (
    FilteredPosterior,
    PosteriorMode,
    SmoothedPosterior,
    require_filtered,
)


def _toy_posterior() -> tuple[np.ndarray, pd.DatetimeIndex]:
    data = np.array([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]])
    timestamps = pd.DatetimeIndex(
        ["2026-05-10 09:30", "2026-05-10 09:31", "2026-05-10 09:32"], tz="UTC"
    )
    return data, timestamps


def test_filtered_posterior_accepted() -> None:
    data, ts = _toy_posterior()
    post = FilteredPosterior(data=data, timestamps=ts)
    out = require_filtered(post)
    assert out is post
    assert out.mode is PosteriorMode.FILTERED


def test_smoothed_posterior_rejected_by_require_filtered() -> None:
    """``require_filtered`` raises ``TypeError`` (message mentions "smoothed")."""
    data, ts = _toy_posterior()
    post = SmoothedPosterior(data=data, timestamps=ts)
    with pytest.raises(TypeError) as exc:
        require_filtered(post)  # type: ignore[arg-type]
    assert "smoothed" in str(exc.value).lower()


def test_non_posterior_type_rejected() -> None:
    """Anything that is not a FilteredPosterior is rejected."""
    with pytest.raises(TypeError):
        require_filtered({"data": [0.5, 0.5], "mode": "filtered"})  # type: ignore[arg-type]


def test_posterior_mode_enum_values() -> None:
    assert PosteriorMode.FILTERED.value == "filtered"
    assert PosteriorMode.SMOOTHED.value == "smoothed"
