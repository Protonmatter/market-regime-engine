# SPDX-License-Identifier: Apache-2.0
"""Baseline tests for :mod:`market_regime_engine.frontier.deep_kernel`.

Phase 5.2 of v1.6.0 (REVIEW_DEEP_V1_5_2.md §4 / §1.3). MLPDeepKernel
ships behind the ``[frontier]`` extra (torch). Skip the entire suite
when torch is missing so the dev-box cleanly reports skipped instead of
errored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from market_regime_engine.frontier.deep_kernel import (
    MLPDeepKernel,
    make_auto_train_kernel,
)


def _toy_panel(n: int = 60, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(n, 3)),
        columns=["a", "b", "c"],
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )


def test_mlp_deep_kernel_forward_shape():
    """The MLP produces an embedding of shape ``(n, hidden_dims[-1])``."""
    kernel = MLPDeepKernel(input_dim=3, hidden_dims=(8, 4), seed=0)
    x = np.random.default_rng(0).normal(size=(10, 3))
    out = kernel(x)
    assert out.shape == (10, 4)
    assert kernel.output_dim == 4


def test_mlp_deep_kernel_forward_handles_1d_input():
    kernel = MLPDeepKernel(input_dim=2, hidden_dims=(4,), seed=0)
    x = np.array([1.0, 2.0])
    out = kernel(x)
    assert out.shape == (1, 4)


def test_mlp_deep_kernel_fit_records_training_losses():
    panel = _toy_panel(n=80)
    kernel = MLPDeepKernel(input_dim=3, hidden_dims=(8, 4), seed=0)
    kernel.fit(panel, n_epochs=3, lr=1e-3, window=16)
    assert kernel.fitted is True
    assert len(kernel.training_losses) >= 1


def test_make_auto_train_kernel_lazily_fits_on_first_call():
    """``_AutoTrainedDeepKernel`` defers the fit until the first forward
    call so :class:`GPBOCPD` can construct it eagerly.
    """
    panel = _toy_panel(n=80)
    auto = make_auto_train_kernel(panel, hidden_dims=(8, 4), n_epochs=2, lr=1e-3)
    assert auto.fitted is False
    out = auto(panel.to_numpy(float))
    assert auto.fitted is True
    assert out.shape == (len(panel), 4)
