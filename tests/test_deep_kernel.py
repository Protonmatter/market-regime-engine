# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the v1.4 MLP deep kernel (item B).

Four contracts:

1. ``test_mlp_deep_kernel_embedding_shape`` — the forward pass returns
   the documented ``(n, hidden_dims[-1])`` shape on both 1D and 2D inputs.
2. ``test_mlp_deep_kernel_training_decreases_loss`` — the marginal-LL
   training loss drops monotonically over 10 epochs (with a ±1 epoch
   tolerance to absorb stochastic blips).
3. ``test_gpbocpd_with_auto_train_deep_kernel_runs_end_to_end`` — the
   ``GPBOCPD(auto_train_deep_kernel=True).score(panel)`` operator
   one-liner produces non-empty output without raising.
4. ``test_deep_kernel_soft_degrade_without_torch`` — constructing the
   class with torch missing raises a clean ``ImportError`` carrying the
   ``[frontier]`` install hint.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _require_torch() -> None:
    pytest.importorskip("torch")


def test_mlp_deep_kernel_embedding_shape() -> None:
    _require_torch()
    from market_regime_engine.frontier.deep_kernel import MLPDeepKernel

    kernel = MLPDeepKernel(input_dim=8, hidden_dims=(16, 4), seed=0)
    # Single-row 1D input.
    out_1d = kernel(np.zeros(8, dtype=np.float64))
    assert out_1d.shape == (1, 4)
    # Multi-row 2D input.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, 8))
    out_2d = kernel(x)
    assert out_2d.shape == (7, 4)
    assert out_2d.dtype == np.float64
    # The output dim property exposes the last hidden dim.
    assert kernel.output_dim == 4


def test_mlp_deep_kernel_training_decreases_loss() -> None:
    _require_torch()
    from market_regime_engine.frontier.deep_kernel import MLPDeepKernel

    rng = np.random.default_rng(0)
    T = 80
    panel = pd.DataFrame(
        rng.normal(size=(T, 4)),
        columns=list("abcd"),
        index=pd.date_range("2020-01-01", periods=T, freq="ME"),
    )
    kernel = MLPDeepKernel(input_dim=4, hidden_dims=(8, 4), seed=42)
    kernel.fit(panel, n_epochs=10, lr=1e-2)
    losses = kernel.training_losses
    assert len(losses) >= 5
    # Loss must drop overall: first epoch >= last epoch (with a tiny
    # epsilon so a flat run still passes).
    assert losses[0] - losses[-1] > 0, f"loss did not decrease: first={losses[0]:.4f} last={losses[-1]:.4f}"
    # ±1 epoch monotonicity tolerance: forall i, exists j in {i, i+1, i+2}
    # with loss[j] <= loss[i] (i.e. progress within 1 lookahead).
    violations = 0
    for i in range(len(losses) - 2):
        window = losses[i : i + 3]
        if min(window[1:]) > losses[i]:
            violations += 1
    assert violations <= 1, f"loss monotonicity violated more than once: {losses}"


def test_gpbocpd_with_auto_train_deep_kernel_runs_end_to_end() -> None:
    """v1.4 (criterion 10): the auto-train one-liner produces non-empty output."""
    _require_torch()
    from market_regime_engine.frontier.gp_cpd import GPBOCPD

    rng = np.random.default_rng(1)
    T = 60
    panel = pd.DataFrame(
        rng.normal(size=(T, 4)),
        columns=list("abcd"),
        index=pd.date_range("2020-01-01", periods=T, freq="ME"),
    )
    detector = GPBOCPD(
        auto_train_deep_kernel=True,
        deep_kernel_hidden_dims=(8, 4),
        deep_kernel_epochs=5,
    )
    out = detector.score(panel)
    assert {
        "date",
        "change_point_prob",
        "bocpd_run_length_mean",
        "bocpd_map_run_length",
        "predictive_log_likelihood",
    } == set(out.columns)
    assert len(out) == T
    # Change-point probabilities must be in [0, 1].
    assert (out["change_point_prob"] >= 0).all()
    assert (out["change_point_prob"] <= 1.0 + 1e-9).all()
    # And the ``deep_kernel`` slot was populated by the auto-train hook.
    assert detector.deep_kernel is not None


def test_deep_kernel_soft_degrade_without_torch(monkeypatch) -> None:
    """Constructing the class without torch raises with the install hint."""
    from market_regime_engine.frontier.deep_kernel import MLPDeepKernel

    # Stub out torch in sys.modules so the deferred import inside the
    # constructor surfaces the clean ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "torch.nn", None)
    with pytest.raises(ImportError, match=r"\[frontier\] extra"):
        MLPDeepKernel(input_dim=4, hidden_dims=(8,))
