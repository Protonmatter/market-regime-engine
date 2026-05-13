# SPDX-License-Identifier: Apache-2.0
"""Multivariate average-patch transformer head (NOT canonical PatchTST).

v1.6.0 honest-naming refactor (REVIEW_DEEP_V1_5_2.md §1.13 / Finding #6):
the previous class name ``PatchTSTHead`` claimed to implement
Nie-Nguyen-Sinthong-Kalagnanam 2023 (ICLR 2023, "A Time Series is
Worth 64 Words: Long-Term Forecasting with Transformers", arXiv
2211.14730). PatchTST's defining contribution is **channel-independent
processing**: each input channel is patched and embedded
independently, then mixed only in the encoder. The shipped
implementation collapses the channel dimension *at the input* via
``mean(axis=1)`` — every cross-channel signal is destroyed before the
patch encoder sees the data.

This module is renamed to :class:`MultivariateAvgPatchHead` to honestly
describe what it does: average the input channels, patch the resulting
1-D series, run a small transformer encoder over the patches, project
to per-quantile forecasts. The behaviour is unchanged from v1.5.x; only
the name and docstring are corrected.

A faithful channel-independent PatchTST implementation per Nie et al.
2023 is tracked as a v1.7.0 TODO.

Hyperparameters (unchanged):

- patch length: 12 (one year of monthly data per patch)
- d_model: 128
- depth: 4 transformer encoder layers
- forecasted quantiles: ``(0.1, 0.25, 0.5, 0.75, 0.9)``

When ``torch`` is missing the module raises a clear ``ImportError`` per
the v1.2 spec so callers can branch on availability via ``HAS_TORCH``.

Public API (matches the other heads):

- ``fit(panel, target, *, horizon)`` — fit on a date-indexed wide panel
  and a target series; ``horizon`` is the number of forward periods
  predicted. Raises :class:`ValueError` when ``n_train < 16`` (the
  previous degenerate empirical-quantile fallback is removed; callers
  decide whether to fall back to a simpler head).
- ``predict(panel)`` — returns a dataframe of per-quantile predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from market_regime_engine.frontier.data_cleaning import NanPolicy, clean_with_policy


def _torch_available() -> tuple[bool, Any]:
    try:  # pragma: no cover - exercised at import time
        import torch

        return True, torch
    except Exception:
        return False, None


HAS_TORCH, _torch_module = _torch_available()


@dataclass
class MultivariateAvgPatchHead:
    """Compact average-channel-then-patch transformer head.

    NOT a faithful PatchTST: the input channels are averaged into a
    single 1-D series before patching (the canonical PatchTST processes
    each channel independently). See module docstring for the rename
    rationale and v1.7.0 follow-up.

    When ``torch`` is missing the module's public methods raise
    :class:`ImportError` with a precise install hint. Callers can probe
    availability via :data:`HAS_TORCH` or by catching the import error
    from :meth:`fit`.

    v1.6.0 (REVIEW_DEEP_V1_5_2.md §1.13 / Finding #6):
    - Renamed from ``PatchTSTHead``.
    - The ``n_features`` parameter on ``_build_model`` is removed (it
      was dead — the transformer is a 1-D encoder over patches).
    - The degenerate ``n_train < 16`` empirical-quantile fallback now
      raises :class:`ValueError`; callers may catch and fall back to a
      simpler head explicitly.
    - The implicit ``fillna(0.0)`` is replaced with ``nan_policy``
      routing through :func:`clean_with_policy` (NaN-To-Zero remains
      the default for backwards compatibility).
    """

    patch_len: int = 12
    d_model: int = 128
    depth: int = 4
    n_heads: int = 4
    horizon: int = 1
    learning_rate: float = 1e-3
    n_epochs: int = 20
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
    random_state: int = 0
    nan_policy: NanPolicy = NanPolicy.NAN_TO_ZERO

    fitted: bool = False
    _model: Any = None
    _input_columns: list[str] = field(default_factory=list)
    _train_quantiles: dict[float, float] = field(default_factory=dict)

    def _require_torch(self) -> Any:
        if not HAS_TORCH:
            raise ImportError(
                "MultivariateAvgPatchHead requires torch. Install the [frontier] extra: "
                "`pip install -e .[frontier]` or `pip install torch>=2.0`."
            )
        return _torch_module

    def _build_model(self):  # pragma: no cover - depends on torch
        torch = self._require_torch()
        nn = torch.nn

        class _PatchEmbedding(nn.Module):
            def __init__(self, patch_len: int, d_model: int) -> None:
                super().__init__()
                self.proj = nn.Linear(patch_len, d_model)

            def forward(self, x):
                return self.proj(x)

        class _PatchTransformer(nn.Module):
            def __init__(
                self, patch_len: int, d_model: int, depth: int, n_heads: int, horizon: int, n_quantiles: int
            ) -> None:
                super().__init__()
                self.patch_len = patch_len
                self.embed = _PatchEmbedding(patch_len, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, batch_first=True, dim_feedforward=d_model * 2, dropout=0.0
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
                self.head = nn.Linear(d_model, horizon * n_quantiles)
                self.horizon = horizon
                self.n_quantiles = n_quantiles

            def forward(self, x):
                b, n = x.shape
                p = self.patch_len
                if n < p:
                    pad = torch.zeros(b, p - n, dtype=x.dtype, device=x.device)
                    x = torch.cat([pad, x], dim=1)
                    n = p
                # Non-overlapping patches for simplicity (TimesFM-style stride=patch_len).
                n_patches = n // p
                x = x[:, : n_patches * p].reshape(b, n_patches, p)
                tokens = self.embed(x)
                encoded = self.encoder(tokens)
                pooled = encoded.mean(dim=1)
                return self.head(pooled).reshape(b, self.horizon, self.n_quantiles)

        return _PatchTransformer(
            self.patch_len, self.d_model, self.depth, self.n_heads, self.horizon, len(self.quantiles)
        )

    def fit(self, panel: pd.DataFrame, target: pd.Series, *, horizon: int = 1) -> MultivariateAvgPatchHead:
        torch = self._require_torch()
        if panel is None or panel.empty or target is None or len(target) == 0:
            return self
        self.horizon = int(horizon)
        self._input_columns = list(panel.columns)
        # v1.6.0: NaN policy applied explicitly. The channel dimension is
        # averaged before patching (NOT canonical PatchTST — see module
        # docstring); v1.7.0 will implement the channel-independent variant.
        cleaned = clean_with_policy(panel, default_policy=self.nan_policy).astype(float)
        cleaned_arr = cleaned.to_numpy(dtype=float)
        cleaned_arr = np.where(np.isfinite(cleaned_arr), cleaned_arr, 0.0)
        x = cleaned_arr.mean(axis=1)
        y = np.asarray(target, dtype=float)
        seq_len = self.patch_len * 4  # use 4 patches per training window
        n = len(x)
        n_train = max(n - seq_len - self.horizon + 1, 0)
        if n_train < 16:
            # v1.6.0 (REVIEW_DEEP_V1_5_2.md §1.13): the prior empirical-
            # quantile fallback was useless as a forecaster (returned the
            # same numbers regardless of input panel). Raise so the
            # caller can decide whether to fall back to a simpler head.
            raise ValueError(
                f"Insufficient training data for MultivariateAvgPatchHead: need >= 16 windows, "
                f"got n_train={n_train} from n={n} samples with seq_len={seq_len} and horizon={self.horizon}."
            )
        model = self._build_model()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        torch.manual_seed(self.random_state)
        X_train = np.stack([x[i : i + seq_len] for i in range(n_train)])
        Y_train = np.stack([y[i + seq_len : i + seq_len + self.horizon] for i in range(n_train)])
        Xt = torch.tensor(X_train, dtype=torch.float32)
        Yt = torch.tensor(Y_train, dtype=torch.float32)
        q_tensor = torch.tensor(self.quantiles, dtype=torch.float32)
        model.train()
        for _ in range(self.n_epochs):
            opt.zero_grad()
            pred = model(Xt)  # (n_train, horizon, n_quantiles)
            err = Yt.unsqueeze(-1) - pred
            loss = torch.maximum(q_tensor * err, (q_tensor - 1) * err).mean()
            loss.backward()
            opt.step()
        self._model = model
        self.fitted = True
        # Empirical-fallback quantiles kept ONLY for the case where
        # fit() succeeded and predict() encounters a panel shorter than
        # seq_len (we still pad with zeros and emit a model prediction).
        self._train_quantiles = {q: float(np.quantile(y, q)) for q in self.quantiles}
        return self

    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("MultivariateAvgPatchHead.predict called before fit().")
        torch = self._require_torch()
        if panel is None or panel.empty:
            return pd.DataFrame()
        cleaned = clean_with_policy(
            panel.reindex(columns=self._input_columns, fill_value=0.0),
            default_policy=self.nan_policy,
        ).astype(float)
        cleaned_arr = cleaned.to_numpy(dtype=float)
        cleaned_arr = np.where(np.isfinite(cleaned_arr), cleaned_arr, 0.0)
        x = cleaned_arr.mean(axis=1)
        seq_len = self.patch_len * 4
        if len(x) < seq_len:
            x = np.concatenate([np.zeros(seq_len - len(x)), x])
        Xt = torch.tensor(x[-seq_len:], dtype=torch.float32).unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            pred = self._model(Xt).squeeze(0).cpu().numpy()  # (horizon, n_quantiles)
        rows = []
        for h_idx in range(pred.shape[0]):
            row: dict[str, float] = {"horizon": float(h_idx + 1)}
            for q_idx, q in enumerate(self.quantiles):
                row[f"q{int(q * 100):02d}"] = float(pred[h_idx, q_idx])
            rows.append(row)
        return pd.DataFrame(rows)


# Backwards-compat alias so v1.5.x callers do not break at import time.
# Tagged for removal in v1.7.0 alongside the channel-independent variant.
PatchTSTHead = MultivariateAvgPatchHead


__all__ = ["HAS_TORCH", "MultivariateAvgPatchHead", "PatchTSTHead"]
