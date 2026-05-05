# SPDX-License-Identifier: Apache-2.0
"""PatchTST sequence model (Nie-Nguyen-Sinthong-Kalagnanam 2023, ICLR 2023).

Implements a small version of PatchTST per "A Time Series is Worth 64 Words:
Long-Term Forecasting with Transformers" (arXiv 2211.14730). The default
hyperparameters are intentionally compact so the model trains in seconds on
a single CPU thread:

- patch length: 12 (one year of monthly data per patch)
- d_model: 128
- depth: 4 transformer encoder layers
- forecasted quantiles: ``(0.1, 0.25, 0.5, 0.75, 0.9)``

When ``torch`` is missing the module raises a clear ``ImportError`` per the
v1.2 spec so callers can branch on availability via ``HAS_TORCH``.

This is one of the candidate models in the BMA mix. The public surface
matches the other heads:

- ``fit(panel, target, *, horizon)`` — fit on a date-indexed wide panel and a
  target series; ``horizon`` is the number of forward periods predicted.
- ``predict(panel)`` — returns a dataframe of per-quantile predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def _torch_available() -> tuple[bool, Any]:
    try:  # pragma: no cover - exercised at import time
        import torch

        return True, torch
    except Exception:
        return False, None


HAS_TORCH, _torch_module = _torch_available()


@dataclass
class PatchTSTHead:
    """Compact PatchTST head with optional torch backend.

    When ``torch`` is missing the module's public methods raise
    :class:`ImportError` with a precise install hint. Callers can probe
    availability via :data:`HAS_TORCH` or by catching the import error from
    :meth:`fit`.
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

    fitted: bool = False
    _model: Any = None
    _input_columns: list[str] = field(default_factory=list)
    _train_quantiles: dict[float, float] = field(default_factory=dict)

    def _require_torch(self) -> Any:
        if not HAS_TORCH:
            raise ImportError(
                "PatchTSTHead requires torch. Install the [frontier] extra: "
                "`pip install -e .[frontier]` or `pip install torch>=2.0`."
            )
        return _torch_module

    def _build_model(self, n_features: int):  # pragma: no cover - depends on torch
        torch = self._require_torch()
        nn = torch.nn

        class _PatchEmbedding(nn.Module):  # type: ignore[name-defined]
            def __init__(self, patch_len: int, d_model: int) -> None:
                super().__init__()
                self.proj = nn.Linear(patch_len, d_model)

            def forward(self, x):
                # x: (batch, n_patches, patch_len) -> (batch, n_patches, d_model)
                return self.proj(x)

        class _PatchTST(nn.Module):  # type: ignore[name-defined]
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
                # x: (batch, seq_len) — split into overlapping patches
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

        return _PatchTST(self.patch_len, self.d_model, self.depth, self.n_heads, self.horizon, len(self.quantiles))

    def fit(self, panel: pd.DataFrame, target: pd.Series, *, horizon: int = 1) -> PatchTSTHead:
        torch = self._require_torch()
        if panel is None or panel.empty or target is None or len(target) == 0:
            return self
        self.horizon = int(horizon)
        self._input_columns = list(panel.columns)
        # Build a single channel sequence by averaging columns (per the
        # PatchTST paper's "channel-independent" trick generalized to 1-D).
        x = panel.fillna(0.0).to_numpy(dtype=float).mean(axis=1)
        y = np.asarray(target, dtype=float)
        seq_len = self.patch_len * 4  # use 4 patches per training window
        n = len(x)
        n_train = max(n - seq_len - self.horizon + 1, 0)
        if n_train < 16:
            # not enough data — record empirical quantiles and stay
            # untrained; predict() will fall back to those.
            self._train_quantiles = {q: float(np.quantile(y, q)) for q in self.quantiles}
            return self
        model = self._build_model(n)
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        torch.manual_seed(self.random_state)
        # Build training windows.
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
            # Pinball loss across quantiles.
            loss = torch.maximum(q_tensor * err, (q_tensor - 1) * err).mean()
            loss.backward()
            opt.step()
        self._model = model
        self.fitted = True
        # Empirical fallback quantiles for short test inputs.
        self._train_quantiles = {q: float(np.quantile(y, q)) for q in self.quantiles}
        return self

    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            if self._train_quantiles:
                rows = [
                    {"horizon": h, **{f"q{int(q * 100):02d}": v for q, v in self._train_quantiles.items()}}
                    for h in range(1, self.horizon + 1)
                ]
                return pd.DataFrame(rows)
            raise RuntimeError("PatchTSTHead.predict called before fit().")
        torch = self._require_torch()
        if panel is None or panel.empty:
            return pd.DataFrame()
        x = panel.fillna(0.0).reindex(columns=self._input_columns, fill_value=0.0).to_numpy(dtype=float).mean(axis=1)
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


__all__ = ["HAS_TORCH", "PatchTSTHead"]
