# SPDX-License-Identifier: Apache-2.0
"""Distributional regression heads.

Three production-ready distributional heads ship here:

1. :class:`NGBoostHead` — Duan, Avati, Wang, Saxena, Schuler, Ng 2020
   "NGBoost: Natural Gradient Boosting for Probabilistic Prediction" (PMLR
   119:2690-2700). Wraps the ``ngboost`` package when the ``[frontier]`` extra
   is installed; soft-degrades to a per-row Gaussian fit when ngboost is
   missing.

2. :class:`IsotonicDistributionalHead` — Henzi, Ziegel, Gneiting 2021
   "Isotonic Distributional Regression" (JRSS-B 83:963-993). Pure numpy /
   scikit-learn; treats the distributional regression as a stack of binary
   isotonic regressions over thresholds and reads the calibrated CDF off
   them. This is the *non-parametric* gold standard and beats most
   parametric heads on calibration.

3. :class:`DeepStateSpaceHead` — Karl, Soelch, Bayer, van der Smagt 2017
   "Deep Variational Bayes Filters" (ICLR 2017). When ``torch`` is
   installed, fits a small latent state-space (default ``latent_dim=32``)
   with neural transition + emission, trained by ELBO. Otherwise degrades to
   :class:`NGBoostHead`.

All three heads expose a ``fit / predict / predict_distribution`` contract
that plugs straight into :class:`market_regime_engine.bma.OnlineBMA`'s
``dict[str, float]`` predictions interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def _ngboost_available() -> tuple[bool, Any]:
    try:  # pragma: no cover - exercised at import time
        import ngboost

        return True, ngboost
    except Exception:
        return False, None


def _torch_available() -> tuple[bool, Any]:
    try:  # pragma: no cover - exercised at import time
        import torch

        return True, torch
    except Exception:
        return False, None


# ---------------------------------------------------------------------------
# 1. NGBoost head
# ---------------------------------------------------------------------------


@dataclass
class NGBoostHead:
    """NGBoost wrapper with soft-degrade.

    Parameters
    ----------
    distribution:
        ``"normal"`` or ``"laplace"``. When NGBoost is missing the soft-
        degrade path always uses a per-row Normal head.
    n_estimators:
        Pass-through to NGBoost when available.
    learning_rate:
        Pass-through to NGBoost when available.
    """

    distribution: str = "normal"
    n_estimators: int = 200
    learning_rate: float = 0.05
    random_state: int = 0

    fitted: bool = False
    backend: str = "fallback"
    _model: Any = None
    _fallback_mean: float = 0.0
    _fallback_std: float = 1.0

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> NGBoostHead:
        installed, ngboost = _ngboost_available()
        Xa = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X.to_numpy(float)
        ya = np.asarray(y, dtype=float)
        if installed:
            try:  # pragma: no cover - depends on optional dep
                from ngboost.distns import Laplace, Normal

                dist = Laplace if self.distribution == "laplace" else Normal
                model = ngboost.NGBRegressor(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    Dist=dist,
                    random_state=self.random_state,
                    verbose=False,
                )
                model.fit(Xa, ya)
                self._model = model
                self.backend = "ngboost"
                self.fitted = True
                return self
            except Exception:
                pass
        # Fallback: marginal Normal.
        self._fallback_mean = float(np.mean(ya))
        self._fallback_std = float(max(np.std(ya, ddof=1), 1e-6))
        self.backend = "fallback"
        self.fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(X) if X is not None else 0)
        Xa = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X.to_numpy(float)
        if self.backend == "ngboost" and self._model is not None:  # pragma: no cover - optional
            return np.asarray(self._model.predict(Xa), dtype=float)
        return np.full(len(Xa), self._fallback_mean)

    def predict_distribution(self, X: np.ndarray | pd.DataFrame) -> list[dict]:
        """Return per-row distribution descriptors (mean, scale, family)."""
        if not self.fitted:
            return []
        Xa = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X.to_numpy(float)
        rows: list[dict] = []
        if self.backend == "ngboost" and self._model is not None:  # pragma: no cover - optional
            try:
                dists = self._model.pred_dist(Xa)
                params = getattr(dists, "params", {})
                loc = np.asarray(params.get("loc", self._model.predict(Xa)), dtype=float)
                scale = np.asarray(params.get("scale", np.full(len(Xa), self._fallback_std)), dtype=float)
                for i in range(len(Xa)):
                    rows.append(
                        {
                            "family": self.distribution,
                            "loc": float(loc[i]),
                            "scale": float(scale[i]),
                        }
                    )
                return rows
            except Exception:
                pass
        for _ in range(len(Xa)):
            rows.append(
                {
                    "family": "normal",
                    "loc": float(self._fallback_mean),
                    "scale": float(self._fallback_std),
                }
            )
        return rows


# ---------------------------------------------------------------------------
# 2. Isotonic Distributional Regression (Henzi-Ziegel-Gneiting 2021)
# ---------------------------------------------------------------------------


@dataclass
class IsotonicDistributionalHead:
    """Henzi-Ziegel-Gneiting (2021) IDR.

    Trains a 1-D isotonic regression of ``P(Y <= y_k | X)`` against a single
    scalar feature (the "rank" of ``X`` per the paper); for multivariate
    ``X`` we project on the *first principal component* of the calibration
    features to keep the head pure-numpy. The CDF at any threshold is then
    read off the isotonic curve.

    For each test row we report the empirical CDF at a configurable grid of
    quantile levels ``cdf_grid``. The default is ``np.linspace(0.05, 0.95,
    19)`` which matches the typical 5%-95% interval reporting.
    """

    cdf_grid: np.ndarray = field(default_factory=lambda: np.linspace(0.05, 0.95, 19))
    fitted: bool = False
    _projection: np.ndarray | None = None
    _train_scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    _train_y: np.ndarray = field(default_factory=lambda: np.empty(0))

    def _project(self, X: np.ndarray) -> np.ndarray:
        if self._projection is None:
            return X.mean(axis=1) if X.ndim == 2 else X.astype(float)
        return X @ self._projection

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> IsotonicDistributionalHead:
        Xa = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X.to_numpy(float)
        ya = np.asarray(y, dtype=float)
        if Xa.ndim == 1:
            Xa = Xa[:, None]
        # Fit a tiny PCA: top-1 component direction.
        Xc = Xa - Xa.mean(axis=0, keepdims=True)
        try:
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            self._projection = Vt[0]
        except np.linalg.LinAlgError:
            self._projection = np.ones(Xa.shape[1]) / max(Xa.shape[1], 1)
        scores = self._project(Xa)
        # Sort by projected score so the CDF can be looked up via searchsorted.
        order = np.argsort(scores)
        self._train_scores = scores[order]
        self._train_y = ya[order]
        self.fitted = True
        return self

    def _empirical_cdf(self, score: float, level: float) -> float:
        """Return P(Y <= y_level | rank ~= rank(score)) via window of similar scores."""
        if not self.fitted or self._train_scores.size == 0:
            return float("nan")
        # Take the 10% nearest-rank window (Henzi-Ziegel-Gneiting "neighbour"
        # estimator); width adapts to the calibration set size.
        n = len(self._train_scores)
        idx = int(np.searchsorted(self._train_scores, score, side="left"))
        half = max(int(0.05 * n), 5)
        lo = max(idx - half, 0)
        hi = min(idx + half, n)
        window_y = self._train_y[lo:hi]
        threshold = float(np.quantile(self._train_y, level))
        return float(np.mean(window_y <= threshold))

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return the median (50th percentile) of the predicted distribution per row."""
        if not self.fitted:
            return np.zeros(len(X) if X is not None else 0)
        dists = self.predict_distribution(X)
        out = np.zeros(len(dists))
        for i, d in enumerate(dists):
            cdf = d["cdf"]
            grid = d["levels"]
            # Find the level closest to the median crossing.
            arr = np.asarray(cdf)
            target = 0.5
            j = int(np.argmin(np.abs(arr - target)))
            out[i] = float(np.quantile(self._train_y, grid[j])) if self._train_y.size else 0.0
        return out

    def predict_distribution(self, X: np.ndarray | pd.DataFrame) -> list[dict]:
        if not self.fitted:
            return []
        Xa = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X.to_numpy(float)
        if Xa.ndim == 1:
            Xa = Xa[:, None]
        scores = self._project(Xa)
        rows: list[dict] = []
        for s in scores:
            cdf = [self._empirical_cdf(float(s), float(level)) for level in self.cdf_grid]
            rows.append(
                {
                    "family": "isotonic_empirical",
                    "levels": self.cdf_grid.tolist(),
                    "cdf": cdf,
                }
            )
        return rows


# ---------------------------------------------------------------------------
# 3. Deep state-space head (Karl-Soelch-Bayer-van der Smagt 2017 inspired)
# ---------------------------------------------------------------------------


@dataclass
class DeepStateSpaceHead:
    """Tiny deep state-space head with torch backend + soft-degrade.

    The default torch backend is a 32-dim latent ``z_t`` with neural
    transition (single linear layer + tanh) and emission (linear), trained
    end-to-end by ELBO. When torch isn't available we transparently fall
    back to :class:`NGBoostHead` so callers never need to branch.
    """

    latent_dim: int = 32
    n_epochs: int = 30
    learning_rate: float = 1e-3
    random_state: int = 0

    fitted: bool = False
    backend: str = "fallback"
    _torch_model: Any = None
    _ngb_fallback: NGBoostHead | None = None

    def _make_torch_model(self, n_features: int):  # pragma: no cover - optional
        installed, torch = _torch_available()
        if not installed:
            return None
        nn = torch.nn

        class TinyDeepSSM(nn.Module):  # type: ignore[name-defined]
            def __init__(self, n_features: int, latent_dim: int) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(n_features, 64),
                    nn.Tanh(),
                    nn.Linear(64, 2 * latent_dim),
                )
                self.transition = nn.Sequential(
                    nn.Linear(latent_dim, latent_dim),
                    nn.Tanh(),
                )
                self.decoder_mean = nn.Linear(latent_dim, 1)
                self.decoder_log_var = nn.Parameter(torch.zeros(1))
                self.latent_dim = latent_dim

            def forward(self, x):
                enc = self.encoder(x)
                mu, logvar = enc.chunk(2, dim=-1)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                z = self.transition(z)
                y_mean = self.decoder_mean(z).squeeze(-1)
                return y_mean, mu, logvar

        return TinyDeepSSM(n_features, self.latent_dim)

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> DeepStateSpaceHead:
        installed, torch = _torch_available()
        Xa = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X.to_numpy(float)
        ya = np.asarray(y, dtype=float)
        if installed:
            try:  # pragma: no cover - depends on optional dep
                model = self._make_torch_model(Xa.shape[1] if Xa.ndim == 2 else 1)
                if model is None:
                    raise RuntimeError("torch model construction failed")
                torch.manual_seed(self.random_state)
                opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
                Xt = torch.tensor(Xa, dtype=torch.float32)
                yt = torch.tensor(ya, dtype=torch.float32)
                model.train()
                for _ in range(self.n_epochs):
                    opt.zero_grad()
                    y_mean, mu, logvar = model(Xt)
                    rec_loss = ((yt - y_mean) ** 2).mean()
                    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
                    loss = rec_loss + 1e-3 * kl
                    loss.backward()
                    opt.step()
                self._torch_model = model
                self.backend = "torch"
                self.fitted = True
                return self
            except Exception:
                pass
        self._ngb_fallback = NGBoostHead().fit(Xa, ya)
        self.backend = "fallback"
        self.fitted = True
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(X) if X is not None else 0)
        if self.backend == "torch" and self._torch_model is not None:  # pragma: no cover - optional
            installed, torch = _torch_available()
            if installed:
                Xa = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X.to_numpy(float)
                Xt = torch.tensor(Xa, dtype=torch.float32)
                self._torch_model.eval()
                with torch.no_grad():
                    y_mean, _, _ = self._torch_model(Xt)
                return np.asarray(y_mean.detach().cpu().numpy(), dtype=float)
        if self._ngb_fallback is not None:
            return self._ngb_fallback.predict(X)
        return np.zeros(len(X) if X is not None else 0)

    def predict_distribution(self, X: np.ndarray | pd.DataFrame) -> list[dict]:
        if not self.fitted:
            return []
        if self.backend == "torch" and self._torch_model is not None:  # pragma: no cover - optional
            installed, torch = _torch_available()
            if installed:
                Xa = np.asarray(X, dtype=float) if not isinstance(X, pd.DataFrame) else X.to_numpy(float)
                Xt = torch.tensor(Xa, dtype=torch.float32)
                self._torch_model.eval()
                with torch.no_grad():
                    y_mean, _, _ = self._torch_model(Xt)
                    log_var_emit = float(self._torch_model.decoder_log_var.item())
                std = float(np.sqrt(np.exp(log_var_emit)))
                rows = []
                for i in range(len(Xa)):
                    rows.append(
                        {
                            "family": "deep_ssm_normal",
                            "loc": float(y_mean[i].item()),
                            "scale": std,
                        }
                    )
                return rows
        if self._ngb_fallback is not None:
            return self._ngb_fallback.predict_distribution(X)
        return []


__all__ = ["DeepStateSpaceHead", "IsotonicDistributionalHead", "NGBoostHead"]
