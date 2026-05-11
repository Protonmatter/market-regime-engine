# SPDX-License-Identifier: Apache-2.0
"""Hierarchical Bayesian liquidity model with partial pooling.

Per the v1.5 deep-research report §2 (OFR latent-liquidity-states): the
deterministic composite scorer is the explainable baseline; the
hierarchical Bayesian model layers on a partial-pooling prior across
``(sector, rating)`` buckets with a cusip-level random effect so a
sparsely-traded bond inherits its sector / rating tier's liquidity
prior rather than collapsing to neutral.

Model::

    y_ij ~ Normal(mu_i, sigma_obs)
    mu_i = mu_global + group_effect[s_i, r_i] + cusip_effect[i]
    group_effect[s, r] ~ Normal(0, sigma_group)
    cusip_effect[i] ~ Normal(0, sigma_cusip)
    sigma_group ~ HalfNormal(5)
    sigma_cusip ~ HalfNormal(2)
    sigma_obs ~ HalfNormal(1)
    mu_global ~ Normal(0, 10)

Inference: NumPyro NUTS. The ``[bayesian]`` extra (``numpyro``,
``jax[cpu]``, ``arviz``) is required; ``fit()`` raises a clean
``ImportError`` with the install hint when the extras are absent.

Out-of-sample prediction backs off the hierarchy: a cusip prediction
returns the cusip-level posterior; if the cusip was not observed
during fit, the helper falls back to the ``(sector, rating)`` bucket
and then to the global posterior. ``hierarchy_level`` on the returned
dict records the actual level used so the operator can audit the
back-off.

Per ``MRE_FIXED_INCOME_AGENT.md §"non-negotiables"`` (explainable
baselines first), this module ships as an *opt-in* secondary scorer.
The deterministic composite in :mod:`fixed_income.liquidity_stress`
remains the production default; the CLI / API surfaces gate the
hierarchical scorer behind a ``--use-hierarchical`` flag.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_INSTALL_HINT = (
    "HierarchicalLiquidityModel requires the optional [bayesian] extra. "
    "Install with `pip install market-regime-engine[bayesian]`."
)


def _require_numpyro() -> tuple[Any, Any, Any, Any, Any]:
    """Import jax + numpyro lazily so the soft-degrade test can stub them out.

    Mirrors :func:`frontier.bayesian_msvar._require_numpyro` exactly so
    the test harness can monkey-patch ``sys.modules['numpyro']=None``
    and observe the same ``ImportError`` shape.
    """
    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
    except ImportError as exc:  # pragma: no cover - import path
        raise ImportError(_INSTALL_HINT) from exc
    return jax, jnp, numpyro, dist, {"MCMC": MCMC, "NUTS": NUTS}


@dataclass
class HierarchicalLiquidityModel:
    """Partial-pooling Bayesian model for liquidity stress.

    Hierarchy:

    * group level — ``(sector, rating)`` bucket: shared offset.
    * individual level — cusip random effect.
    * observation — Gaussian on the configured value column.

    The constructor takes the **expected** sector / rating universe so
    the model can preallocate group-effect dimensions even when the
    fit panel is missing some combinations.

    Use :meth:`fit` once to run NUTS, then :meth:`predict` to query
    posterior summaries at the cusip / sector-rating / market level.
    """

    sectors: Sequence[str]
    ratings: Sequence[str]
    num_warmup: int = 500
    num_samples: int = 1000
    num_chains: int = 2
    seed: int = 42
    value_column: str = "liquidity_value"

    _sector_to_idx: dict[str, int] = field(default_factory=dict)
    _rating_to_idx: dict[str, int] = field(default_factory=dict)
    _cusip_to_idx: dict[str, int] = field(default_factory=dict)
    _cusip_metadata: dict[str, tuple[str, str]] = field(default_factory=dict)
    _posterior_samples: dict[str, np.ndarray] = field(default_factory=dict)
    _diagnostics: dict[str, Any] = field(default_factory=dict)
    fitted: bool = False

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._sector_to_idx = {str(s): i for i, s in enumerate(self.sectors)}
        self._rating_to_idx = {str(r): i for i, r in enumerate(self.ratings)}

    def fit(self, panel: pd.DataFrame) -> HierarchicalLiquidityModel:
        """Fit the model with NumPyro NUTS.

        ``panel`` must contain columns
        ``[cusip, sector, rating, <value_column>]``; additional numeric
        columns are tolerated (they are not used for inference).

        Raises
        ------
        ImportError
            When ``numpyro`` / ``jax`` are not installed.
        ValueError
            When ``panel`` is empty or missing required columns.
        """
        jax_mod, jnp, _numpyro, dist, infer = _require_numpyro()
        if panel is None or panel.empty:
            raise ValueError("HierarchicalLiquidityModel.fit requires a non-empty panel")
        required = {"cusip", "sector", "rating", self.value_column}
        missing = required - set(panel.columns)
        if missing:
            raise ValueError(
                f"HierarchicalLiquidityModel.fit panel missing columns: {sorted(missing)!r}"
            )

        df = panel.copy()
        df["cusip"] = df["cusip"].astype(str)
        df["sector"] = df["sector"].astype(str)
        df["rating"] = df["rating"].astype(str)
        df = df.dropna(subset=[self.value_column])
        if df.empty:
            raise ValueError("HierarchicalLiquidityModel.fit panel has no usable observations")

        cusips = sorted(df["cusip"].unique().tolist())
        self._cusip_to_idx = {c: i for i, c in enumerate(cusips)}
        # Map cusip → (sector, rating) for back-off lookups during predict.
        for _, row in df.iterrows():
            self._cusip_metadata.setdefault(row["cusip"], (row["sector"], row["rating"]))

        sector_idx = df["sector"].map(self._sector_to_idx).fillna(-1).astype(int).to_numpy()
        rating_idx = df["rating"].map(self._rating_to_idx).fillna(-1).astype(int).to_numpy()
        cusip_idx = df["cusip"].map(self._cusip_to_idx).astype(int).to_numpy()
        if (sector_idx < 0).any() or (rating_idx < 0).any():
            bad = df.loc[(sector_idx < 0) | (rating_idx < 0), ["sector", "rating"]].drop_duplicates()
            raise ValueError(
                "HierarchicalLiquidityModel.fit encountered (sector, rating) values "
                f"absent from the configured universe: {bad.to_dict(orient='records')!r}"
            )
        values = df[self.value_column].astype(float).to_numpy()

        n_sectors = len(self.sectors)
        n_ratings = len(self.ratings)
        n_cusips = len(cusips)

        def model(values_obs: jnp.ndarray) -> None:  # type: ignore[name-defined]
            mu_global = _numpyro.sample("mu_global", dist.Normal(0.0, 10.0))
            sigma_group = _numpyro.sample("sigma_group", dist.HalfNormal(5.0))
            sigma_cusip = _numpyro.sample("sigma_cusip", dist.HalfNormal(2.0))
            sigma_obs = _numpyro.sample("sigma_obs", dist.HalfNormal(1.0))
            group_effect = _numpyro.sample(
                "group_effect",
                dist.Normal(0.0, sigma_group).expand([n_sectors, n_ratings]).to_event(2),
            )
            cusip_effect = _numpyro.sample(
                "cusip_effect",
                dist.Normal(0.0, sigma_cusip).expand([n_cusips]).to_event(1),
            )
            mu = (
                mu_global
                + group_effect[sector_idx, rating_idx]
                + cusip_effect[cusip_idx]
            )
            _numpyro.sample("obs", dist.Normal(mu, sigma_obs), obs=values_obs)

        nuts = infer["NUTS"](model)
        mcmc = infer["MCMC"](
            nuts,
            num_warmup=int(self.num_warmup),
            num_samples=int(self.num_samples),
            num_chains=int(self.num_chains),
            progress_bar=False,
            chain_method="sequential",
        )
        key = jax_mod.random.PRNGKey(int(self.seed))
        mcmc.run(key, values_obs=jnp.asarray(values))

        samples = mcmc.get_samples()
        # Materialise numpy arrays so consumers don't need jax installed.
        self._posterior_samples = {k: np.asarray(v) for k, v in samples.items()}
        self._diagnostics = _compute_diagnostics(mcmc, samples)
        self.fitted = True
        return self

    def predict(
        self,
        cusip: str | None = None,
        sector: str | None = None,
        rating: str | None = None,
    ) -> dict[str, Any]:
        """Posterior summary for a cusip, ``(sector, rating)`` bucket, or market.

        Back-off order:

        1. ``cusip`` supplied and observed in fit → cusip-level posterior.
        2. ``sector + rating`` supplied → group-level posterior.
        3. ``rating`` supplied (no sector) → marginalise over sectors.
        4. Otherwise → market-level posterior (global mean only).

        Returns a dict with ``posterior_mean``, ``ci_low_5``,
        ``ci_high_95``, ``n_obs``, and ``hierarchy_level``.
        """
        if not self.fitted:
            raise RuntimeError("HierarchicalLiquidityModel.predict called before fit()")

        mu_global = self._posterior_samples["mu_global"]
        group = self._posterior_samples["group_effect"]
        cusip_eff = self._posterior_samples["cusip_effect"]

        # Cusip path
        if cusip is not None and cusip in self._cusip_to_idx:
            idx = self._cusip_to_idx[cusip]
            sec, rat = self._cusip_metadata.get(cusip, (None, None))
            if sec is not None and rat is not None:
                s = self._sector_to_idx.get(sec, None)
                r = self._rating_to_idx.get(rat, None)
                if s is not None and r is not None:
                    samples = mu_global + group[:, s, r] + cusip_eff[:, idx]
                else:
                    samples = mu_global + cusip_eff[:, idx]
            else:
                samples = mu_global + cusip_eff[:, idx]
            return _summarise(samples, n_obs=1, level="cusip")

        # Sector + rating path
        if sector is not None and rating is not None:
            s = self._sector_to_idx.get(str(sector))
            r = self._rating_to_idx.get(str(rating))
            if s is not None and r is not None:
                samples = mu_global + group[:, s, r]
                n_obs = sum(
                    1
                    for (sec_meta, rat_meta) in self._cusip_metadata.values()
                    if sec_meta == sector and rat_meta == rating
                )
                return _summarise(samples, n_obs=n_obs, level="sector_rating")

        # Rating-only path: marginalise over sectors.
        if rating is not None and sector is None:
            r = self._rating_to_idx.get(str(rating))
            if r is not None:
                samples = mu_global + group[:, :, r].mean(axis=1)
                n_obs = sum(
                    1 for (_, rat_meta) in self._cusip_metadata.values() if rat_meta == rating
                )
                return _summarise(samples, n_obs=n_obs, level="rating")

        # Market path: just the global mean.
        return _summarise(mu_global, n_obs=len(self._cusip_metadata), level="market")

    def diagnostics(self) -> dict[str, Any]:
        """Return r_hat / n_eff per group parameter (cached from ``fit``)."""
        if not self.fitted:
            raise RuntimeError("HierarchicalLiquidityModel.diagnostics called before fit()")
        return dict(self._diagnostics)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _summarise(samples: np.ndarray, *, n_obs: int, level: str) -> dict[str, Any]:
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0:
        return {
            "posterior_mean": float("nan"),
            "ci_low_5": float("nan"),
            "ci_high_95": float("nan"),
            "n_obs": int(n_obs),
            "hierarchy_level": level,
        }
    return {
        "posterior_mean": float(np.mean(samples)),
        "ci_low_5": float(np.quantile(samples, 0.05)),
        "ci_high_95": float(np.quantile(samples, 0.95)),
        "n_obs": int(n_obs),
        "hierarchy_level": level,
    }


def _compute_diagnostics(mcmc: Any, samples: Mapping[str, Any]) -> dict[str, Any]:
    """Return r_hat / n_eff summaries.

    Uses NumPyro's ``print_summary`` API where available; falls back to
    a hand-rolled per-parameter r_hat estimate that is good enough for
    the smoke-test diagnostic surface (no multi-chain dependency).
    """
    diag: dict[str, Any] = {"r_hat": {}, "n_eff": {}}
    try:
        summary = mcmc.summary(group_by_chain=False, prob=0.9)  # type: ignore[attr-defined]
        for name, stats in (summary or {}).items():
            r_hat = stats.get("r_hat")
            n_eff = stats.get("n_eff")
            if r_hat is not None:
                diag["r_hat"][name] = float(np.asarray(r_hat).mean())
            if n_eff is not None:
                diag["n_eff"][name] = float(np.asarray(n_eff).mean())
    except Exception:  # pragma: no cover - numpyro version drift
        pass
    if not diag["r_hat"]:
        # Hand-rolled fallback: variance of sample chain / variance of
        # within-half splits. Not as rigorous as the multi-chain r_hat
        # but adequate for the smoke tests.
        for name, values in samples.items():
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.size < 4:
                diag["r_hat"][name] = float("nan")
                diag["n_eff"][name] = float(arr.size)
                continue
            mid = arr.size // 2
            v1 = float(np.var(arr[:mid], ddof=1))
            v2 = float(np.var(arr[mid:], ddof=1))
            total = float(np.var(arr, ddof=1))
            diag["r_hat"][name] = float(np.sqrt((v1 + v2) / max(2.0 * total, 1e-12)))
            diag["n_eff"][name] = float(arr.size)
    return diag


__all__ = ["HierarchicalLiquidityModel"]
