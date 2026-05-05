# SPDX-License-Identifier: Apache-2.0
"""Sequential / anytime-valid testing primitives.

Two anytime-valid e-value primitives ship here, both selected for the v1.2
"safe testing" gate:

1. :class:`EValueLogScore` — sequential e-value test of "model A's log-score
   dominates model B's" using the running likelihood-ratio e-process per
   Howard-Ramdas (2021) "Time-uniform, nonparametric, nonasymptotic
   confidence sequences" (AOS 2021). The test rejects ``H_0: E[L_A - L_B] <=
   0`` whenever the running e-value crosses ``1 / alpha``, and the rejection
   is *anytime-valid* — you may stop whenever convenient without inflating
   the type-I error.

2. :class:`SafeTestPromotion` — wraps the e-value into a promotion gate per
   Grünwald-de Heide-Koolen 2024 "Safe Testing" (JRSS-B 86:1091-1128).
   Replaces the Hansen MCS path in :func:`market_regime_engine.release_gates.
   evaluate_release_gate` when the operator selects
   ``promotion_method="e_values"``.

Both classes are pure-numpy and have zero soft dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass
class EValueLogScore:
    """Anytime-valid sequential e-value of "model A beats model B".

    The e-process is the running product of likelihood-ratio increments under
    the null ``H_0: E[L_A - L_B] <= 0`` (where larger ``L`` is *worse* — we
    follow the loss convention). The test statistic is

        E_t = prod_{s <= t} exp(-eta * (L_A_s - L_B_s)),

    parametrized by a tuning constant ``eta``. By Ville's inequality,
    ``P(sup_t E_t >= 1/alpha) <= alpha`` under the null, so rejecting at
    ``E_t >= 1/alpha`` controls the type-I error at any stopping time.

    Default ``eta`` is set to the GROW-style optimal value
    ``1 / (max(|L_A - L_B|) * 2)`` updated online; if the user provides a
    fixed ``eta`` we use that instead.
    """

    alpha: float = 0.05
    eta: float | None = None  # None => online estimate
    e_value: float = 1.0
    n: int = 0
    _abs_max: float = 1.0
    history: list[float] = field(default_factory=list)

    def update(self, loss_a: float, loss_b: float) -> float:
        """Roll the e-statistic forward with a single (loss_a, loss_b) pair."""
        d = float(loss_a) - float(loss_b)
        if not np.isfinite(d):
            return self.e_value
        self.n += 1
        self._abs_max = max(self._abs_max, abs(d))
        eta = float(self.eta) if self.eta is not None else 1.0 / (2.0 * max(self._abs_max, 1e-9))
        # Negative d means A is better than B (smaller loss). The e-process
        # rewards that case (multiplier > 1) under the alternative.
        log_inc = -eta * d
        # Cap the increment to keep the statistic bounded under heavy-tailed
        # loss differences.
        log_inc = float(np.clip(log_inc, -10.0, 10.0))
        self.e_value = float(self.e_value * math.exp(log_inc))
        self.e_value = float(max(self.e_value, EPS))
        self.history.append(float(self.e_value))
        return self.e_value

    def is_significant(self, level: float | None = None) -> bool:
        threshold = 1.0 / max(level if level is not None else self.alpha, EPS)
        return bool(self.e_value >= threshold)


@dataclass
class SafeTestPromotion:
    """Safe-testing promotion gate (Grünwald-de Heide-Koolen 2024).

    Wraps an :class:`EValueLogScore` so production pipelines can ask "should
    the challenger be promoted?" with an anytime-valid guarantee. The gate
    fires (returns ``True``) the first time the e-value crosses ``1 /
    alpha``; once fired it stays fired across subsequent updates so the
    decision is monotone.
    """

    alpha: float = 0.05
    eta: float | None = None
    e_value: float = 1.0
    fired: bool = False
    fired_at_n: int | None = None
    _e_test: EValueLogScore = field(default_factory=lambda: EValueLogScore(alpha=0.05))

    def __post_init__(self) -> None:
        self._e_test = EValueLogScore(alpha=self.alpha, eta=self.eta)

    def update(self, loss_challenger: float, loss_champion: float) -> dict:
        """Roll the gate forward with one (challenger, champion) loss pair.

        ``loss_challenger`` plays the role of ``loss_a`` (we want to detect
        when *challenger* is better, i.e. has smaller loss). Returns a status
        dict with ``e_value``, ``fired``, ``fired_at_n``.
        """
        e = self._e_test.update(loss_challenger, loss_champion)
        self.e_value = e
        if not self.fired and self._e_test.is_significant():
            self.fired = True
            self.fired_at_n = int(self._e_test.n)
        return {
            "e_value": float(self.e_value),
            "fired": bool(self.fired),
            "fired_at_n": self.fired_at_n,
            "n": int(self._e_test.n),
            "level": float(self.alpha),
        }

    @classmethod
    def run(
        cls,
        challenger_losses: Sequence[float],
        champion_losses: Sequence[float],
        *,
        alpha: float = 0.05,
        eta: float | None = None,
    ) -> dict:
        """Convenience: run the gate over two batches of losses and return the final status."""
        gate = cls(alpha=alpha, eta=eta)
        last: dict[str, object] = {}
        for la, lb in zip(challenger_losses, champion_losses, strict=False):
            last = gate.update(float(la), float(lb))
        last.setdefault("e_value", float(gate.e_value))
        last.setdefault("fired", bool(gate.fired))
        last.setdefault("fired_at_n", gate.fired_at_n)
        last.setdefault("n", int(gate._e_test.n))
        last.setdefault("level", float(alpha))
        return last


def evaluate_with_e_values(
    challenger_losses: pd.Series,
    champion_losses: pd.Series,
    *,
    alpha: float = 0.05,
) -> dict:
    """Compute the e-value gate over aligned challenger/champion loss series.

    Returns a JSON-serializable dict suitable for the warehouse / release
    gate. Missing values in either series are dropped.
    """
    a = pd.Series(challenger_losses, dtype=float)
    b = pd.Series(champion_losses, dtype=float)
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if joined.empty:
        return {
            "n": 0,
            "e_value": 1.0,
            "fired": False,
            "fired_at_n": None,
            "level": float(alpha),
        }
    return SafeTestPromotion.run(joined["a"], joined["b"], alpha=alpha)


__all__ = ["EValueLogScore", "SafeTestPromotion", "evaluate_with_e_values"]
