# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Arc:
    src: str
    dst: str
    cost: float
    label: str = "normal"


DEFAULT_STATES = {
    "risk_on_expansion",
    "late_cycle",
    "soft_landing",
    "sticky_inflation",
    "credit_stress",
    "energy_shock",
    "recessionary_bear",
    "stagflation",
    "liquidity_meltup",
}


# Hand-coded prior arcs. These remain the prior over the regime grammar even
# after empirical learning so that improbable arcs (e.g. recessionary_bear ->
# liquidity_meltup) stay closed unless the data screams about them.
PRIOR_ARCS: tuple[Arc, ...] = (
    Arc("risk_on_expansion", "late_cycle", 0.35),
    Arc("risk_on_expansion", "soft_landing", 0.70),
    Arc("risk_on_expansion", "liquidity_meltup", 0.65),
    Arc("liquidity_meltup", "late_cycle", 0.45),
    Arc("liquidity_meltup", "risk_on_expansion", 0.35),
    Arc("late_cycle", "sticky_inflation", 0.38, "inflation_break"),
    Arc("late_cycle", "soft_landing", 0.46, "inflation_cools"),
    Arc("late_cycle", "credit_stress", 0.55, "credit_break"),
    Arc("late_cycle", "recessionary_bear", 1.20, "labor_break"),
    Arc("sticky_inflation", "energy_shock", 0.35, "oil_break"),
    Arc("sticky_inflation", "stagflation", 0.45, "labor_break"),
    Arc("sticky_inflation", "recessionary_bear", 0.90, "labor_break"),
    Arc("credit_stress", "recessionary_bear", 0.20, "credit_break"),
    Arc("credit_stress", "soft_landing", 0.95, "credit_heals"),
    Arc("energy_shock", "stagflation", 0.38, "oil_break"),
    Arc("energy_shock", "recessionary_bear", 0.70, "labor_break"),
    Arc("stagflation", "recessionary_bear", 0.45, "labor_break"),
    Arc("stagflation", "soft_landing", 1.10, "inflation_cools"),
    Arc("recessionary_bear", "soft_landing", 0.55, "policy_pivot"),
    Arc("soft_landing", "risk_on_expansion", 0.35, "growth_recovers"),
    Arc("soft_landing", "late_cycle", 0.55),
)


@dataclass
class RegimeWFST:
    """Weighted finite-state regime decoder.

    The decoder consumes observed candidate regimes, optional event labels, and
    optional posterior probabilities from the HMM. It returns the least-cost
    valid regime path. This is not a market predictor by itself; it is the
    grammar layer that prevents noisy one-month labels from producing absurd
    regime jumps.

    Costs can either come from the hand-coded prior (``PRIOR_ARCS``) or be
    learned from data via :meth:`fit_costs_from_transition_matrix` and
    :meth:`fit_event_bonus`. Learning always preserves the prior arc set;
    transitions that the prior denies stay disabled.
    """

    start: str = "risk_on_expansion"
    default_cost: float = 3.0
    stay_cost: float = 0.12
    event_bonus: float = 0.35
    states: set[str] = field(default_factory=lambda: DEFAULT_STATES.copy())
    prior_blend: float = 0.5

    def __post_init__(self) -> None:
        self.arcs = list(PRIOR_ARCS)
        self._arc_map = defaultdict(list)
        for arc in self.arcs:
            self._arc_map[(arc.src, arc.dst)].append(arc)
        self.fitted = False
        self.fit_log: dict[str, float] = {}

    def transition_cost(self, src: str, dst: str, labels: set[str] | None = None) -> float:
        if src == dst:
            return self.stay_cost
        labels = labels or set()
        arcs = self._arc_map.get((src, dst), [])
        if not arcs:
            return self.default_cost
        best = self.default_cost
        for arc in arcs:
            cost = arc.cost
            if arc.label in labels:
                cost = max(0.01, cost - self.event_bonus)
            best = min(best, cost)
        return best

    def decode(
        self,
        observed: list[str],
        *,
        event_labels: list[set[str]] | None = None,
        posterior_rows: list[dict[str, float]] | None = None,
    ) -> list[str]:
        if not observed:
            return []
        states = sorted(self.states)
        event_labels = event_labels or [set() for _ in observed]
        posterior_rows = posterior_rows or [{} for _ in observed]

        dp: list[dict[str, float]] = []
        back: list[dict[str, str | None]] = []

        first = {}
        first_back = {}
        for s in states:
            posterior_penalty = -math.log(max(posterior_rows[0].get(s, 0.0), 1e-6)) if posterior_rows[0] else 0.0
            emission = 0.0 if s == observed[0] else 1.0
            first[s] = (0.0 if s == self.start else 0.8) + emission + 0.2 * posterior_penalty
            first_back[s] = None
        dp.append(first)
        back.append(first_back)

        for i in range(1, len(observed)):
            layer = {}
            layer_back = {}
            labels = event_labels[i] if i < len(event_labels) else set()
            post = posterior_rows[i] if i < len(posterior_rows) else {}
            for dst in states:
                posterior_penalty = -math.log(max(post.get(dst, 0.0), 1e-6)) if post else 0.0
                emission = 0.0 if dst == observed[i] else 1.0
                choices = [
                    (
                        dp[i - 1][src] + self.transition_cost(src, dst, labels) + emission + 0.2 * posterior_penalty,
                        src,
                    )
                    for src in states
                ]
                cost, src = min(choices, key=lambda x: x[0])
                layer[dst] = cost
                layer_back[dst] = src
            dp.append(layer)
            back.append(layer_back)

        cur = min(dp[-1], key=dp[-1].get)
        path = [cur]
        for i in range(len(observed) - 1, 0, -1):
            prev = back[i][cur]
            if prev is None:
                break
            cur = prev
            path.append(cur)
        return list(reversed(path))

    # ------------------------------------------------------------------
    # Learning methods
    # ------------------------------------------------------------------

    def fit_costs_from_transition_matrix(
        self,
        transition: pd.DataFrame | np.ndarray,
        states: list[str] | None = None,
        *,
        smoothing: float = 1.0,
    ) -> RegimeWFST:
        """Re-cost the prior arcs using empirical transition probabilities.

        Each prior arc's cost is replaced by the negative log of the empirical
        ``P(dst | src)``, blended with the prior's hand-coded cost via
        ``self.prior_blend``. Laplace smoothing avoids ``-log 0`` for unseen
        transitions.

        ``transition`` may be either a count matrix or an already-normalized
        probability matrix. We auto-detect: if all row sums are close to 1.0
        the input is treated as a probability matrix and smoothing is applied
        only as a numerical floor (clipped to a small epsilon and renormalized
        after blending). Otherwise smoothing is interpreted as Dirichlet
        pseudocounts.

        Parameters
        ----------
        transition:
            Either a square dataframe indexed by state name, or a numpy array
            paired with an explicit ``states`` list.
        states:
            Required when ``transition`` is a numpy array.
        smoothing:
            Pseudocount or epsilon depending on input mode (see above).
        """
        if isinstance(transition, pd.DataFrame):
            states = list(transition.index)
            mat = transition.to_numpy(dtype=float)
        else:
            if states is None:
                raise ValueError("states required when transition is an ndarray")
            mat = np.asarray(transition, dtype=float)

        K = len(states)
        if mat.shape != (K, K):
            raise ValueError(f"transition must be square ({K}, {K}); got {mat.shape}")
        idx = {s: i for i, s in enumerate(states)}

        row_sums = mat.sum(axis=1)
        looks_like_probabilities = bool(np.all(np.abs(row_sums - 1.0) < 1e-3))
        if looks_like_probabilities:
            # Treat input as already-normalized P(dst|src). Apply only a small
            # numerical floor so log is finite, then renormalize per row.
            eps = max(1e-9, float(smoothing) * 1e-3)
            prob = np.maximum(mat, eps)
            prob = prob / prob.sum(axis=1, keepdims=True)
        else:
            smoothed = mat + smoothing
            row_total = smoothed.sum(axis=1, keepdims=True)
            prob = smoothed / np.where(row_total > 0, row_total, 1.0)

        new_arcs: list[Arc] = []
        new_map: dict[tuple[str, str], list[Arc]] = defaultdict(list)
        for arc in PRIOR_ARCS:
            i = idx.get(arc.src)
            j = idx.get(arc.dst)
            if i is None or j is None:
                new_arcs.append(arc)
                new_map[(arc.src, arc.dst)].append(arc)
                continue
            empirical_cost = float(-np.log(max(prob[i, j], 1e-9)))
            blended = (1.0 - self.prior_blend) * empirical_cost + self.prior_blend * arc.cost
            new_arc = Arc(arc.src, arc.dst, max(0.01, blended), arc.label)
            new_arcs.append(new_arc)
            new_map[(arc.src, arc.dst)].append(new_arc)

        # Empirical self-stay cost.
        stays = np.array([float(prob[i, i]) for i in range(K)], dtype=float)
        stay_emp_cost = float(-np.log(max(stays.mean(), 1e-9)))
        self.stay_cost = max(0.01, (1.0 - self.prior_blend) * stay_emp_cost + self.prior_blend * self.stay_cost)

        self.arcs = new_arcs
        self._arc_map = new_map
        self.fitted = True
        self.fit_log["transition_smoothing"] = float(smoothing)
        self.fit_log["mean_stay_prob"] = float(stays.mean())
        return self

    def fit_event_bonus(
        self,
        observed: list[str],
        gold: list[str],
        *,
        event_labels: list[set[str]] | None = None,
        posterior_rows: list[dict[str, float]] | None = None,
        candidates: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0),
    ) -> RegimeWFST:
        """Pick the ``event_bonus`` that minimizes Hamming distance to a gold path.

        The gold path is typically the in-sample HMM Viterbi labelling on a
        held-out window. We grid-search a small candidate set and keep the
        bonus that produces the closest decoded path. This is a deliberate
        single-scalar fit so it cannot overfit.
        """
        if not observed or not gold or len(observed) != len(gold):
            return self
        best_bonus = self.event_bonus
        best_loss = float("inf")
        for c in candidates:
            self.event_bonus = float(c)
            decoded = self.decode(observed, event_labels=event_labels, posterior_rows=posterior_rows)
            loss = sum(1 for a, b in zip(decoded, gold, strict=False) if a != b)
            if loss < best_loss:
                best_loss = loss
                best_bonus = float(c)
        self.event_bonus = best_bonus
        self.fit_log["event_bonus"] = float(best_bonus)
        self.fit_log["event_bonus_loss"] = float(best_loss)
        return self


def event_labels_from_scores(scores: dict[str, float], cp_prob: float = 0.0) -> set[str]:
    labels: set[str] = set()
    if scores.get("credit", 0.0) > 1.1 or (scores.get("credit", 0.0) > 0.8 and cp_prob > 0.45):
        labels.add("credit_break")
    if scores.get("housing", 0.0) > 0.9:
        labels.add("housing_break")
    if scores.get("labor", 0.0) > 0.9:
        labels.add("labor_break")
    if scores.get("inflation", 0.0) > 1.2:
        labels.add("inflation_break")
    if scores.get("energy", 0.0) > 1.0:
        labels.add("oil_break")
    if scores.get("inflation", 0.0) < 0.7 and scores.get("labor", 0.0) < 0.5:
        labels.add("inflation_cools")
    if scores.get("credit", 0.0) < 0.5 and scores.get("labor", 0.0) < 0.5:
        labels.add("credit_heals")
    if cp_prob > 0.65:
        labels.add("structural_break")
    return labels
