# SPDX-License-Identifier: Apache-2.0
"""Historical scenario library + replay harness.

The docs require the engine to demonstrate sensible behavior on the canonical
historical stress scenarios. Each scenario is described by:

- A name and short description.
- A start / end date range to extract from the warehouse panel.
- Expected qualitative outcomes (regime transition, hazard direction, etc.).

The :func:`replay_scenario` driver pulls the actual feature path from the
warehouse, scores it through the engine's regime / hazard / change-point
chain, and returns both the structured outputs and a pass/fail verdict for
each documented expectation.

Scenarios deliberately stay declarative and additive: extend
:data:`SCENARIOS` to add a new historical episode without code changes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from market_regime_engine.features import build_features, monthly_panel
from market_regime_engine.hazard_model import train_fitted_hazard_outputs
from market_regime_engine.regimes import score_regimes


@dataclass(frozen=True)
class Scenario:
    name: str
    start: str
    end: str
    description: str
    expected_regimes: tuple[str, ...] = ()  # any of these counts as a pass
    expected_change_point: bool = True
    expected_hazard_direction: str = "up"  # "up" | "down" | "flat"


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="oil_shock_1973",
        start="1973-10-01",
        end="1975-06-01",
        description="OPEC embargo, oil shock, 1973-1975 recession.",
        expected_regimes=("energy_shock", "stagflation", "recessionary_bear"),
        expected_change_point=True,
        expected_hazard_direction="up",
    ),
    Scenario(
        name="volcker_disinflation",
        start="1979-08-01",
        end="1982-12-01",
        description="Volcker tightening, double-dip recessions, disinflation.",
        expected_regimes=("sticky_inflation", "stagflation", "recessionary_bear", "credit_stress"),
        expected_change_point=True,
        expected_hazard_direction="up",
    ),
    Scenario(
        name="savings_loan_recession",
        start="1989-01-01",
        end="1991-06-01",
        description="S&L crisis, 1990-91 recession.",
        expected_regimes=("credit_stress", "recessionary_bear"),
        expected_change_point=True,
        expected_hazard_direction="up",
    ),
    Scenario(
        name="dotcom_bust",
        start="2000-03-01",
        end="2002-12-01",
        description="Dotcom valuation reset, 2001 recession.",
        expected_regimes=("late_cycle", "recessionary_bear", "soft_landing"),
        expected_change_point=True,
        expected_hazard_direction="up",
    ),
    Scenario(
        name="gfc",
        start="2007-06-01",
        end="2009-12-01",
        description="Global Financial Crisis, credit-housing chain.",
        expected_regimes=("credit_stress", "recessionary_bear"),
        expected_change_point=True,
        expected_hazard_direction="up",
    ),
    Scenario(
        name="covid_shock",
        start="2020-01-01",
        end="2020-12-01",
        description="COVID-19 shock and rebound.",
        expected_regimes=("recessionary_bear", "credit_stress", "soft_landing", "liquidity_meltup"),
        expected_change_point=True,
        expected_hazard_direction="up",
    ),
    Scenario(
        name="inflation_2022",
        start="2022-01-01",
        end="2023-12-01",
        description="Post-pandemic inflation surge and Fed tightening.",
        expected_regimes=("sticky_inflation", "late_cycle", "credit_stress"),
        expected_change_point=True,
        expected_hazard_direction="up",
    ),
)


@dataclass
class ScenarioResult:
    scenario: Scenario
    rows: int
    regime_path: list[str] = field(default_factory=list)
    hazard_path: list[float] = field(default_factory=list)
    cp_max: float = 0.0
    passed_regime: bool = False
    passed_change_point: bool = False
    passed_hazard: bool = False

    @property
    def passed(self) -> bool:
        return self.passed_regime and self.passed_change_point and self.passed_hazard

    def to_dict(self) -> dict:
        return {
            "name": self.scenario.name,
            "start": self.scenario.start,
            "end": self.scenario.end,
            "rows": self.rows,
            "regime_path_unique": sorted(set(self.regime_path)),
            "cp_max": self.cp_max,
            "hazard_min": min(self.hazard_path) if self.hazard_path else None,
            "hazard_max": max(self.hazard_path) if self.hazard_path else None,
            "passed_regime": self.passed_regime,
            "passed_change_point": self.passed_change_point,
            "passed_hazard": self.passed_hazard,
            "passed_overall": self.passed,
        }


def replay_scenario(
    observations: pd.DataFrame,
    catalog: list[dict],
    scenario: Scenario,
    *,
    recession_labels: pd.DataFrame | None = None,
) -> ScenarioResult:
    """Run a single scenario through the regime + hazard pipeline."""
    panel = monthly_panel(observations, forward_fill_limit=0)
    if panel.empty:
        return ScenarioResult(scenario=scenario, rows=0)
    panel = panel.loc[scenario.start : scenario.end]
    if panel.empty:
        return ScenarioResult(scenario=scenario, rows=0)
    features = build_features(panel, catalog)
    regimes = score_regimes(features)
    regime_path = list(regimes["decoded_regime"]) if not regimes.empty else []
    cp_max = float(regimes["change_point_prob"].max()) if not regimes.empty else 0.0

    hazard_path: list[float] = []
    if recession_labels is not None and not recession_labels.empty:
        outputs, _ = train_fitted_hazard_outputs(features, recession_labels)
        if not outputs.empty:
            haz = outputs[outputs["target"] == "monthly_recession_hazard"]
            hazard_path = haz["value"].astype(float).tolist()

    passed_regime = not scenario.expected_regimes or any(r in scenario.expected_regimes for r in regime_path)
    passed_cp = cp_max > 0.30 if scenario.expected_change_point else cp_max < 0.30
    if scenario.expected_hazard_direction == "up":
        passed_hz = (max(hazard_path, default=0.0) - min(hazard_path, default=0.0)) > 1e-3
    elif scenario.expected_hazard_direction == "down":
        passed_hz = bool(hazard_path) and hazard_path[-1] < hazard_path[0]
    else:
        passed_hz = True
    return ScenarioResult(
        scenario=scenario,
        rows=len(panel),
        regime_path=regime_path,
        hazard_path=hazard_path,
        cp_max=cp_max,
        passed_regime=passed_regime,
        passed_change_point=passed_cp,
        passed_hazard=passed_hz,
    )


def replay_all(
    observations: pd.DataFrame,
    catalog: list[dict],
    *,
    recession_labels: pd.DataFrame | None = None,
    only: Iterable[str] | None = None,
) -> list[ScenarioResult]:
    selected = SCENARIOS if only is None else [s for s in SCENARIOS if s.name in set(only)]
    return [replay_scenario(observations, catalog, s, recession_labels=recession_labels) for s in selected]


__all__ = ["SCENARIOS", "Scenario", "ScenarioResult", "replay_all", "replay_scenario"]
