# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date

import pandas as pd

from market_regime_engine.data_sources import FredClient


def monthly_vintage_grid(start: str, end: str, freq: str = "MS") -> list[str]:
    dates = pd.date_range(start, end, freq=freq)
    return [d.strftime("%Y-%m-%d") for d in dates]


@dataclass
class FredVintageIngestionPlan:
    series_ids: list[str]
    observation_start: str = "1960-01-01"
    vintage_start: str = "1990-01-01"
    vintage_end: str | None = None
    vintage_frequency: str = "MS"

    def vintages(self) -> list[str]:
        end = self.vintage_end or date.today().strftime("%Y-%m-%d")
        return monthly_vintage_grid(self.vintage_start, end, self.vintage_frequency)


def fetch_fred_vintage_plan(plan: FredVintageIngestionPlan, api_key: str | None = None) -> pd.DataFrame:
    key = api_key or os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is required for vintage ingestion")
    client = FredClient(key)
    frames = []
    for sid in plan.series_ids:
        frames.append(client.fetch_vintage_grid(sid, start=plan.observation_start, vintage_dates=plan.vintages()))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
