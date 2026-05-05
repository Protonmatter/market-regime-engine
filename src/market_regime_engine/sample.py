# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def generate_sample_observations(start: str = "1990-01-01", end: str = "2026-03-01", seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="MS")
    n = len(dates)
    t = np.arange(n)

    stress = np.zeros(n)
    stress[(dates >= "2000-01-01") & (dates <= "2002-12-01")] += 0.5
    stress[(dates >= "2007-07-01") & (dates <= "2009-06-01")] += 1.3
    stress[(dates >= "2020-03-01") & (dates <= "2020-12-01")] += 1.1
    stress[(dates >= "2022-01-01") & (dates <= "2023-06-01")] += 0.8

    inflation = 2.0 + 1.5 * stress + 0.3 * np.sin(t / 18) + rng.normal(0, 0.12, n)
    fed = np.clip(0.6 + 0.8 * inflation + rng.normal(0, 0.25, n), 0.05, None)
    dgs10 = np.clip(2.1 + 0.45 * inflation + rng.normal(0, 0.25, n), 0.2, None)
    term = dgs10 - fed + rng.normal(0, 0.15, n)
    unrate = np.clip(4.1 + 2.0 * stress + rng.normal(0, 0.18, n), 3.0, 14.0)
    u6 = unrate + 3.0 + 1.1 * stress + rng.normal(0, 0.15, n)
    payems = 110000 + np.cumsum(120 + rng.normal(0, 45, n) - 280 * stress)

    cpi = 100 * np.exp(np.cumsum((inflation / 100) / 12))
    core_cpi = 100 * np.exp(np.cumsum(((inflation - 0.15 * stress) / 100) / 12))
    credit = np.clip(1.7 + 1.4 * stress + rng.normal(0, 0.16, n), 0.5, None)
    permits = 1450 * np.exp(np.cumsum(0.001 + rng.normal(0, 0.016, n) - 0.025 * stress))
    starts = 1420 * np.exp(np.cumsum(0.001 + rng.normal(0, 0.018, n) - 0.024 * stress))
    mortgage = np.clip(dgs10 + 1.7 + 0.25 * stress + rng.normal(0, 0.18, n), 2.0, None)
    oil = 35 * np.exp(np.cumsum(0.002 + rng.normal(0, 0.052, n) + 0.018 * stress))
    dollar = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n) + 0.002 * stress))
    debt = 45 + 0.18 * t + 15 * (dates >= "2008-01-01") + 20 * (dates >= "2020-01-01")

    market_ret = 0.006 + rng.normal(0, 0.035, n) - 0.035 * stress - 0.004 * np.maximum(fed - 4.0, 0)
    spx = 350 * np.exp(np.cumsum(market_ret))

    values = {
        "SPX": spx,
        "FEDFUNDS": fed,
        "DGS10": dgs10,
        "T10Y3M": term,
        "UNRATE": unrate,
        "U6RATE": u6,
        "PAYEMS": payems,
        "CPIAUCSL": cpi,
        "CPILFESL": core_cpi,
        "BAA10Y": credit,
        "PERMIT": permits,
        "HOUST": starts,
        "MORTGAGE30US": mortgage,
        "DCOILWTICO": oil,
        "DTWEXBGS": dollar,
        "GFDEGDQ188S": debt,
    }

    rows = []
    for series_id, arr in values.items():
        for date, value in zip(dates, arr, strict=False):
            rows.append(
                {
                    "series_id": series_id,
                    "date": date,
                    "value": float(value),
                    "vintage_date": date,
                    "source": "sample",
                    "metadata_json": json.dumps({"synthetic": True}),
                }
            )
    return pd.DataFrame(rows)
