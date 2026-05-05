# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os

import pandas as pd
import requests


def fetch_fred_recession_indicator(
    series_id: str = "USREC", api_key: str | None = None, observation_start: str = "1960-01-01"
) -> pd.DataFrame:
    """Fetch a FRED recession indicator, usually USREC.

    Requires FRED_API_KEY unless the local/network FRED endpoint allows unauthenticated access.
    """
    api_key = api_key or os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is required for live FRED recession ingestion")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    rows = []
    for item in obs:
        val = item.get("value")
        if val in (None, "."):
            continue
        rows.append(
            {
                "date": item["date"],
                "recession": float(val),
                "source": f"fred:{series_id}",
                "metadata_json": json.dumps(
                    {"realtime_start": item.get("realtime_start"), "realtime_end": item.get("realtime_end")},
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(rows)
