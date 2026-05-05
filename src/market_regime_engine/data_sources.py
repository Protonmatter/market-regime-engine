# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
import requests


@dataclass
class FredClient:
    api_key: str
    base_url: str = "https://api.stlouisfed.org/fred/series/observations"

    def fetch_series(self, series_id: str, start: str = "1960-01-01") -> pd.DataFrame:
        return self.fetch_series_as_of(series_id, start=start)

    def fetch_series_as_of(
        self,
        series_id: str,
        *,
        start: str = "1960-01-01",
        end: str | None = None,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch FRED observations with realtime/vintage metadata.

        If realtime_start/end are supplied, FRED returns observations for that vintage window.
        This is the ingestion hook used for point-in-time backtests. For true ALFRED-style
        reconstruction, schedule multiple vintages and store each realtime_start as vintage_date.
        """
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start,
        }
        if end:
            params["observation_end"] = end
        if realtime_start:
            params["realtime_start"] = realtime_start
        if realtime_end:
            params["realtime_end"] = realtime_end
        r = requests.get(self.base_url, params=params, timeout=30)
        r.raise_for_status()
        rows = []
        for obs in r.json().get("observations", []):
            if obs.get("value") in (None, ".", ""):
                continue
            rows.append(
                {
                    "series_id": series_id,
                    "date": obs["date"],
                    "value": float(obs["value"]),
                    "vintage_date": obs.get("realtime_start") or realtime_start or obs["date"],
                    "source": "fred",
                    "metadata_json": "{}",
                }
            )
        return pd.DataFrame(rows)

    def fetch_vintage_grid(
        self,
        series_id: str,
        *,
        start: str = "1960-01-01",
        vintage_dates: Iterable[str],
    ) -> pd.DataFrame:
        """Fetch repeated point-in-time vintages for one series.

        This is intentionally simple and explicit. It avoids pretending that one revised endpoint
        is enough for institutional backtesting. Humans did invent revisions, because apparently
        one version of truth felt too easy.
        """
        frames = []
        for vintage in vintage_dates:
            frames.append(self.fetch_series_as_of(series_id, start=start, realtime_start=vintage, realtime_end=vintage))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@dataclass
class BlsClient:
    api_key: str | None = None

    def fetch_series(self, series_ids: Iterable[str], start_year: int, end_year: int) -> pd.DataFrame:
        payload = {"seriesid": list(series_ids), "startyear": str(start_year), "endyear": str(end_year)}
        if self.api_key:
            payload["registrationkey"] = self.api_key
        r = requests.post("https://api.bls.gov/publicAPI/v2/timeseries/data/", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError(data)
        rows = []
        for series in data.get("Results", {}).get("series", []):
            sid = series["seriesID"]
            for item in series.get("data", []):
                period = item["period"]
                if not period.startswith("M"):
                    continue
                month = int(period[1:])
                # BLS API does not provide full historical vintages here. Approximate release
                # lag as following month for PIT-safe scaffolding; production should use exact
                # release calendar metadata.
                date = pd.Timestamp(year=int(item["year"]), month=month, day=1)
                vintage = date + pd.DateOffset(months=1)
                rows.append(
                    {
                        "series_id": sid,
                        "date": date.strftime("%Y-%m-%d"),
                        "value": float(item["value"]),
                        "vintage_date": vintage.strftime("%Y-%m-%d"),
                        "source": "bls",
                        "metadata_json": "{}",
                    }
                )
        return pd.DataFrame(rows)
