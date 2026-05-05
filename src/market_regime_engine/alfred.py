# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
import requests

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"


@dataclass(frozen=True)
class AlfredRequest:
    series_id: str
    observation_start: str = "1960-01-01"
    observation_end: str | None = None
    realtime_start: str = "1990-01-01"
    realtime_end: str | None = None
    vintage_frequency: str = "MS"


def vintage_grid(start: str, end: str | None = None, frequency: str = "MS") -> list[str]:
    """Build a stable vintage-date grid for ALFRED/FRED real-time requests."""
    end = end or pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    dates = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=frequency)
    if not len(dates) or dates[-1] < pd.Timestamp(end):
        dates = dates.append(pd.DatetimeIndex([pd.Timestamp(end)]))
    return [d.strftime("%Y-%m-%d") for d in dates]


def build_alfred_request_matrix(
    series_ids: Iterable[str],
    *,
    observation_start: str = "1960-01-01",
    observation_end: str | None = None,
    vintage_start: str = "1990-01-01",
    vintage_end: str | None = None,
    vintage_frequency: str = "MS",
) -> pd.DataFrame:
    rows = []
    vintages = vintage_grid(vintage_start, vintage_end, vintage_frequency)
    for sid in series_ids:
        for v in vintages:
            rows.append(
                {
                    "series_id": sid,
                    "observation_start": observation_start,
                    "observation_end": observation_end or "",
                    "realtime_start": v,
                    "realtime_end": v,
                    "endpoint": FRED_OBSERVATIONS_URL,
                }
            )
    return pd.DataFrame(rows)


def _parse_value(value: str) -> float | None:
    if value in {".", "", None}:  # FRED missing value convention
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def observations_to_vintage_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Convert legacy ALFRED `observations` rows into `vintage_observations` shape.

    The v0.7 ingestion path wrote per-vintage observations into the `observations` table.
    The v0.8+ canonical path is `vintage_observations`, which carries explicit
    `realtime_start`/`realtime_end` and `observation_date` columns so the as-of
    materializer can prove lineage. This helper converts the legacy shape to the
    canonical shape so a single ALFRED unification can be done at the call site.
    """
    if observations is None or observations.empty:
        return pd.DataFrame(
            columns=[
                "series_id",
                "observation_date",
                "value",
                "realtime_start",
                "realtime_end",
                "vintage_date",
                "source",
                "ingested_at_utc",
                "metadata_json",
            ]
        )
    df = observations.copy()
    if "observation_date" not in df:
        df["observation_date"] = df["date"]
    if "realtime_start" not in df:
        df["realtime_start"] = df["vintage_date"]
    if "realtime_end" not in df:
        df["realtime_end"] = pd.Timestamp("9999-12-31").strftime("%Y-%m-%d")
    if "ingested_at_utc" not in df:
        df["ingested_at_utc"] = datetime.now(UTC).isoformat()
    if "source" not in df:
        df["source"] = "alfred_fred_realtime"
    if "metadata_json" not in df:
        df["metadata_json"] = "{}"
    cols = [
        "series_id",
        "observation_date",
        "value",
        "realtime_start",
        "realtime_end",
        "vintage_date",
        "source",
        "ingested_at_utc",
        "metadata_json",
    ]
    for c in cols:
        if c not in df:
            df[c] = None
    return df[cols].copy()


def fetch_alfred_vintages(
    series_ids: Iterable[str],
    *,
    api_key: str | None = None,
    observation_start: str = "1960-01-01",
    observation_end: str | None = None,
    vintage_start: str = "1990-01-01",
    vintage_end: str | None = None,
    vintage_frequency: str = "MS",
    timeout: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch point-in-time observations through the FRED/ALFRED observations API.

    The API uses `realtime_start`/`realtime_end` to request observations as of a vintage date.
    This function returns `(observations, manifest)` and intentionally keeps one row per
    `(series_id, observation date, vintage date)` so downstream point-in-time filtering can
    enforce what was actually knowable at forecast time.

    The returned `observations` frame carries the legacy shape used by the
    `observations` table; convert it to the canonical `vintage_observations`
    shape via `observations_to_vintage_observations` before persisting.
    """
    api_key = api_key or os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED_API_KEY is required for live ALFRED/FRED vintage ingestion. Use build_alfred_request_matrix for dry-run planning."
        )

    matrix = build_alfred_request_matrix(
        series_ids,
        observation_start=observation_start,
        observation_end=observation_end,
        vintage_start=vintage_start,
        vintage_end=vintage_end,
        vintage_frequency=vintage_frequency,
    )
    obs_rows: list[dict] = []
    manifest_rows: list[dict] = []
    session = requests.Session()
    started = datetime.now(UTC).isoformat()

    for _, req in matrix.iterrows():
        params = {
            "series_id": req["series_id"],
            "api_key": api_key,
            "file_type": "json",
            "observation_start": req["observation_start"],
            "realtime_start": req["realtime_start"],
            "realtime_end": req["realtime_end"],
        }
        if req["observation_end"]:
            params["observation_end"] = req["observation_end"]
        status = "ok"
        error = ""
        count = 0
        try:
            response = session.get(FRED_OBSERVATIONS_URL, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            observations = payload.get("observations", [])
            for item in observations:
                val = _parse_value(item.get("value"))
                if val is None:
                    continue
                vintage = item.get("realtime_start") or req["realtime_start"]
                obs_rows.append(
                    {
                        "series_id": req["series_id"],
                        "date": item.get("date"),
                        "value": val,
                        "vintage_date": vintage,
                        "source": "alfred_fred_realtime",
                        "metadata_json": json.dumps(
                            {
                                "realtime_start": item.get("realtime_start"),
                                "realtime_end": item.get("realtime_end"),
                                "ingested_at_utc": started,
                            },
                            sort_keys=True,
                        ),
                    }
                )
                count += 1
        except Exception as exc:  # pragma: no cover - network path
            status = "error"
            error = str(exc)
        manifest_rows.append(
            {
                "series_id": req["series_id"],
                "realtime_start": req["realtime_start"],
                "realtime_end": req["realtime_end"],
                "observation_start": req["observation_start"],
                "observation_end": req["observation_end"],
                "rows": count,
                "status": status,
                "error": error,
                "ingested_at_utc": started,
                "metadata_json": json.dumps({"endpoint": FRED_OBSERVATIONS_URL}, sort_keys=True),
            }
        )

    return pd.DataFrame(obs_rows), pd.DataFrame(manifest_rows)
