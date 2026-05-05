# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlencode

import pandas as pd
import requests

FRED_BASE = "https://api.stlouisfed.org/fred"
FRED_OBSERVATIONS_URL = f"{FRED_BASE}/series/observations"
FRED_VINTAGE_DATES_URL = f"{FRED_BASE}/series/vintagedates"


@dataclass(frozen=True)
class RealAlfredPlan:
    series_ids: list[str]
    observation_start: str = "1960-01-01"
    observation_end: str | None = None
    vintage_start: str = "1990-01-01"
    vintage_end: str | None = None
    max_vintages_per_series: int | None = None
    sleep_seconds: float = 0.0


def _api_key(api_key: str | None = None) -> str:
    key = api_key or os.getenv("FRED_API_KEY")
    if not key:
        raise ValueError("FRED_API_KEY is required for live ALFRED/FRED vintage ingestion.")
    return key


def _parse_float(value: object) -> float | None:
    if value in (None, "", "."):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def fetch_series_vintage_dates(
    series_id: str,
    *,
    api_key: str | None = None,
    realtime_start: str = "1776-07-04",
    realtime_end: str | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch true FRED/ALFRED vintage dates for a series.

    A vintage date is a date on which the series changed or received a new value.
    This is intentionally different from v0.7's synthetic monthly vintage grid.
    """
    key = _api_key(api_key)
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "realtime_start": realtime_start,
        "realtime_end": realtime_end or pd.Timestamp.now("UTC").strftime("%Y-%m-%d"),
    }
    response = requests.get(FRED_VINTAGE_DATES_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    dates = payload.get("vintage_dates", [])
    rows = [
        {
            "series_id": series_id,
            "vintage_date": d,
            "source": "fred_series_vintagedates",
            "ingested_at_utc": datetime.now(UTC).isoformat(),
            "metadata_json": json.dumps({"endpoint": FRED_VINTAGE_DATES_URL}, sort_keys=True),
        }
        for d in dates
    ]
    return pd.DataFrame(rows)


def build_real_alfred_plan(
    series_ids: Iterable[str],
    *,
    api_key: str | None = None,
    observation_start: str = "1960-01-01",
    observation_end: str | None = None,
    vintage_start: str = "1990-01-01",
    vintage_end: str | None = None,
    max_vintages_per_series: int | None = None,
    timeout: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (series_vintages, request_plan) from real ALFRED vintage dates."""
    vintage_frames: list[pd.DataFrame] = []
    plan_rows: list[dict] = []
    vend = vintage_end or pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    for sid in series_ids:
        vintages = fetch_series_vintage_dates(
            sid,
            api_key=api_key,
            realtime_start=vintage_start,
            realtime_end=vend,
            timeout=timeout,
        )
        if not vintages.empty:
            vintages["vintage_date"] = pd.to_datetime(vintages["vintage_date"])
            vintages = vintages[
                (vintages["vintage_date"] >= pd.Timestamp(vintage_start))
                & (vintages["vintage_date"] <= pd.Timestamp(vend))
            ]
            vintages = vintages.sort_values("vintage_date")
            if max_vintages_per_series:
                vintages = vintages.tail(int(max_vintages_per_series))
            vintage_frames.append(vintages.assign(vintage_date=lambda d: d["vintage_date"].dt.strftime("%Y-%m-%d")))
            for v in vintages["vintage_date"].dt.strftime("%Y-%m-%d"):
                url_params = {
                    "series_id": sid,
                    "observation_start": observation_start,
                    "realtime_start": v,
                    "realtime_end": v,
                }
                if observation_end:
                    url_params["observation_end"] = observation_end
                plan_rows.append(
                    {
                        "series_id": sid,
                        "vintage_date": v,
                        "observation_start": observation_start,
                        "observation_end": observation_end or "",
                        "realtime_start": v,
                        "realtime_end": v,
                        "endpoint": FRED_OBSERVATIONS_URL,
                        "request_url_without_key": FRED_OBSERVATIONS_URL + "?" + urlencode(url_params),
                    }
                )
    series_vintages = (
        pd.concat(vintage_frames, ignore_index=True)
        if vintage_frames
        else pd.DataFrame(columns=["series_id", "vintage_date", "source", "ingested_at_utc", "metadata_json"])
    )
    return series_vintages, pd.DataFrame(plan_rows)


def fetch_observations_for_vintage(
    series_id: str,
    vintage_date: str,
    *,
    api_key: str | None = None,
    observation_start: str = "1960-01-01",
    observation_end: str | None = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch observation values exactly as-of one vintage date."""
    key = _api_key(api_key)
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": observation_start,
        "realtime_start": vintage_date,
        "realtime_end": vintage_date,
    }
    if observation_end:
        params["observation_end"] = observation_end
    response = requests.get(FRED_OBSERVATIONS_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    rows = []
    now = datetime.now(UTC).isoformat()
    for item in payload.get("observations", []):
        value = _parse_float(item.get("value"))
        if value is None:
            continue
        rt_start = item.get("realtime_start") or vintage_date
        rt_end = item.get("realtime_end") or vintage_date
        rows.append(
            {
                "series_id": series_id,
                "observation_date": item.get("date"),
                "date": item.get("date"),
                "value": value,
                "realtime_start": rt_start,
                "realtime_end": rt_end,
                "vintage_date": rt_start,
                "source": "alfred_real_observation_vintage",
                "ingested_at_utc": now,
                "metadata_json": json.dumps(
                    {
                        "endpoint": FRED_OBSERVATIONS_URL,
                        "requested_vintage_date": vintage_date,
                        "api_realtime_start": rt_start,
                        "api_realtime_end": rt_end,
                    },
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def fetch_real_alfred_vintage_observations(
    series_ids: Iterable[str],
    *,
    api_key: str | None = None,
    observation_start: str = "1960-01-01",
    observation_end: str | None = None,
    vintage_start: str = "1990-01-01",
    vintage_end: str | None = None,
    max_vintages_per_series: int | None = None,
    sleep_seconds: float = 0.0,
    timeout: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch real vintage dates and vintage observations.

    Returns (series_vintages, vintage_observations, manifest).
    """
    vintages, plan = build_real_alfred_plan(
        series_ids,
        api_key=api_key,
        observation_start=observation_start,
        observation_end=observation_end,
        vintage_start=vintage_start,
        vintage_end=vintage_end,
        max_vintages_per_series=max_vintages_per_series,
        timeout=timeout,
    )
    observation_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict] = []
    started = datetime.now(UTC).isoformat()
    for _, req in plan.iterrows():
        status = "ok"
        error = ""
        rows = 0
        try:
            obs = fetch_observations_for_vintage(
                req["series_id"],
                req["vintage_date"],
                api_key=api_key,
                observation_start=req["observation_start"],
                observation_end=req["observation_end"] or None,
                timeout=timeout,
            )
            rows = len(obs)
            if not obs.empty:
                observation_frames.append(obs)
        except Exception as exc:  # pragma: no cover - live network branch
            status = "error"
            error = str(exc)
        manifest_rows.append(
            {
                "series_id": req["series_id"],
                "realtime_start": req["vintage_date"],
                "realtime_end": req["vintage_date"],
                "observation_start": req["observation_start"],
                "observation_end": req["observation_end"],
                "rows": rows,
                "status": status,
                "error": error,
                "ingested_at_utc": started,
                "metadata_json": json.dumps(
                    {"mode": "real_alfred_vintage", "endpoint": FRED_OBSERVATIONS_URL}, sort_keys=True
                ),
            }
        )
        if sleep_seconds > 0:
            time.sleep(float(sleep_seconds))
    observations = (
        pd.concat(observation_frames, ignore_index=True)
        if observation_frames
        else pd.DataFrame(
            columns=[
                "series_id",
                "observation_date",
                "date",
                "value",
                "realtime_start",
                "realtime_end",
                "vintage_date",
                "source",
                "ingested_at_utc",
                "metadata_json",
            ]
        )
    )
    return vintages, observations, pd.DataFrame(manifest_rows)


def seed_vintage_observations_from_latest(
    observations: pd.DataFrame, revision_bps: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic vintage tables from existing latest observations for local tests.

    This does not pretend to be official ALFRED data. It only allows the v0.8 point-in-time
    pipeline to be exercised without a live API key.
    """
    if observations is None or observations.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = observations.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "vintage_date" not in df:
        df["vintage_date"] = df["date"] + pd.offsets.MonthBegin(1)
    else:
        df["vintage_date"] = pd.to_datetime(df["vintage_date"], errors="coerce").fillna(
            df["date"] + pd.offsets.MonthBegin(1)
        )
    df = df.sort_values(["series_id", "date", "vintage_date"])
    now = datetime.now(UTC).isoformat()
    out = df.assign(
        observation_date=df["date"],
        realtime_start=df["vintage_date"],
        realtime_end=pd.Timestamp("9999-12-31"),
        source="seeded_vintage_from_observations",
        ingested_at_utc=now,
        metadata_json=json.dumps({"warning": "seeded from latest observations; not official ALFRED"}, sort_keys=True),
    )
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
    vintages = out[["series_id", "vintage_date", "source", "ingested_at_utc", "metadata_json"]].drop_duplicates()
    vintages["vintage_date"] = pd.to_datetime(vintages["vintage_date"]).dt.strftime("%Y-%m-%d")
    vo = out[cols].copy()
    for c in ["observation_date", "realtime_start", "realtime_end", "vintage_date"]:
        vo[c] = pd.to_datetime(vo[c], errors="coerce").dt.strftime("%Y-%m-%d")
    return vintages, vo
