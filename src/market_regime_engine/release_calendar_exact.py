# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DEFAULT_LAGS = {
    "labor": 5,
    "inflation": 14,
    "rates": 1,
    "credit": 1,
    "housing": 25,
    "energy": 1,
    "fx": 1,
    "fiscal": 20,
    "market": 1,
}


@dataclass(frozen=True)
class ExactReleaseRule:
    series_id: str
    domain: str
    lag_days: int
    release_hour_utc: int = 13


def _load_yaml_calendar_indexed() -> dict[tuple[str, str], dict[str, str]]:
    """Index every YAML calendar entry by (series_id, observation_date).

    Falls back silently to an empty index when the optional
    ``frontier.release_calendars`` module isn't available (the engine
    is still importable without it). The index is rebuilt on every
    ``build_exact_release_calendar`` call so updates to the YAML
    cache are picked up immediately.
    """
    try:
        from market_regime_engine.frontier.release_calendars import load_yaml_calendar
    except Exception:
        return {}
    out: dict[tuple[str, str], dict[str, str]] = {}
    for agency in ("bls", "bea", "census", "fed"):
        for entry in load_yaml_calendar(agency):
            key = (str(entry.get("series_id", "")), str(entry.get("observation_date", "")))
            if not all(key):
                continue
            out[key] = {
                "release_timestamp_utc": str(entry.get("release_timestamp_utc", "")),
                "agency": agency,
            }
    return out


def build_exact_release_calendar(observations: pd.DataFrame, catalog: list[dict] | None = None) -> pd.DataFrame:
    """Create deterministic exact-release timestamps for point-in-time enforcement.

    v1.4 update: when a series is covered by the v1.4 hand-curated YAML
    cache under ``config/release_calendars/{bls,bea,census,fed}.yaml``,
    the calendar entry's ``release_timestamp_utc`` wins over the v1.3
    ``DEFAULT_LAGS`` rule. The ``source`` column records provenance:

    - ``bls_real`` / ``bea_real`` / ``census_real`` / ``fed_real`` — the
      release timestamp came from a (cached) live calendar.
    - ``v0.6_conservative_rule`` — fell through to ``DEFAULT_LAGS``.
    """
    if observations is None or observations.empty:
        return pd.DataFrame(
            columns=[
                "series_id",
                "observation_date",
                "release_timestamp_utc",
                "domain",
                "lag_days",
                "source",
                "metadata_json",
            ]
        )
    domain_by_series = {c.get("series_id"): c.get("domain", "unknown") for c in (catalog or [])}
    yaml_index = _load_yaml_calendar_indexed()
    obs = observations.copy()
    obs["date"] = pd.to_datetime(obs["date"])
    rows = []
    for (series_id, date), _grp in obs.groupby(["series_id", "date"]):
        domain = str(domain_by_series.get(series_id, "unknown"))
        date_iso = pd.Timestamp(date).strftime("%Y-%m-%d")
        yaml_hit = yaml_index.get((str(series_id), date_iso))
        if yaml_hit and yaml_hit.get("release_timestamp_utc"):
            release_iso = str(yaml_hit["release_timestamp_utc"])
            try:
                release_ts = pd.Timestamp(release_iso, tz="UTC").tz_convert(None)
            except Exception:
                release_ts = pd.Timestamp(date) + pd.Timedelta(days=15)
            lag_days = int((release_ts.normalize() - pd.Timestamp(date).normalize()).days)
            source = f"{yaml_hit.get('agency', 'unknown')}_real"
        else:
            lag = DEFAULT_LAGS.get(domain, 15)
            release_ts = pd.Timestamp(date) + pd.Timedelta(days=lag) + pd.Timedelta(hours=13)
            release_iso = release_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
            lag_days = int(lag)
            source = "v0.6_conservative_rule"
        rows.append(
            {
                "series_id": series_id,
                "observation_date": date_iso,
                "release_timestamp_utc": release_iso,
                "domain": domain,
                "lag_days": lag_days,
                "source": source,
                "metadata_json": "{}",
            }
        )
    return pd.DataFrame(rows).sort_values(["series_id", "observation_date"])


def enforce_exact_release_calendar(observations: pd.DataFrame, release_calendar: pd.DataFrame) -> pd.DataFrame:
    if observations is None or observations.empty or release_calendar is None or release_calendar.empty:
        return observations.copy() if observations is not None else pd.DataFrame()
    obs = observations.copy()
    cal = release_calendar.copy()
    obs["date"] = pd.to_datetime(obs["date"]).dt.strftime("%Y-%m-%d")
    if "vintage_date" not in obs:
        obs["vintage_date"] = obs["date"]
    cal["observation_date"] = pd.to_datetime(cal["observation_date"]).dt.strftime("%Y-%m-%d")
    merged = obs.merge(
        cal[["series_id", "observation_date", "release_timestamp_utc"]],
        left_on=["series_id", "date"],
        right_on=["series_id", "observation_date"],
        how="left",
    )
    fallback = pd.to_datetime(merged["date"]) + pd.to_timedelta(15, unit="D")
    release_date = (
        pd.to_datetime(merged["release_timestamp_utc"], errors="coerce").dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    )
    merged["vintage_date"] = release_date.fillna(fallback.dt.strftime("%Y-%m-%d"))
    return merged[obs.columns]


def audit_exact_release_calendar(observations: pd.DataFrame, release_calendar: pd.DataFrame) -> pd.DataFrame:
    if observations is None or observations.empty:
        return pd.DataFrame()
    obs = observations.copy()
    obs["date"] = pd.to_datetime(obs["date"]).dt.strftime("%Y-%m-%d")
    obs["vintage_date"] = pd.to_datetime(obs.get("vintage_date", obs["date"])).dt.strftime("%Y-%m-%d")
    if release_calendar is None or release_calendar.empty:
        return pd.DataFrame(
            [{"series_id": s, "rows": len(g), "violations": None, "coverage": 0} for s, g in obs.groupby("series_id")]
        )
    cal = release_calendar.copy()
    cal["observation_date"] = pd.to_datetime(cal["observation_date"]).dt.strftime("%Y-%m-%d")
    cal["release_date"] = pd.to_datetime(cal["release_timestamp_utc"], errors="coerce").dt.strftime("%Y-%m-%d")
    merged = obs.merge(
        cal[["series_id", "observation_date", "release_date"]],
        left_on=["series_id", "date"],
        right_on=["series_id", "observation_date"],
        how="left",
    )
    merged["covered"] = merged["release_date"].notna()
    merged["violation"] = merged["covered"] & (
        pd.to_datetime(merged["vintage_date"]) < pd.to_datetime(merged["release_date"])
    )
    return (
        merged.groupby("series_id")
        .agg(rows=("date", "size"), coverage=("covered", "sum"), violations=("violation", "sum"))
        .reset_index()
    )
