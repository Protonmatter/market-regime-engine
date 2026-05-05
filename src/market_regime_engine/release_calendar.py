# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


@dataclass(frozen=True)
class ReleaseCalendarRule:
    series_id: str
    lag_days: int = 0
    lag_months: int = 0
    release_family: str = "unknown"
    domain: str = "unknown"

    def release_date(self, observation_date: str | pd.Timestamp) -> pd.Timestamp:
        date = pd.Timestamp(observation_date)
        return date + pd.DateOffset(months=int(self.lag_months), days=int(self.lag_days))


def load_release_calendar(path: str | Path = "config/release_calendar.yaml") -> dict[str, ReleaseCalendarRule]:
    path = Path(path)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, ReleaseCalendarRule] = {}
    for sid, item in (raw.get("series") or {}).items():
        item = item or {}
        out[str(sid)] = ReleaseCalendarRule(
            series_id=str(sid),
            lag_days=int(item.get("lag_days", 0) or 0),
            lag_months=int(item.get("lag_months", 0) or 0),
            release_family=str(item.get("release_family", "unknown")),
            domain=str(item.get("domain", "unknown")),
        )
    return out


def enforce_release_calendar(
    observations: pd.DataFrame, rules: dict[str, ReleaseCalendarRule] | None = None
) -> pd.DataFrame:
    """Return observations with vintage_date moved forward to the configured release date minimum."""
    if observations.empty:
        return observations.copy()
    rules = rules or load_release_calendar()
    frame = observations.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if "vintage_date" not in frame:
        frame["vintage_date"] = pd.NaT
    frame["vintage_date"] = pd.to_datetime(frame["vintage_date"], errors="coerce")

    def fix(row: pd.Series) -> pd.Timestamp:
        sid = str(row["series_id"])
        rule = rules.get(sid, ReleaseCalendarRule(sid))
        min_release = rule.release_date(row["date"])
        vintage = row["vintage_date"] if pd.notna(row["vintage_date"]) else min_release
        return max(pd.Timestamp(vintage), pd.Timestamp(min_release))

    frame["vintage_date"] = frame.apply(fix, axis=1)
    if "metadata_json" not in frame:
        frame["metadata_json"] = "{}"
    else:
        frame["metadata_json"] = frame["metadata_json"].fillna("{}")

    def add_meta(row: pd.Series) -> str:
        sid = str(row["series_id"])
        rule = rules.get(sid, ReleaseCalendarRule(sid))
        try:
            meta = json.loads(row.get("metadata_json", "{}") or "{}")
        except Exception:
            meta = {}
        meta["release_calendar"] = {
            "release_family": rule.release_family,
            "lag_days": rule.lag_days,
            "lag_months": rule.lag_months,
            "domain": rule.domain,
        }
        return json.dumps(meta, sort_keys=True)

    frame["metadata_json"] = frame.apply(add_meta, axis=1)
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    frame["vintage_date"] = frame["vintage_date"].dt.strftime("%Y-%m-%d")
    return frame


def audit_release_calendar(
    observations: pd.DataFrame, rules: dict[str, ReleaseCalendarRule] | None = None
) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame(
            columns=["series_id", "rows", "violations", "coverage", "min_required_release", "max_actual_vintage"]
        )
    rules = rules or load_release_calendar()
    frame = observations.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["vintage_date"] = pd.to_datetime(frame.get("vintage_date", frame["date"]), errors="coerce")
    rows = []
    for sid, group in frame.groupby("series_id"):
        rule = rules.get(str(sid), ReleaseCalendarRule(str(sid)))
        required = group["date"].apply(rule.release_date)
        violations = int((group["vintage_date"] < required).sum())
        rows.append(
            {
                "series_id": str(sid),
                "rows": len(group),
                "violations": violations,
                "coverage": bool(str(sid) in rules),
                "release_family": rule.release_family,
                "domain": rule.domain,
                "min_required_release": required.min().strftime("%Y-%m-%d") if len(required) else None,
                "max_actual_vintage": group["vintage_date"].max().strftime("%Y-%m-%d")
                if group["vintage_date"].notna().any()
                else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["violations", "series_id"], ascending=[False, True])
