# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_panel(
    observations: pd.DataFrame,
    *,
    forward_fill_limit: int = 0,
    asof: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Resample observations onto a month-start grid.

    Parameters
    ----------
    observations:
        Long-format observation frame with ``series_id``, ``date``, ``value``,
        and optional ``vintage_date``.
    forward_fill_limit:
        How many consecutive missing months to forward-fill. Defaults to 0 in
        v1.0+ to remove the v0.8 silent leakage hazard. Values >0 are
        explicit opt-in for legacy behavior. The point-in-time path
        (``feature_asof_values``) never relies on this fill.
    asof:
        Optional cutoff: rows with ``date > asof`` or ``vintage_date > asof``
        are dropped before resampling. Use this for any panel that feeds a
        forecast trained at ``asof``; the engine routes through
        ``feature_asof_values`` for production-grade PIT, this is the
        secondary safety net.
    """
    if observations.empty:
        return pd.DataFrame()
    df = observations.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "vintage_date" in df:
        df["vintage_date"] = pd.to_datetime(df["vintage_date"], errors="coerce")
    if asof is not None:
        cutoff = pd.Timestamp(asof)
        df = df[df["date"] <= cutoff]
        if "vintage_date" in df:
            df = df[df["vintage_date"].isna() | (df["vintage_date"] <= cutoff)]
    df = df.sort_values(["series_id", "date"] + (["vintage_date"] if "vintage_date" in df else []))
    df = df.drop_duplicates(["series_id", "date"], keep="last")
    panel = df.pivot(index="date", columns="series_id", values="value").sort_index()
    panel = panel.resample("MS").last()
    if forward_fill_limit and forward_fill_limit > 0:
        panel = panel.ffill(limit=int(forward_fill_limit))
    return panel


def log_growth(s: pd.Series, periods: int) -> pd.Series:
    positive = s.where(s > 0)
    return np.log(positive).diff(periods)


def rolling_z(s: pd.Series, window: int = 60, min_periods: int = 24) -> pd.Series:
    mu = s.rolling(window, min_periods=min_periods).mean().shift(1)
    sd = s.rolling(window, min_periods=min_periods).std(ddof=1).shift(1)
    return (s - mu) / sd.replace(0, np.nan)


DEFAULT_TRANSFORMS: tuple[str, ...] = (
    "level",
    "diff_3m",
    "diff_12m",
    "pct_1m",
    "pct_12m",
    "log_3m",
    "log_yoy",
    "z_60m",
)


def _build_transform_map(s: pd.Series) -> dict[str, pd.Series]:
    return {
        "level": s,
        "diff_3m": s.diff(3),
        "diff_12m": s.diff(12),
        "pct_1m": s.pct_change(1),
        "pct_12m": s.pct_change(12),
        "pct_change": s.pct_change(1),
        "log_3m": log_growth(s, 3),
        "log_yoy": log_growth(s, 12),
        "z_60m": rolling_z(s, 60),
    }


def _resolve_transforms(item: dict) -> tuple[str, ...]:
    """Resolve the transform list for a catalog entry.

    Precedence:
    1. Explicit `transforms: [...]` list in the catalog (multi-value).
    2. Single `transform:` string in the catalog (kept narrow to honor the spec).
    3. Default to all eight legacy transforms for backward compatibility when no
       transform field is present.

    `transform: pct_change` from the existing catalog is mapped to `pct_change`
    (alias for `pct_1m`) so existing yaml files are honored exactly as written.
    """
    listed = item.get("transforms")
    if listed:
        return tuple(str(x).strip() for x in listed if str(x).strip())
    single = item.get("transform")
    if single:
        return (str(single).strip(),)
    return DEFAULT_TRANSFORMS


def build_features(panel: pd.DataFrame, catalog: list[dict], *, honor_catalog_transforms: bool = True) -> pd.DataFrame:
    """Build a long-format feature frame from a wide monthly panel.

    Parameters
    ----------
    honor_catalog_transforms:
        When True (default in v1.0+), respect the `transform` / `transforms` field
        in the catalog entry. When False, fall back to the legacy behavior of
        applying every default transform to every series. The flag exists to
        preserve any callers that depended on the old superset behavior.
    """
    frames = []
    for item in catalog:
        sid = item["series_id"]
        domain = item.get("domain")
        if sid not in panel:
            continue
        s = panel[sid].astype(float)
        transform_map = _build_transform_map(s)
        names = _resolve_transforms(item) if honor_catalog_transforms else DEFAULT_TRANSFORMS
        for name in names:
            if name not in transform_map:
                continue
            vals = transform_map[name]
            f = vals.rename("value").reset_index()
            f["feature_name"] = f"{sid}.{name}"
            f["domain"] = domain
            frames.append(f[["feature_name", "date", "value", "domain"]])
    if not frames:
        return pd.DataFrame(columns=["feature_name", "date", "value", "domain"])
    return pd.concat(frames, ignore_index=True).dropna(subset=["value"])


def feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    f = features.copy()
    f["date"] = pd.to_datetime(f["date"])
    return f.pivot(index="date", columns="feature_name", values="value").sort_index()
