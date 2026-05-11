# SPDX-License-Identifier: Apache-2.0
"""Point-in-time materialisation of feature snapshots.

The pre-v1.2.1 implementation rebuilt the full vintage panel and re-applied
every feature transform once per as-of date. On the synthetic sample
(435 monthly observations × 16 series × 8 transforms × 399 as-of dates)
that loop took ~2 minutes wall-clock and timed out the reviewer's smoke
pipeline. v1.2.1 rewrites the inner loop as a set-based pipeline:

1. Resolve the as-of grid once.
2. Detect whether any (series_id, observation_date) pair has multiple
   vintages.
3. **No-revisions fast path.** When the data is "single-vintage" (the
   synthetic sample, ALFRED snapshots before any revision lands, etc.)
   the legal vintage at any as-of date is fully determined by
   ``vintage_date <= as_of_date AND observation_date <= as_of_date``.
   We build the wide monthly panel exactly once, compute every transform
   exactly once, attach lineage via a single vectorised merge, then for
   each as-of date we slice the precomputed feature frame and pick the
   latest row per ``feature_name``. Cost goes from O(N_asof × N_series ×
   N_transforms) Python iterations to O(N_asof) vectorised slices.
4. **Revisions path.** When the same (series, observation_date) appears
   under multiple vintages, the legal table varies per as-of. We still
   eliminate the inner ``iterrows`` over feature names by building the
   per-as-of feature frame in one pass and joining lineage with a single
   merge.

The lineage / output schema is byte-identical to the pre-v1.2.1 frame
(modulo float rounding within ``1e-9``); see
``tests/test_asof_perf.py`` for the correctness regression.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum

import pandas as pd

from market_regime_engine.features import build_features, monthly_panel


class Cadence(str, Enum):
    """Round-tripable cadence labels used by :func:`_resolve_asof_grid`.

    The v1.4.1 implementation hard-coded ``freq="MS"`` (month-start),
    which is correct for the existing macro panels but blocks every
    Fixed-Income feature builder that operates intraday or on a daily
    business-day grid (ASK-2 / P0 in
    :file:`market-regime-engine-review/REVIEW.md`).

    Values map to the pandas frequency strings used by
    :func:`pandas.date_range`. ``MIN_15`` / ``MIN_1`` use the lowercase
    ``"min"`` alias that pandas 2.x recommends over the legacy ``T``.

    PR-1 lands the enum + a default-preserving ``freq=`` kwarg on
    :func:`_resolve_asof_grid`; PR-3 / PR-4 wire FI callers through
    explicit ``Cadence.DAILY`` / ``Cadence.MIN_15`` selections.
    """

    MONTHLY = "MS"
    DAILY = "D"
    HOURLY = "h"
    MIN_15 = "15min"
    MIN_1 = "1min"

    @classmethod
    def from_pandas_freq(cls, freq: str) -> Cadence:
        """Coerce a pandas frequency alias into a :class:`Cadence` member.

        Accepts the canonical aliases (``"MS"``, ``"D"``, ``"h"``,
        ``"15min"``, ``"1min"``) and a small set of legacy synonyms
        (``"H"`` → ``HOURLY``, ``"T"``/``"min"`` → ``MIN_1``,
        ``"15T"`` → ``MIN_15``). Anything else raises ``ValueError`` so
        callers do not silently fall through to monthly.
        """
        if freq is None:
            raise ValueError("freq is required (got None)")
        normalised = str(freq).strip()
        # Direct hit first; pandas accepts both ``"h"`` and ``"H"`` so we
        # tolerate either case for the single-letter aliases.
        for member in cls:
            if member.value == normalised:
                return member
        legacy_map = {
            "M": cls.MONTHLY,
            "BMS": cls.MONTHLY,
            "B": cls.DAILY,
            "H": cls.HOURLY,
            "T": cls.MIN_1,
            "min": cls.MIN_1,
            "15T": cls.MIN_15,
        }
        if normalised in legacy_map:
            return legacy_map[normalised]
        raise ValueError(f"Unknown cadence freq: {freq!r}")


# v1.4.1 monthly-equivalent shift mapping. The legacy
# ``_resolve_asof_grid(df, min_history_months)`` shifted by
# ``pd.DateOffset(months=min_history_months)`` regardless of cadence.
# For sub-daily / daily callers, we still anchor on a monthly-equivalent
# horizon so the "min_history" parameter keeps the same semantic
# (years-of-data warmup) — only the grid spacing changes.
_DEFAULT_HISTORY_OFFSET_MONTHS = 1

_OUTPUT_COLUMNS = [
    "as_of_date",
    "feature_name",
    "source_series_id",
    "observation_date",
    "vintage_date",
    "value",
    "transform_name",
    "created_at_utc",
    "metadata_json",
]


def _empty_feature_asof_values() -> pd.DataFrame:
    return pd.DataFrame(columns=_OUTPUT_COLUMNS)


def _date_str(x: object) -> str:
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def latest_vintage_observations_asof(vintage_observations: pd.DataFrame, asof_date: str | pd.Timestamp) -> pd.DataFrame:
    """Return one legal vintage value per (series, observation_date) as of forecast date."""
    if vintage_observations is None or vintage_observations.empty:
        return pd.DataFrame(columns=["series_id", "date", "value", "vintage_date", "source", "metadata_json"])
    t = pd.Timestamp(asof_date).normalize()
    df = vintage_observations.copy()
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df["vintage_date"] = pd.to_datetime(df["vintage_date"], errors="coerce")
    if "realtime_start" in df:
        df["realtime_start"] = pd.to_datetime(df["realtime_start"], errors="coerce")
    else:
        df["realtime_start"] = df["vintage_date"]
    # An observation dated after the forecast date and a vintage after forecast date are both illegal.
    legal = df[(df["observation_date"] <= t) & (df["vintage_date"] <= t) & (df["realtime_start"] <= t)].copy()
    if legal.empty:
        return pd.DataFrame(columns=["series_id", "date", "value", "vintage_date", "source", "metadata_json"])
    legal = legal.sort_values(["series_id", "observation_date", "vintage_date", "realtime_start"])
    latest = legal.drop_duplicates(["series_id", "observation_date"], keep="last")
    out = latest.rename(columns={"observation_date": "date"}).copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["vintage_date"] = pd.to_datetime(out["vintage_date"]).dt.strftime("%Y-%m-%d")
    if "source" not in out:
        out["source"] = "vintage_observations"
    if "metadata_json" not in out:
        out["metadata_json"] = "{}"
    return out[["series_id", "date", "value", "vintage_date", "source", "metadata_json"]]


def latest_vintage_observations_per_asof_grid(
    vintage_observations: pd.DataFrame,
    asof_dates: list[str | pd.Timestamp],
) -> pd.DataFrame:
    """Multi-date variant of :func:`latest_vintage_observations_asof`.

    Returns a long frame with one row per (series_id, observation_date,
    as_of_date) holding the latest legal vintage for that triple. The
    frame is keyed on ``(series_id, observation_date, as_of_date)`` and
    the legal-vintage selection is monotone: a row is "live" at as-of
    ``t`` iff ``t`` is in the half-open interval ``[vintage_date,
    next_vintage_date)`` for that (series, obs_date) pair.

    The intended caller is :func:`materialize_feature_asof_values`; the
    helper is exposed because it's the natural set-based primitive for
    any other PIT computation that wants the full vintage panel grid in
    one pass.
    """
    if vintage_observations is None or vintage_observations.empty or not asof_dates:
        return pd.DataFrame(columns=["series_id", "observation_date", "as_of_date", "vintage_date", "value"])
    df = vintage_observations.copy()
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df["vintage_date"] = pd.to_datetime(df["vintage_date"], errors="coerce")
    if "realtime_start" in df:
        df["realtime_start"] = pd.to_datetime(df["realtime_start"], errors="coerce").fillna(df["vintage_date"])
    else:
        df["realtime_start"] = df["vintage_date"]
    df = df.sort_values(["series_id", "observation_date", "vintage_date", "realtime_start"])
    df["next_vintage_date"] = df.groupby(["series_id", "observation_date"])["vintage_date"].shift(-1)

    asof_ts = pd.Index(sorted({pd.Timestamp(d).normalize() for d in asof_dates}))
    rows: list[pd.DataFrame] = []
    sentinel = pd.Timestamp("2999-12-31")
    next_eff = df["next_vintage_date"].fillna(sentinel)
    for t in asof_ts:
        mask = (df["observation_date"] <= t) & (df["vintage_date"] <= t) & (df["realtime_start"] <= t) & (next_eff > t)
        if not mask.any():
            continue
        chunk = df.loc[mask, ["series_id", "observation_date", "vintage_date", "value"]].copy()
        chunk["as_of_date"] = t
        rows.append(chunk)
    if not rows:
        return pd.DataFrame(columns=["series_id", "observation_date", "as_of_date", "vintage_date", "value"])
    return pd.concat(rows, ignore_index=True)


def _resolve_asof_grid(
    df: pd.DataFrame,
    min_history_months: int,
    freq: str = "MS",
) -> list[pd.Timestamp]:
    """Build the as-of grid used by :func:`materialize_feature_asof_values`.

    v1.4.1 hard-coded ``freq="MS"`` (REVIEW.md ASK-2 / P0). v1.5 adds
    the ``freq`` kwarg with the same default so existing monthly
    callers stay byte-for-byte identical; FI callers pass an explicit
    ``Cadence.DAILY.value`` / ``Cadence.MIN_15.value`` to drive
    sub-daily PIT materialisation.

    For monthly grids the warmup is the legacy
    ``min_history_months``-month offset. For sub-daily / daily grids
    we keep the monthly-equivalent shift so the warmup window stays a
    function of calendar months (a 36-month warmup at daily cadence
    still requires ~3 years of history before the first as-of, the
    grid is just denser inside that window).
    """
    vmin = df["vintage_date"].min()
    vmax = df["vintage_date"].max()
    if pd.isna(vmin) or pd.isna(vmax):
        return []
    start = max(vmin, df["observation_date"].min() + pd.DateOffset(months=min_history_months))
    # Resolve the cadence so we can decide whether to normalise to
    # midnight. Day-grained cadences keep the v1.4.1 ``.normalize()``
    # behaviour so existing monthly/daily callers get bit-identical
    # output; intraday cadences preserve the wall-clock time so an FI
    # 15-min grid does not collapse to midnight.
    try:
        cadence: Cadence | None = Cadence.from_pandas_freq(freq)
    except ValueError:
        # Fall through to pandas so a niche pandas alias still works;
        # pandas will raise if the freq is truly invalid.
        cadence = None
    resolved_freq = cadence.value if cadence is not None else freq
    asof_idx = pd.date_range(start=start, end=vmax, freq=resolved_freq)
    day_grained = cadence in (Cadence.MONTHLY, Cadence.DAILY, None)
    if day_grained:
        return [pd.Timestamp(d).normalize() for d in asof_idx]
    return [pd.Timestamp(d) for d in asof_idx]


def _materialize_no_revisions(
    df: pd.DataFrame,
    catalog: list[dict],
    asof_ts_list: list[pd.Timestamp],
) -> pd.DataFrame:
    """Fast path: no (series, obs_date) pair has more than one vintage.

    Build the panel + features ONCE on the full data, then per-asof slice
    and pick latest per feature_name.

    This is mathematically equivalent to the per-asof rebuild because, in
    the no-revisions case, the legal panel at as-of ``t`` is just the
    full panel with rows filtered to ``observation_date <= t`` AND
    ``vintage_date <= t``. Both filters trim history; neither changes the
    *value* in any cell. Feature transforms are pure functions of the
    panel rows that pass the filter — and since the feature value at a
    given date depends only on observations at or before that date, the
    feature value at date ``d`` (when ``d <= t``) is identical whether
    we compute it on the full panel or on the truncated one.
    """
    obs_for_panel = df.rename(columns={"observation_date": "date"})[
        ["series_id", "date", "value", "vintage_date"]
    ].copy()
    panel = monthly_panel(obs_for_panel)
    if panel.empty:
        return _empty_feature_asof_values()
    global_feats = build_features(panel, catalog)
    if global_feats.empty:
        return _empty_feature_asof_values()
    global_feats = global_feats.copy()
    global_feats["date"] = pd.to_datetime(global_feats["date"])

    # Annotate per-feature lineage in one vectorised pass.
    # Catalog entries with the longest series_id wins on prefix matching
    # so that "SPX.level" maps to "SPX" rather than getting confused with
    # a hypothetical longer-prefix match.
    catalog_series = {item["series_id"] for item in catalog}
    parts = global_feats["feature_name"].astype(str).str.split(".", n=1, expand=True)
    raw_sid = parts[0]
    raw_transform = parts[1].fillna("unknown")
    global_feats["source_series_id"] = raw_sid.where(raw_sid.isin(catalog_series), raw_sid)
    global_feats["transform_name"] = raw_transform

    # Single-merge lineage lookup: for each (series, obs_date) pull the
    # vintage_date observed at that pair. Since we're in the no-revisions
    # path each (series, obs_date) has exactly one vintage_date.
    vintage_lookup = (
        df[["series_id", "observation_date", "vintage_date"]]
        .drop_duplicates(["series_id", "observation_date"])
        .rename(columns={"series_id": "source_series_id", "observation_date": "_obs_date"})
    )
    # The feature's "observation_date" in the original output schema is
    # the panel-resampled date that the feature value sits on. Resampling
    # to month-start can shift observations forward by a few days when
    # they aren't already MS-aligned; for lineage we want the original
    # observation_date that produced the panel cell, which the merge_asof
    # below recovers. ``merge_asof`` requires the *left* keys to be
    # globally sorted by the ``on`` column (it ignores the ``by`` group
    # for sort-validation), so we sort by ``date`` first.
    global_feats = global_feats.sort_values("date").reset_index(drop=True)
    vintage_lookup = vintage_lookup.sort_values("_obs_date").reset_index(drop=True)
    merged = pd.merge_asof(
        global_feats,
        vintage_lookup,
        left_on="date",
        right_on="_obs_date",
        by="source_series_id",
        direction="backward",
    )
    # Drop rows where the lookup failed entirely (no observation at or
    # before the feature date for that series — should be rare and was
    # silently skipped by the legacy loop too).
    merged = merged[merged["_obs_date"].notna()].copy()
    merged["observation_date"] = merged["_obs_date"]
    merged.drop(columns=["_obs_date"], inplace=True)

    created = datetime.now(UTC).isoformat()
    obs_str = merged["observation_date"].dt.strftime("%Y-%m-%d")
    vint_str = pd.to_datetime(merged["vintage_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    feat_date_str = merged["date"].dt.strftime("%Y-%m-%d")
    domain_vals = merged.get("domain")
    if domain_vals is None:
        domain_vals = pd.Series([None] * len(merged), index=merged.index)
    metadata_json = [
        json.dumps({"feature_date": d, "domain": dom}, sort_keys=True)
        for d, dom in zip(feat_date_str.values, domain_vals.values, strict=False)
    ]
    merged = merged.assign(
        _obs_str=obs_str.values,
        _vint_str=vint_str.values,
        _metadata_json=metadata_json,
    )

    # Per-asof slice + pick latest per feature_name. The work inside the
    # loop is now dominated by an integer mask and a drop_duplicates on
    # ~N_features rows — orders of magnitude faster than rebuilding the
    # panel + features per as-of.
    out_rows: list[pd.DataFrame] = []
    for asof_ts in asof_ts_list:
        feats_t = merged[merged["date"] <= asof_ts]
        if feats_t.empty:
            continue
        latest = feats_t.sort_values("date").drop_duplicates("feature_name", keep="last")
        if latest.empty:
            continue
        n = len(latest)
        out = pd.DataFrame(
            {
                "as_of_date": [asof_ts.strftime("%Y-%m-%d")] * n,
                "feature_name": latest["feature_name"].values,
                "source_series_id": latest["source_series_id"].values,
                "observation_date": latest["_obs_str"].values,
                "vintage_date": latest["_vint_str"].values,
                "value": latest["value"].astype(float).values,
                "transform_name": latest["transform_name"].values,
                "created_at_utc": [created] * n,
                "metadata_json": latest["_metadata_json"].values,
            }
        )
        out_rows.append(out)
    if not out_rows:
        return _empty_feature_asof_values()
    return pd.concat(out_rows, ignore_index=True)


def _materialize_with_revisions(
    df: pd.DataFrame,
    catalog: list[dict],
    asof_ts_list: list[pd.Timestamp],
) -> pd.DataFrame:
    """Slow(er) path: at least one (series, obs_date) carries multiple
    vintages, so the panel value at a cell can change as ``as_of``
    advances. We still avoid the per-feature ``iterrows`` of the legacy
    code by building lineage via a single ``merge`` per as-of, and we
    cache the panel/feature build whenever the legal table is identical
    to the previous as-of.
    """
    catalog_series = {item["series_id"] for item in catalog}
    out_rows: list[pd.DataFrame] = []
    created = datetime.now(UTC).isoformat()

    df_sorted = df.sort_values(["series_id", "observation_date", "vintage_date"]).copy()

    last_obs_sig: tuple | None = None
    cached_feats: pd.DataFrame | None = None
    cached_legal_obs: pd.DataFrame | None = None

    for asof_ts in asof_ts_list:
        legal_obs = latest_vintage_observations_asof(df_sorted, asof_ts.strftime("%Y-%m-%d"))
        if legal_obs.empty:
            continue
        # Cheap fingerprint: (row count, max obs_date, max vintage_date).
        # When it matches the prior as-of we reuse the cached panel and
        # features instead of rebuilding them.
        sig = (
            len(legal_obs),
            str(legal_obs["date"].max()),
            str(legal_obs["vintage_date"].max()),
        )
        if sig != last_obs_sig:
            panel = monthly_panel(legal_obs)
            feats = build_features(panel, catalog)
            if feats.empty:
                last_obs_sig = sig
                cached_feats = None
                cached_legal_obs = legal_obs
                continue
            feats = feats.copy()
            feats["date"] = pd.to_datetime(feats["date"])
            parts = feats["feature_name"].astype(str).str.split(".", n=1, expand=True)
            raw_sid = parts[0]
            feats["source_series_id"] = raw_sid.where(raw_sid.isin(catalog_series), raw_sid)
            feats["transform_name"] = parts[1].fillna("unknown")
            cached_feats = feats
            cached_legal_obs = legal_obs.copy()
            cached_legal_obs["date"] = pd.to_datetime(cached_legal_obs["date"])
            cached_legal_obs["vintage_date"] = pd.to_datetime(cached_legal_obs["vintage_date"], errors="coerce")
            last_obs_sig = sig
        if cached_feats is None or cached_legal_obs is None:
            continue
        feats_t = cached_feats[cached_feats["date"] <= asof_ts]
        if feats_t.empty:
            continue
        latest = feats_t.sort_values("date").drop_duplicates("feature_name", keep="last")
        if latest.empty:
            continue
        # Vectorised lineage lookup via merge_asof (backward) so the
        # source observation is the latest legal obs at or before the
        # feature date. ``merge_asof`` requires both sides sorted by the
        # ``on`` column globally.
        latest_sorted = latest.sort_values("date").reset_index(drop=True)
        legal_sorted = cached_legal_obs.sort_values("date").reset_index(drop=True)
        merged = pd.merge_asof(
            latest_sorted,
            legal_sorted[["series_id", "date", "vintage_date"]].rename(
                columns={"series_id": "source_series_id", "date": "_obs_date", "vintage_date": "_vintage_date"}
            ),
            left_on="date",
            right_on="_obs_date",
            by="source_series_id",
            direction="backward",
        )
        merged = merged[merged["_obs_date"].notna()].copy()
        if merged.empty:
            continue
        n = len(merged)
        feat_date_str = merged["date"].dt.strftime("%Y-%m-%d")
        obs_str = merged["_obs_date"].dt.strftime("%Y-%m-%d")
        vint_str = merged["_vintage_date"].dt.strftime("%Y-%m-%d")
        domain_vals = merged.get("domain")
        if domain_vals is None:
            domain_vals = pd.Series([None] * n, index=merged.index)
        metadata_json = [
            json.dumps({"feature_date": d, "domain": dom}, sort_keys=True)
            for d, dom in zip(feat_date_str.values, domain_vals.values, strict=False)
        ]
        out = pd.DataFrame(
            {
                "as_of_date": [asof_ts.strftime("%Y-%m-%d")] * n,
                "feature_name": merged["feature_name"].values,
                "source_series_id": merged["source_series_id"].values,
                "observation_date": obs_str.values,
                "vintage_date": vint_str.values,
                "value": merged["value"].astype(float).values,
                "transform_name": merged["transform_name"].values,
                "created_at_utc": [created] * n,
                "metadata_json": metadata_json,
            }
        )
        out_rows.append(out)
    if not out_rows:
        return _empty_feature_asof_values()
    return pd.concat(out_rows, ignore_index=True)


def materialize_feature_asof_values(
    vintage_observations: pd.DataFrame,
    catalog: list[dict],
    *,
    asof_dates: list[str] | None = None,
    min_history_months: int = 36,
) -> pd.DataFrame:
    """Build one feature snapshot per as-of date using only legal vintages.

    v1.2.1: Vectorised pipeline. Detects whether the input has any
    revisions per (series, observation_date). When there are none, the
    panel and feature transforms are computed once on the full data and
    every as-of becomes a slice + ``drop_duplicates`` over a few hundred
    feature rows. When revisions exist, the per-as-of rebuild is kept but
    the inner per-feature ``iterrows`` is replaced by a single vectorised
    ``merge_asof`` for lineage. Result schema is byte-identical to the
    pre-v1.2.1 frame (modulo float rounding within ``1e-9``).
    """
    if vintage_observations is None or vintage_observations.empty:
        return _empty_feature_asof_values()
    df = vintage_observations.copy()
    df["vintage_date"] = pd.to_datetime(df["vintage_date"], errors="coerce")
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    if asof_dates is None:
        asof_ts_list = _resolve_asof_grid(df, min_history_months)
    else:
        asof_ts_list = [pd.Timestamp(d).normalize() for d in asof_dates]
    if not asof_ts_list:
        return _empty_feature_asof_values()
    has_revisions = bool(df.duplicated(["series_id", "observation_date"], keep=False).any())
    if has_revisions:
        return _materialize_with_revisions(df, catalog, asof_ts_list)
    return _materialize_no_revisions(df, catalog, asof_ts_list)


def feature_asof_to_features(feature_asof_values: pd.DataFrame) -> pd.DataFrame:
    """Convert as-of feature snapshots into the existing feature table shape."""
    if feature_asof_values is None or feature_asof_values.empty:
        return pd.DataFrame(columns=["feature_name", "date", "value", "domain", "metadata_json"])
    frame = feature_asof_values.copy()
    frame["date"] = frame["as_of_date"]
    frame["domain"] = frame["metadata_json"].apply(
        lambda s: json.loads(s).get("domain") if isinstance(s, str) and s else None
    )
    frame["metadata_json"] = frame.apply(
        lambda r: json.dumps(
            {
                "as_of_date": r["as_of_date"],
                "source_series_id": r["source_series_id"],
                "observation_date": r["observation_date"],
                "vintage_date": r["vintage_date"],
                "transform_name": r["transform_name"],
                "point_in_time": True,
            },
            sort_keys=True,
        ),
        axis=1,
    )
    return frame[["feature_name", "date", "value", "domain", "metadata_json"]]


def audit_feature_asof_lineage(
    feature_asof_values: pd.DataFrame, *, asof_timestamp_required: bool = False
) -> pd.DataFrame:
    """Audit hard point-in-time invariants for feature snapshots."""
    if feature_asof_values is None or feature_asof_values.empty:
        return pd.DataFrame(
            [
                {
                    "audit": "feature_asof_lineage",
                    "rows": 0,
                    "violations": 1,
                    "status": "FAIL",
                    "details": "No feature_asof_values rows found",
                    "metadata_json": "{}",
                }
            ]
        )
    df = feature_asof_values.copy()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df["vintage_date"] = pd.to_datetime(df["vintage_date"], errors="coerce")
    obs_future = int((df["observation_date"] > df["as_of_date"]).sum())
    vintage_future = int((df["vintage_date"] > df["as_of_date"]).sum())
    null_lineage = int(
        df[["source_series_id", "observation_date", "vintage_date", "transform_name"]].isna().any(axis=1).sum()
    )
    dupes = int(df.duplicated(["as_of_date", "feature_name"]).sum())
    violations = obs_future + vintage_future + null_lineage + dupes
    return pd.DataFrame(
        [
            {
                "audit": "feature_asof_lineage",
                "rows": len(df),
                "violations": int(violations),
                "status": "PASS" if violations == 0 else "FAIL",
                "details": json.dumps(
                    {
                        "future_observation_date": obs_future,
                        "future_vintage_date": vintage_future,
                        "null_lineage": null_lineage,
                        "duplicate_feature_asof": dupes,
                    },
                    sort_keys=True,
                ),
                "metadata_json": json.dumps({"asof_timestamp_required": asof_timestamp_required}, sort_keys=True),
            }
        ]
    )


def audit_vintage_observations(vintage_observations: pd.DataFrame) -> pd.DataFrame:
    if vintage_observations is None or vintage_observations.empty:
        return pd.DataFrame(
            [
                {
                    "audit": "vintage_observations",
                    "rows": 0,
                    "violations": 1,
                    "status": "FAIL",
                    "details": "No vintage observations found",
                    "metadata_json": "{}",
                }
            ]
        )
    df = vintage_observations.copy()
    for c in ["observation_date", "vintage_date", "realtime_start"]:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    nulls = int(df[["series_id", "observation_date", "vintage_date", "value"]].isna().any(axis=1).sum())
    obs_after_vintage = int((df["observation_date"] > df["vintage_date"]).sum())
    dupes = int(df.duplicated(["series_id", "observation_date", "vintage_date"]).sum())
    violations = nulls + obs_after_vintage + dupes
    return pd.DataFrame(
        [
            {
                "audit": "vintage_observations",
                "rows": len(df),
                "violations": int(violations),
                "status": "PASS" if violations == 0 else "FAIL",
                "details": json.dumps(
                    {"nulls": nulls, "observation_after_vintage": obs_after_vintage, "duplicate_keys": dupes},
                    sort_keys=True,
                ),
                "metadata_json": "{}",
            }
        ]
    )
