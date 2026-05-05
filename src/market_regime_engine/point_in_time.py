# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ReleaseRule:
    series_id: str
    lag_days: int = 0
    lag_months: int = 0

    def release_date_for_observation(self, date: pd.Timestamp) -> pd.Timestamp:
        return pd.Timestamp(date) + pd.DateOffset(months=self.lag_months, days=self.lag_days)


DEFAULT_RELEASE_RULES: dict[str, ReleaseRule] = {
    # v1.3 (item B4): conservative defaults that cover every series in
    # ``config/series_catalog.yaml``. Exact release calendar ingestion
    # belongs in production via ``release_calendar.yaml`` /
    # ``exact_release_calendar``; the dict below is the floor used by
    # ``apply_release_lag`` when no exact calendar is loaded.
    #
    # Daily-availability series get a 1-day lag so the vintage_date is
    # never the same calendar day as the observation_date (a 0-lag rule
    # would technically pass ``assert_no_future_vintages`` while still
    # encoding a same-day publish, which is wrong for nearly every
    # macro / market series). Monthly series get a 1-month lag (CPI,
    # employment), quarterly series get a 3-month lag.
    # ----- Labor / employment -----
    "UNRATE": ReleaseRule("UNRATE", lag_months=1),
    "U6RATE": ReleaseRule("U6RATE", lag_months=1),
    "PAYEMS": ReleaseRule("PAYEMS", lag_months=1),
    # ----- Inflation -----
    "CPIAUCSL": ReleaseRule("CPIAUCSL", lag_months=1),
    "CPILFESL": ReleaseRule("CPILFESL", lag_months=1),
    "PCEPI": ReleaseRule("PCEPI", lag_months=1),
    # ----- Rates -----
    "FEDFUNDS": ReleaseRule("FEDFUNDS", lag_months=1),
    "DGS10": ReleaseRule("DGS10", lag_days=1),
    "T10Y3M": ReleaseRule("T10Y3M", lag_days=1),
    "BAA10Y": ReleaseRule("BAA10Y", lag_days=1),
    # ----- Housing -----
    "PERMIT": ReleaseRule("PERMIT", lag_months=1),
    "HOUST": ReleaseRule("HOUST", lag_months=1),
    "MORTGAGE30US": ReleaseRule("MORTGAGE30US", lag_days=1),
    # ----- Energy / FX -----
    "DCOILWTICO": ReleaseRule("DCOILWTICO", lag_days=1),
    "DTWEXBGS": ReleaseRule("DTWEXBGS", lag_days=1),
    # ----- Markets / fiscal -----
    "SPX": ReleaseRule("SPX", lag_days=0),  # market index synthetic series
    "GFDEGDQ188S": ReleaseRule("GFDEGDQ188S", lag_months=3),
}


def assert_no_future_vintages(observations: pd.DataFrame) -> None:
    """Raise if observation vintage/release chronology is impossible."""
    if observations.empty:
        return
    frame = observations.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["vintage_date"] = pd.to_datetime(frame.get("vintage_date", frame["date"]))
    bad = frame[frame["vintage_date"] < frame["date"]]
    if not bad.empty:
        sample = bad[["series_id", "date", "vintage_date"]].head().to_dict(orient="records")
        raise ValueError(f"Invalid point-in-time data: vintage_date before observation date: {sample}")


def apply_release_lag(
    observations: pd.DataFrame,
    rules: dict[str, ReleaseRule] | None = None,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Fill missing/too-early vintages using conservative release-lag rules.

    v1.3 (item B4): ``strict`` defaults to ``True``. When a ``series_id``
    in ``observations`` is not in ``rules`` (or in
    :data:`DEFAULT_RELEASE_RULES` when ``rules`` is left as default), the
    function raises ``RuntimeError`` instead of silently treating the
    series as zero-lag. The pre-v1.3 behaviour (silent zero-lag) is
    available via ``strict=False`` and will emit a one-line warning.

    The breaking change is intentional: the v1.2.1 catalog only covered
    ~9 series, so any series in ``config/series_catalog.yaml`` that
    didn't appear in DEFAULT_RELEASE_RULES received an effective
    ``lag_months=0, lag_days=0`` — which is a subtle PIT leak (no
    release lag is still a vintage; a fresh observation date is not the
    same as a same-day vintage_date). Production deployments that need
    this knob can either extend ``DEFAULT_RELEASE_RULES`` or pass an
    explicit ``rules`` mapping that covers every series id.
    """
    if observations.empty:
        return observations.copy()
    rules = rules or DEFAULT_RELEASE_RULES

    # Identify any series that lack an explicit rule and surface the
    # gap loudly. The audit list is deduplicated and sorted so the
    # error message is stable across runs.
    series_in_frame = (
        (observations["series_id"].astype(str).dropna().unique().tolist()) if "series_id" in observations else []
    )
    missing = sorted({sid for sid in series_in_frame if sid not in rules})
    if missing:
        if strict:
            sample = ", ".join(missing[:5])
            more = "" if len(missing) <= 5 else f" (and {len(missing) - 5} more)"
            raise RuntimeError(
                f"series '{sample}'{more} have no release rule; add to "
                "DEFAULT_RELEASE_RULES or release_calendar.yaml, or pass "
                "strict=False to fall back to the legacy zero-lag default."
            )
        # Non-strict: warn loudly so the silent zero-lag isn't invisible
        # in production logs.
        try:
            from market_regime_engine.logging_setup import get_logger as _get_logger

            _get_logger("mre.point_in_time").warning(
                "apply_release_lag strict=False: %d series lack release rules; "
                "falling back to zero-lag (vintage = observation_date).",
                len(missing),
            )
        except Exception:  # pragma: no cover - defensive
            pass

    frame = observations.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if "vintage_date" not in frame:
        frame["vintage_date"] = pd.NaT
    frame["vintage_date"] = pd.to_datetime(frame["vintage_date"], errors="coerce")

    def effective_vintage(row: pd.Series) -> pd.Timestamp:
        rule = rules.get(str(row["series_id"]), ReleaseRule(str(row["series_id"])))
        min_release = rule.release_date_for_observation(row["date"])
        vintage = row["vintage_date"] if pd.notna(row["vintage_date"]) else min_release
        return max(pd.Timestamp(vintage), pd.Timestamp(min_release))

    frame["vintage_date"] = frame.apply(effective_vintage, axis=1)
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    frame["vintage_date"] = frame["vintage_date"].dt.strftime("%Y-%m-%d")
    return frame


def observations_as_of(observations: pd.DataFrame, as_of: str | pd.Timestamp) -> pd.DataFrame:
    """Return latest observation vintages known as of the supplied forecast date."""
    if observations.empty:
        return observations.copy()
    as_of_ts = pd.to_datetime(as_of)
    frame = observations.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["vintage_date"] = pd.to_datetime(frame.get("vintage_date", frame["date"]))
    frame = frame[(frame["date"] <= as_of_ts) & (frame["vintage_date"] <= as_of_ts)]
    frame = frame.sort_values(["series_id", "date", "vintage_date"])
    return frame.groupby(["series_id", "date"], as_index=False).tail(1).reset_index(drop=True)


def point_in_time_panel_builder(
    observations: pd.DataFrame, as_of_dates: list[str | pd.Timestamp]
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Build a dictionary of observation slices as of forecast dates."""
    return {pd.to_datetime(as_of): observations_as_of(observations, as_of) for as_of in as_of_dates}
