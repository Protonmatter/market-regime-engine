# SPDX-License-Identifier: Apache-2.0
"""Canonical governed-signal contract shared by all external adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

GOVERNED_SIGNAL_COLUMNS: tuple[str, ...] = (
    "date",
    "regime_state",
    "regime_confidence",
    "change_point_prob",
    "drawdown_prob",
    "recession_prob",
    "confidence_score",
    "release_gate_decision",
    "release_gate_approved",
    "model_run_id",
    "artifact_hash",
    "metadata_json",
)

_TRUE_VALUES = {"true", "1", "yes", "y", "approved", "release"}
_FALSE_VALUES = {"false", "0", "no", "n", "", "none", "null", "nan", "hold", "blocked", "rejected"}


@dataclass(frozen=True)
class GovernedSignalExport:
    """Result object returned by adapter export helpers."""

    path: str
    rows: int
    format: str
    columns: tuple[str, ...]


def _first_existing(frame: pd.DataFrame, names: tuple[str, ...], default: object = None) -> pd.Series:
    for name in names:
        if name in frame:
            return frame[name]
    if isinstance(default, pd.Series):
        return default.reindex(frame.index)
    if isinstance(default, pd.Index) or (hasattr(default, "__len__") and not isinstance(default, str)):
        try:
            if len(default) == len(frame):
                return pd.Series(default, index=frame.index)
        except TypeError:
            pass
    return pd.Series([default] * len(frame), index=frame.index)


def parse_bool_series(values: pd.Series, *, default: bool = False) -> pd.Series:
    """Parse external boolean-like values without Python truthiness traps.

    ``astype(bool)`` treats every non-empty string as True, so the string
    ``"False"`` becomes ``True``. That is unacceptable for release-gate signals.
    """

    if values.empty:
        return pd.Series([], dtype=bool, index=values.index)
    normalized = values.fillna(str(default)).astype(str).str.strip().str.lower()
    valid = _TRUE_VALUES | _FALSE_VALUES
    bad = normalized[~normalized.isin(valid)]
    if not bad.empty:
        raise ValueError(f"invalid boolean values: {sorted(bad.unique())}")
    return normalized.isin(_TRUE_VALUES)


def normalize_governed_signals(
    frame: pd.DataFrame,
    *,
    default_release_gate_decision: str = "unknown",
) -> pd.DataFrame:
    """Normalize regime/model output frames into the external signal contract."""

    if frame is None or frame.empty:
        return pd.DataFrame(columns=GOVERNED_SIGNAL_COLUMNS)

    src = frame.copy()
    out = pd.DataFrame(index=src.index)
    dates = _first_existing(src, ("date", "as_of_date", "timestamp"), src.index)
    out["date"] = pd.to_datetime(dates, errors="coerce").dt.strftime("%Y-%m-%d")
    out["regime_state"] = _first_existing(
        src,
        ("regime_state", "decoded_regime", "regime", "msvar_regime", "state"),
        "unknown",
    ).astype(str)
    out["regime_confidence"] = pd.to_numeric(
        _first_existing(src, ("regime_confidence", "msvar_confidence", "score", "confidence"), 0.0),
        errors="coerce",
    ).fillna(0.0)
    out["change_point_prob"] = pd.to_numeric(
        _first_existing(src, ("change_point_prob", "cp_prob", "bocpd_prob"), 0.0),
        errors="coerce",
    ).fillna(0.0)
    out["drawdown_prob"] = pd.to_numeric(
        _first_existing(src, ("drawdown_prob", "p_drawdown", "drawdown_probability"), 0.0),
        errors="coerce",
    ).fillna(0.0)
    out["recession_prob"] = pd.to_numeric(
        _first_existing(src, ("recession_prob", "p_recession", "recession_probability"), 0.0),
        errors="coerce",
    ).fillna(0.0)
    out["confidence_score"] = pd.to_numeric(
        _first_existing(src, ("confidence_score", "confidence"), out["regime_confidence"]),
        errors="coerce",
    ).fillna(out["regime_confidence"])
    out["release_gate_decision"] = _first_existing(
        src,
        ("release_gate_decision", "decision"),
        default_release_gate_decision,
    ).astype(str)
    if "release_gate_approved" in src or "approved" in src:
        approved = _first_existing(src, ("release_gate_approved", "approved"), False)
        out["release_gate_approved"] = parse_bool_series(approved)
    else:
        out["release_gate_approved"] = parse_bool_series(out["release_gate_decision"])
    out["model_run_id"] = _first_existing(src, ("model_run_id", "run_id"), "").astype(str)
    out["artifact_hash"] = _first_existing(src, ("artifact_hash",), "").astype(str)

    if "metadata_json" in src:
        out["metadata_json"] = src["metadata_json"].astype(str)
    else:
        out["metadata_json"] = [
            json.dumps({"adapter_contract": "governed_macro_regime_signal_v1"}, sort_keys=True)
            for _ in range(len(out))
        ]

    return out.loc[:, GOVERNED_SIGNAL_COLUMNS].sort_values("date").reset_index(drop=True)


def assert_governed_signal_contract(frame: pd.DataFrame) -> None:
    """Raise when the normalized signal frame is unsafe for export."""

    missing = [col for col in GOVERNED_SIGNAL_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"governed signal frame is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("governed signal frame is empty")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("governed signal frame contains invalid dates")
    numeric_cols = (
        "regime_confidence",
        "change_point_prob",
        "drawdown_prob",
        "recession_prob",
        "confidence_score",
    )
    for col in numeric_cols:
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{col} contains non-numeric values")
        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"{col} must stay within [0, 1]")


def export_governed_signals(
    frame: pd.DataFrame,
    out_path: str | Path,
    *,
    fmt: Literal["csv", "json", "jsonl"] | None = None,
) -> GovernedSignalExport:
    """Normalize, validate, and export governed signals."""

    out = Path(out_path)
    resolved_fmt: str = fmt or out.suffix.lower().lstrip(".") or "csv"
    signals = normalize_governed_signals(frame)
    assert_governed_signal_contract(signals)
    out.parent.mkdir(parents=True, exist_ok=True)

    if resolved_fmt == "csv":
        signals.to_csv(out, index=False)
    elif resolved_fmt == "json":
        out.write_text(signals.to_json(orient="records", indent=2), encoding="utf-8")
    elif resolved_fmt == "jsonl":
        out.write_text(signals.to_json(orient="records", lines=True) + "\n", encoding="utf-8")
    else:
        raise ValueError(f"unsupported governed signal export format: {resolved_fmt!r}")

    return GovernedSignalExport(
        path=str(out),
        rows=len(signals),
        format=resolved_fmt,
        columns=tuple(signals.columns),
    )


__all__ = [
    "GOVERNED_SIGNAL_COLUMNS",
    "GovernedSignalExport",
    "assert_governed_signal_contract",
    "export_governed_signals",
    "normalize_governed_signals",
    "parse_bool_series",
]
