# SPDX-License-Identifier: Apache-2.0
"""OpenBB adapter for governed macro regime signal records."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from market_regime_engine.adapters.core import (
    GovernedSignalExport,
    assert_governed_signal_contract,
    normalize_governed_signals,
)


def to_openbb_records(frame: pd.DataFrame) -> list[dict]:
    """Return JSON-serializable records shaped for OpenBB/provider extensions."""

    governed = normalize_governed_signals(frame)
    assert_governed_signal_contract(governed)
    return governed.to_dict(orient="records")


def to_openbb_obbject_like(frame: pd.DataFrame, *, provider: str = "market_regime_engine") -> dict:
    """Return an OBBject-like dictionary without requiring OpenBB as a dependency."""

    records = to_openbb_records(frame)
    return {
        "results": records,
        "provider": provider,
        "warnings": None,
        "chart": None,
        "extra": {"metadata": {"contract": "governed_macro_regime_signal_v1", "rows": len(records)}},
    }


def export_openbb_json(frame: pd.DataFrame, out_path: str | Path) -> GovernedSignalExport:
    obj = to_openbb_obbject_like(frame)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return GovernedSignalExport(
        path=str(path),
        rows=len(obj["results"]),
        format="json",
        columns=tuple(obj["results"][0].keys()) if obj["results"] else (),
    )


__all__ = ["export_openbb_json", "to_openbb_obbject_like", "to_openbb_records"]
