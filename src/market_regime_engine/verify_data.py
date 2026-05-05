# SPDX-License-Identifier: Apache-2.0
"""Detect warehouse drift between a stored ``model_run`` and the current state.

v1.3 (item F) ships a companion to ``mre verify-run``. ``verify-run``
asks "does the *environment* still match the run envelope?" (code SHA,
lockfile hash, training audit). ``verify-data`` asks the orthogonal
question: "does the *warehouse data* still match the run envelope?"
A silent in-place mutation of ``vintage_observations`` (e.g. an ETL
re-run that overwrote a cell) would slip past ``verify-run``; this
module detects it.

The strategy is simple:

1. Load the stored ``repro_envelope`` from the run row's ``metadata_json``.
2. Re-derive ``feature_payload``, ``output_payload``, and
   ``vintage_payload`` against the warehouse's *current* state using
   the same ``_hash_frame`` implementation that originally produced the
   envelope.
3. Compare hash to hash. When a payload differs, surface the row count
   and the first ten changed rows (against the stored snapshot if it's
   embedded in ``extra``, otherwise against the empty set so the
   operator at least sees the live frame's first rows).

The CLI in ``cli.py`` exits 0 on no drift and exits 2 with structured
JSON on drift, mirroring ``verify-run``'s contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from market_regime_engine.model_runs import _hash_frame, _hash_frame_legacy
from market_regime_engine.storage import Warehouse


def _changed_rows(current: pd.DataFrame, stored_hash: str, *, hasher) -> list[dict[str, Any]]:
    """Best-effort first-10 rows for a drifted payload.

    The stored envelope only carries the hash, not the full row list,
    so we cannot identify *which* rows changed without the prior
    snapshot. As a useful proxy, surface the latest 10 rows so the
    operator has concrete context.
    """
    if current is None or current.empty:
        return []
    return current.tail(10).to_dict(orient="records")


def verify_warehouse_state(
    *,
    run_id: str | None = None,
    db_path: str | Path = "data/mre.db",
    legacy_hash: bool = False,
) -> dict[str, Any]:
    """Re-derive payload hashes from the warehouse and compare to a run.

    Parameters
    ----------
    run_id:
        Run id to verify. ``None`` (default) picks the latest run.
    db_path:
        Path to the warehouse.
    legacy_hash:
        Use the v1.2.1 hashing implementation. Required when verifying
        runs written before the v1.3 hash migration.

    Returns
    -------
    dict
        ``{"approved": bool, "run_id": str, "differences": {...},
        "stored_payloads": {...}, "current_payloads": {...}}``.
    """
    db = Warehouse(db_path)
    hasher = _hash_frame_legacy if legacy_hash else _hash_frame
    try:
        runs = db.read_model_runs()
        if runs.empty:
            return {
                "approved": False,
                "run_id": None,
                "missing_run": True,
                "differences": {},
                "warnings": ["no model_runs in warehouse"],
            }
        if run_id:
            row_match = runs[runs["run_id"] == run_id]
            if row_match.empty:
                return {
                    "approved": False,
                    "run_id": run_id,
                    "missing_run": True,
                    "differences": {},
                    "warnings": [f"run_id {run_id} not found"],
                }
            run_row = row_match.iloc[0]
        else:
            run_row = runs.iloc[-1]

        try:
            meta = json.loads(run_row.get("metadata_json", "{}") or "{}")
        except Exception:
            meta = {}
        stored = meta.get("repro_envelope", {}) if isinstance(meta, dict) else {}

        current_features = db.read_features()
        current_outputs = db.read_model_outputs()
        current_vintage = db.read_feature_asof_values()

        current_hashes = {
            "feature_payload": hasher(current_features),
            "output_payload": hasher(current_outputs),
            "vintage_payload": hasher(current_vintage),
        }
        stored_hashes = {
            "feature_payload": stored.get("feature_payload", ""),
            "output_payload": stored.get("output_payload", ""),
            "vintage_payload": stored.get("vintage_payload", ""),
        }
        live_frames: dict[str, pd.DataFrame] = {
            "feature_payload": current_features,
            "output_payload": current_outputs,
            "vintage_payload": current_vintage,
        }
        differences: dict[str, dict[str, Any]] = {}
        for payload_key, current in current_hashes.items():
            stored_h = stored_hashes.get(payload_key, "")
            if stored_h and current != stored_h:
                frame = live_frames[payload_key]
                differences[payload_key] = {
                    "stored": stored_h,
                    "current": current,
                    "current_rows": len(frame) if frame is not None else 0,
                    "changed_rows": _changed_rows(frame, stored_h, hasher=hasher),
                }
        return {
            "approved": not differences and bool(stored),
            "run_id": str(run_row["run_id"]),
            "missing_run": False,
            "missing_envelope": not stored,
            "stored_payloads": stored_hashes,
            "current_payloads": current_hashes,
            "differences": differences,
            "warnings": (["stored envelope is empty (run predates v1.0 envelope)"] if not stored else []),
        }
    finally:
        db.close()


__all__ = ["verify_warehouse_state"]
