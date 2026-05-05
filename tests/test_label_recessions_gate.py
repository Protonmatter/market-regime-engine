"""Regression test for the recession-label staleness gate ordering.

The original ``label_recessions_cmd`` wrote labels to the warehouse and
*then* checked the staleness gate, so a tripped gate left the warehouse
poisoned with stale rows. v1.1 evaluates the gate first and refuses to
write when it trips.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from market_regime_engine.cli import label_recessions_cmd
from market_regime_engine.nber import LabelStaleness
from market_regime_engine.sample import generate_sample_observations
from market_regime_engine.storage import Warehouse


def test_stale_gate_trips_before_write_keeps_warehouse_clean(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "mre_stale_test.db"
        db = Warehouse(db_path)
        try:
            db.write_observations(generate_sample_observations())
            assert db.read_recession_labels().empty
        finally:
            db.close()

        # Patch the labeller to return a fixed "very stale" report so the
        # gate trips deterministically without touching the network.
        stale = LabelStaleness(
            source="built_in_nber_windows",
            last_label_date="2020-04-01",
            panel_last_date="2024-06-01",
            months_stale=50,
            fetch_error="",
        )
        fake_labels = pd.DataFrame(
            [
                {
                    "date": "2020-04-01",
                    "recession": 1.0,
                    "source": "built_in_nber_windows",
                    "metadata_json": "{}",
                }
            ]
        )

        def fake_label(*args, **kwargs):
            return fake_labels, stale

        monkeypatch.setattr("market_regime_engine.nber.label_recessions_with_fallback", fake_label)

        ns = argparse.Namespace(
            db=str(db_path),
            force_builtin=True,
            max_stale_months=24,
        )
        with pytest.raises(SystemExit) as exc_info:
            label_recessions_cmd(ns)
        msg = str(exc_info.value)
        assert "stale" in msg.lower()
        assert "24" in msg

        db = Warehouse(db_path)
        try:
            assert db.read_recession_labels().empty, "stale labels must NOT be written when the gate trips"
        finally:
            db.close()


def test_gate_disabled_writes_normally(monkeypatch) -> None:
    """Sanity: when ``max_stale_months`` is None, labels are written."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "mre_clean_test.db"
        db = Warehouse(db_path)
        try:
            db.write_observations(generate_sample_observations())
        finally:
            db.close()

        ns = argparse.Namespace(
            db=str(db_path),
            force_builtin=True,
            max_stale_months=None,
        )
        label_recessions_cmd(ns)

        db = Warehouse(db_path)
        try:
            assert not db.read_recession_labels().empty
        finally:
            db.close()
