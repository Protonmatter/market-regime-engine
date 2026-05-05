# SPDX-License-Identifier: Apache-2.0
"""v1.4.1 — `verify_run` rejects arbitrary `extra` envelope drift (item D).

Pre-v1.4.1, ``verify_run`` only inspected the ``extra.training_audit``
sub-key of the stored envelope and ignored the rest of ``extra``. That
let arbitrary operator-supplied fields (compliance IDs, tenant labels,
custom run tags, etc.) drift between the stored envelope and the
re-derived current envelope without surfacing in the verify-run report.

v1.4.1 closes that gap: every key in ``stored.extra`` (other than
``training_audit``, which keeps its v1.2.1 friendly handling) is
structurally compared against the corresponding key in
``current_envelope.extra`` and any divergence flips ``approved`` to
``False`` and is recorded under ``differences["extra:<key>"]``.

These regression tests pin the contract.
"""

from __future__ import annotations

import pandas as pd

from market_regime_engine.model_runs import (
    build_repro_envelope,
    create_model_run,
    verify_run,
)


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"feature_name": "f1", "date": "2024-01-01", "value": 1.0},
            {"feature_name": "f2", "date": "2024-01-01", "value": 2.0},
        ]
    )


def _outputs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": "m",
                "date": "2024-01-01",
                "horizon": "3m",
                "target": "t",
                "value": 0.5,
            }
        ]
    )


def _stored_run(extra: dict) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Persist a model run whose envelope.extra contains the supplied dict.

    ``extra`` here is layered on top of the auto-stamped
    ``engine_version`` + ``purpose`` keys that ``create_model_run`` adds.
    The returned row's ``metadata_json`` carries the full envelope.
    """
    features = _features()
    outputs = _outputs()
    run = create_model_run(
        engine_version="1.4.1-test",
        purpose="extra-drift regression test",
        features=features,
        model_outputs=outputs,
    )
    # ``create_model_run`` auto-stamps ``extra={engine_version, purpose}``
    # but does not expose a hook for arbitrary extras. We mutate the
    # serialised metadata in-place to inject the test's ``extra`` dict
    # into the stored envelope, mimicking what an operator who wraps
    # ``create_model_run`` and post-processes the metadata JSON would
    # see.
    import json as _json

    meta = _json.loads(run.metadata_json)
    repro = meta["repro_envelope"]
    repro_extra = repro.setdefault("extra", {})
    for k, v in extra.items():
        repro_extra[k] = v
    meta["repro_envelope"] = repro
    row = pd.Series({"run_id": run.run_id, "metadata_json": _json.dumps(meta, sort_keys=True)})
    return row, features, outputs


def _current_envelope(features: pd.DataFrame, outputs: pd.DataFrame, extra: dict):
    return build_repro_envelope(
        features=features,
        model_outputs=outputs,
        extra={
            "engine_version": "1.4.1-test",
            "purpose": "extra-drift regression test",
            **extra,
        },
    )


def test_verify_run_detects_arbitrary_extra_field_drift() -> None:
    """Stored extra `foo: bar` vs current `foo: baz` → drift on `extra:foo`."""
    row, features, outputs = _stored_run({"foo": "bar"})
    current = _current_envelope(features, outputs, {"foo": "baz"})
    report = verify_run(str(row["run_id"]), row, current_envelope=current)
    assert report["approved"] is False, report
    assert "extra:foo" in report["differences"], report["differences"]
    diff = report["differences"]["extra:foo"]
    assert diff["stored"] == "bar"
    assert diff["current"] == "baz"


def test_verify_run_detects_added_extra_field() -> None:
    """Stored extra has no `foo`; current adds `foo` → drift."""
    row, features, outputs = _stored_run({})  # no foo
    current = _current_envelope(features, outputs, {"foo": "baz"})
    report = verify_run(str(row["run_id"]), row, current_envelope=current)
    assert report["approved"] is False, report
    assert "extra:foo" in report["differences"], report["differences"]
    diff = report["differences"]["extra:foo"]
    assert diff["stored"] is None
    assert diff["current"] == "baz"


def test_verify_run_detects_removed_extra_field() -> None:
    """Stored has extra `foo`; current does not → drift."""
    row, features, outputs = _stored_run({"foo": "bar"})
    current = _current_envelope(features, outputs, {})  # no foo
    report = verify_run(str(row["run_id"]), row, current_envelope=current)
    assert report["approved"] is False, report
    assert "extra:foo" in report["differences"], report["differences"]
    diff = report["differences"]["extra:foo"]
    assert diff["stored"] == "bar"
    assert diff["current"] is None


def test_verify_run_training_mode_drift_still_friendly() -> None:
    """v1.2.1 ``training_mode_drift`` / ``legacy_fallback_authorized``
    semantics are preserved verbatim under the v1.4.1 strict-extra
    compare. The friendly per-key handling for ``training_audit``
    stays."""
    row, features, outputs = _stored_run({})
    # Patch the stored extra so it now carries a legacy training_audit.
    import json as _json

    meta = _json.loads(row["metadata_json"])
    meta["repro_envelope"]["extra"]["training_audit"] = {
        "mode": "point_in_time",
        "mode_used": "legacy",
        "fallback_authorized": False,
        "fallback_reason": "feature_asof_values empty",
    }
    row = pd.Series({"run_id": row["run_id"], "metadata_json": _json.dumps(meta, sort_keys=True)})
    current = _current_envelope(features, outputs, {})
    report = verify_run(str(row["run_id"]), row, current_envelope=current)
    # Friendly handling still surfaces training_mode_drift, NOT extra:training_audit.
    assert "training_mode_drift" in report["differences"], report
    drift = report["differences"]["training_mode_drift"]
    assert drift["stored_mode"] == "legacy"
    assert drift["expected"] == "point_in_time"
    assert "extra:training_audit" not in report["differences"], (
        "training_audit must not double-report under the structural extra compare; it has its own friendly handling."
    )


def test_verify_run_training_audit_friendly_handling_with_arbitrary_extra() -> None:
    """A run with both training_audit (legacy) AND arbitrary extra drift
    surfaces both signals: training_mode_drift (friendly) AND
    extra:<key> for the arbitrary key."""
    row, features, outputs = _stored_run({"compliance_id": "XYZ-123"})
    import json as _json

    meta = _json.loads(row["metadata_json"])
    meta["repro_envelope"]["extra"]["training_audit"] = {
        "mode": "point_in_time",
        "mode_used": "legacy",
        "fallback_authorized": True,
    }
    row = pd.Series({"run_id": row["run_id"], "metadata_json": _json.dumps(meta, sort_keys=True)})
    current = _current_envelope(features, outputs, {"compliance_id": "XYZ-789"})
    report = verify_run(str(row["run_id"]), row, current_envelope=current)
    assert report["approved"] is False
    assert "training_mode_drift" in report["differences"]
    assert "extra:compliance_id" in report["differences"]
    assert "legacy_fallback_authorized" in report["warnings"]


def test_verify_run_extra_compare_is_canonical_under_dict_key_order() -> None:
    """Two dicts with identical mappings but different insertion order
    must NOT report drift — the canonicalisation step in verify_run
    sort-keys both sides before comparing."""
    extra_a = {"alpha": 1, "beta": 2, "gamma": 3}
    row, features, outputs = _stored_run({"nested": extra_a})
    current = _current_envelope(features, outputs, {"nested": {"gamma": 3, "alpha": 1, "beta": 2}})
    report = verify_run(str(row["run_id"]), row, current_envelope=current)
    assert report["approved"] is True, report
    assert "extra:nested" not in report["differences"], report["differences"]
