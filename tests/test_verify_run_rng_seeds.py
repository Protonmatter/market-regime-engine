# SPDX-License-Identifier: Apache-2.0
"""v1.4.1 — `verify_run` compares `rng_seeds` instead of skipping (item E).

Pre-v1.4.1, ``verify_run`` had ``if key == "rng_seeds": continue`` —
the dict was unconditionally skipped, with the stated justification
that "the dict is unordered after JSON round-trip and we don't want
false drift". That justification was wrong: Python dict equality is
order-insensitive for equivalent mappings, and the JSON round-trip
that the doc-string referenced is exactly the canonicalisation step
that proves it.

The skip silently let stochastic-rerun workflows (different seeds →
different model outputs) pass that part of the envelope check.
v1.4.1 makes the comparison real and exposes a deliberate
``--ignore-rng-seeds`` opt-out for callers who legitimately re-derive
with different seeds.
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


def _stored_run_with_seeds(seeds: dict[str, int]) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    features = _features()
    outputs = _outputs()
    run = create_model_run(
        engine_version="1.4.1-test",
        purpose="rng-seeds regression test",
        features=features,
        model_outputs=outputs,
        rng_seeds=seeds,
    )
    row = pd.Series({"run_id": run.run_id, "metadata_json": run.metadata_json})
    return row, features, outputs


def _current_envelope(features: pd.DataFrame, outputs: pd.DataFrame, seeds: dict[str, int]):
    return build_repro_envelope(
        features=features,
        model_outputs=outputs,
        rng_seeds=seeds,
        extra={
            "engine_version": "1.4.1-test",
            "purpose": "rng-seeds regression test",
        },
    )


def test_verify_run_detects_rng_seed_drift() -> None:
    """Stored seeds {a: 1, b: 2}; current {a: 1, b: 999} → exit 2 + drift."""
    row, features, outputs = _stored_run_with_seeds({"a": 1, "b": 2})
    current = _current_envelope(features, outputs, {"a": 1, "b": 999})
    report = verify_run(str(row["run_id"]), row, current_envelope=current)
    assert report["approved"] is False, report
    assert "rng_seeds" in report["differences"], report["differences"]
    diff = report["differences"]["rng_seeds"]
    assert diff["stored"]["b"] == 2
    assert diff["current"]["b"] == 999


def test_verify_run_ignore_rng_seeds_passes_drifted_seeds() -> None:
    """``ignore_rng_seeds=True`` restores the v1.2.1 skip behaviour."""
    row, features, outputs = _stored_run_with_seeds({"a": 1, "b": 2})
    current = _current_envelope(features, outputs, {"a": 1, "b": 999})
    report = verify_run(
        str(row["run_id"]),
        row,
        current_envelope=current,
        ignore_rng_seeds=True,
    )
    assert report["approved"] is True, report
    assert "rng_seeds" not in report["differences"], report["differences"]


def test_verify_run_dict_key_order_in_rng_seeds_does_not_cause_false_drift() -> None:
    """The ``json.loads(json.dumps(d, sort_keys=True))`` canonicalisation
    proves that the v1.2.1 stated concern ("the dict is unordered after
    JSON round-trip") was not a real reason to skip the field.

    Build two dicts with the same mappings but inserted in opposite
    orders. After canonicalisation the comparison is identity, so no
    drift surfaces.
    """
    seeds_a = {"alpha": 1, "beta": 2, "gamma": 3}
    seeds_b = {"gamma": 3, "alpha": 1, "beta": 2}
    assert seeds_a == seeds_b  # Python dict equality is order-insensitive.
    row, features, outputs = _stored_run_with_seeds(seeds_a)
    current = _current_envelope(features, outputs, seeds_b)
    report = verify_run(str(row["run_id"]), row, current_envelope=current)
    assert report["approved"] is True, report
    assert "rng_seeds" not in report["differences"], report["differences"]


def test_verify_run_matching_rng_seeds_not_in_differences() -> None:
    """Identical seeds → no drift, no extra rows in differences."""
    seeds = {"global": 42, "model": 7}
    row, features, outputs = _stored_run_with_seeds(seeds)
    current = _current_envelope(features, outputs, dict(seeds))
    report = verify_run(str(row["run_id"]), row, current_envelope=current)
    assert report["approved"] is True, report
    assert "rng_seeds" not in report["differences"], report["differences"]


def test_verify_run_empty_seeds_match_cleanly() -> None:
    """Empty seeds on both sides is the v1.2.1 default; must still pass."""
    row, features, outputs = _stored_run_with_seeds({})
    current = _current_envelope(features, outputs, {})
    report = verify_run(str(row["run_id"]), row, current_envelope=current)
    assert report["approved"] is True, report
