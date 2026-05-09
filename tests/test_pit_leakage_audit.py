# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pandas as pd

from market_regime_engine.cli_dispatch import main as cli_main
from market_regime_engine.leakage_checks import audit_pit_frames
from market_regime_engine.snapshot_manifest import (
    build_snapshot_manifest,
    verify_snapshot_manifest,
    write_snapshot_manifest,
)


def _features(**overrides: object) -> pd.DataFrame:
    data: dict[str, object] = {
        "series_id": ["cpi"],
        "entity_id": ["US"],
        "forecast_origin": ["2020-02-01T00:00:00Z"],
        "observation_date": ["2020-01-01T00:00:00Z"],
        "observed_at": ["2020-01-15T00:00:00Z"],
        "available_at": ["2020-01-20T00:00:00Z"],
        "as_of": ["2020-01-25T00:00:00Z"],
        "value": [1.0],
        "source": ["fixture"],
        "source_revision_id": ["rev-1"],
        "source_revision_available_at": ["2020-01-22T00:00:00Z"],
        "snapshot_id": ["snap-1"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def _labels(**overrides: object) -> pd.DataFrame:
    data: dict[str, object] = {
        "entity_id": ["US"],
        "forecast_origin": ["2020-02-01T00:00:00Z"],
        "label_time": ["2020-05-01T00:00:00Z"],
        "horizon": ["3m"],
        "target": ["drawdown_gt_10pct"],
        "label_value": [0],
        "label_available_at": ["2020-05-02T00:00:00Z"],
        "joined_at": ["2020-05-03T00:00:00Z"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_clean_point_in_time_frames_pass() -> None:
    report = audit_pit_frames(_features(), _labels())

    assert report.passed is True, report.to_json()
    assert report.matched_pairs == 1
    assert report.summary()["blockers"] == 0


def test_feature_as_of_after_forecast_origin_fails() -> None:
    report = audit_pit_frames(
        _features(as_of=["2020-02-02T00:00:00Z"]),
        _labels(),
    )

    assert report.passed is False
    checks = {issue.check for issue in report.issues}
    assert "feature_as_of_lte_forecast_origin" in checks


def test_label_joined_before_available_fails() -> None:
    report = audit_pit_frames(
        _features(),
        _labels(joined_at=["2020-05-01T00:00:00Z"]),
    )

    assert report.passed is False
    checks = {issue.check for issue in report.issues}
    assert "label_available_lte_join_time" in checks or "label_joined_before_available" in checks


def test_revision_used_before_available_fails() -> None:
    report = audit_pit_frames(
        _features(source_revision_available_at=["2020-01-30T00:00:00Z"]),
        _labels(),
    )

    assert report.passed is False
    checks = {issue.check for issue in report.issues}
    assert "revision_available_lte_as_of" in checks or "vintage_revision_used_before_available" in checks


def test_snapshot_manifest_round_trip_and_detects_mutation(tmp_path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    (root / "a.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (root / "b.txt").write_text("hello\n", encoding="utf-8")

    manifest = build_snapshot_manifest(root, snapshot_id="fixture-snapshot")
    manifest_path = tmp_path / "manifest.json"
    write_snapshot_manifest(manifest, manifest_path)

    ok = verify_snapshot_manifest(manifest_path)
    assert ok.passed is True, ok.to_json()
    assert ok.checked_files == 2

    (root / "b.txt").write_text("hello, mutated universe\n", encoding="utf-8")
    bad = verify_snapshot_manifest(manifest_path)
    assert bad.passed is False
    assert any(issue.check in {"sha256", "size_bytes"} for issue in bad.issues)


def test_cli_pit_audit_writes_json_and_enforces_failure(tmp_path) -> None:
    features_path = tmp_path / "features.csv"
    labels_path = tmp_path / "labels.csv"
    out_json = tmp_path / "pit_report.json"

    _features(as_of=["2020-02-02T00:00:00Z"]).to_csv(features_path, index=False)
    _labels().to_csv(labels_path, index=False)

    rc = cli_main(
        [
            "pit-audit",
            "--features",
            str(features_path),
            "--labels",
            str(labels_path),
            "--out-json",
            str(out_json),
            "--fail-on-leakage",
        ]
    )

    assert rc == 2
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["summary"]["blockers"] > 0


def test_cli_snapshot_build_and_verify_enforce_mismatch(tmp_path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    source = root / "input.csv"
    source.write_text("x,y\n1,2\n", encoding="utf-8")
    manifest_path = tmp_path / "snapshot.json"

    build_rc = cli_main(["snapshot-build", "--input", str(root), "--out", str(manifest_path)])
    assert build_rc == 0
    assert manifest_path.exists()

    ok_rc = cli_main(["snapshot-verify", "--manifest", str(manifest_path), "--fail-on-mismatch"])
    assert ok_rc == 0

    source.write_text("x,y\n9,9\n", encoding="utf-8")
    bad_rc = cli_main(["snapshot-verify", "--manifest", str(manifest_path), "--fail-on-mismatch"])
    assert bad_rc == 2
