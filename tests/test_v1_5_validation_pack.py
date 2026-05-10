from __future__ import annotations

from pathlib import Path

import pytest

from market_regime_engine.validation_pack import build_evidence_pack, verify_evidence_pack


def test_evidence_pack_build_verify_and_detect_tamper(tmp_path) -> None:
    artifact = tmp_path / "validation.json"
    artifact.write_text('{"brier": 0.12}\n', encoding="utf-8")

    pack = build_evidence_pack(
        includes=[artifact],
        out_dir=tmp_path / "pack",
        metadata={"model": "candidate"},
        hmac_key="secret",
        require_signed=True,
    )
    assert pack.file_count == 1
    assert pack.signed is True

    ok = verify_evidence_pack(pack.path, hmac_key="secret", require_signed=True)
    assert ok["approved"] is True
    assert ok["signed"] is True

    copied = Path(pack.path) / "artifacts" / "validation.json"
    copied.write_text('{"brier": 0.99}\n', encoding="utf-8")

    bad = verify_evidence_pack(pack.path, hmac_key="secret", require_signed=True)
    assert bad["approved"] is False
    assert any(key.startswith("sha256:") for key in bad["differences"])


def test_evidence_pack_detects_extra_payload_files(tmp_path) -> None:
    artifact = tmp_path / "validation.json"
    artifact.write_text("{}\n", encoding="utf-8")
    pack = build_evidence_pack(includes=[artifact], out_dir=tmp_path / "pack", hmac_key="secret")
    extra = Path(pack.path) / "artifacts" / "extra.json"
    extra.write_text("{}\n", encoding="utf-8")
    report = verify_evidence_pack(pack.path, hmac_key="secret")
    assert report["approved"] is False
    assert "extra:artifacts/extra.json" in report["differences"]


def test_evidence_pack_requires_signature_when_requested(tmp_path) -> None:
    artifact = tmp_path / "validation.json"
    artifact.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        build_evidence_pack(includes=[artifact], out_dir=tmp_path / "pack", require_signed=True)


def test_evidence_pack_redacts_absolute_source_paths_by_default(tmp_path) -> None:
    artifact = tmp_path / "validation.json"
    artifact.write_text("{}\n", encoding="utf-8")
    pack = build_evidence_pack(includes=[artifact], out_dir=tmp_path / "pack", hmac_key="secret")
    manifest = Path(pack.manifest_path).read_text(encoding="utf-8")
    assert str(tmp_path) not in manifest
