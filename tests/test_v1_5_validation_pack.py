from __future__ import annotations

from pathlib import Path

from market_regime_engine.validation_pack import build_evidence_pack, verify_evidence_pack


def test_evidence_pack_build_verify_and_detect_tamper(tmp_path) -> None:
    artifact = tmp_path / "validation.json"
    artifact.write_text('{"brier": 0.12}\n', encoding="utf-8")

    pack = build_evidence_pack(
        includes=[artifact],
        out_dir=tmp_path / "pack",
        metadata={"model": "candidate"},
        hmac_key="secret",
    )
    assert pack.file_count == 1
    assert pack.signed is True

    ok = verify_evidence_pack(pack.path, hmac_key="secret")
    assert ok["approved"] is True
    assert ok["signed"] is True

    copied = Path(pack.path) / "artifacts" / "validation.json"
    copied.write_text('{"brier": 0.99}\n', encoding="utf-8")

    bad = verify_evidence_pack(pack.path, hmac_key="secret")
    assert bad["approved"] is False
    assert any(key.startswith("sha256:") for key in bad["differences"])
