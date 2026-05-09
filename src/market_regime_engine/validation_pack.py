# SPDX-License-Identifier: Apache-2.0
"""Tamper-evident empirical validation evidence packs.

The evidence pack is intentionally boring: copy selected artifacts into a
single directory, hash every byte, write a canonical manifest, then hash and
optionally HMAC-sign that manifest. It does not make results true. It makes
quiet post-hoc edits obvious, which is the best software can do while humans
continue trying to negotiate with probability.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Mapping

from market_regime_engine import __version__

MANIFEST_NAME = "manifest.json"
MANIFEST_HASH_NAME = "manifest.sha256"
MANIFEST_HMAC_NAME = "manifest.hmac.sha256"


@dataclass(frozen=True)
class EvidencePackResult:
    path: str
    manifest_path: str
    manifest_hash: str
    file_count: int
    signed: bool


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _git_revision(short: bool = False) -> str:
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                yield child


def _copy_inputs(includes: Iterable[str | Path], pack_dir: Path) -> list[dict]:
    artifacts_dir = pack_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict] = []
    for include in includes:
        src = Path(include).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"evidence input does not exist: {src}")
        if src.is_file():
            dest = artifacts_dir / src.name
            if dest.exists():
                dest = artifacts_dir / f"{src.stem}.{hashlib.sha256(str(src).encode()).hexdigest()[:8]}{src.suffix}"
            shutil.copy2(src, dest)
            copied.append({"source": str(src), "path": str(dest.relative_to(pack_dir))})
        else:
            root_dest = artifacts_dir / src.name
            for file_path in _iter_files(src):
                rel = file_path.relative_to(src)
                dest = root_dest / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest)
                copied.append({"source": str(file_path), "path": str(dest.relative_to(pack_dir))})
    return copied


def build_evidence_pack(
    *,
    includes: Iterable[str | Path],
    out_dir: str | Path,
    metadata: Mapping[str, object] | None = None,
    force: bool = False,
    hmac_key: str | None = None,
) -> EvidencePackResult:
    """Build a tamper-evident evidence pack from selected files/directories."""

    pack_dir = Path(out_dir).expanduser().resolve()
    if pack_dir.exists():
        if not force:
            raise FileExistsError(f"evidence pack already exists: {pack_dir}")
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)

    copied = _copy_inputs(includes, pack_dir)
    file_entries: list[dict] = []
    for rel in sorted(item["path"] for item in copied):
        path = pack_dir / rel
        file_entries.append(
            {
                "path": rel.replace(os.sep, "/"),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )

    manifest = {
        "schema": "mre.validation_evidence_pack.v1",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "engine_version": __version__,
        "git_sha": _git_revision(short=False),
        "git_short_sha": _git_revision(short=True),
        "file_count": len(file_entries),
        "files": file_entries,
        "source_map": copied,
        "metadata": dict(metadata or {}),
    }

    manifest_path = pack_dir / MANIFEST_NAME
    manifest_path.write_bytes(_canonical_json(manifest) + b"\n")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (pack_dir / MANIFEST_HASH_NAME).write_text(f"{manifest_hash}  {MANIFEST_NAME}\n", encoding="utf-8")

    signed = False
    key = hmac_key if hmac_key is not None else os.getenv("MRE_EVIDENCE_HMAC_KEY", "")
    if key:
        sig = hmac.new(key.encode("utf-8"), manifest_path.read_bytes(), hashlib.sha256).hexdigest()
        (pack_dir / MANIFEST_HMAC_NAME).write_text(f"{sig}  {MANIFEST_NAME}\n", encoding="utf-8")
        signed = True

    return EvidencePackResult(
        path=str(pack_dir),
        manifest_path=str(manifest_path),
        manifest_hash=manifest_hash,
        file_count=len(file_entries),
        signed=signed,
    )


def verify_evidence_pack(path: str | Path, *, hmac_key: str | None = None) -> dict:
    """Verify an evidence pack's manifest hash, file hashes, and optional HMAC."""

    pack_dir = Path(path).expanduser().resolve()
    manifest_path = pack_dir / MANIFEST_NAME
    hash_path = pack_dir / MANIFEST_HASH_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {MANIFEST_NAME}: {manifest_path}")
    if not hash_path.exists():
        raise FileNotFoundError(f"missing {MANIFEST_HASH_NAME}: {hash_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_hash = hash_path.read_text(encoding="utf-8").split()[0]
    actual_manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    differences: dict[str, object] = {}
    if expected_manifest_hash != actual_manifest_hash:
        differences["manifest_hash"] = {
            "expected": expected_manifest_hash,
            "actual": actual_manifest_hash,
        }

    for entry in manifest.get("files", []):
        rel = str(entry["path"])
        file_path = pack_dir / rel
        if not file_path.exists():
            differences[f"missing:{rel}"] = {"expected_sha256": entry.get("sha256")}
            continue
        actual = _sha256_file(file_path)
        if actual != entry.get("sha256"):
            differences[f"sha256:{rel}"] = {
                "expected": entry.get("sha256"),
                "actual": actual,
            }
        size = file_path.stat().st_size
        if size != int(entry.get("size_bytes", -1)):
            differences[f"size:{rel}"] = {
                "expected": entry.get("size_bytes"),
                "actual": size,
            }

    signed = False
    hmac_path = pack_dir / MANIFEST_HMAC_NAME
    key = hmac_key if hmac_key is not None else os.getenv("MRE_EVIDENCE_HMAC_KEY", "")
    if hmac_path.exists():
        signed = True
        if not key:
            differences["hmac"] = {"error": "pack is signed but no HMAC key was provided"}
        else:
            expected_sig = hmac_path.read_text(encoding="utf-8").split()[0]
            actual_sig = hmac.new(key.encode("utf-8"), manifest_path.read_bytes(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected_sig, actual_sig):
                differences["hmac"] = {"expected": expected_sig, "actual": actual_sig}

    return {
        "approved": not differences,
        "schema": manifest.get("schema"),
        "path": str(pack_dir),
        "manifest_hash": actual_manifest_hash,
        "file_count": len(manifest.get("files", [])),
        "signed": signed,
        "differences": differences,
    }


__all__ = [
    "EvidencePackResult",
    "build_evidence_pack",
    "verify_evidence_pack",
]
