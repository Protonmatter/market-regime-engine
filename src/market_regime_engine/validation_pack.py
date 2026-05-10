# SPDX-License-Identifier: Apache-2.0
"""Tamper-evident empirical validation evidence packs."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from market_regime_engine import __version__
from market_regime_engine.production import is_production_env

MANIFEST_NAME = "manifest.json"
MANIFEST_HASH_NAME = "manifest.sha256"
MANIFEST_HMAC_NAME = "manifest.hmac.sha256"
_CONTROL_FILES = {MANIFEST_NAME, MANIFEST_HASH_NAME, MANIFEST_HMAC_NAME}


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


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(["git", "status", "--porcelain"], check=False, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        return bool(result.stdout.strip())
    except Exception:
        return None


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                yield child


def _safe_rmtree_target(path: Path) -> None:
    resolved = path.resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    try:
        repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.DEVNULL)
        forbidden.add(Path(repo_root.strip()).resolve())
    except Exception:
        pass
    if resolved in forbidden or len(resolved.parts) < 3:
        raise ValueError(f"refusing to delete unsafe evidence-pack path: {resolved}")


def _copy_inputs(includes: Iterable[str | Path], pack_dir: Path, *, absolute_source_map: bool) -> list[dict]:
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
            source = str(src) if absolute_source_map else src.name
            copied.append({"source": source, "path": str(dest.relative_to(pack_dir)).replace(os.sep, "/")})
        else:
            root_dest = artifacts_dir / src.name
            for file_path in _iter_files(src):
                rel = file_path.relative_to(src)
                dest = root_dest / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, dest)
                source = str(file_path) if absolute_source_map else str(Path(src.name) / rel)
                copied.append({"source": source, "path": str(dest.relative_to(pack_dir)).replace(os.sep, "/")})
    return copied


def _lockfile_hashes(lockfiles: Sequence[str | Path] | None = None) -> dict[str, str]:
    candidates = [Path(p) for p in lockfiles] if lockfiles else sorted(Path.cwd().glob("requirements-lock*.txt"))
    out: dict[str, str] = {}
    for candidate in candidates:
        path = candidate.expanduser()
        if path.exists() and path.is_file():
            out[str(path.name)] = _sha256_file(path.resolve())
    return out


def _observed_payload_files(pack_dir: Path) -> set[str]:
    observed: set[str] = set()
    for file_path in _iter_files(pack_dir):
        rel = str(file_path.relative_to(pack_dir)).replace(os.sep, "/")
        if rel in _CONTROL_FILES:
            continue
        observed.add(rel)
    return observed


def build_evidence_pack(
    *,
    includes: Iterable[str | Path],
    out_dir: str | Path,
    metadata: Mapping[str, object] | None = None,
    force: bool = False,
    hmac_key: str | None = None,
    require_signed: bool | None = None,
    absolute_source_map: bool = False,
    lockfiles: Sequence[str | Path] | None = None,
    command_line: Sequence[str] | None = None,
) -> EvidencePackResult:
    """Build a tamper-evident evidence pack from selected files/directories."""

    pack_dir = Path(out_dir).expanduser().resolve()
    if pack_dir.exists():
        if not force:
            raise FileExistsError(f"evidence pack already exists: {pack_dir}")
        _safe_rmtree_target(pack_dir)
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)

    require_signed = is_production_env() if require_signed is None else bool(require_signed)
    key = hmac_key if hmac_key is not None else os.getenv("MRE_EVIDENCE_HMAC_KEY", "")
    if require_signed and not key:
        raise RuntimeError("signed evidence pack required but no HMAC key was provided")

    copied = _copy_inputs(includes, pack_dir, absolute_source_map=absolute_source_map)
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
        "schema": "mre.validation_evidence_pack.v2",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "engine_version": __version__,
        "git_sha": _git_revision(short=False),
        "git_short_sha": _git_revision(short=True),
        "git_dirty": _git_dirty(),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "command_line": list(command_line) if command_line is not None else sys.argv,
        },
        "lockfile_hashes": _lockfile_hashes(lockfiles),
        "file_count": len(file_entries),
        "files": file_entries,
        "source_map_redacted": not absolute_source_map,
        "source_map": copied,
        "metadata": dict(metadata or {}),
    }

    manifest_path = pack_dir / MANIFEST_NAME
    manifest_path.write_bytes(_canonical_json(manifest) + b"\n")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (pack_dir / MANIFEST_HASH_NAME).write_text(f"{manifest_hash}  {MANIFEST_NAME}\n", encoding="utf-8")

    signed = False
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


def verify_evidence_pack(path: str | Path, *, hmac_key: str | None = None, require_signed: bool = False) -> dict:
    """Verify an evidence pack's manifest hash, file hashes, optional HMAC, and extra-file state."""

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
        differences["manifest_hash"] = {"expected": expected_manifest_hash, "actual": actual_manifest_hash}

    declared_files = manifest.get("files", [])
    declared_paths = {str(entry["path"]).replace(os.sep, "/") for entry in declared_files}
    for entry in declared_files:
        rel = str(entry["path"]).replace(os.sep, "/")
        file_path = pack_dir / rel
        if not file_path.exists():
            differences[f"missing:{rel}"] = {"expected_sha256": entry.get("sha256")}
            continue
        actual = _sha256_file(file_path)
        if actual != entry.get("sha256"):
            differences[f"sha256:{rel}"] = {"expected": entry.get("sha256"), "actual": actual}
        size = file_path.stat().st_size
        if size != int(entry.get("size_bytes", -1)):
            differences[f"size:{rel}"] = {"expected": entry.get("size_bytes"), "actual": size}

    observed_paths = _observed_payload_files(pack_dir)
    for rel in sorted(observed_paths - declared_paths):
        differences[f"extra:{rel}"] = {"actual_sha256": _sha256_file(pack_dir / rel)}

    if int(manifest.get("file_count", -1)) != len(declared_files):
        differences["file_count"] = {"expected": manifest.get("file_count"), "actual": len(declared_files)}

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
    elif require_signed:
        differences["hmac"] = {"error": "signature required but manifest.hmac.sha256 is missing"}

    return {
        "approved": not differences,
        "schema": manifest.get("schema"),
        "path": str(pack_dir),
        "manifest_hash": actual_manifest_hash,
        "file_count": len(declared_files),
        "signed": signed,
        "git_dirty": manifest.get("git_dirty"),
        "lockfile_hashes": manifest.get("lockfile_hashes", {}),
        "differences": differences,
    }


__all__ = ["EvidencePackResult", "build_evidence_pack", "verify_evidence_pack"]
