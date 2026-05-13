# SPDX-License-Identifier: Apache-2.0
"""Tamper-evident empirical validation evidence packs.

Sister module to :mod:`market_regime_engine.fixed_income.evidence_pack` —
the two share the canonical-JSON encoding and the generic
HMAC-SHA256-hex primitive via :mod:`market_regime_engine.evidence_common`,
but layer different higher-level signing schemes on top:

- FI evidence packs sign the canonical JSON of the *pack dataclass*
  under a versioned (``v1`` / ``v2`` / ...) key prefix; the verifier
  routes the signature back to the right key by version.
- Validation evidence packs sign the *whole manifest file* under a
  single key; rotations are handled at the file level (re-sign or
  re-build the pack).

Operators that need per-signal audit trail (every Auto-X firing) want
the FI pack; operators that need per-release-run audit bundle (every
release-gate or back-test sweep) want this one. See
``docs/V1_6_ENGINEERING_PLAN.md`` for the full reconciliation matrix.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from market_regime_engine import __version__
from market_regime_engine.evidence_common import (
    canonical_json as _shared_canonical_json,
)
from market_regime_engine.evidence_common import (
    coerce_for_canonical as _shared_coerce,
)
from market_regime_engine.evidence_common import (
    git_dirty as _shared_git_dirty,
)
from market_regime_engine.evidence_common import (
    git_revision as _shared_git_revision,
)
from market_regime_engine.evidence_common import (
    hmac_sha256_hex,
)
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


def _canonical_json(obj: object, *, version: str = "v2") -> bytes:
    """Local thin wrapper that emits bytes -- calls the shared encoder.

    v1.6 PR-22: the canonical-JSON encoding is shared with the FI
    evidence-pack subpack via :mod:`market_regime_engine.evidence_common`.
    We keep this wrapper to preserve the local byte-returning signature
    that this module's manifest writer expects (the shared helper
    returns a ``str``).

    v1.6.0 (REVIEW_DEEP_V1_5_2.md section 2.5): defaults to the RFC 8785
    encoder (``version="v2"``) so a new validation evidence pack
    produces cross-language verifiable canonical bytes. Floats,
    non-ASCII strings, datetime / Decimal / Path values are pre-coerced
    via :func:`evidence_common.coerce_for_canonical` so the strict
    encoder's reject-on-non-native rule does not surprise existing
    manifest builders. ``version="v1"`` is retained for tests that
    need byte-identical reproduction of v1.5.x packs.
    """
    if version == "v2":
        obj = _shared_coerce(obj)
    return _shared_canonical_json(obj, version=version).encode("utf-8")  # type: ignore[arg-type]


_git_revision = _shared_git_revision
_git_dirty = _shared_git_dirty


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


def _redact_command_line(command_line: Sequence[str], *, absolute_source_map: bool) -> list[str]:
    if absolute_source_map:
        return [str(arg) for arg in command_line]
    redacted: list[str] = []
    for arg in command_line:
        text = str(arg)
        try:
            candidate = Path(text).expanduser()
        except Exception:
            redacted.append(text)
            continue
        if candidate.is_absolute() or candidate.exists():
            redacted.append(candidate.name or "<path>")
        else:
            redacted.append(text)
    return redacted


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

    raw_command_line = list(command_line) if command_line is not None else sys.argv
    manifest = {
        "schema": "mre.validation_evidence_pack.v2",
        # v1.6.0 (REVIEW_DEEP_V1_5_2.md section 2.5): canonical-JSON
        # encoder version stamped into the manifest. New packs default to
        # v2 (RFC 8785). Verifier hashes the file bytes verbatim so the
        # stamp is informational -- a cross-language verifier can read it
        # to choose the right re-derivation algorithm if it ever wants
        # to recompute the manifest hash from the underlying artifact
        # set.
        "canonical_version": "v2",
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
            "command_line": _redact_command_line(raw_command_line, absolute_source_map=absolute_source_map),
            "command_line_redacted": not absolute_source_map,
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
        sig = hmac_sha256_hex(key.encode("utf-8"), manifest_path.read_bytes())
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
            actual_sig = hmac_sha256_hex(key.encode("utf-8"), manifest_path.read_bytes())
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
