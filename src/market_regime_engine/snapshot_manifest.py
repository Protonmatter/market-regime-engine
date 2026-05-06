# SPDX-License-Identifier: Apache-2.0
"""Deterministic snapshot manifests for raw market-data inputs.

A snapshot manifest is the boring receipt for a dataset: every file, byte size,
and SHA-256 hash. When a run claims it used a given raw input snapshot, this is
how we make that claim auditable instead of vibes-based financial archaeology.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SnapshotFile:
    """One hashed file entry in a snapshot manifest."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class SnapshotManifest:
    """Deterministic manifest for a file or directory snapshot."""

    schema_version: str
    snapshot_id: str
    input_root: str
    files: tuple[SnapshotFile, ...]
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "input_root": self.input_root,
            "files": [asdict(f) for f in self.files],
            "manifest_sha256": self.manifest_sha256,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"


@dataclass(frozen=True)
class SnapshotVerificationIssue:
    path: str
    check: str
    message: str
    expected: str | int | None = None
    actual: str | int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SnapshotVerificationReport:
    """Result of verifying a snapshot manifest against the filesystem."""

    manifest_path: str
    input_root: str
    checked_files: int
    passed: bool
    issues: tuple[SnapshotVerificationIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "input_root": self.input_root,
            "checked_files": self.checked_files,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"# Snapshot Verification — {status}",
            "",
            f"- **manifest_path:** {self.manifest_path}",
            f"- **input_root:** {self.input_root}",
            f"- **checked_files:** {self.checked_files}",
            f"- **issues:** {len(self.issues)}",
            "",
            "## Issues",
            "",
        ]
        if not self.issues:
            lines.append("_No snapshot mismatches detected._")
        else:
            lines.append("| Path | Check | Expected | Actual | Message |")
            lines.append("|---|---|---|---|---|")
            for issue in self.issues:
                lines.append(
                    f"| {issue.path} | {issue.check} | {issue.expected} | {issue.actual} | "
                    f"{issue.message.replace('|', '\\|')} |"
                )
        return "\n".join(lines).rstrip() + "\n"


def build_snapshot_manifest(input_path: str | Path, *, snapshot_id: str | None = None) -> SnapshotManifest:
    """Build a deterministic manifest for a file or directory."""

    root = Path(input_path).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    entries = tuple(_snapshot_file(root, path) for path in _iter_files(root))
    manifest_hash = _manifest_hash(entries)
    return SnapshotManifest(
        schema_version="mre.snapshot_manifest.v1",
        snapshot_id=snapshot_id or manifest_hash[:24],
        input_root=str(root),
        files=entries,
        manifest_sha256=manifest_hash,
    )


def write_snapshot_manifest(manifest: SnapshotManifest, out: str | Path) -> Path:
    """Write a manifest JSON document."""

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json(), encoding="utf-8")
    return path


def load_snapshot_manifest(path: str | Path) -> SnapshotManifest:
    """Load a manifest JSON document."""

    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    files = tuple(SnapshotFile(path=str(f["path"]), size_bytes=int(f["size_bytes"]), sha256=str(f["sha256"])) for f in data["files"])
    return SnapshotManifest(
        schema_version=str(data["schema_version"]),
        snapshot_id=str(data["snapshot_id"]),
        input_root=str(data["input_root"]),
        files=files,
        manifest_sha256=str(data["manifest_sha256"]),
    )


def verify_snapshot_manifest(path: str | Path) -> SnapshotVerificationReport:
    """Verify a manifest against current files."""

    manifest_path = Path(path).resolve()
    manifest = load_snapshot_manifest(manifest_path)
    root = Path(manifest.input_root)
    if not root.is_absolute():
        root = (manifest_path.parent / root).resolve()
    issues: list[SnapshotVerificationIssue] = []

    expected_manifest_hash = _manifest_hash(manifest.files)
    if expected_manifest_hash != manifest.manifest_sha256:
        issues.append(
            SnapshotVerificationIssue(
                path=str(manifest_path),
                check="manifest_sha256",
                message="Manifest content hash does not match manifest_sha256",
                expected=manifest.manifest_sha256,
                actual=expected_manifest_hash,
            )
        )

    for entry in manifest.files:
        file_path = root / entry.path
        if not file_path.exists():
            issues.append(
                SnapshotVerificationIssue(
                    path=entry.path,
                    check="exists",
                    message="Manifest file is missing from snapshot root",
                    expected="present",
                    actual="missing",
                )
            )
            continue
        actual_size = file_path.stat().st_size
        if actual_size != entry.size_bytes:
            issues.append(
                SnapshotVerificationIssue(
                    path=entry.path,
                    check="size_bytes",
                    message="File size differs from manifest",
                    expected=entry.size_bytes,
                    actual=actual_size,
                )
            )
        actual_sha = sha256_file(file_path)
        if actual_sha != entry.sha256:
            issues.append(
                SnapshotVerificationIssue(
                    path=entry.path,
                    check="sha256",
                    message="File SHA-256 differs from manifest",
                    expected=entry.sha256,
                    actual=actual_sha,
                )
            )

    return SnapshotVerificationReport(
        manifest_path=str(manifest_path),
        input_root=str(root),
        checked_files=len(manifest.files),
        passed=not issues,
        issues=tuple(issues),
    )


def sha256_file(path: str | Path) -> str:
    """Hash a file using SHA-256."""

    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix())


def _snapshot_file(root: Path, path: Path) -> SnapshotFile:
    rel = path.name if root.is_file() else path.relative_to(root).as_posix()
    return SnapshotFile(path=rel, size_bytes=int(path.stat().st_size), sha256=sha256_file(path))


def _manifest_hash(files: tuple[SnapshotFile, ...]) -> str:
    payload = json.dumps([asdict(f) for f in files], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "SnapshotFile",
    "SnapshotManifest",
    "SnapshotVerificationIssue",
    "SnapshotVerificationReport",
    "build_snapshot_manifest",
    "load_snapshot_manifest",
    "sha256_file",
    "verify_snapshot_manifest",
    "write_snapshot_manifest",
]
