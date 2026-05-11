# SPDX-License-Identifier: Apache-2.0
"""PR-1 acceptance tests for the 5-lockfile-hashes dict (REVIEW.md AF-9)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from market_regime_engine.model_runs import (
    _LOCKFILE_FILES,
    _lockfile_hash,
    _lockfile_hashes_dict,
    build_repro_envelope,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _seed_lockfile(root: Path, name: str, content: bytes) -> None:
    (root / name).write_bytes(content)


def test_lockfile_hashes_dict_includes_all_five_lockfiles(tmp_path: Path) -> None:
    contents = {name: f"content-{name}".encode() for name in _LOCKFILE_FILES}
    for name, payload in contents.items():
        _seed_lockfile(tmp_path, name, payload)

    out = _lockfile_hashes_dict(root=tmp_path)
    assert set(out.keys()) == set(_LOCKFILE_FILES)
    for name, payload in contents.items():
        assert out[name] == _sha256_bytes(payload)


def test_missing_lockfile_maps_to_none(tmp_path: Path) -> None:
    _seed_lockfile(tmp_path, "requirements-lock.txt", b"only-canonical-present")
    out = _lockfile_hashes_dict(root=tmp_path)
    assert out["requirements-lock.txt"] is not None
    for missing in _LOCKFILE_FILES[1:]:
        assert out[missing] is None, missing


def test_legacy_scalar_lockfile_hash_still_emitted_for_back_compat(
    tmp_path: Path,
) -> None:
    """``_lockfile_hash`` (scalar) and ``_lockfile_hashes_dict`` must agree
    on the canonical ``requirements-lock.txt`` entry."""
    _seed_lockfile(tmp_path, "requirements-lock.txt", b"identical-canonical-payload")
    scalar = _lockfile_hash(root=tmp_path)
    dict_form = _lockfile_hashes_dict(root=tmp_path)
    assert scalar == dict_form["requirements-lock.txt"]


def test_envelope_carries_both_scalar_and_dict() -> None:
    """``build_repro_envelope`` populates both ``lockfile_hash`` and
    ``lockfile_hashes`` so v1.4 verify-run consumers keep working."""
    env = build_repro_envelope(features=pd.DataFrame(), model_outputs=pd.DataFrame())
    # The dict carries at least the canonical entry (real repo has the
    # file checked in); missing-platform entries can be None.
    assert isinstance(env.lockfile_hashes, dict)
    assert "requirements-lock.txt" in env.lockfile_hashes
    # Scalar/dict agree.
    assert env.lockfile_hash == env.lockfile_hashes.get("requirements-lock.txt", "")


def test_extra_lockfile_via_glob_is_captured(tmp_path: Path) -> None:
    """A novel ``requirements-lock.gpu-linux.txt`` is picked up by glob."""
    _seed_lockfile(tmp_path, "requirements-lock.txt", b"core")
    _seed_lockfile(tmp_path, "requirements-lock.gpu-linux.txt", b"gpu-platform")
    out = _lockfile_hashes_dict(root=tmp_path)
    assert "requirements-lock.gpu-linux.txt" in out
    assert out["requirements-lock.gpu-linux.txt"] == _sha256_bytes(b"gpu-platform")
