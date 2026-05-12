# SPDX-License-Identifier: Apache-2.0
"""Shared primitives for the engine's two evidence-pack subpackages.

The engine ships two complementary evidence-pack systems:

- :mod:`market_regime_engine.fixed_income.evidence_pack` — per-signal audit
  trail. Every FI Auto-X firing produces a single dataclass-shaped pack
  signed with a versioned HMAC (``v1`` / ``v2`` / ...) over the canonical
  JSON bytestream of the pack itself.
- :mod:`market_regime_engine.validation_pack` — per-release-run audit
  bundle. Each validation run produces a directory of artifacts plus a
  ``manifest.json`` and an optional whole-file HMAC over that manifest.

The two packs serve different consumers (one row of audit trail per
signal vs one directory bundle per release run) and therefore have
different HMAC scopes. They share two primitives — canonical JSON
encoding and a generic HMAC-SHA256-hex helper — which now live here.
``fixed_income.hashing`` keeps the same public symbols (re-exported
from this module) so existing FI callers stay binary-compatible across
the v1.6 refactor.

Migration notes (v1.6 PR-22)::

    # Before (v1.5.x):
    from market_regime_engine.fixed_income.hashing import canonical_json

    # After (v1.6+):
    from market_regime_engine.evidence_common import canonical_json
    # (the fixed_income.hashing path keeps working as a thin alias)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import subprocess
from typing import Any

_HASH_PREFIX = "sha256:"


def canonical_json(payload: Any) -> str:
    """Canonical JSON encoding for hashing / HMAC inputs.

    Mirrors the v1.0 ``model_runs.envelope_to_json`` shape (and the
    ``MRE_FIXED_INCOME_AGENT.md §"Hashing rules"`` rule) so callers can
    hash arbitrary dict / dataclass / list payloads and get a stable
    byte sequence across runs and Python minor versions. ``default=str``
    ensures ``datetime``, ``Decimal``, ``Path``, and other non-JSON-
    native types round-trip via ``str(...)`` instead of raising.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def canonical_sha256(payload: Any) -> str:
    """Return ``"sha256:" + hex(sha256(canonical_json(payload)))``.

    The prefix lets downstream consumers detect (and migrate from) the
    legacy bare-hex form without ambiguity.
    """
    data = canonical_json(payload).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    return f"{_HASH_PREFIX}{digest}"


def strip_hash_prefix(value: str) -> str:
    """Return the hex digest with the ``"sha256:"`` prefix removed."""
    if value.startswith(_HASH_PREFIX):
        return value[len(_HASH_PREFIX) :]
    return value


def hmac_sha256_hex(key: bytes, payload: bytes) -> str:
    """Return the lower-case hex HMAC-SHA256 of ``payload`` under ``key``.

    Generic byte-level helper used by both evidence-pack subpacks. The
    higher-level pack-specific signing layers (versioned HMAC with
    ``v<ver>:`` prefix in FI; whole-manifest HMAC in validation) build
    on top of this primitive.
    """
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def git_revision(short: bool = False) -> str:
    """Return the current git HEAD SHA, or ``"unknown"`` when unavailable.

    Used by both evidence-pack subpacks to stamp the code revision into
    the pack manifest. ``short=True`` returns the abbreviated SHA;
    otherwise the full 40-character SHA is returned. Subprocess errors
    (no git, detached worktree, etc.) degrade to ``"unknown"`` rather
    than raising so a pack build never fails on a CI runner without
    git installed.
    """
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool | None:
    """Return ``True`` when the working tree has uncommitted changes.

    ``False`` when clean; ``None`` when git is unavailable or the call
    fails (so callers can distinguish "definitely clean" from "couldn't
    tell"). Stamped into evidence-pack manifests so a reviewer can
    detect a pack built from a dirty worktree.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


__all__ = [
    "canonical_json",
    "canonical_sha256",
    "git_dirty",
    "git_revision",
    "hmac_sha256_hex",
    "strip_hash_prefix",
]
