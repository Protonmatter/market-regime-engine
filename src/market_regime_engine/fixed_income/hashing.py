# SPDX-License-Identifier: Apache-2.0
"""Canonical-JSON SHA-256 hashing for FI evidence packs and signals.

The canonical form mirrors :func:`model_runs.envelope_to_json`
(``model_runs.py:267-272``, REVIEW flag F-7) so the FI hashing path and
the macro repro-envelope path produce identical bytes for identical
inputs. This is what lets :func:`canonical_sha256` be reused for both
``ReproEnvelope`` artifact hashes and FI evidence-pack artifact hashes
without keeping two separate canonicalisation rules in sync.

Hashing rule (per ``MRE_FIXED_INCOME_AGENT.md §"Hashing rules"``)::

    json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

The ``sha256`` digest is returned with the ``"sha256:"`` prefix so the
hash field is self-describing on the wire (e.g.
``"sha256:c0ffee..."``). PR-7 introduces HMAC signing on top of this
canonical bytestream.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

_HASH_PREFIX = "sha256:"


def canonical_json(payload: Any) -> str:
    """Canonical JSON encoding for hashing / HMAC inputs.

    Mirrors the v1.0 ``model_runs.envelope_to_json`` shape (and the
    AGENT.md §"Hashing rules" rule) so callers can hash arbitrary
    dict / dataclass / list payloads and get a stable byte sequence
    across runs and Python minor versions. ``default=str`` ensures
    ``datetime``, ``Decimal``, ``Path``, and other non-JSON-native
    types round-trip via ``str(...)`` instead of raising.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def canonical_sha256(payload: Any) -> str:
    """Return ``"sha256:" + hex(sha256(canonical_json(payload)))``.

    The prefix lets downstream consumers detect (and migrate from) the
    legacy bare-hex form without ambiguity. PR-7 HMAC signing also uses
    this canonical bytestream as the message body, excluding the
    ``hmac_signature`` field of the evidence pack itself.
    """
    data = canonical_json(payload).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    return f"{_HASH_PREFIX}{digest}"


def strip_hash_prefix(value: str) -> str:
    """Return the hex digest with the ``"sha256:"`` prefix removed."""
    if value.startswith(_HASH_PREFIX):
        return value[len(_HASH_PREFIX) :]
    return value


__all__ = ["canonical_json", "canonical_sha256", "strip_hash_prefix"]
