# SPDX-License-Identifier: Apache-2.0
"""Canonical-JSON SHA-256 hashing — back-compat alias for ``evidence_common``.

v1.6 PR-22: the canonical-JSON and SHA-256 helpers moved up to
:mod:`market_regime_engine.evidence_common` so the FI evidence-pack
subpack and the validation evidence-pack subpack share one
implementation. This module remains as a thin re-export shim so the
existing import path

::

    from market_regime_engine.fixed_income.hashing import canonical_json

keeps working verbatim. New code should import from
``market_regime_engine.evidence_common`` directly.

Hashing rule (per ``MRE_FIXED_INCOME_AGENT.md §"Hashing rules"``)::

    json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

The ``sha256`` digest is returned with the ``"sha256:"`` prefix so the
hash field is self-describing on the wire (e.g.
``"sha256:c0ffee..."``). PR-7 introduced HMAC signing on top of this
canonical bytestream; the HMAC helper now also lives in
``evidence_common`` (as ``hmac_sha256_hex``).
"""

from __future__ import annotations

from market_regime_engine.evidence_common import (
    canonical_json,
    canonical_sha256,
    strip_hash_prefix,
)

__all__ = ["canonical_json", "canonical_sha256", "strip_hash_prefix"]
