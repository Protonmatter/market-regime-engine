# SPDX-License-Identifier: Apache-2.0
"""Fixed-Income evidence-pack construction (PR-1 scaffolding).

Per ``MRE_FIXED_INCOME_AGENT.md §"FixedIncomeEvidencePack"``: every FI
signal that goes external must be reproducible from a tamper-evident
pack. PR-1 ships hash construction + verification helpers; PR-7
hardens by adding HMAC sign/verify around the same canonical
bytestream and threads ``data_vintages`` capture through the warehouse
read path.

Until PR-7 lands, ``hmac_signature`` is accepted as ``None`` and the
``verify_pack_hash`` helper covers integrity only (not authenticity).

The canonical bytestream for hashing is the AGENT.md hashing rule
(``canonical_json``); the pack-hash is the SHA-256 of that bytestream
**excluding** the ``hmac_signature`` field so future HMAC re-signing
under a rotated key does not invalidate the historical hash.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from market_regime_engine.fixed_income.hashing import canonical_json, canonical_sha256
from market_regime_engine.fixed_income.schemas import FixedIncomeEvidencePack

_HMAC_FIELD = "hmac_signature"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _python_version() -> str:
    return ".".join(str(x) for x in sys.version_info[:3])


def _pack_to_canonical_dict(pack: FixedIncomeEvidencePack) -> dict[str, Any]:
    """Return the pack as a canonical dict, dropping ``hmac_signature``.

    The HMAC signature wraps the canonical bytestream, so it cannot
    itself participate in the hash without making the construction
    fixed-point. AGENT.md §"Hashing rules" spells this out: "HMAC
    signing should sign the canonical pack JSON excluding
    ``hmac_signature`` itself."
    """
    raw = asdict(pack)
    raw.pop(_HMAC_FIELD, None)
    return raw


def build_evidence_pack(
    *,
    model_run_id: str,
    component_name: str,
    model_version: str,
    code_sha: str | None,
    model_hash: str,
    input_features_hash: str,
    output_hash: str,
    release_gate: bool,
    data_vintages: dict[str, Any] | None = None,
    validation_results: dict[str, Any] | None = None,
    random_seeds: dict[str, Any] | None = None,
    lockfile_hash: str | None = None,
    hmac_signature: str | None = None,
    metadata: dict[str, Any] | None = None,
    timestamp: str | None = None,
    python_version: str | None = None,
) -> FixedIncomeEvidencePack:
    """Construct a :class:`FixedIncomeEvidencePack`.

    Optional fields fall back to safe defaults: empty dicts for the
    nested dict fields, current UTC ISO-8601 for ``timestamp``, and
    the running interpreter's ``sys.version_info`` for
    ``python_version`` so PR-7 ``verify-run`` can detect a Python
    version drift after the fact.

    ``hmac_signature`` defaults to ``None`` in PR-1; PR-7's
    ``sign_pack`` returns a new pack with the signature attached.
    """
    return FixedIncomeEvidencePack(
        model_run_id=model_run_id,
        component_name=component_name,
        model_version=model_version,
        timestamp=timestamp or _now_iso(),
        code_sha=code_sha,
        model_hash=model_hash,
        input_features_hash=input_features_hash,
        output_hash=output_hash,
        data_vintages=dict(data_vintages or {}),
        validation_results=dict(validation_results or {}),
        release_gate=bool(release_gate),
        random_seeds=dict(random_seeds or {}),
        python_version=python_version or _python_version(),
        lockfile_hash=lockfile_hash,
        hmac_signature=hmac_signature,
        metadata=dict(metadata or {}),
    )


def compute_pack_hash(pack: FixedIncomeEvidencePack) -> str:
    """Return ``"sha256:..."`` over the canonical JSON of the pack.

    The ``hmac_signature`` field is excluded so the same pack can be
    re-signed under a rotated HMAC key without invalidating the
    artifact hash. PR-7 ``verify-run`` cross-references this hash
    against the stored value in ``fixed_income_evidence_packs``.
    """
    return canonical_sha256(_pack_to_canonical_dict(pack))


def canonical_pack_payload(pack: FixedIncomeEvidencePack) -> str:
    """Return the exact canonical JSON bytestream used for hashing/signing.

    Exposed so PR-7 HMAC signing can compute ``hmac.digest(key, body)``
    over the same bytes that :func:`compute_pack_hash` SHA-256s. Keeps
    the two derivations in lockstep without callers needing to know
    which dict keys are excluded.
    """
    return canonical_json(_pack_to_canonical_dict(pack))


def verify_pack_hash(pack: FixedIncomeEvidencePack, expected_hash: str) -> bool:
    """Recompute the pack's canonical hash and compare to ``expected_hash``.

    Returns ``True`` iff the recomputed hash matches verbatim (case-
    insensitive on the hex tail; the ``"sha256:"`` prefix must match
    exactly). PR-7 introduces an HMAC verify variant on top of this
    integrity check.
    """
    computed = compute_pack_hash(pack)
    return computed.lower() == expected_hash.lower()


__all__ = [
    "build_evidence_pack",
    "canonical_pack_payload",
    "compute_pack_hash",
    "verify_pack_hash",
]
