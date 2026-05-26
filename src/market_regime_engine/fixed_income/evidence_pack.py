# SPDX-License-Identifier: Apache-2.0
"""Fixed-Income evidence-pack construction (PR-1 scaffolding + PR-7 HMAC).

Per ``MRE_FIXED_INCOME_AGENT.md §"FixedIncomeEvidencePack"``: every FI
signal that goes external must be reproducible from a tamper-evident
pack. PR-1 ships hash construction + verification helpers; PR-7
hardens by adding HMAC sign/verify around the same canonical
bytestream and threads ``data_vintages`` capture through the warehouse
read path.

The canonical bytestream for hashing is the AGENT.md hashing rule
(``canonical_json``); the pack-hash is the SHA-256 of that bytestream
**excluding** the ``hmac_signature`` field so future HMAC re-signing
under a rotated key does not invalidate the historical hash.

PR-7 HMAC operations
--------------------

- :func:`get_hmac_keys` parses ``MRE_FI_HMAC_KEY_VERSIONS`` (JSON
  ``{"v1": "<base64>", "v2": "..."}``) — or the singleton
  ``MRE_FI_HMAC_KEY`` (registered as ``v1``) — into a per-version map
  of decoded key bytes.
- :func:`sign_pack` signs the canonical JSON (excluding the
  ``hmac_signature`` field itself) under the latest key version and
  returns a new pack with ``hmac_signature = "v<ver>:<hex>"``.
- :func:`verify_pack` parses the version prefix, looks up the key,
  re-derives the HMAC, and compares with :func:`hmac.compare_digest`
  (constant-time, side-channel-resistant).
- :func:`require_production_hmac` returns True when ``MRE_ENV=production``
  or ``MRE_FI_REQUIRE_HMAC=1``; in those modes :func:`sign_pack` raises
  rather than returning a pack with ``hmac_signature=None`` so a
  production worker cannot accidentally publish unsigned packs.
- :func:`capture_data_vintages` snapshots the latest ``timestamp`` /
  ``source_timestamp`` per FI source table at or below ``asof`` so a
  downstream replay can verify the same vintages were available at
  decision time (review §4.3 data lineage).
"""

from __future__ import annotations

import base64
import hmac
import json
import logging
import os
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from market_regime_engine.evidence_common import (
    CanonicalVersion,
    canonical_json,
    canonical_sha256,
    coerce_for_canonical,
    hmac_sha256_hex,
)
from market_regime_engine.fixed_income.schemas import FixedIncomeEvidencePack

log = logging.getLogger(__name__)

_HMAC_FIELD = "hmac_signature"

# v1.5 PR-8 (Tier-1 fix C-AUTO-1): the canonical SHA-256 of the pack
# (computed via :func:`compute_pack_hash`) is persisted into
# ``pack.metadata`` under this key at write time so ``verify-run`` can
# detect row-level tampering by comparing the recomputed pack hash to
# the stored envelope hash. The key is stripped from
# :func:`_pack_to_canonical_dict` (alongside ``hmac_signature``) so the
# canonical bytestream is invariant under presence/absence of the
# envelope hash — both HMAC signing and pack-hash recomputation produce
# the same bytes whether the envelope hash has been stamped or not.
_ENVELOPE_HASH_METADATA_KEY = "_envelope_hash"

# v1.5.1 (PR-9 FIX 3): when this flag is True in ``pack.metadata`` the
# canonical bytestream binds ``pack.request_id`` into the HMAC payload.
# When absent / False the canonical bytestream excludes ``request_id``
# (the v1.5.0 wire format). The flag itself is part of canonical bytes
# (just regular metadata), so a v2-signed pack that toggles the flag
# would fail verify automatically — the flag is tamper-evident.
#
# Legacy v1.5.0 packs lacked any concept of ``request_id`` binding;
# their metadata does not carry this flag and their HMAC payload does
# not contain ``request_id``. Verification of those packs is therefore
# preserved verbatim under the new code.
_REQUEST_ID_BOUND_METADATA_KEY = "_request_id_bound"

# v1.6.0 (REVIEW_DEEP_V1_5_2.md section 2.5): canonical-JSON encoder
# version. When this metadata key is absent, the pack is encoded under
# the legacy ``json.dumps(default=str)`` form (v1). When set to ``"v2"``
# the pack is encoded under :func:`evidence_common._canonical_json_v2`
# (RFC 8785 / JCS). The key is part of canonical bytes so an attacker
# cannot strip the key to downgrade the encoding silently -- removing it
# changes the byte sequence the verifier will recompute and HMAC
# verification fails. Legacy v1.5.x packs carry no key and continue to
# verify under v1; v1.6.0+ packs default to v2 (see
# :func:`build_evidence_pack`).
_CANONICAL_VERSION_METADATA_KEY = "_canonical_version"

# Env vars (per AGENT.md PR-7 + INSTRUCTIONS.md §10 governance rules).
_HMAC_KEY_VERSIONS_ENV = "MRE_FI_HMAC_KEY_VERSIONS"
_HMAC_KEY_SINGLETON_ENV = "MRE_FI_HMAC_KEY"
_HMAC_ACTIVE_VERSION_ENV = "MRE_FI_HMAC_ACTIVE_VERSION"
_REQUIRE_HMAC_ENV = "MRE_FI_REQUIRE_HMAC"
_ENV_NAME_ENV = "MRE_ENV"
_MIN_PRODUCTION_HMAC_KEY_BYTES = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pack_canonical_version(pack: FixedIncomeEvidencePack) -> CanonicalVersion:
    """Read the canonical-JSON encoder version stamped into a pack.

    v1.6.0 (REVIEW_DEEP_V1_5_2.md section 2.5): legacy packs carry no
    ``_canonical_version`` metadata key and use the v1 encoder; new
    packs stamp ``"v2"`` and use the RFC 8785 encoder. Any other
    value is treated as v1 so a forward-rev pack does not silently
    bypass verification when an older verifier reads it (the
    canonical bytes would mismatch and HMAC verification would fail
    -- but we want the failure to come from the integrity check, not
    from an :class:`KeyError` raised mid-decode).
    """
    metadata = pack.metadata or {}
    if isinstance(metadata, dict):
        value = metadata.get(_CANONICAL_VERSION_METADATA_KEY)
        if value == "v2":
            return "v2"
    return "v1"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _python_version() -> str:
    return ".".join(str(x) for x in sys.version_info[:3])


def _pack_to_canonical_dict(
    pack: FixedIncomeEvidencePack,
    *,
    version: CanonicalVersion = "v1",
) -> dict[str, Any]:
    """Return the pack as a canonical dict, dropping ``hmac_signature``
    and the self-referential envelope-hash key from ``metadata``.

    The HMAC signature wraps the canonical bytestream, so it cannot
    itself participate in the hash without making the construction
    fixed-point. AGENT.md §"Hashing rules" spells this out: "HMAC
    signing should sign the canonical pack JSON excluding
    ``hmac_signature`` itself."

    v1.5 PR-8 (Tier-1 fix C-AUTO-1): ``metadata[_ENVELOPE_HASH_METADATA_KEY]``
    is stamped at write time so :func:`verify-run` can detect tampering
    by recomputing :func:`compute_pack_hash` and comparing to the
    stored value. The key is excluded from the canonical bytestream so
    the same compute_pack_hash result is invariant under whether the
    envelope hash has been stamped or not (avoiding a second fixed
    point on top of the HMAC one).

    v1.5.1 (PR-9 FIX 3): the new optional ``request_id`` field is
    included in the canonical bytestream **iff**
    ``metadata[_request_id_bound]`` is truthy. Legacy v1 packs persisted
    before PR-9 carry no such flag and we drop ``request_id`` entirely
    so the canonical bytes are byte-identical to what the v1.5.0 signer
    produced. New v2 packs built via :func:`build_evidence_pack` with a
    non-``None`` ``request_id`` automatically receive the flag and
    therefore bind the id into the HMAC payload.

    The flag itself is part of canonical bytes (regular metadata), so a
    pack-on-disk that flipped the flag would fail :func:`verify_pack`
    automatically — the binding is tamper-evident in both directions.
    """
    raw = asdict(pack)
    raw.pop(_HMAC_FIELD, None)
    metadata = raw.get("metadata")
    request_id_bound = False
    if isinstance(metadata, dict):
        request_id_bound = bool(metadata.get(_REQUEST_ID_BOUND_METADATA_KEY))
    if not request_id_bound:
        raw.pop("request_id", None)
    if isinstance(metadata, dict) and _ENVELOPE_HASH_METADATA_KEY in metadata:
        raw["metadata"] = {
            k: v for k, v in metadata.items() if k != _ENVELOPE_HASH_METADATA_KEY
        }
    # v1.6.0 (REVIEW_DEEP_V1_5_2.md section 2.5): under the RFC 8785
    # encoder the dict must be JSON-native (no datetime / Decimal /
    # Path / set). ``coerce_for_canonical`` is the documented
    # pre-coercion hook -- it converts datetime -> isoformat, Decimal
    # -> str, Path -> POSIX string, set -> sorted list. The pack's
    # ``timestamp`` is already a string, and all numeric dict fields
    # in evidence packs are JSON-native (float / int) by contract,
    # so the coercion is usually a no-op -- but the hook is the
    # right place to handle a stray non-native value rather than
    # letting the v2 encoder raise from a deep call stack.
    if version == "v2":
        raw = coerce_for_canonical(raw)
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
    request_id: str | None = None,
    canonical_version: CanonicalVersion = "v2",
) -> FixedIncomeEvidencePack:
    """Construct a :class:`FixedIncomeEvidencePack`.

    Optional fields fall back to safe defaults: empty dicts for the
    nested dict fields, current UTC ISO-8601 for ``timestamp``, and
    the running interpreter's ``sys.version_info`` for
    ``python_version`` so PR-7 ``verify-run`` can detect a Python
    version drift after the fact.

    ``hmac_signature`` defaults to ``None`` in PR-1; PR-7's
    ``sign_pack`` returns a new pack with the signature attached.

    v1.5.1 (PR-9 FIX 3): ``request_id`` MUST be set in production for
    execution-confidence packs. Setting it threads the value into the
    HMAC canonical bytestream so a replay of the same
    ``(model_run_id, output_hash)`` under a different request id no
    longer verifies. Legacy v1.5.0 packs were built with
    ``request_id=None`` and continue to verify under the v1 key.

    v1.6.0 (REVIEW_DEEP_V1_5_2.md section 2.5): ``canonical_version``
    selects the canonical-JSON encoder: ``"v2"`` (default for new
    packs) emits RFC 8785 / JCS-conformant bytes -- numbers via
    ECMA-262 Number::toString, raw UTF-8 strings, UTF-16 code-point
    key ordering, NaN/Infinity rejected. ``"v1"`` is the legacy
    ``json.dumps(default=str)`` form retained for verifying historic
    packs (and for tests that need byte-identical reproduction of
    v1.5.x persisted hashes). The choice is stamped into
    ``pack.metadata[_canonical_version]`` so :func:`compute_pack_hash`
    and :func:`verify_pack` route to the same encoder later.
    """
    # v1.5.1 (PR-9 FIX 3): when ``request_id`` is set, stamp the
    # metadata flag that ``_pack_to_canonical_dict`` reads to decide
    # whether to bind ``request_id`` into the HMAC payload. Legacy
    # callers (``request_id=None``) get no flag and the canonical
    # bytestream stays byte-identical to v1.5.0.
    metadata_dict: dict[str, Any] = dict(metadata or {})
    if request_id is not None and not metadata_dict.get(_REQUEST_ID_BOUND_METADATA_KEY):
        metadata_dict[_REQUEST_ID_BOUND_METADATA_KEY] = True
    # v1.6.0 (REVIEW_DEEP_V1_5_2.md section 2.5): stamp the canonical-
    # JSON encoder version into metadata when the new RFC 8785 (``v2``)
    # encoder is used. Legacy callers can opt back to v1 explicitly
    # (``canonical_version="v1"``) and the stamp is omitted so the
    # bytestream stays byte-identical to v1.5.x.
    if canonical_version == "v2":
        metadata_dict[_CANONICAL_VERSION_METADATA_KEY] = "v2"
    # Explicit downgrade by a caller (test fixture) wins -- strip a
    # stale "v2" stamp if the caller passed ``canonical_version="v1"``.
    elif canonical_version == "v1" and metadata_dict.get(_CANONICAL_VERSION_METADATA_KEY) is not None:
        metadata_dict.pop(_CANONICAL_VERSION_METADATA_KEY, None)
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
        metadata=metadata_dict,
        request_id=request_id,
    )


def compute_pack_hash(
    pack: FixedIncomeEvidencePack,
    *,
    version: CanonicalVersion | None = None,
) -> str:
    """Return ``"sha256:..."`` over the canonical JSON of the pack.

    The ``hmac_signature`` field is excluded so the same pack can be
    re-signed under a rotated HMAC key without invalidating the
    artifact hash. PR-7 ``verify-run`` cross-references this hash
    against the stored value in ``fixed_income_evidence_packs``.

    v1.6.0 (REVIEW_DEEP_V1_5_2.md section 2.5): when ``version`` is
    ``None`` (the default) the encoder version is read from
    ``pack.metadata[_canonical_version]``; legacy packs without that
    key fall back to ``"v1"`` so historical verify continues to
    match. Passing an explicit ``version="v2"`` overrides the
    metadata (used by ``fi-evidence-resign --to-version v2`` to
    transition an existing pack to the new encoder).
    """
    resolved = version if version is not None else _pack_canonical_version(pack)
    return canonical_sha256(
        _pack_to_canonical_dict(pack, version=resolved), version=resolved
    )


def canonical_pack_payload(
    pack: FixedIncomeEvidencePack,
    *,
    version: CanonicalVersion | None = None,
) -> str:
    """Return the exact canonical JSON bytestream used for hashing/signing.

    Exposed so PR-7 HMAC signing can compute ``hmac.digest(key, body)``
    over the same bytes that :func:`compute_pack_hash` SHA-256s. Keeps
    the two derivations in lockstep without callers needing to know
    which dict keys are excluded.

    v1.6.0: version-aware -- see :func:`compute_pack_hash` for the
    encoder-version resolution rules.
    """
    resolved = version if version is not None else _pack_canonical_version(pack)
    return canonical_json(
        _pack_to_canonical_dict(pack, version=resolved), version=resolved
    )


def verify_pack_hash(pack: FixedIncomeEvidencePack, expected_hash: str) -> bool:
    """Recompute the pack's canonical hash and compare to ``expected_hash``.

    Returns ``True`` iff the recomputed hash matches verbatim (case-
    insensitive on the hex tail; the ``"sha256:"`` prefix must match
    exactly). PR-7 introduces an HMAC verify variant on top of this
    integrity check.

    v1.6.0: routes through :func:`compute_pack_hash` so the version
    stamped into ``pack.metadata[_canonical_version]`` is honoured.
    """
    computed = compute_pack_hash(pack)
    return computed.lower() == expected_hash.lower()


# ---------------------------------------------------------------------------
# PR-7 §A.1 — HMAC key resolution + sign / verify
# ---------------------------------------------------------------------------


def _decode_key_material(value: str) -> bytes:
    """Decode an HMAC key from base64 (preferred) or raw UTF-8 bytes.

    Operators may legitimately use either: base64-encoded random bytes
    (``MRE_FI_HMAC_KEY_VERSIONS={"v1": "<base64>"}``) or a high-entropy
    passphrase. We try base64 first (with padding tolerance) and fall
    back to UTF-8 bytes if the base64 alphabet check fails.
    """
    raw = value.strip()
    if not raw:
        raise ValueError("HMAC key material must not be empty")
    try:
        # validate=True ensures we don't silently accept arbitrary text
        return base64.b64decode(raw, validate=True)
    except Exception:
        pass
    # Tolerate base64 without padding (common in env vars copied from
    # secret managers).
    try:
        padded = raw + "=" * (-len(raw) % 4)
        return base64.b64decode(padded, validate=True)
    except Exception:
        return raw.encode("utf-8")


def _enforce_hmac_key_strength(version: str, key: bytes) -> None:
    """Fail closed on weak FI HMAC keys in production-signing mode."""
    if require_production_hmac() and len(key) < _MIN_PRODUCTION_HMAC_KEY_BYTES:
        raise RuntimeError(
            f"HMAC key {version!r} decodes to {len(key)} bytes; "
            f"production requires at least {_MIN_PRODUCTION_HMAC_KEY_BYTES} bytes"
        )


def get_hmac_keys() -> dict[str, bytes]:
    """Parse ``MRE_FI_HMAC_KEY_VERSIONS`` (JSON env var) into ``{version: bytes}``.

    Schema: ``{"v1": "base64-encoded-key", "v2": "...", ...}``.

    If ``MRE_FI_HMAC_KEY_VERSIONS`` is unset *and* ``MRE_FI_HMAC_KEY``
    (singleton) is set, the singleton is registered under version
    ``"v1"`` so a single-key deployment can rotate to multi-key without
    code changes.

    Returns an empty dict if neither is set. Bad JSON raises
    :class:`RuntimeError` rather than silently degrading; a typo in the
    env var must NOT produce a worker that signs nothing.
    """
    raw = os.environ.get(_HMAC_KEY_VERSIONS_ENV, "").strip()
    if raw:
        try:
            mapping = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{_HMAC_KEY_VERSIONS_ENV} must be a JSON object: {exc}"
            ) from exc
        if not isinstance(mapping, dict):
            raise RuntimeError(
                f"{_HMAC_KEY_VERSIONS_ENV} must decode to a JSON object; got {type(mapping).__name__}"
            )
        out: dict[str, bytes] = {}
        for version, value in mapping.items():
            if not isinstance(version, str) or not version:
                raise RuntimeError("HMAC key version must be a non-empty string")
            if not isinstance(value, str):
                raise RuntimeError(f"HMAC key {version!r} value must be a string")
            decoded = _decode_key_material(value)
            _enforce_hmac_key_strength(version, decoded)
            out[version] = decoded
        return out
    singleton = os.environ.get(_HMAC_KEY_SINGLETON_ENV, "").strip()
    if singleton:
        decoded = _decode_key_material(singleton)
        _enforce_hmac_key_strength("v1", decoded)
        return {"v1": decoded}
    return {}


def _hmac_version_sort_key(version: str) -> tuple[int, str, int | str]:
    """Natural-sort HMAC versions so ``v10`` sorts after ``v9``."""
    stripped = version.strip()
    if len(stripped) > 1 and stripped[0].lower() == "v" and stripped[1:].isdigit():
        return (1, "v", int(stripped[1:]))
    return (0, stripped, stripped)


def latest_hmac_version() -> str | None:
    """Return the most recent key version, or ``None`` if no keys are configured.

    The active key can be pinned with ``MRE_FI_HMAC_ACTIVE_VERSION``.
    Otherwise versions of the conventional form ``v<N>`` are natural-
    sorted so ``v10`` sorts after ``v9``; arbitrary version strings keep
    a stable lexical fallback.
    """
    keys = get_hmac_keys()
    if not keys:
        return None
    active = os.environ.get(_HMAC_ACTIVE_VERSION_ENV, "").strip()
    if active:
        if active not in keys:
            raise RuntimeError(
                f"{_HMAC_ACTIVE_VERSION_ENV}={active!r} is not present in "
                f"{_HMAC_KEY_VERSIONS_ENV}; configured versions={sorted(keys)!r}"
            )
        return active
    return max(keys.keys(), key=_hmac_version_sort_key)


def require_production_hmac() -> bool:
    """Returns True iff production-mode HMAC enforcement is on.

    Production mode is signalled by EITHER

    1. ``MRE_ENV=production`` (case-insensitive), OR
    2. ``MRE_FI_REQUIRE_HMAC`` set to any truthy string.

    v1.6.0 (REVIEW_DEEP_V1_5_2.md F5 / Finding §3.11): the v1.5.x
    check used exact ``=='1'`` matching for ``MRE_FI_REQUIRE_HMAC``
    which was inconsistent with the rest of the codebase’s
    ``rate_limit_enabled()`` style — an operator who set
    ``MRE_FI_REQUIRE_HMAC=true`` (natural-language truthy) would
    silently disable HMAC enforcement. The matcher now accepts
    the same ``{"1", "true", "yes", "on"}`` truthy set
    (case-insensitive, whitespace-stripped) as ``rate_limit_enabled``.
    """
    raw = os.environ.get(_REQUIRE_HMAC_ENV, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    return os.environ.get(_ENV_NAME_ENV, "").lower() == "production"


def _hmac_hex(key: bytes, payload: str) -> str:
    """Thin alias for back-compat; new code uses ``evidence_common.hmac_sha256_hex``."""
    return hmac_sha256_hex(key, payload.encode("utf-8"))


def sign_pack(
    pack: FixedIncomeEvidencePack,
    *,
    key_version: str | None = None,
) -> FixedIncomeEvidencePack:
    """Sign the pack's canonical JSON and return a new pack.

    The signature format is ``"v<ver>:<hex(hmac-sha256)>"`` so the
    verifier can route to the right key version without an out-of-band
    contract.

    Behaviour matrix:

    - keys configured (any number of versions) → sign with
      ``key_version`` if supplied, else :func:`latest_hmac_version`,
      return a new pack with ``hmac_signature`` populated.
    - no keys configured AND :func:`require_production_hmac` is True →
      raise :class:`RuntimeError` (production must not publish
      unsigned packs).
    - no keys configured AND production mode is False → return the
      pack unchanged (``hmac_signature`` stays ``None``); dev-mode
      pass-through.

    v1.5.1 (PR-9 FIX 3): in production
    (:func:`require_production_hmac`) every execution-confidence pack
    MUST carry ``request_id`` (the HMAC-bound replay token). Signing
    an execution-confidence pack with ``request_id=None`` under
    production mode raises :class:`RuntimeError` so the operator
    cannot accidentally publish a replay-vulnerable pack. Other
    components (e.g. ``credit_regime``) are exempt because they do
    not consume an inbound request id.
    """
    if (
        require_production_hmac()
        and pack.component_name == "execution_confidence"
        and not pack.request_id
    ):
        raise RuntimeError(
            "FI HMAC production mode (MRE_ENV=production or "
            "MRE_FI_REQUIRE_HMAC=1) requires request_id on "
            "execution_confidence evidence packs; pass --request-id on "
            "fi-evidence-pack or thread the FastAPI X-Request-ID through "
            "build_evidence_pack(...)"
        )
    keys = get_hmac_keys()
    if not keys:
        if require_production_hmac():
            raise RuntimeError(
                "FI HMAC required (MRE_ENV=production or MRE_FI_REQUIRE_HMAC=1) "
                f"but no keys are configured via {_HMAC_KEY_VERSIONS_ENV} / "
                f"{_HMAC_KEY_SINGLETON_ENV}"
            )
        return pack
    if key_version is None:
        key_version = latest_hmac_version()
    if key_version not in keys:
        raise RuntimeError(
            f"HMAC key version {key_version!r} is not in the configured "
            f"versions {sorted(keys)!r}"
        )
    payload = canonical_pack_payload(pack)
    digest = _hmac_hex(keys[key_version], payload)
    signature = f"{key_version}:{digest}"
    return replace(pack, hmac_signature=signature)


def verify_pack(pack: FixedIncomeEvidencePack) -> bool:
    """Verify the HMAC signature on a pack.

    Parses ``"v<ver>:<hex>"`` from ``pack.hmac_signature``, looks up
    the key, recomputes the HMAC over the canonical JSON (excluding
    the signature field itself), and compares using
    :func:`hmac.compare_digest`.

    Implementation note (do not refactor without preserving):
    verification uses :func:`hmac.compare_digest` for **constant-time
    comparison** so an attacker that can submit guesses against a
    pack does not learn how many leading hex digits matched via
    timing side channels. Replacing the call with ``==`` (or any
    short-circuiting equality) would silently re-introduce a timing
    oracle on the digest comparison and is explicitly forbidden by
    AGENT.md §"HMAC operations". The
    ``test_module_uses_compare_digest_in_verify`` introspection
    test pins this so a refactor cannot remove the guarantee
    silently.

    Returns ``False`` when the signature is missing, malformed, or
    signed under a key version that is not currently configured.
    Returns ``True`` (dev-mode pass-through) only when no keys are
    configured *and* ``hmac_signature`` is ``None``: a deployment that
    deliberately runs unsigned must not see a wall of False from this
    helper.

    v1.5 PR-8 (Tier-1 fix C-AUTO-3): every False return path
    increments ``fi_hmac_signature_failures_total{reason=...}`` so the
    runbook in ``docs/V1_5_HMAC_OPERATIONS.md`` can alert on
    aggregate failure rates without needing per-call-site
    instrumentation. Labels are stable strings to keep cardinality
    bounded.
    """
    from market_regime_engine.fixed_income.observability_ext import (
        incr_hmac_signature_failures,
    )

    keys = get_hmac_keys()
    sig = pack.hmac_signature
    if sig is None:
        if keys:
            incr_hmac_signature_failures(reason="missing_signature")
            return False
        return True
    if not isinstance(sig, str) or ":" not in sig:
        incr_hmac_signature_failures(reason="malformed_signature")
        return False
    version, _, hex_digest = sig.partition(":")
    if not version or not hex_digest:
        incr_hmac_signature_failures(reason="malformed_signature")
        return False
    key = keys.get(version)
    if key is None:
        incr_hmac_signature_failures(reason="key_not_found")
        return False
    payload = canonical_pack_payload(pack)
    expected = _hmac_hex(key, payload)
    try:
        if hmac.compare_digest(hex_digest, expected):
            return True
        incr_hmac_signature_failures(reason="compare_digest_mismatch")
        return False
    except Exception:
        incr_hmac_signature_failures(reason="compare_digest_error")
        return False


# ---------------------------------------------------------------------------
# PR-7 §A.2 — data_vintages capture
# ---------------------------------------------------------------------------

# Per-table column to read for the vintage timestamp (review §4.3 point 1).
# When a column is absent the helper falls back to the next listed name.
_FI_VINTAGE_TABLES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("trace_trades", "read_trace_trades", ("source_timestamp", "timestamp")),
    ("rfq_events", "read_rfq_events", ("source_timestamp", "timestamp")),
    ("curve_snapshots", "read_curve_snapshots", ("source_timestamp", "timestamp")),
    ("cds_curve_snapshots", "read_cds_curve_snapshots", ("source_timestamp", "timestamp")),
    ("bond_reference", "read_bond_reference", ("valid_from", "issue_date")),
    ("dealer_quotes", "read_dealer_quotes", ("source_timestamp", "timestamp")),
    ("dealer_response_stats", "read_dealer_response_stats", ("window_end", "window_start")),
)


_EPOCH_VINTAGE: str = "1970-01-01T00:00:00Z"


def _coerce_iso8601_z(ts: Any) -> str:
    """Coerce a timestamp scalar to an ISO-8601 string with ``Z`` suffix."""
    if ts is None:
        return _EPOCH_VINTAGE
    try:
        parsed = pd.Timestamp(ts)
    except Exception:
        return _EPOCH_VINTAGE
    if pd.isna(parsed):
        return _EPOCH_VINTAGE
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _latest_vintage_for_table(
    warehouse: Any,
    *,
    reader_name: str,
    columns: tuple[str, ...],
    asof: pd.Timestamp | None,
) -> str:
    """Return the latest vintage ISO-8601 string for one FI table."""
    reader = getattr(warehouse, reader_name, None)
    if reader is None or not callable(reader):
        return _EPOCH_VINTAGE
    try:
        df = reader()
    except Exception as exc:  # pragma: no cover - reader-side failure
        log.warning("read for %s failed during vintage capture: %s", reader_name, exc)
        return _EPOCH_VINTAGE
    if df is None or df.empty:
        return _EPOCH_VINTAGE
    column = next((c for c in columns if c in df.columns), None)
    if column is None:
        return _EPOCH_VINTAGE
    series = pd.to_datetime(df[column], errors="coerce", utc=True)
    series = series.dropna()
    if asof is not None:
        cap = pd.Timestamp(asof)
        if cap.tzinfo is None:
            cap = cap.tz_localize("UTC")
        else:
            cap = cap.tz_convert("UTC")
        series = series[series <= cap]
    if series.empty:
        return _EPOCH_VINTAGE
    return _coerce_iso8601_z(series.max())


def capture_data_vintages(
    warehouse: Any,
    *,
    asof: pd.Timestamp | None = None,
) -> dict[str, str]:
    """Capture per-source latest vintage timestamps from the warehouse.

    Returns a stable dict keyed by FI table name with ISO-8601 ``Z``
    timestamps. Missing tables, tables without a recognised vintage
    column, or tables with no rows ≤ ``asof`` produce
    ``"1970-01-01T00:00:00Z"`` so the dict shape is invariant across
    fresh deployments and full-history runs.

    Per review §4.3 point 1: this snapshot is what allows
    ``mre verify-run --model-run-id <id>`` to assert that the same
    vintages were available when the pack was signed and when an
    auditor rebuilds it later.
    """
    out: dict[str, str] = {}
    for table_name, reader_name, columns in _FI_VINTAGE_TABLES:
        out[table_name] = _latest_vintage_for_table(
            warehouse,
            reader_name=reader_name,
            columns=columns,
            asof=asof,
        )
    return out


# ---------------------------------------------------------------------------
# PR-7 §A.3 — write evidence pack to warehouse
# ---------------------------------------------------------------------------


def evidence_pack_to_row(
    pack: FixedIncomeEvidencePack,
    *,
    request_id: str,
) -> dict[str, Any]:
    """Project a pack onto the ``fixed_income_evidence_packs`` row schema.

    JSON-encoded fields use ``canonical_json`` so the row hash is
    deterministic and matches what :func:`compute_pack_hash` saw.

    v1.5.1 (PR-9 FIX 3): when ``pack.request_id`` is set it MUST equal
    the writer-provided ``request_id`` parameter — they're the same
    value semantically and a mismatch indicates a code bug
    (e.g. caller threaded the row-level id but built the pack with the
    wrong id). The mismatch raises :class:`ValueError` so the divergence
    surfaces immediately.
    """
    if pack.request_id is not None and str(pack.request_id) != str(request_id):
        raise ValueError(
            f"evidence_pack_to_row: pack.request_id={pack.request_id!r} but "
            f"row request_id={request_id!r}; pass the same id to "
            f"build_evidence_pack() and to evidence_pack_to_row()"
        )
    return {
        "model_run_id": pack.model_run_id,
        "request_id": request_id,
        "component_name": pack.component_name,
        "model_version": pack.model_version,
        "timestamp": pack.timestamp,
        "code_sha": pack.code_sha,
        "model_hash": pack.model_hash,
        "input_features_hash": pack.input_features_hash,
        "output_hash": pack.output_hash,
        "data_vintages_json": canonical_json(dict(pack.data_vintages)),
        "validation_results_json": canonical_json(dict(pack.validation_results)),
        "release_gate": 1 if pack.release_gate else 0,
        "random_seeds_json": canonical_json(dict(pack.random_seeds)),
        "python_version": pack.python_version,
        "lockfile_hash": pack.lockfile_hash,
        "hmac_signature": pack.hmac_signature,
        "metadata_json": canonical_json(dict(pack.metadata)),
    }


def _stamp_envelope_hash(pack: FixedIncomeEvidencePack) -> FixedIncomeEvidencePack:
    """Return a new pack with ``metadata[_ENVELOPE_HASH_METADATA_KEY]``
    populated with the canonical pack-hash.

    The envelope hash is the SHA-256 over the canonical bytestream
    *excluding* both ``hmac_signature`` and this key itself (see
    :func:`_pack_to_canonical_dict`), so the value is stable under
    re-signing and under repeated stamping.

    v1.5 PR-8 (Tier-1 fix C-AUTO-1): ``verify-run`` compares the
    recomputed hash against this stamped value to detect row-level
    tampering of any pack field (including ``output_hash``) without
    requiring HMAC keys.
    """
    envelope_hash = compute_pack_hash(pack)
    new_metadata = dict(pack.metadata or {})
    new_metadata[_ENVELOPE_HASH_METADATA_KEY] = envelope_hash
    return replace(pack, metadata=new_metadata)


def write_evidence_pack(
    warehouse: Any,
    pack: FixedIncomeEvidencePack,
    *,
    request_id: str,
    sign: bool | None = None,
) -> FixedIncomeEvidencePack:
    """Persist an evidence pack to ``fixed_income_evidence_packs``.

    Behaviour:

    - ``sign`` is ``None`` (default) → sign when keys are configured;
      pass through unsigned otherwise (subject to
      :func:`require_production_hmac`).
    - ``sign=True`` → always attempt to sign; raise if no keys
      configured.
    - ``sign=False`` → never sign; raise if production mode requires
      HMAC.

    Returns the (possibly newly-signed) pack so the caller can compare
    the persisted ``hmac_signature`` against the in-memory one.

    v1.5 PR-8 (Tier-1 fix C-AUTO-1): stamps the canonical pack-hash
    into ``pack.metadata[_envelope_hash]`` before signing so
    :func:`verify_run` can compare the recomputed hash against the
    stored envelope hash to detect row-level tampering.
    """
    stamped = _stamp_envelope_hash(pack)
    if sign is True:
        signed = sign_pack(stamped)
        if signed.hmac_signature is None:
            raise RuntimeError(
                "sign=True requested but no HMAC keys are configured "
                "(set MRE_FI_HMAC_KEY_VERSIONS or MRE_FI_HMAC_KEY)"
            )
    elif sign is False:
        if require_production_hmac():
            raise RuntimeError(
                "sign=False requested but production mode requires HMAC; "
                "either disable production mode or pass sign=True"
            )
        signed = stamped
    else:
        signed = sign_pack(stamped)
    row = evidence_pack_to_row(signed, request_id=request_id)
    warehouse.write_evidence_pack(pd.DataFrame([row]))
    return signed


def read_evidence_pack(
    warehouse: Any,
    *,
    model_run_id: str,
    request_id: str | None = None,
) -> FixedIncomeEvidencePack | None:
    """Read an evidence pack row → :class:`FixedIncomeEvidencePack`.

    When ``request_id`` is omitted, returns the most recent pack for
    ``model_run_id`` (ordered by ``timestamp``). Returns ``None`` if no
    matching row exists.

    v1.6.0 (REVIEW_DEEP_V1_5_2.md A8 / Finding §3.3): now uses the
    indexed :meth:`Warehouse.latest_evidence_pack` SQL fast path
    instead of reading the whole ``fixed_income_evidence_packs``
    table and filtering in pandas. Falls back to the legacy full-
    table read when the warehouse object does not yet expose
    ``latest_evidence_pack`` (older callers / test mocks).
    """
    if hasattr(warehouse, "latest_evidence_pack"):
        sub = warehouse.latest_evidence_pack(
            model_run_id, request_id=request_id
        )
        if sub is None or sub.empty:
            return None
        row = sub.iloc[0]
        return _row_to_pack(row)
    # Legacy fallback for callers / test mocks that only expose
    # read_evidence_packs(); preserves the v1.5.x semantics.
    df = warehouse.read_evidence_packs()
    if df is None or df.empty:
        return None
    sub = df[df["model_run_id"] == model_run_id]
    if request_id is not None:
        sub = sub[sub["request_id"] == request_id]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    return _row_to_pack(row)


def _parse_json_field(value: Any) -> dict[str, Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        if not value:
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _normalize_timestamp_for_roundtrip(value: Any) -> str:
    """Return the timestamp in ISO-8601 ``Z`` form so a DuckDB roundtrip
    is bit-identical to the signed canonical bytestream.

    DuckDB's TIMESTAMP type drops the ``Z`` suffix and uses a space
    separator on read, which would otherwise change the canonical
    JSON and break HMAC verification. We re-emit the same shape that
    :func:`build_evidence_pack` produces.
    """
    if value is None:
        return ""
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return str(value)
    if pd.isna(ts):
        return ""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _row_to_pack(row: pd.Series) -> FixedIncomeEvidencePack:
    """Hydrate a :class:`FixedIncomeEvidencePack` from a warehouse row.

    v1.5.1 (PR-9 FIX 3): the row's ``request_id`` is always read back
    into ``pack.request_id`` for completeness, but it is the metadata
    flag ``_request_id_bound`` (stamped into ``pack.metadata`` at
    :func:`build_evidence_pack` time) that decides whether
    :func:`verify_pack` binds the value into the canonical bytestream.
    Legacy v1 packs lack the flag and verify exactly as before; new v2
    packs built via :func:`build_evidence_pack` with a non-``None``
    ``request_id`` carry the flag and bind the id.
    """
    hmac_sig = (
        None
        if pd.isna(row.get("hmac_signature"))
        or row.get("hmac_signature") in ("", None)
        else str(row["hmac_signature"])
    )
    row_rid = row.get("request_id")
    has_row_rid = (
        row_rid is not None
        and not (isinstance(row_rid, float) and pd.isna(row_rid))
        and str(row_rid) != ""
    )
    request_id_for_pack: str | None = str(row_rid) if has_row_rid else None
    return FixedIncomeEvidencePack(
        model_run_id=str(row["model_run_id"]),
        component_name=str(row["component_name"]),
        model_version=str(row["model_version"]),
        timestamp=_normalize_timestamp_for_roundtrip(row["timestamp"]),
        code_sha=(None if pd.isna(row.get("code_sha")) else str(row["code_sha"])),
        model_hash=str(row["model_hash"]),
        input_features_hash=str(row["input_features_hash"]),
        output_hash=str(row["output_hash"]),
        data_vintages=_parse_json_field(row.get("data_vintages_json")),
        validation_results=_parse_json_field(row.get("validation_results_json")),
        release_gate=bool(int(row["release_gate"])),
        random_seeds=_parse_json_field(row.get("random_seeds_json")),
        python_version=(
            None if pd.isna(row.get("python_version")) else str(row["python_version"])
        )
        or "",
        lockfile_hash=(
            None if pd.isna(row.get("lockfile_hash")) else str(row["lockfile_hash"])
        ),
        hmac_signature=hmac_sig,
        metadata=_parse_json_field(row.get("metadata_json")),
        request_id=request_id_for_pack,
    )


def evidence_pack_to_dict(pack: FixedIncomeEvidencePack) -> dict[str, Any]:
    """JSON-serialisable dict form of a pack — used by the API + CLI."""
    out = asdict(pack)
    out["data_vintages"] = dict(pack.data_vintages)
    out["validation_results"] = dict(pack.validation_results)
    out["random_seeds"] = dict(pack.random_seeds)
    out["metadata"] = dict(pack.metadata)
    return out


def stored_envelope_hash(pack: FixedIncomeEvidencePack) -> str | None:
    """Return the envelope hash stamped into ``pack.metadata`` at write time.

    v1.5 PR-8 (Tier-1 fix C-AUTO-1): ``write_evidence_pack`` stamps the
    canonical pack hash here; ``verify-run`` reads it back and compares
    to a fresh :func:`compute_pack_hash` to detect row-level tampering.
    Returns ``None`` when the pack was written before the stamping
    rollout (or by a path that bypassed :func:`write_evidence_pack`)
    so ``verify-run`` fails closed with ``"envelope_hash_missing"``.
    """
    metadata = pack.metadata or {}
    value = metadata.get(_ENVELOPE_HASH_METADATA_KEY)
    if not isinstance(value, str) or not value:
        return None
    return value


__all__ = [
    "build_evidence_pack",
    "canonical_pack_payload",
    "capture_data_vintages",
    "compute_pack_hash",
    "evidence_pack_to_dict",
    "evidence_pack_to_row",
    "get_hmac_keys",
    "latest_hmac_version",
    "read_evidence_pack",
    "require_production_hmac",
    "sign_pack",
    "stored_envelope_hash",
    "verify_pack",
    "verify_pack_hash",
    "write_evidence_pack",
]
