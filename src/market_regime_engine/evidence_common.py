# SPDX-License-Identifier: Apache-2.0
"""Shared primitives for the engine's two evidence-pack subpackages.

The engine ships two complementary evidence-pack systems:

- :mod:`market_regime_engine.fixed_income.evidence_pack` -- per-signal audit
  trail. Every FI Auto-X firing produces a single dataclass-shaped pack
  signed with a versioned HMAC (``v1`` / ``v2`` / ...) over the canonical
  JSON bytestream of the pack itself.
- :mod:`market_regime_engine.validation_pack` -- per-release-run audit
  bundle. Each validation run produces a directory of artifacts plus a
  ``manifest.json`` and an optional whole-file HMAC over that manifest.

The two packs serve different consumers (one row of audit trail per
signal vs one directory bundle per release run) and therefore have
different HMAC scopes. They share two primitives -- canonical JSON
encoding and a generic HMAC-SHA256-hex helper -- which now live here.
``fixed_income.hashing`` keeps the same public symbols (re-exported
from this module) so existing FI callers stay binary-compatible across
the v1.6 refactor.

Migration notes (v1.6 PR-22)::

    # Before (v1.5.x):
    from market_regime_engine.fixed_income.hashing import canonical_json

    # After (v1.6+):
    from market_regime_engine.evidence_common import canonical_json
    # (the fixed_income.hashing path keeps working as a thin alias)

v1.6.0 (REVIEW_DEEP_V1_5_2.md section 2.5): two encoders coexist.

- **``v1`` (legacy)** -- ``json.dumps(payload, sort_keys=True,
  separators=(",", ":"), default=str)``. Stable across same-CPython
  runs (and same-version Pythons in general) but diverges from RFC
  8785 on numbers, non-ASCII strings, and NaN/Inf handling. Kept for
  backward-compatibility verification of v1.5.x packs.
- **``v2`` (RFC 8785, JCS)** -- pure-Python implementation of the
  JSON Canonicalization Scheme. Numbers go through ECMA-262 7.1.12.1
  Number::toString (``1.0`` becomes ``"1"``), strings use minimal
  Unicode escapes (only U+0000-U+001F, ``"``, backslash), object
  keys sort lexicographically by UTF-16 code points (Python's
  default ``sort_keys`` is identical for BMP keys), and NaN /
  Infinity / non-JSON-native types are rejected.

Callers select the version explicitly; new evidence packs stamp
``_canonical_version="v2"`` into their metadata so :func:`verify_pack`
routes correctly. Legacy packs without that metadata key continue to
verify under v1.

BMP-only assumption (v2): RFC 8785 prescribes UTF-16 code-point
ordering on object keys. For keys in the Basic Multilingual Plane
(``U+0000``-``U+FFFF``), Python's default ``sorted()`` on ``str``
gives the same order. Keys above the BMP (supplementary planes,
``U+10000``-``U+10FFFF``) would require explicit UTF-16 code-unit
comparison; the FI evidence pack schema uses ASCII keys exclusively
so the BMP path is sufficient for our consumers.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import subprocess
from datetime import date, datetime
from decimal import Decimal
from pathlib import PurePath
from typing import Any, Literal

CanonicalVersion = Literal["v1", "v2"]
DEFAULT_CANONICAL_VERSION: CanonicalVersion = "v2"

_HASH_PREFIX = "sha256:"

# ---------------------------------------------------------------------------
# RFC 8785 (JCS) primitives -- v2 encoder
# ---------------------------------------------------------------------------

# Per RFC 8785 section 3.2.2.2 (and ECMA-262 7.1.12.1 step 8): only the
# named escapes plus the C0 control range receive ``\uXXXX`` form.
# Everything else is emitted raw (the encoder runs with
# ``ensure_ascii=False``).
_RFC8785_ESCAPES: dict[int, str] = {
    0x08: "\\b",
    0x09: "\\t",
    0x0A: "\\n",
    0x0C: "\\f",
    0x0D: "\\r",
    0x22: "\\\"",
    0x5C: "\\\\",
}


def _ecma262_number_tostring(x: float | int) -> str:
    """Implement ECMA-262 section 7.1.12.1 ``Number::toString`` for RFC 8785.

    Returns the canonical decimal / scientific string per the
    algorithm spec; raises :class:`ValueError` on NaN / Infinity
    (which RFC 8785 explicitly forbids encoders from emitting).

    Key invariants the FI cross-language verifier relies on:

    - Integers -> plain decimal: ``1`` not ``1.0`` and not ``"1"``.
    - Whole-number floats with ``|x| < 1e21`` -> integer form.
    - Floats are emitted as the shortest decimal string that
      round-trips to the same float (``repr(x)`` semantics).
    - Range ``[1e-6, 1e21)`` -- emit as decimal (no scientific).
    - Outside that range (excluding zero) -- emit scientific
      ``<mantissa>e<sign><magnitude>`` with the exponent stripped of
      any leading zero (so Python's ``1e-07`` becomes RFC 8785
      ``1e-7``).
    - Negative zero collapses to ``"0"`` (ECMA-262 step 2).
    """
    if isinstance(x, bool):
        # bool is a subclass of int; the encoder dispatches booleans
        # separately (``true`` / ``false``) so a bool reaching this
        # helper is a programming error.
        raise TypeError("_ecma262_number_tostring received bool; route via dispatcher")
    if isinstance(x, int):
        return str(x)
    if not isinstance(x, float):
        raise TypeError(
            f"_ecma262_number_tostring: unsupported type {type(x).__name__}"
        )
    if math.isnan(x):
        raise ValueError("RFC 8785 rejects NaN/Infinity")
    if math.isinf(x):
        raise ValueError("RFC 8785 rejects NaN/Infinity")
    if x == 0.0:
        return "0"
    if x < 0.0:
        return "-" + _ecma262_number_tostring(-x)
    # x > 0 and finite.
    if x.is_integer() and x < 1e21:
        return str(int(x))
    # Shortest round-trip representation. Python's ``repr`` follows
    # PEP 3101 / dtoa to produce the shortest decimal that survives
    # round-trip -- the same property RFC 8785 requires.
    raw = repr(x)
    if "e" in raw:
        mantissa, _, exp_str = raw.partition("e")
        exp = int(exp_str)
    else:
        mantissa, exp = raw, 0
    if "." in mantissa:
        int_part, frac_part = mantissa.split(".")
    else:
        int_part, frac_part = mantissa, ""
    raw_digits = int_part + frac_part
    leading_zeros = len(raw_digits) - len(raw_digits.lstrip("0"))
    sig_digits = raw_digits.lstrip("0").rstrip("0") or "0"
    if sig_digits == "0":
        return "0"
    k = len(sig_digits)
    # ECMA section 7.1.12.1 step 5: choose ``n`` such that
    # 10^(n-1) <= s < 10^n, where ``s`` is the integer of
    # significant digits.
    if leading_zeros < len(int_part):
        digit_exp = len(int_part) - 1 - leading_zeros + exp
    else:
        digit_exp = -1 - (leading_zeros - len(int_part)) + exp
    n = digit_exp + 1
    # Step 6: k <= n <= 21 -> pad with zeros on the right.
    if k <= n <= 21:
        return sig_digits + "0" * (n - k)
    # Step 7: 0 < n <= 21 -> split with decimal point.
    if 0 < n <= 21:
        return sig_digits[:n] + "." + sig_digits[n:]
    # Step 8: -6 < n <= 0 -> leading-zero decimal form.
    if -6 < n <= 0:
        return "0." + "0" * (-n) + sig_digits
    # Step 9: scientific notation; ECMA strips the leading zero on
    # the exponent (Python's ``repr`` emits ``1e-07``, RFC 8785
    # canonical form is ``1e-7``).
    if len(sig_digits) > 1:
        mantissa_out = sig_digits[0] + "." + sig_digits[1:]
    else:
        mantissa_out = sig_digits
    sign = "+" if digit_exp >= 0 else "-"
    return mantissa_out + "e" + sign + str(abs(digit_exp))


def _escape_string_rfc8785(value: str) -> str:
    """Encode a Python ``str`` per RFC 8785 section 3.2.2.2.

    Only U+0000-U+001F, U+0022 (``"``), and U+005C (backslash) receive
    escapes. Everything else (including all non-ASCII code points)
    is emitted as raw UTF-8 -- the encoder itself drops
    ``ensure_ascii``.
    """
    chunks: list[str] = ["\""]
    for ch in value:
        code = ord(ch)
        if code in _RFC8785_ESCAPES:
            chunks.append(_RFC8785_ESCAPES[code])
        elif code < 0x20:
            chunks.append(f"\\u{code:04x}")
        else:
            chunks.append(ch)
    chunks.append("\"")
    return "".join(chunks)


def _canonical_json_v2(payload: Any) -> str:
    """RFC 8785 (JCS) canonical JSON encoder.

    Reference: https://www.rfc-editor.org/rfc/rfc8785
    ECMA-262 7.1.12.1: https://tc39.es/ecma262/#sec-numeric-types-number-tostring

    Differences vs the v1 ``json.dumps(default=str)`` path:

    1. Numbers honour ECMA-262 Number::toString -- ``1.0`` becomes
       ``"1"``; no scientific notation for ``|x| in [1e-6, 1e21)``.
    2. Strings emit raw UTF-8; only the RFC 8785 mandatory escape
       set receives ``\\uXXXX`` form.
    3. Object keys sort by UTF-16 code points (BMP-only -- see module
       docstring for the supplementary-plane caveat).
    4. NaN, Infinity, and ``-0.0`` are rejected (the latter
       collapses to ``0`` per ECMA-262).
    5. Non-JSON-native types (datetime, Decimal, Path, set, etc.)
       are **rejected** with :class:`TypeError`. Callers must
       pre-coerce via :func:`coerce_for_canonical`.

    This is the encoder a Java JCS or Rust ``serde_jcs`` verifier
    will agree with byte-for-byte.
    """
    return _encode_v2(payload)


def _encode_v2(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _ecma262_number_tostring(value)
    if isinstance(value, str):
        return _escape_string_rfc8785(value)
    if isinstance(value, dict):
        # RFC 8785 section 3.2.3: sort by UTF-16 code units; for BMP
        # keys Python's ``sorted`` is byte-identical to the spec.
        items = sorted(value.items(), key=lambda kv: kv[0])
        pieces: list[str] = []
        for k, v in items:
            if not isinstance(k, str):
                raise TypeError(
                    f"RFC 8785 object keys must be strings; got {type(k).__name__}"
                )
            pieces.append(_escape_string_rfc8785(k) + ":" + _encode_v2(v))
        return "{" + ",".join(pieces) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_encode_v2(item) for item in value) + "]"
    raise TypeError(
        f"RFC 8785 encoder rejects non-JSON-native type {type(value).__name__}; "
        "use evidence_common.coerce_for_canonical(payload) to pre-coerce "
        "datetime / Decimal / Path / set inputs"
    )


def coerce_for_canonical(payload: Any) -> Any:
    """Recursively coerce a payload to a JSON-native shape for v2.

    The v1 encoder accepted arbitrary types via ``default=str``;
    cross-language verifiers cannot rely on Python's ``str(...)``
    output for ``datetime`` / ``Decimal`` / ``Path`` so the v2
    encoder rejects them. This helper provides the documented
    coercion path so callers can opt in without sprinkling
    ``str(...)`` casts at every emit site:

    - ``datetime`` / ``date`` -> ISO-8601 string (``isoformat()``).
    - ``Decimal`` -> ``str(Decimal)`` (preserves precision).
    - ``PurePath`` -> forward-slash POSIX string.
    - ``set`` / ``frozenset`` -> sorted list (deterministic order
      independent of ``PYTHONHASHSEED``).
    - ``bytes`` -> :class:`TypeError` (RFC 8785 has no bytes type;
      callers must hex-or-base64-encode bytes themselves).
    - dict / list / tuple -> recurse.
    - native JSON types (bool, int, float, str, None) -> unchanged.

    Any other type falls through to ``str(value)`` so a stray object
    doesn't block the migration; callers should not rely on this
    fallback (the encoder will accept the string but the value's
    cross-language interpretation is undefined).
    """
    if payload is None or isinstance(payload, (bool, int, float, str)):
        return payload
    if isinstance(payload, datetime):
        return payload.isoformat()
    if isinstance(payload, date):
        return payload.isoformat()
    if isinstance(payload, Decimal):
        return str(payload)
    if isinstance(payload, PurePath):
        return payload.as_posix()
    if isinstance(payload, bytes):
        raise TypeError(
            "coerce_for_canonical: bytes are not RFC 8785-native; hex/base64-"
            "encode the value explicitly before passing it to the encoder."
        )
    if isinstance(payload, (set, frozenset)):
        return [coerce_for_canonical(item) for item in sorted(payload, key=repr)]
    if isinstance(payload, dict):
        return {str(k): coerce_for_canonical(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [coerce_for_canonical(item) for item in payload]
    return str(payload)


# ---------------------------------------------------------------------------
# Public canonical-JSON / SHA-256 API
# ---------------------------------------------------------------------------


def canonical_json(
    payload: Any,
    *,
    version: CanonicalVersion = "v1",
) -> str:
    """Canonical JSON encoding for hashing / HMAC inputs.

    The default version is ``"v1"`` so an unannotated caller (the
    v1.5.x wire form) keeps the legacy ``json.dumps(default=str)``
    output verbatim. New callers -- every v1.6.0 pack-builder path
    -- should pass ``version="v2"`` so the bytestream conforms to
    RFC 8785 (cross-language JCS) and a future Java / Rust verifier
    re-derives the same hash byte-for-byte.

    v1 (legacy):
        ``json.dumps(payload, sort_keys=True, separators=(",", ":"),
        default=str)``. Same behaviour as v1.5.x; stable across CPython
        minor versions but not RFC 8785-compliant on numbers,
        non-ASCII strings, NaN/Inf, or non-JSON-native types
        (datetime / Decimal / Path -- the ``default=str`` fallback).

    v2 (RFC 8785, JCS):
        Numbers via ECMA-262 Number::toString; strings with minimal
        escapes and raw UTF-8 bytes; keys sorted by UTF-16 code
        points (BMP-only -- see module docstring); rejects NaN /
        Infinity / non-JSON-native types. Callers with datetime /
        Decimal / Path inputs MUST pre-coerce via
        :func:`coerce_for_canonical`.
    """
    if version == "v1":
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        )
    if version == "v2":
        return _canonical_json_v2(payload)
    raise ValueError(f"canonical_json version must be 'v1' or 'v2'; got {version!r}")


def canonical_sha256(
    payload: Any,
    *,
    version: CanonicalVersion = "v1",
) -> str:
    """Return ``"sha256:" + hex(sha256(canonical_json(payload, version=...)))``.

    The prefix lets downstream consumers detect (and migrate from) the
    legacy bare-hex form without ambiguity. The default version is
    ``"v1"`` for the same back-compat reason as :func:`canonical_json`.
    """
    data = canonical_json(payload, version=version).encode("utf-8")
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


_BLAS_VARIANTS: tuple[str, ...] = (
    "mkl",
    "accelerate",
    "openblas",
    "blis",
    "atlas",
    "netlib",
)


def _detect_numpy_blas() -> str:
    """Return the BLAS variant linked into the running NumPy build.

    v1.6.0 (REVIEW_DEEP_V1_5_2.md section 3.7): the reproducibility
    envelope captures the BLAS variant so a verifier can detect
    OpenBLAS vs. MKL vs. Accelerate divergence. ``np.linalg.solve``
    and related LAPACK routines differ by 1-2 ULP between variants;
    a reviewer who refits a model on a non-matching BLAS sees that
    in the verify-run drift report rather than chasing a phantom
    numerical regression.

    Resolution order:

    1. ``numpy.show_config(mode="dicts")`` (NumPy >= 1.26):
       structured output -- look for ``blas.name``.
    2. ``numpy.show_config()`` redirected to a buffer (older NumPy):
       fall back to string-matching the printed output.
    3. ``"unknown"`` if neither path resolves a known variant (the
       envelope keeps the field shape stable even when capture
       fails).

    Returns one of ``mkl`` / ``accelerate`` / ``openblas`` /
    ``blis`` / ``atlas`` / ``netlib`` / ``unknown``. ``openblas``
    subsumes both upstream OpenBLAS and the SciPy-bundled
    ``scipy-openblas`` repackaging because they ship the same
    kernel.
    """
    try:
        import numpy as _np
    except Exception:
        return "unknown"

    # NumPy >= 1.26 exposes structured output via mode="dicts".
    try:
        cfg = _np.show_config(mode="dicts")
    except Exception:
        cfg = None
    if isinstance(cfg, dict):
        bd = cfg.get("Build Dependencies") or {}
        if isinstance(bd, dict):
            blas = bd.get("blas")
            if isinstance(blas, dict):
                name = str(blas.get("name", "")).lower()
                if name:
                    return _normalise_blas_name(name)

    # Fallback: capture printed output (legacy NumPy < 1.26).
    import contextlib as _ctx
    import io as _io

    buf = _io.StringIO()
    try:
        with _ctx.redirect_stdout(buf):
            _np.show_config()
    except Exception:
        return "unknown"
    text = buf.getvalue().lower()
    # Order matters: ``mkl_info`` may appear inside an
    # ``openblas_info`` section header on legacy NumPy builds.
    # Check the more-specific variants first (mkl, accelerate)
    # before the generic openblas key.
    for variant in _BLAS_VARIANTS:
        # Each variant's section header is ``<variant>_info`` on
        # legacy NumPy (< 2.0); the new structured output emits
        # a ``name: <variant>`` line.
        if f"{variant}_info" in text or f"name: {variant}" in text:
            return variant
    # scipy-openblas is upstream OpenBLAS repackaged by SciPy;
    # treat them identically for envelope purposes.
    if "scipy-openblas" in text or "scipy_openblas" in text:
        return "openblas"
    return "unknown"


def _normalise_blas_name(name: str) -> str:
    """Collapse upstream BLAS names onto the envelope vocabulary."""
    lowered = name.lower().strip()
    if "scipy-openblas" in lowered or "scipy_openblas" in lowered:
        return "openblas"
    for variant in _BLAS_VARIANTS:
        if variant in lowered:
            return variant
    return "unknown"


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
    "DEFAULT_CANONICAL_VERSION",
    "CanonicalVersion",
    "canonical_json",
    "canonical_sha256",
    "coerce_for_canonical",
    "detect_numpy_blas",
    "git_dirty",
    "git_revision",
    "hmac_sha256_hex",
    "strip_hash_prefix",
]


# Public alias for the BLAS-variant detection helper -- the leading
# underscore is preserved on the implementation function for the
# call site (see ``model_runs.build_repro_envelope``) but a public
# name lets external integrators query the same value without
# touching a private symbol.
detect_numpy_blas = _detect_numpy_blas
