# SPDX-License-Identifier: Apache-2.0
"""v1.6.0 (REVIEW_DEEP_V1_5_2.md section 2.5): RFC 8785 canonical JSON v2.

Pins both the cross-encoder regression (v1 vs v2 byte difference) and
the conformance vectors a Java JCS / Rust ``serde_jcs`` verifier would
exercise against this encoder. The latter is the value-add of the
versioned encoder -- without it the v1 packs can be verified
in-process but no external auditor can re-derive the bytes.
"""

from __future__ import annotations

import math

import pytest

from market_regime_engine.evidence_common import (
    _canonical_json_v2,
    _ecma262_number_tostring,
    _escape_string_rfc8785,
    canonical_json,
    canonical_sha256,
    coerce_for_canonical,
)

# ---------------------------------------------------------------------------
# ECMA-262 7.1.12.1 Number::toString conformance vectors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Integers - emit as plain decimal.
        (0, "0"),
        (1, "1"),
        (42, "42"),
        (-5, "-5"),
        # Whole-number floats - no trailing .0.
        (1.0, "1"),
        (100.0, "100"),
        (-3.0, "-3"),
        (0.0, "0"),
        # Pre-step "-0.0 -> 0" rule.
        (-0.0, "0"),
        # Fractional floats - shortest round-trip.
        (1.5, "1.5"),
        (3.14, "3.14"),
        (-2.5, "-2.5"),
        # Decimal range [1e-6, 1e21) - no scientific.
        (0.001, "0.001"),
        (0.0001, "0.0001"),
        (1e-6, "0.000001"),
        (1e20, "100000000000000000000"),
        # Outside the decimal range - scientific, no leading zero on exp.
        (1e-7, "1e-7"),
        (1e-10, "1e-10"),
        (1e21, "1e+21"),
        (1e22, "1e+22"),
        # 1.5 * 10^21 stays scientific (above the integer threshold).
        (1.5e21, "1.5e+21"),
        # Negative scientific.
        (-1e-7, "-1e-7"),
        (-1e21, "-1e+21"),
    ],
)
def test_ecma262_number_tostring_conformance(value: float | int, expected: str) -> None:
    assert _ecma262_number_tostring(value) == expected


def test_ecma262_rejects_nan() -> None:
    with pytest.raises(ValueError, match="NaN/Infinity"):
        _ecma262_number_tostring(float("nan"))


def test_ecma262_rejects_pos_inf() -> None:
    with pytest.raises(ValueError, match="NaN/Infinity"):
        _ecma262_number_tostring(float("inf"))


def test_ecma262_rejects_neg_inf() -> None:
    with pytest.raises(ValueError, match="NaN/Infinity"):
        _ecma262_number_tostring(float("-inf"))


def test_ecma262_rejects_bool_dispatch() -> None:
    """Booleans must be routed through the dispatcher (true/false), not the
    number formatter (which would emit 1/0)."""
    with pytest.raises(TypeError, match="bool"):
        _ecma262_number_tostring(True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# String escape vectors (RFC 8785 section 3.2.2.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", '""'),
        ("hello", '"hello"'),
        ('with "quotes"', '"with \\"quotes\\""'),
        ("back\\slash", '"back\\\\slash"'),
        ("tab\there", '"tab\\there"'),
        ("new\nline", '"new\\nline"'),
        # Raw UTF-8 for everything outside the mandatory escape set:
        # non-ASCII characters are NOT escaped to \\uXXXX.
        ("caf\u00e9", '"caf\u00e9"'),
        ("\u4e2d\u6587", '"\u4e2d\u6587"'),
        # Control characters below 0x20 get \\u escaped.
        ("\x01", '"\\u0001"'),
        ("\x1f", '"\\u001f"'),
        # Form feed / backspace / CR get the named short escapes.
        ("\x08", '"\\b"'),
        ("\x0c", '"\\f"'),
        ("\r", '"\\r"'),
    ],
)
def test_escape_string_rfc8785_conformance(value: str, expected: str) -> None:
    assert _escape_string_rfc8785(value) == expected


# ---------------------------------------------------------------------------
# v1 vs v2 cross-encoder regression
# ---------------------------------------------------------------------------


def test_v1_vs_v2_differ_on_floats() -> None:
    """A payload with float values produces different v1 / v2 bytes."""
    payload = {"value": 1.0, "ratio": 1.5}
    v1 = canonical_json(payload, version="v1")
    v2 = canonical_json(payload, version="v2")
    assert v1 != v2
    # v1 keeps `1.0`; v2 collapses to `1`.
    assert "1.0" in v1
    # v2 strips the trailing .0 from whole-number floats.
    assert v2 == '{"ratio":1.5,"value":1}'


def test_v1_vs_v2_differ_on_non_ascii() -> None:
    """A payload with non-ASCII strings produces different v1 / v2 bytes."""
    payload = {"name": "caf\u00e9"}
    v1 = canonical_json(payload, version="v1")
    v2 = canonical_json(payload, version="v2")
    assert v1 != v2
    # v1 escapes the non-ASCII codepoint.
    assert "\\u00e9" in v1
    # v2 emits raw UTF-8.
    assert "caf\u00e9" in v2


def test_v1_vs_v2_differ_on_nested_dict() -> None:
    """A payload with nested float / string / dict produces different bytes
    -- the primary cross-encoder regression the deep review flagged."""
    payload = {
        "a": {"price": 100.5, "name": "caf\u00e9"},
        "b": [1.0, 2.5, 3.0],
    }
    v1 = canonical_json(payload, version="v1")
    v2 = canonical_json(payload, version="v2")
    assert v1 != v2


def test_v2_rejects_nan_and_inf() -> None:
    with pytest.raises(ValueError, match="NaN/Infinity"):
        _canonical_json_v2({"x": float("nan")})
    with pytest.raises(ValueError, match="NaN/Infinity"):
        _canonical_json_v2({"x": float("inf")})
    with pytest.raises(ValueError, match="NaN/Infinity"):
        _canonical_json_v2([float("-inf")])


def test_v1_accepts_nan_and_inf() -> None:
    """v1 path retains the legacy behaviour -- Python's ``json.dumps``
    accepts NaN/Infinity by default and emits non-standard tokens
    (``NaN`` / ``Infinity``). The legacy v1.5.x verifiers expect this
    so we must NOT change the v1 behaviour."""
    payload = {"x": float("nan"), "y": float("inf")}
    out = canonical_json(payload, version="v1")
    assert "NaN" in out
    assert "Infinity" in out


def test_v2_rejects_non_json_native() -> None:
    """The v2 encoder rejects datetime / Decimal / Path / set so a
    cross-language verifier never sees an undefined coercion."""
    from datetime import datetime
    from decimal import Decimal
    from pathlib import Path

    with pytest.raises(TypeError, match="RFC 8785 encoder rejects"):
        _canonical_json_v2({"ts": datetime(2026, 5, 12)})
    with pytest.raises(TypeError, match="RFC 8785 encoder rejects"):
        _canonical_json_v2({"d": Decimal("1.0")})
    with pytest.raises(TypeError, match="RFC 8785 encoder rejects"):
        _canonical_json_v2({"p": Path("/tmp")})
    with pytest.raises(TypeError, match="RFC 8785 encoder rejects"):
        _canonical_json_v2({"s": {1, 2, 3}})


def test_v2_object_key_sort_order() -> None:
    """Object keys sort lexicographically by Python string order, which
    matches UTF-16 code points for BMP keys (the only case the FI
    schema uses)."""
    payload = {"b": 1, "a": 2, "c": 3, "A": 4}
    out = _canonical_json_v2(payload)
    # Python sorts: uppercase before lowercase ('A' < 'a' < 'b' < 'c').
    assert out == '{"A":4,"a":2,"b":1,"c":3}'


def test_v2_canonical_sha256_routes_correctly() -> None:
    """``canonical_sha256(payload, version='v2')`` hashes v2 bytes;
    v1 hashes v1 bytes; they differ on payloads with floats."""
    payload = {"x": 1.0}
    v1_hash = canonical_sha256(payload, version="v1")
    v2_hash = canonical_sha256(payload, version="v2")
    assert v1_hash != v2_hash
    assert v1_hash.startswith("sha256:")
    assert v2_hash.startswith("sha256:")


def test_canonical_json_v1_is_default() -> None:
    """``canonical_json(payload)`` with no version arg gives the legacy
    v1 bytes -- the backward-compat guarantee for v1.5.x callers."""
    payload = {"x": 1.0, "name": "caf\u00e9"}
    default = canonical_json(payload)
    explicit_v1 = canonical_json(payload, version="v1")
    assert default == explicit_v1


def test_canonical_json_rejects_bad_version() -> None:
    with pytest.raises(ValueError, match="version must be"):
        canonical_json({}, version="v3")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# coerce_for_canonical helper
# ---------------------------------------------------------------------------


def test_coerce_datetime_to_isoformat() -> None:
    from datetime import datetime

    out = coerce_for_canonical({"ts": datetime(2026, 5, 12, 23, 0, 0)})
    assert out == {"ts": "2026-05-12T23:00:00"}


def test_coerce_decimal_to_string() -> None:
    from decimal import Decimal

    out = coerce_for_canonical({"price": Decimal("1.50")})
    assert out == {"price": "1.50"}


def test_coerce_path_to_posix_string() -> None:
    from pathlib import PurePosixPath, PureWindowsPath

    posix = coerce_for_canonical(PurePosixPath("/tmp/foo"))
    assert posix == "/tmp/foo"
    win = coerce_for_canonical(PureWindowsPath("C:\\foo\\bar"))
    assert win == "C:/foo/bar"


def test_coerce_set_to_sorted_list() -> None:
    out = coerce_for_canonical({3, 1, 2})
    assert out == [1, 2, 3]


def test_coerce_bytes_raises() -> None:
    with pytest.raises(TypeError, match="bytes are not RFC 8785"):
        coerce_for_canonical(b"hello")


def test_coerce_recurses_into_nested() -> None:
    from datetime import datetime
    from decimal import Decimal

    payload = {
        "a": {"ts": datetime(2026, 1, 1), "d": Decimal("0.5")},
        "b": [datetime(2026, 1, 2)],
    }
    out = coerce_for_canonical(payload)
    assert out == {
        "a": {"ts": "2026-01-01T00:00:00", "d": "0.5"},
        "b": ["2026-01-02T00:00:00"],
    }


# ---------------------------------------------------------------------------
# RFC 8785 official conformance fixtures (from the RFC's appendix and
# common JCS test corpus). The canonical bytes a Java JCS / Rust
# serde_jcs verifier produces for these inputs are byte-identical to
# the strings below.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        # Empty object / array.
        ({}, "{}"),
        ([], "[]"),
        # Booleans + null.
        ({"a": True, "b": False, "c": None}, '{"a":true,"b":false,"c":null}'),
        # Integer rendering.
        ({"n": 0}, '{"n":0}'),
        ({"n": -1}, '{"n":-1}'),
        # Float rendering matches ECMA-262.
        ({"x": 1.0}, '{"x":1}'),
        ({"x": 1.5}, '{"x":1.5}'),
        ({"x": 0.001}, '{"x":0.001}'),
        # Strings with raw UTF-8 and minimal escapes.
        ({"k": 'a\\b"c'}, '{"k":"a\\\\b\\"c"}'),
        # Nested structure with sorted keys.
        (
            {"z": 1, "a": [1.0, 2.0, {"y": "x", "a": "b"}]},
            '{"a":[1,2,{"a":"b","y":"x"}],"z":1}',
        ),
    ],
)
def test_rfc8785_conformance_vectors(payload: object, expected: str) -> None:
    """Pin RFC 8785 canonical bytes for inputs a cross-language
    verifier (Java JCS, Rust ``serde_jcs``) would re-derive identically."""
    assert _canonical_json_v2(payload) == expected


def test_rfc8785_object_key_sorted_by_codepoint() -> None:
    """Section 3.2.3 of RFC 8785 prescribes UTF-16 code-point sort. For
    BMP-only keys this is identical to Python's default string sort."""
    payload = {"b": 1, "\u00e9": 2, "a": 3}
    out = _canonical_json_v2(payload)
    # Expected order: a < b < \u00e9 (ASCII before Latin-1 Supplement).
    assert out == '{"a":3,"b":1,"\u00e9":2}'


# ---------------------------------------------------------------------------
# Internal: integer ECMA-262 conformance
# ---------------------------------------------------------------------------


def test_ecma262_signed_zero_collapses() -> None:
    """ECMA-262 step 2: both +0 and -0 are emitted as the single token
    ``"0"`` so the canonical bytes are identical regardless of sign."""
    assert _ecma262_number_tostring(0.0) == "0"
    assert _ecma262_number_tostring(-0.0) == "0"
    assert _ecma262_number_tostring(0) == "0"
    # And the float-equal value goes through.
    assert math.copysign(1.0, -0.0) < 0  # sanity: -0.0 is signed
