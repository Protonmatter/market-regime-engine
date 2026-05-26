# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from typing import Any

import pandas as pd

try:  # pragma: no cover - optional import guard for minimal installs
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


@dataclass(frozen=True)
class NumericPolicy:
    """Wire-level numeric contract for XPro fixed-income artifacts."""

    prob_scale: int = 1_000_000
    bps_scale: int = 10_000
    price_scale: int = 1_000_000
    money_scale: int = 100
    rounding: str = "half_even"
    canonical_json: str = "rfc8785-jcs-v2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_NUMERIC_POLICY = NumericPolicy()


def _decimal(value: Any, *, label: str) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric, not bool")
    if isinstance(value, Decimal):
        dec = value
    elif isinstance(value, int):
        dec = Decimal(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
        dec = Decimal(str(value))
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError(f"{label} must not be empty")
        try:
            dec = Decimal(raw)
        except InvalidOperation as exc:
            raise ValueError(f"{label} must be decimal-compatible: {value!r}") from exc
    else:
        try:
            if pd.isna(value):
                raise ValueError(f"{label} must be finite")
        except TypeError:
            pass
        try:
            dec = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"{label} must be decimal-compatible: {value!r}") from exc
    if not dec.is_finite():
        raise ValueError(f"{label} must be finite")
    return dec


def _scaled_int(value: Any, scale: int, *, label: str) -> int:
    dec = _decimal(value, label=label)
    return int((dec * Decimal(scale)).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))


def prob_to_ppm(value: Any, policy: NumericPolicy = DEFAULT_NUMERIC_POLICY) -> int:
    dec = _decimal(value, label="probability")
    if dec < Decimal("0") or dec > Decimal("1"):
        raise ValueError("probability must be in [0, 1]")
    return _scaled_int(dec, policy.prob_scale, label="probability")


def bps_to_q4(value: Any, policy: NumericPolicy = DEFAULT_NUMERIC_POLICY) -> int:
    return _scaled_int(value, policy.bps_scale, label="basis_points")


def price_to_q6(value: Any, policy: NumericPolicy = DEFAULT_NUMERIC_POLICY) -> int:
    dec = _decimal(value, label="price")
    if dec <= Decimal("0"):
        raise ValueError("price must be positive")
    return _scaled_int(dec, policy.price_scale, label="price")


def money_to_cents(value: Any, policy: NumericPolicy = DEFAULT_NUMERIC_POLICY) -> int:
    dec = _decimal(value, label="money")
    if dec < Decimal("0"):
        raise ValueError("money must be non-negative")
    return _scaled_int(dec, policy.money_scale, label="money")


def timestamp_to_epoch_ns_str(value: Any) -> str:
    try:
        ts = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"timestamp must be parseable: {value!r}") from exc
    if ts.tzinfo is None:
        raise ValueError("timestamp must be tz-aware")
    ts = ts.tz_convert("UTC")
    return str(int(ts.value))


def _is_float(value: Any) -> bool:
    if isinstance(value, float):
        return True
    return bool(np is not None and isinstance(value, np.floating))


def assert_no_float_artifact(payload: Any, *, path: str = "root") -> None:
    """Raise when an artifact payload contains a Python/numpy float."""

    if _is_float(payload):
        raise TypeError(f"float value at {path}")
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            assert_no_float_artifact(value, path=f"{path}.{key}")
        return
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for idx, value in enumerate(payload):
            assert_no_float_artifact(value, path=f"{path}[{idx}]")


__all__ = [
    "DEFAULT_NUMERIC_POLICY",
    "NumericPolicy",
    "assert_no_float_artifact",
    "bps_to_q4",
    "money_to_cents",
    "price_to_q6",
    "prob_to_ppm",
    "timestamp_to_epoch_ns_str",
]
