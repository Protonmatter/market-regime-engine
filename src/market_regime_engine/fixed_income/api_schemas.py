# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from market_regime_engine.fixed_income.liquidity_stress import (
    output_to_dict as liquidity_output_to_dict,
)
from market_regime_engine.fixed_income.schemas import (
    CreditRegimeOutput,
    ExecutionConfidenceRequest,
    ExecutionConfidenceResponse,
    LiquidityStressOutput,
    TcaRegimeSegment,
)


class ExecutionConfidenceRequestModel(BaseModel):
    """Pydantic v2 validation model for POST /v1/execution_confidence body.

    The dataclass :class:`ExecutionConfidenceRequest` is the internal
    contract; this Pydantic shim wraps it for the FastAPI request body so
    type errors at the boundary surface as 422 rather than slipping into
    the scorer as ``TypeError``."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    timestamp: str = Field(..., description="ISO-8601 UTC timestamp with explicit tz info")
    cusip: str = Field(..., min_length=8, max_length=12)
    side: Literal["buy", "sell"]
    notional: float = Field(..., gt=0, le=500_000_000.0)
    protocol: Literal["Auto-X", "RFQ", "Manual"]
    limit_price: float | None = Field(default=None, gt=0)
    urgency: Literal["low", "normal", "high"] = "normal"
    request_id: str = Field(..., min_length=1, max_length=128)
    sector: str | None = None
    rating: str | None = None
    maturity_bucket: str | None = None
    client_request_id: str | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("timestamp")
    @classmethod
    def _ts_must_be_utc_iso8601(cls, v: str) -> str:
        """Coerce inbound timestamp to canonical UTC ``...Z`` form.

        v1.6.0 (REVIEW_DEEP_V1_5_2.md A9 / Finding §3.4): the
        v1.5.x validator accepted any tz-aware ISO-8601 string
        (``+05:30``, ``-08:00``, ``Z``) without normalisation, so
        the same logical instant submitted from different
        operator timezones produced different canonical bytes and
        therefore different artifact hashes. The validator now
        rewrites every accepted timestamp to ``YYYY-MM-DDTHH:MM:SS[.ffffff]Z`` (UTC, microseconds-when-present), so two requests
        for the same instant under different offsets produce
        byte-identical canonical payloads and therefore identical
        ``artifact_hash`` values.
        """
        import pandas as pd

        try:
            parsed = pd.Timestamp(v)
        except Exception as exc:
            raise ValueError(f"timestamp must be ISO-8601: {v!r}") from exc
        if parsed.tzinfo is None:
            raise ValueError(f"timestamp must carry explicit tz info (e.g. 'Z' suffix): {v!r}")
        utc_ts = parsed.tz_convert("UTC")
        canonical = utc_ts.strftime("%Y-%m-%dT%H:%M:%S")
        if utc_ts.microsecond:
            canonical += f".{utc_ts.microsecond:06d}"
        canonical += "Z"
        return canonical

    @field_validator("cusip")
    @classmethod
    def _cusip_must_be_alphanumeric(cls, v: str) -> str:
        if not v.isalnum():
            raise ValueError(f"cusip must be alphanumeric: {v!r}")
        return v.upper()

    @field_validator("metadata")
    @classmethod
    def _metadata_size_and_depth(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """Cap metadata payload at 8 KiB canonical-JSON / depth 5.

        v1.6.0 (REVIEW_DEEP_V1_5_2.md F9 / Finding §3.13): the
        v1.5.x contract was unbounded. Production metadata stays
        well below 1 KB so the 8192-byte canonical-JSON cap and
        depth-5 nesting cap reject pathological payloads (deeply
        nested dicts, MBs of free-form blobs) that would otherwise
        inflate every downstream canonical-JSON / artifact-hash
        computation. The body-size middleware already caps the
        full request body at 32 KB; this validator narrows the
        contract to metadata specifically.
        """
        if v is None:
            return v
        import json as _json

        try:
            encoded = _json.dumps(v, sort_keys=True, default=str)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metadata must be JSON-serialisable: {exc}") from exc
        if len(encoded) > 8192:
            raise ValueError(f"metadata too large: {len(encoded)} bytes > 8192 byte cap")

        def _depth(obj: Any, current: int = 0) -> int:
            if current > 5:
                return current
            if isinstance(obj, dict):
                if not obj:
                    return current + 1
                return max(_depth(val, current + 1) for val in obj.values())
            if isinstance(obj, list):
                if not obj:
                    return current + 1
                return max(_depth(val, current + 1) for val in obj)
            return current + 1

        if _depth(v) > 5:
            raise ValueError("metadata nesting depth exceeds 5 levels")
        return v

    def to_dataclass(self) -> ExecutionConfidenceRequest:
        """Project the Pydantic model onto the internal dataclass."""
        return ExecutionConfidenceRequest(
            timestamp=self.timestamp,
            cusip=self.cusip,
            side=self.side,
            notional=float(self.notional),
            protocol=self.protocol,
            limit_price=float(self.limit_price) if self.limit_price is not None else None,
            urgency=self.urgency,
            sector=self.sector,
            rating=self.rating,
            maturity_bucket=self.maturity_bucket,
            client_request_id=self.client_request_id or self.request_id,
            metadata=dict(self.metadata or {}),
        )


XProProtocol = Literal["Auto-X", "RFQ", "Manual"]


def _default_candidate_protocols() -> list[XProProtocol]:
    return ["Auto-X", "RFQ", "Manual"]


class XProDecisionRequestModel(ExecutionConfidenceRequestModel):
    """Pydantic body for POST /v1/xpro/decision."""

    candidate_protocols: list[XProProtocol] = Field(
        default_factory=_default_candidate_protocols,
        min_length=1,
        max_length=3,
    )
    decision_id: str | None = Field(default=None, min_length=1, max_length=128)


def credit_regime_output_to_dict(output: CreditRegimeOutput) -> dict[str, Any]:
    """JSON-serialisable dict form of :class:`CreditRegimeOutput`.

    Drivers are exposed as a list (not a tuple) so ``json.dumps`` does
    not need ``default=str``. Mirrors the AGENT.md §6.1 output example
    exactly.

    PR-7 §N (PR-13): the response also exposes
    ``metadata.signal_age_seconds`` (computed against the current UTC
    clock) so Auto-X consumers can check the SLA without parsing the
    timestamp twice.
    """
    out = asdict(output)
    out["drivers"] = list(output.drivers)
    # v1.6.0 (REVIEW_DEEP_V1_5_2.md F2 / Finding §3.15):
    # ``output.metadata`` is a read-only ``_ReadOnlyMetadata``
    # (dict subclass). ``dataclasses.asdict`` preserves the
    # subclass and the subclass raises on mutation, so coerce
    # to a plain dict before attaching derived fields.
    out["metadata"] = dict(out.get("metadata", {}) or {})
    out["metadata"].setdefault("signal_age_seconds", _signal_age_seconds_now(output.timestamp))
    return out


def _signal_age_seconds_now(ts: str | None) -> float:
    """Return seconds between ``ts`` (ISO-8601) and now (UTC).

    Returns ``float('inf')`` when ``ts`` is ``None`` so consumers that
    rely on the SLA gate (≤ MRE_FI_MAX_SIGNAL_STALENESS_SEC) trip
    automatically on a missing timestamp.
    """
    if ts is None:
        return float("inf")
    try:
        import pandas as pd

        parsed = pd.Timestamp(ts)
    except Exception:
        return float("inf")
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    now = pd.Timestamp.now(tz="UTC")
    return float((now - parsed).total_seconds())


def liquidity_stress_output_to_dict(output: LiquidityStressOutput) -> dict[str, Any]:
    """JSON-serialisable dict form of :class:`LiquidityStressOutput`.

    Re-export of :func:`fixed_income.liquidity_stress.output_to_dict`
    on the API namespace; PR-7 §N enriches the dict with
    ``metadata.signal_age_seconds`` so Auto-X consumers see the same
    staleness signal across all FI endpoints.
    """
    out = liquidity_output_to_dict(output)
    # v1.6.0 F2: coerce read-only metadata to plain dict before mutation.
    out["metadata"] = dict(out.get("metadata", {}) or {})
    out["metadata"].setdefault("signal_age_seconds", _signal_age_seconds_now(output.timestamp))
    return out


def execution_confidence_response_to_dict(
    response: ExecutionConfidenceResponse,
) -> dict[str, Any]:
    """JSON-serialisable dict form of :class:`ExecutionConfidenceResponse`."""
    return asdict(response)


def tca_regime_segment_to_dict(segment: TcaRegimeSegment) -> dict[str, Any]:
    """JSON-serialisable dict form of :class:`TcaRegimeSegment`.

    The dataclass already round-trips through ``asdict``; this wrapper
    coerces the ``timestamp`` to an ISO-8601 string with the Z suffix
    (mirrors the other FI output converters).
    """
    out = asdict(segment)
    ts = segment.timestamp
    out["timestamp"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    return out


__all__ = [
    "ExecutionConfidenceRequestModel",
    "XProDecisionRequestModel",
    "_signal_age_seconds_now",
    "credit_regime_output_to_dict",
    "execution_confidence_response_to_dict",
    "liquidity_stress_output_to_dict",
    "tca_regime_segment_to_dict",
]
