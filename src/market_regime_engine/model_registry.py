# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict


@dataclass(frozen=True)
class ModelCard:
    model_name: str
    version: str
    target: str
    horizon: str
    training_start: str
    training_end: str
    feature_count: int
    observations: int
    objective: str
    known_limitations: list[str]
    validation_metrics: dict[str, float]
    created_at_utc: str
    artifact_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelCardKwargs(TypedDict):
    """Typed payload of the data fields of :class:`ModelCard`.

    Used to narrow the kwargs dict at the :func:`create_model_card` call
    site so that ``ModelCard(**base, ...)`` type-checks under mypy strict
    arg-type inference. v1.6.0 (REVIEW_DEEP_V1_5_2.md §4.2): without
    this TypedDict the literal dict is inferred as ``dict[str, object]``
    and mypy flags four ``arg-type`` errors at the unpack site.
    """

    model_name: str
    version: str
    target: str
    horizon: str
    training_start: str
    training_end: str
    feature_count: int
    observations: int
    objective: str
    known_limitations: list[str]
    validation_metrics: dict[str, float]


def stable_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def create_model_card(
    *,
    model_name: str,
    version: str,
    target: str,
    horizon: str,
    training_start: str,
    training_end: str,
    feature_count: int,
    observations: int,
    objective: str,
    known_limitations: list[str],
    validation_metrics: dict[str, float],
) -> ModelCard:
    base: ModelCardKwargs = {
        "model_name": model_name,
        "version": version,
        "target": target,
        "horizon": horizon,
        "training_start": training_start,
        "training_end": training_end,
        "feature_count": feature_count,
        "observations": observations,
        "objective": objective,
        "known_limitations": known_limitations,
        "validation_metrics": validation_metrics,
    }
    return ModelCard(
        **base,
        created_at_utc=datetime.now(UTC).isoformat(),
        artifact_hash=stable_hash(dict(base)),
    )


def write_model_card(card: ModelCard, directory: str | Path = "data/model_cards") -> Path:
    outdir = Path(directory)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{card.model_name}_{card.target}_{card.horizon}_{card.version}.json".replace("/", "_")
    path.write_text(json.dumps(card.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path
