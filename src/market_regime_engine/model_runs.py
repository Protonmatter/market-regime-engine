"""Immutable model-run records with a full reproducibility envelope.

The envelope captures everything an external auditor needs to re-derive a
forecast bit-for-bit:

- ``code_version``     short git revision (``git rev-parse --short HEAD``).
- ``code_sha``         long git revision; empty if the repo is not git.
- ``code_dirty``       True if the working tree has uncommitted changes.
- ``lockfile_hash``    sha256 of ``requirements-lock.txt`` (when present).
- ``platform``         OS / arch string from :func:`platform.platform`.
- ``python_version``   ``sys.version_info`` packed.
- ``feature_payload``  sha256 of the feature frame.
- ``output_payload``   sha256 of model outputs.
- ``vintage_payload``  sha256 of the ``feature_asof_values`` frame, when the
  forecast trains from the point-in-time materialization.
- ``rng_seeds``        explicit dict of named RNG seeds used by the engine.
- ``artifact_hash``    sha256 over the whole envelope above.

The historical ``ModelRun`` dataclass is preserved for back-compat (the storage
schema only requires the columns it already had); the additional reproducibility
fields land in ``metadata_json`` so older databases continue to work without
migration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from market_regime_engine.training_data import TrainingMode

logger = logging.getLogger(__name__)

_LOCKFILE_NAME = "requirements-lock.txt"

# v1.5 (PR-1 AF-9 / P1): the reproducibility envelope hashes ALL five
# platform lockfiles, not just the canonical one. The order below is
# deterministic so the dict keys hash stably across runs; missing
# files map to ``None``.
_LOCKFILE_FILES: tuple[str, ...] = (
    "requirements-lock.txt",
    "requirements-lock.core.txt",
    "requirements-lock.frontier-cpu-linux.txt",
    "requirements-lock.bayesian-cpu-linux.txt",
    "requirements-lock.dashboard.txt",
)

# v1.5 (PR-1 AF-13 / ASK-12): MRE_BUILD_SHA / MRE_BUILD_DIRTY env-var
# overrides make the repro envelope buildable inside container images
# where ``git`` is not on PATH. Truthy values follow the same set as
# Docker / CI conventions.
_BUILD_SHA_ENV = "MRE_BUILD_SHA"
_BUILD_DIRTY_ENV = "MRE_BUILD_DIRTY"
_TRUTHY = frozenset({"1", "true", "yes", "y", "on", "True", "TRUE", "Yes", "YES"})


@dataclass(frozen=True)
class ModelRun:
    run_id: str
    created_at_utc: str
    engine_version: str
    purpose: str
    data_start: str
    data_end: str
    feature_count: int
    observation_count: int
    model_count: int
    code_version: str
    artifact_hash: str
    metadata_json: str = "{}"


@dataclass(frozen=True)
class ReproEnvelope:
    """Full reproducibility envelope embedded into ``metadata_json``.

    v1.5 (PR-1 AF-9): ``lockfile_hashes`` is the dict of per-lockfile
    SHA-256 hashes covering all five platform lockfiles. ``lockfile_hash``
    is preserved as the canonical ``requirements-lock.txt`` scalar
    hash for v1.4-and-earlier consumers; new consumers should prefer
    the dict for cross-platform reproducibility.

    rng_seeds contract (v1.5+ — REVIEW.md §3.4 Q-3 / Q-11)
    ------------------------------------------------------
    Keys are RNG namespace strings. Recommended namespaces:

        "numpy"    → np.random.default_rng(seed)
        "jax"      → jax.random.PRNGKey(seed)
        "torch"    → torch.manual_seed(seed)
        "sklearn"  → sklearn estimators that accept random_state=

    Models using multiple namespaces MUST register all seeds in this
    dict so :func:`verify_run` can detect a silent change in the RNG
    source. Concretely the FI surface registers:

    - **execution_confidence (PR-5 deterministic baseline)**: no random
      state — the logit is closed-form. ``rng_seeds`` is left empty.
    - **execution_confidence (v1.5.1 calibrated logistic, planned)**:
      ``{"numpy", "sklearn"}``.
    - **Bayesian MS-VAR (frontier.bayesian_msvar)**: ``{"numpy", "jax"}``.
    - **PatchTST (frontier.patchtst_quantile, when enabled)**:
      ``{"numpy", "torch"}``.
    - **Hierarchical liquidity (frontier.hierarchical_liquidity)**:
      ``{"numpy", "jax"}``.
    - **HMM / BOCPD core (model_runs primary path)**: ``{"numpy"}``.

    Down-stream ``verify_run`` rejects an envelope whose ``rng_seeds``
    namespaces drift from the registered set on the same code SHA, so
    operators can detect a quiet seed-namespace migration after the
    fact.
    """

    code_version: str
    code_sha: str
    code_dirty: bool
    lockfile_hash: str
    platform: str
    python_version: str
    feature_payload: str
    output_payload: str
    vintage_payload: str
    rng_seeds: dict[str, int] = field(default_factory=dict)
    extra: dict[str, object] = field(default_factory=dict)
    lockfile_hashes: dict[str, str | None] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


_LEGACY_HASH_TAG = "v1_2_1_csv_str_round_trip"


def _hash_frame_legacy(df: pd.DataFrame | None) -> str:
    """v1.2.1 (and earlier) hash function.

    Kept verbatim for back-compat: ``mre verify-run --legacy-hash`` falls
    back to this implementation when the operator is verifying a run
    from before the v1.3 hash migration. Do NOT call this for new
    envelopes.
    """
    if df is None or df.empty:
        return hashlib.sha256(b"empty").hexdigest()
    stable = df.copy()
    for col in stable.columns:
        stable[col] = stable[col].astype(str)
    payload = stable.sort_index(axis=1).sort_values(list(stable.columns)).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_frame(df: pd.DataFrame | None) -> str:
    """v1.3 stable payload hash.

    The v1.2.1 ``_hash_frame`` cast every column to str then hashed the
    CSV. That is fragile across pandas versions: a float64 column may
    round-trip with different formatting (``1.0`` vs ``1``) on a pandas
    minor bump, which would trip ``verify-run`` against a previously-
    valid envelope. The v1.3 implementation instead:

    1. Sorts columns by name (canonical column order),
    2. Sorts rows by every column (canonical row order; stable across
       writes that preserve row content but not insertion order),
    3. For each column, streams ``(name, dtype_str, raw_bytes)`` into
       sha256 where ``raw_bytes`` is the column's underlying buffer
       via ``numpy.frombuffer`` for numeric columns or a length-prefixed
       UTF-8 byte stream for object/string columns.

    The resulting hash is invariant under ``df.copy()``, column
    re-ordering, and row-order shuffles (after canonical sort). It
    DOES change when a column's dtype changes — that is intentional:
    a silent float→int dtype migration is not a no-op for downstream
    code, and the audit envelope should detect it.

    Note: this is a breaking change for stored envelopes. v1.2.1 runs
    will fail ``verify-run`` until they are re-hashed; the
    ``--legacy-hash`` flag on ``mre verify-run`` falls back to
    :func:`_hash_frame_legacy` for forensic comparisons.
    """
    if df is None or df.empty:
        return hashlib.sha256(b"empty:" + _LEGACY_HASH_TAG.replace("_", "-").encode()).hexdigest()
    h = hashlib.sha256()
    # Canonical column order.
    cols = sorted(df.columns.astype(str).tolist())
    h.update(b"v1.3-frame-hash:")
    h.update(",".join(cols).encode("utf-8"))
    h.update(b"\x00")

    # Canonical row order: sort by every column (lexicographic) so that
    # writes that preserve row content but not insertion order produce
    # the same hash. ``kind="stable"`` keeps ties deterministic.
    sortable = df.reindex(columns=cols).copy()
    if not sortable.empty:
        try:
            sortable = sortable.sort_values(by=cols, kind="stable", na_position="last").reset_index(drop=True)
        except TypeError:
            # Mixed-type columns can break sort_values; fall back to a
            # string projection for ordering only (the actual hash
            # bytes still come from the original column buffers).
            order = sortable[cols].astype(str).agg("|".join, axis=1).argsort(kind="stable").to_numpy()
            sortable = sortable.iloc[order].reset_index(drop=True)

    for col in cols:
        series = sortable[col]
        dtype_str = str(series.dtype)
        h.update(f"{col}::{dtype_str}::".encode())
        kind = series.dtype.kind
        # Numeric (int / unsigned / float / complex) → raw NumPy buffer
        # so the hash is stable as long as the dtype is preserved.
        if kind in {"i", "u", "f", "c", "b"}:
            arr = np.ascontiguousarray(series.to_numpy())
            h.update(np.array([arr.shape[0]], dtype=np.int64).tobytes())
            h.update(arr.tobytes())
        elif kind == "M":
            arr = (
                series.values.view(np.int64)
                if hasattr(series.values, "view")
                else np.asarray(series).astype("datetime64[ns]").view(np.int64)
            )
            h.update(np.array([arr.shape[0]], dtype=np.int64).tobytes())
            h.update(np.ascontiguousarray(arr).tobytes())
        elif kind == "m":
            arr = (
                series.values.view(np.int64)
                if hasattr(series.values, "view")
                else np.asarray(series).astype("timedelta64[ns]").view(np.int64)
            )
            h.update(np.array([arr.shape[0]], dtype=np.int64).tobytes())
            h.update(np.ascontiguousarray(arr).tobytes())
        else:
            # object / string / category: length-prefixed UTF-8 stream.
            # NaN is mapped to a sentinel byte so it does not collide
            # with the empty string.
            count = len(series)
            h.update(np.array([count], dtype=np.int64).tobytes())
            for value in series:
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    h.update(b"\x01\x00\x00\x00\x00\x00\x00\x00")
                    continue
                buf = str(value).encode("utf-8")
                h.update(np.array([len(buf)], dtype=np.int64).tobytes())
                h.update(buf)
        h.update(b"\x00")
    return h.hexdigest()


def _git_revision(short: bool = True) -> str:
    """Resolve the running build's git revision.

    v1.5 (PR-1 AF-13 / ASK-12): consults ``MRE_BUILD_SHA`` first so a
    container image baked at build time without ``git`` on PATH can
    still produce a meaningful repro envelope. When the env var is
    set, ``short=True`` truncates to 7 chars to match
    ``git rev-parse --short HEAD``; ``short=False`` returns the full
    string verbatim.

    Falls back to ``git rev-parse`` then ``"unknown"`` with a
    WARNING log so an operational outage (git missing, repo
    detached) surfaces in the logs instead of silently degrading.
    """
    env_value = os.environ.get(_BUILD_SHA_ENV, "").strip()
    if env_value:
        return env_value[:7] if short else env_value
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception as exc:
        logger.warning("git rev-parse failed (%s); returning 'unknown'", exc)
        return "unknown"


def _git_dirty() -> bool:
    """Detect uncommitted changes in the working tree.

    v1.5 (PR-1 AF-13): consults ``MRE_BUILD_DIRTY`` first so a
    pre-baked container image can declare its dirty bit explicitly.
    Truthy values follow ``_TRUTHY``; anything else (including unset)
    falls back to the ``git status --porcelain`` probe.
    """
    env_value = os.environ.get(_BUILD_DIRTY_ENV)
    if env_value is not None and env_value != "":
        return env_value in _TRUTHY
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True)
        return bool(out.strip())
    except Exception as exc:
        logger.warning("git status failed (%s); assuming clean", exc)
        return False


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _lockfile_hash(root: Path | None = None) -> str:
    """Return the canonical ``requirements-lock.txt`` SHA-256 hash.

    Kept for v1.4 back-compat. Downstream consumers should prefer
    :func:`_lockfile_hashes_dict` which covers all five platform
    lockfiles (AF-9 / P1). Returns ``""`` when the canonical lockfile
    is missing so the v1.4.1 ``verify_run`` comparison path keeps
    its semantics.
    """
    root = root or _project_root()
    candidate = root / _LOCKFILE_NAME
    if not candidate.exists():
        return ""
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def _lockfile_hashes_dict(root: Path | None = None) -> dict[str, str | None]:
    """Return SHA-256 hashes for all platform lockfiles in deterministic order.

    Keys are the lockfile filenames (relative to the project root) and
    values are the hex SHA-256 digests, or ``None`` when the file is
    not present. Missing-file → ``None`` keeps the schema stable
    across CI runners that ship only a subset of lockfiles (the
    container build, for example, may only have ``requirements-lock.txt``
    + ``requirements-lock.dashboard.txt``).

    Any extra ``requirements-lock*.txt`` files discovered via glob
    are appended after the canonical five in sorted order so a
    future platform lockfile (e.g. ``requirements-lock.gpu-linux.txt``)
    is captured automatically without a code change.

    AF-9 / P1 (REVIEW.md section 3.1).
    """
    root = root or _project_root()
    out: dict[str, str | None] = {}
    for name in _LOCKFILE_FILES:
        candidate = root / name
        out[name] = hashlib.sha256(candidate.read_bytes()).hexdigest() if candidate.exists() else None
    # Pick up any additional lockfiles via glob; sort for determinism.
    known = set(_LOCKFILE_FILES)
    for extra in sorted(root.glob("requirements-lock*.txt")):
        name = extra.name
        if name in known:
            continue
        out[name] = hashlib.sha256(extra.read_bytes()).hexdigest()
    return out


def _python_version() -> str:
    return ".".join(str(x) for x in sys.version_info[:3])


# ---------------------------------------------------------------------------
# envelope construction
# ---------------------------------------------------------------------------


def build_repro_envelope(
    *,
    features: pd.DataFrame | None,
    model_outputs: pd.DataFrame | None,
    vintage_features: pd.DataFrame | None = None,
    rng_seeds: dict[str, int] | None = None,
    extra: dict[str, object] | None = None,
    legacy_hash: bool = False,
) -> ReproEnvelope:
    """Build a reproducibility envelope for the supplied frames.

    ``legacy_hash`` (default False) selects the v1.2.1 ``_hash_frame``
    implementation. Use only when forensically reconstructing an
    envelope that was originally written by a v1.2.1-or-earlier run;
    new runs should always use the default v1.3 stable hash.
    """
    hasher = _hash_frame_legacy if legacy_hash else _hash_frame
    return ReproEnvelope(
        code_version=_git_revision(short=True),
        code_sha=_git_revision(short=False),
        code_dirty=_git_dirty(),
        lockfile_hash=_lockfile_hash(),
        platform=platform.platform(),
        python_version=_python_version(),
        feature_payload=hasher(features),
        output_payload=hasher(model_outputs),
        vintage_payload=hasher(vintage_features),
        rng_seeds=dict(rng_seeds or {}),
        extra=dict(extra or {}),
        lockfile_hashes=_lockfile_hashes_dict(),
    )


def envelope_to_json(envelope: ReproEnvelope) -> str:
    return json.dumps(asdict(envelope), sort_keys=True, default=str)


def envelope_artifact_hash(envelope: ReproEnvelope) -> str:
    return hashlib.sha256(envelope_to_json(envelope).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def create_model_run(
    *,
    engine_version: str,
    purpose: str,
    features: pd.DataFrame,
    model_outputs: pd.DataFrame,
    metadata: dict | None = None,
    vintage_features: pd.DataFrame | None = None,
    rng_seeds: dict[str, int] | None = None,
    training_audit: dict | None = None,
) -> ModelRun:
    """Build an immutable model-run record.

    ``training_audit`` is the structured audit dict emitted by
    :func:`training_data.load_training_panel`. When provided, it is embedded
    in ``metadata.training_audit`` AND copied onto
    ``ReproEnvelope.extra["training_audit"]`` so ``mre verify-run`` can
    detect a silent LEGACY-fallback after the fact.
    """
    extra: dict[str, object] = {"engine_version": engine_version, "purpose": purpose}
    if training_audit:
        extra["training_audit"] = dict(training_audit)
    envelope = build_repro_envelope(
        features=features,
        model_outputs=model_outputs,
        vintage_features=vintage_features,
        rng_seeds=rng_seeds,
        extra=extra,
    )
    artifact_hash = envelope_artifact_hash(envelope)

    dates = (
        pd.to_datetime(features["date"])
        if features is not None and not features.empty and "date" in features
        else pd.Series(dtype="datetime64[ns]")
    )
    run_seed = {
        "artifact_hash": artifact_hash,
        "created_at_hint": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    run_id = hashlib.sha256(json.dumps(run_seed, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    user_metadata = dict(metadata or {})
    user_metadata.setdefault("repro_envelope", asdict(envelope))
    if training_audit:
        user_metadata["training_audit"] = dict(training_audit)

    return ModelRun(
        run_id=run_id,
        created_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        engine_version=engine_version,
        purpose=purpose,
        data_start=dates.min().strftime("%Y-%m-%d") if not dates.empty else "unknown",
        data_end=dates.max().strftime("%Y-%m-%d") if not dates.empty else "unknown",
        feature_count=int(features["feature_name"].nunique())
        if features is not None and not features.empty and "feature_name" in features
        else 0,
        observation_count=int(features["date"].nunique())
        if features is not None and not features.empty and "date" in features
        else 0,
        model_count=int(model_outputs["model_name"].nunique())
        if model_outputs is not None and not model_outputs.empty and "model_name" in model_outputs
        else 0,
        code_version=envelope.code_version,
        artifact_hash=artifact_hash,
        metadata_json=json.dumps(user_metadata, sort_keys=True, default=str),
    )


def model_run_frame(run: ModelRun) -> pd.DataFrame:
    return pd.DataFrame([asdict(run)])


def _canonicalise(value: object) -> object:
    """Round-trip ``value`` through ``json.dumps(..., sort_keys=True)``.

    The v1.2.1 ``verify_run`` skipped ``rng_seeds`` outright on the
    grounds that "the dict is unordered after JSON round-trip and we
    don't want false drift". That justification was wrong on two
    counts: Python dict equality has been order-insensitive since
    forever, and the JSON round-trip is exactly the canonicalisation
    step that proves it. v1.4.1 keeps the round-trip but uses it as
    the canonical form for comparison instead of an excuse to skip
    the field.

    For non-JSON-serialisable values, fall back to ``str(value)`` so
    the comparison is always defined.
    """
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def verify_run(
    run_id: str,
    run_row: pd.Series,
    *,
    current_envelope: ReproEnvelope,
    ignore_rng_seeds: bool = False,
) -> dict:
    """Compare a stored run's repro envelope to the current environment.

    Returns a small report describing which fields drifted; ``approved=True`` is
    only set when **every** envelope field matches **and** the embedded
    ``extra.training_audit`` (when present) shows the run was trained against
    point-in-time features.

    v1.2.1 changes:

    - ``extra`` is no longer in the skip set. The training audit lives in
      ``extra.training_audit`` and is now structurally compared so the
      verify-run report cannot tell an operator "approved" while the run
      was secretly trained on legacy / revised data.
    - ``training_mode_drift`` is appended to ``differences`` when the
      stored ``training_audit.mode_used`` is not ``"point_in_time"``. The
      report ``approved`` is then ``False``.
    - ``warnings`` carries non-fatal advisories. When
      ``training_audit.fallback_authorized`` is True the run was trained on
      a deliberately-authorized legacy fallback; the run is still approved
      but ``"legacy_fallback_authorized"`` is appended to ``warnings`` so a
      change-management gate can see the conscious downgrade.

    v1.4.1 changes (items D, E):

    - ``extra`` is now compared **structurally** for *every* sub-key
      (not just ``training_audit``). Arbitrary other fields (e.g.
      ``engine_version``, ``purpose``, or any operator-supplied extra)
      that diverge between stored and current envelopes surface as
      ``differences["extra:<key>"] = {"stored": ..., "current": ...}``
      and flip ``approved`` to ``False``. The friendly per-key
      ``training_mode_drift`` / ``legacy_fallback_authorized``
      semantics from v1.2.1 are preserved verbatim — the new
      structural compare is *additive* on top of them.
    - ``rng_seeds`` is no longer in the unconditional skip set; it is
      compared via canonicalised JSON round-trip so dict insertion
      order does not produce false drift. Pass
      ``ignore_rng_seeds=True`` (CLI: ``--ignore-rng-seeds``) to
      restore the v1.2.1 skip behaviour for stochastic-seed-rerun
      workflows.

    Caller contract: ``current_envelope.extra`` is whatever the caller
    re-derived for the *current* run. The CLI builds the current
    envelope from the live frames and does not re-attach the stored
    ``training_audit`` — so an operator who stores arbitrary keys
    under ``extra`` and re-runs ``verify-run`` *will* see those keys
    flagged unless they re-attach the same ``extra`` dict, which is
    by design.
    """
    try:
        meta = json.loads(run_row.get("metadata_json", "{}") or "{}")
    except Exception:
        meta = {}
    stored = meta.get("repro_envelope", {})
    diffs: dict[str, object] = {}
    warnings_list: list[str] = []
    for key, stored_value in stored.items():
        if key == "lockfile_hashes":
            # v1.5 AF-9: dict comparison when both sides have the new
            # field; non-fatal warning when only one side has it
            # (legacy v1.4 envelopes still carry the scalar
            # ``lockfile_hash`` only).
            current_lh = getattr(current_envelope, "lockfile_hashes", {}) or {}
            stored_lh = stored_value if isinstance(stored_value, dict) else {}
            if not stored_lh and current_lh:
                warnings_list.append("lockfile_hashes_legacy_envelope")
                continue
            if stored_lh and not current_lh:
                warnings_list.append("lockfile_hashes_current_envelope_missing")
                continue
            if _canonicalise(stored_lh) != _canonicalise(current_lh):
                diffs["lockfile_hashes"] = {
                    "stored": stored_lh,
                    "current": current_lh,
                }
            continue
        if key == "rng_seeds":
            if ignore_rng_seeds:
                continue
            current_seeds = getattr(current_envelope, "rng_seeds", {}) or {}
            if _canonicalise(stored_value) != _canonicalise(current_seeds):
                diffs["rng_seeds"] = {
                    "stored": stored_value,
                    "current": current_seeds,
                }
            continue
        if key == "extra":
            stored_extra = stored_value if isinstance(stored_value, dict) else {}
            current_extra_raw = getattr(current_envelope, "extra", {}) or {}
            current_extra = current_extra_raw if isinstance(current_extra_raw, dict) else {}
            # Friendly per-key handling for ``training_audit`` is preserved
            # exactly as v1.2.1 specified — the structural compare below
            # then compares *every other* extra key on top.
            audit = stored_extra.get("training_audit") if isinstance(stored_extra.get("training_audit"), dict) else None
            if audit is not None:
                stored_mode = str(audit.get("mode_used", "unknown"))
                if stored_mode != TrainingMode.POINT_IN_TIME.value:
                    diffs["training_mode_drift"] = {
                        "stored_mode": stored_mode,
                        "expected": TrainingMode.POINT_IN_TIME.value,
                    }
                if audit.get("fallback_authorized") is True:
                    warnings_list.append("legacy_fallback_authorized")
            # Structural compare of every non-``training_audit`` key.
            # ``training_audit`` itself has its own friendly handling
            # above so we exclude it from the structural diff to avoid
            # double-reporting.
            keys = set(stored_extra.keys()) | set(current_extra.keys())
            for sub_key in sorted(keys):
                if sub_key == "training_audit":
                    continue
                stored_sub = stored_extra.get(sub_key)
                current_sub = current_extra.get(sub_key)
                if _canonicalise(stored_sub) != _canonicalise(current_sub):
                    diffs[f"extra:{sub_key}"] = {
                        "stored": stored_sub,
                        "current": current_sub,
                    }
            continue
        current_value = getattr(current_envelope, key, None)
        if isinstance(current_value, bool) and not isinstance(stored_value, bool):
            current_value = bool(current_value)
        if stored_value != current_value:
            diffs[key] = {"stored": stored_value, "current": current_value}
    return {
        "run_id": run_id,
        "approved": not diffs and bool(stored),
        "missing_envelope": not stored,
        "differences": diffs,
        "warnings": warnings_list,
        "lockfile_present": bool(os.path.exists(_LOCKFILE_NAME)),
    }


__all__ = [
    "ModelRun",
    "ReproEnvelope",
    "build_repro_envelope",
    "create_model_run",
    "envelope_artifact_hash",
    "envelope_to_json",
    "model_run_frame",
    "verify_run",
]
