"""Soft import wrapper for the optional Rust kernels.

The compiled extension module ships as ``mre_rust_ext`` and is built via::

    cd rust_ext
    maturin develop --release   # or `maturin build --release`

When the extension is not present, every helper here returns ``None`` so the
caller falls back to the Python reference implementation. The
:func:`is_available` predicate is the canonical "is Rust live?" check used by
both the bench harness and the production hot-path call sites.

Parity tests in ``tests/test_rust_parity.py`` are marked with the ``rust``
pytest marker so CI matrices that don't ship the Rust toolchain can skip them
cleanly.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only when the extension is built
    import mre_rust_ext

    _AVAILABLE = True
    # v1.3 (item J): log the compiled wheel's version when present so an
    # operator can confirm which platform-specific wheel got installed.
    # The attribute is optional (older wheels did not export ``__version__``).
    _WHEEL_VERSION = getattr(mre_rust_ext, "__version__", None) or getattr(mre_rust_ext, "VERSION", None)
    if _WHEEL_VERSION:
        log.info("mre_rust_ext loaded: version=%s", _WHEEL_VERSION)
    else:
        log.info("mre_rust_ext loaded (no __version__ attribute exported)")
except Exception as exc:  # pragma: no cover - default in CI without Rust
    mre_rust_ext = None
    _AVAILABLE = False
    _WHEEL_VERSION = None
    log.debug("mre_rust_ext unavailable: %s", exc)


def is_available() -> bool:
    return _AVAILABLE


def wheel_version() -> str | None:
    """Return the loaded Rust wheel's version, or ``None`` if not built."""
    return _WHEEL_VERSION


def population_stability_index_rust(expected_pct: np.ndarray, actual_pct: np.ndarray) -> float | None:
    if not _AVAILABLE:
        return None
    return float(
        mre_rust_ext.population_stability_index_kernel(
            np.ascontiguousarray(expected_pct, dtype=np.float64),
            np.ascontiguousarray(actual_pct, dtype=np.float64),
        )
    )


def rolling_mahalanobis_distance_rust(
    x: np.ndarray, mean: np.ndarray, cov: np.ndarray, ridge: float = 1e-4
) -> float | None:
    if not _AVAILABLE:
        return None
    return float(
        mre_rust_ext.rolling_mahalanobis_distance_kernel(
            np.ascontiguousarray(x, dtype=np.float64),
            np.ascontiguousarray(mean, dtype=np.float64),
            np.ascontiguousarray(cov, dtype=np.float64),
            float(ridge),
        )
    )


def wfst_viterbi_decode_rust(cost: np.ndarray, start_costs: np.ndarray, emission: np.ndarray) -> np.ndarray | None:
    if not _AVAILABLE:
        return None
    return np.asarray(
        mre_rust_ext.wfst_viterbi_decode(
            np.ascontiguousarray(cost, dtype=np.float64),
            np.ascontiguousarray(start_costs, dtype=np.float64),
            np.ascontiguousarray(emission, dtype=np.float64),
        )
    )


def bocpd_diag_update_rust(
    x: np.ndarray,
    log_joint: np.ndarray,
    state_n: np.ndarray,
    state_mean: np.ndarray,
    state_m2: np.ndarray,
    *,
    prior_var: float = 1.0,
    hazard: float = 1.0 / 48.0,
    max_run: int = 96,
) -> tuple | None:
    if not _AVAILABLE:
        return None
    return mre_rust_ext.bocpd_diag_update(
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(log_joint, dtype=np.float64),
        np.ascontiguousarray(state_n, dtype=np.int64),
        np.ascontiguousarray(state_mean, dtype=np.float64),
        np.ascontiguousarray(state_m2, dtype=np.float64),
        float(prior_var),
        float(hazard),
        int(max_run),
    )


__all__ = [
    "bocpd_diag_update_rust",
    "is_available",
    "population_stability_index_rust",
    "rolling_mahalanobis_distance_rust",
    "wfst_viterbi_decode_rust",
    "wheel_version",
]
