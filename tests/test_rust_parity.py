"""Parity tests for the optional Rust kernels.

Skipped when ``mre_rust_ext`` is not built. CI runs them on the matrix slot
that ships the Rust toolchain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.rust_kernels import (
    bocpd_diag_update_rust,
    is_available,
    population_stability_index_rust,
    rolling_mahalanobis_distance_rust,
    wfst_viterbi_decode_rust,
)

if not is_available():  # pragma: no cover - skip when extension is missing
    pytest.skip("mre_rust_ext not built; run `maturin develop` in rust_ext/", allow_module_level=True)


pytestmark = pytest.mark.rust


def _python_psi(expected_pct: np.ndarray, actual_pct: np.ndarray) -> float:
    e = np.maximum(expected_pct, 1e-6)
    a = np.maximum(actual_pct, 1e-6)
    return float(np.sum((a - e) * np.log(a / e)))


def test_psi_parity():
    rng = np.random.default_rng(0)
    bins = 10
    e_counts = rng.integers(1, 100, size=bins).astype(float)
    a_counts = rng.integers(1, 100, size=bins).astype(float)
    e_pct = e_counts / e_counts.sum()
    a_pct = a_counts / a_counts.sum()
    py_psi = _python_psi(e_pct, a_pct)
    rs_psi = population_stability_index_rust(e_pct, a_pct)
    assert rs_psi is not None
    assert abs(py_psi - rs_psi) < 1e-12


def test_mahalanobis_parity():
    rng = np.random.default_rng(0)
    d = 6
    n = 60
    hist = rng.normal(size=(n, d))
    mu = hist.mean(axis=0)
    cov = np.cov(hist, rowvar=False)
    x = rng.normal(size=d)
    diff = x - mu
    py_md = float(np.sqrt(max(diff @ np.linalg.pinv(cov + 1e-4 * np.eye(d)) @ diff.T, 0.0)))
    rs_md = rolling_mahalanobis_distance_rust(x, mu, cov, ridge=1e-4)
    assert rs_md is not None
    assert abs(py_md - rs_md) < 1e-9


def test_wfst_viterbi_parity():
    from market_regime_engine.wfst import RegimeWFST

    states = sorted(RegimeWFST().states)
    s = len(states)
    rng = np.random.default_rng(0)
    obs = [states[int(i)] for i in rng.integers(0, s, size=20)]
    decoder = RegimeWFST()

    # Materialize cost / emission matrices to feed the kernel.
    cost = np.array(
        [[decoder.transition_cost(states[i], states[j]) for j in range(s)] for i in range(s)],
        dtype=np.float64,
    )
    emission = np.array(
        [[0.0 if states[j] == obs[t] else 1.0 for j in range(s)] for t in range(len(obs))],
        dtype=np.float64,
    )
    start_costs = np.array(
        [0.0 if states[j] == decoder.start else 0.8 for j in range(s)],
        dtype=np.float64,
    )
    py_path = decoder.decode(obs)
    rs_indices = wfst_viterbi_decode_rust(cost, start_costs, emission)
    assert rs_indices is not None
    rs_path = [states[int(i)] for i in rs_indices]
    assert py_path == rs_path


def test_bocpd_diag_update_parity():
    from market_regime_engine.bocpd import DiagonalStudentTBOCPD, RunningDiagState

    rng = np.random.default_rng(0)
    n = 24
    d = 4
    x = rng.normal(size=(n, d))
    detector = DiagonalStudentTBOCPD(max_run=16)

    # Run python reference to capture per-step probabilities.
    df = pd.DataFrame(x, index=pd.date_range("2010-01-01", periods=n, freq="MS"))
    py_out = detector.score(df)
    py_cp = py_out["change_point_prob"].to_numpy()

    # Now drive the Rust kernel one step at a time with the same prior.
    states = [RunningDiagState.prior(d, 1.0)]
    log_joint = np.array([0.0])
    rs_cp = []
    for t in range(n):
        # Pack states into arrays.
        n_states = len(states)
        state_n = np.array([s.n for s in states], dtype=np.int64)
        state_mean = np.zeros((n_states, d), dtype=np.float64)
        state_m2 = np.zeros((n_states, d), dtype=np.float64)
        for i, s in enumerate(states):
            state_mean[i] = s.mean
            state_m2[i] = s.m2
        result = bocpd_diag_update_rust(
            x[t],
            log_joint,
            state_n,
            state_mean,
            state_m2,
            prior_var=1.0,
            hazard=1.0 / 48.0,
            max_run=16,
        )
        assert result is not None
        new_log_joint, cp_prob, _, _, _, new_state_n, new_state_mean, new_state_m2, r_out, d_out = result
        rs_cp.append(float(cp_prob))
        new_log_joint = np.asarray(new_log_joint).reshape(int(r_out))
        new_state_n = np.asarray(new_state_n).reshape(int(r_out))
        new_state_mean = np.asarray(new_state_mean).reshape(int(r_out), int(d_out))
        new_state_m2 = np.asarray(new_state_m2).reshape(int(r_out), int(d_out))
        # Reconstruct python states from the Rust output.
        states = [
            RunningDiagState(
                n=int(new_state_n[i]),
                mean=new_state_mean[i].copy(),
                m2=new_state_m2[i].copy(),
                prior_var=1.0,
            )
            for i in range(int(r_out))
        ]
        log_joint = new_log_joint
    rs_cp_arr = np.asarray(rs_cp, dtype=float)
    np.testing.assert_allclose(py_cp, rs_cp_arr, atol=1e-9)
