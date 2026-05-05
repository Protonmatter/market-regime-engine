//! High-performance kernels for the market regime engine.
//!
//! Each function is parity-tested against the Python reference implementation
//! in `tests/test_rust_parity.py`. Promotion of a kernel into the production
//! Python path requires the parity test to pass with `atol=1e-9`.
//!
//! Kernels exposed here:
//!
//! - `bocpd_diag_update` — single-step diagonal Student-t BOCPD update: takes
//!   the current run-length log-joint, the per-state running diagonal stats,
//!   and a new observation; returns the new posterior probabilities and
//!   updated state. Equivalent to one iteration of
//!   `DiagonalStudentTBOCPD.score`'s inner loop.
//! - `wfst_viterbi_decode` — log-space Viterbi over a precomputed transition
//!   cost matrix. Equivalent to `RegimeWFST.decode` once the cost matrix has
//!   been materialised.
//! - `population_stability_index` — PSI between two histograms.
//! - `rolling_mahalanobis_distance` — single-row Mahalanobis distance against
//!   a window mean and ridge-stabilised covariance.

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

const LN_PI: f64 = 1.144_729_885_849_400_2_f64;
const PI_F: f64 = std::f64::consts::PI;

#[inline]
fn lgamma(x: f64) -> f64 {
    libm_lgamma(x)
}

// We avoid pulling libm as a dependency by re-implementing lgamma via the
// Lanczos approximation. Accuracy is ~1e-14 in the regime we care about.
fn libm_lgamma(x: f64) -> f64 {
    // Use the Lanczos g=7 approximation. Numerical Recipes Ch. 6.
    const G: f64 = 7.0;
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula.
        return (PI_F / (PI_F * x).sin()).ln() - libm_lgamma(1.0 - x);
    }
    let mut a = COEFFS[0];
    let xx = x - 1.0;
    for (i, c) in COEFFS.iter().enumerate().skip(1) {
        a += c / (xx + i as f64);
    }
    let t = xx + G + 0.5;
    0.5 * (2.0 * PI_F).ln() + (xx + 0.5) * t.ln() - t + a.ln()
}

#[inline]
fn logsumexp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }
    let m = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return f64::NEG_INFINITY;
    }
    let s: f64 = values.iter().map(|v| (v - m).exp()).sum();
    m + s.ln()
}

#[inline]
fn student_t_logpdf_diag(
    x: &[f64],
    mean: &[f64],
    variance: &[f64],
    n_obs: usize,
    min_df: f64,
) -> f64 {
    let df = min_df.max((n_obs + 1) as f64);
    let scale_factor = 1.0 + 1.0 / ((n_obs + 1).max(1) as f64);
    let c = lgamma((df + 1.0) / 2.0) - lgamma(df / 2.0) - 0.5 * (df * PI_F).ln();
    let mut total = 0.0;
    for (i, xi) in x.iter().enumerate() {
        let var = (variance[i] * scale_factor).max(1e-8);
        let scale = var.sqrt();
        let z = (xi - mean[i]) / scale;
        total += c - scale.ln() - ((df + 1.0) / 2.0) * (1.0 + (z * z) / df).ln();
    }
    total
}

/// Single BOCPD update step using a diagonal Student-t predictive density.
///
/// Inputs (all 1-D float64 arrays unless noted):
/// - `x`               : observation, shape (d,)
/// - `log_joint`       : log run-length joint, shape (R,)
/// - `state_n`         : per-state observation counts, shape (R,)
/// - `state_mean`      : per-state running mean, shape (R, d)
/// - `state_m2`        : per-state running M2 (Welford), shape (R, d)
/// - `prior_var`       : prior variance scalar
/// - `hazard`          : change-point hazard
/// - `max_run`         : truncation length
///
/// Returns a tuple `(new_log_joint, cp_prob, run_length_mean, map_run_length,
/// pred_log_likelihood, new_state_n, new_state_mean, new_state_m2)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bocpd_diag_update<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    log_joint: PyReadonlyArray1<'py, f64>,
    state_n: PyReadonlyArray1<'py, i64>,
    state_mean: PyReadonlyArray2<'py, f64>,
    state_m2: PyReadonlyArray2<'py, f64>,
    prior_var: f64,
    hazard: f64,
    max_run: usize,
) -> PyResult<Bound<'py, PyTuple>> {
    let xv = x.as_slice()?;
    let d = xv.len();
    let log_joint_v = log_joint.as_slice()?.to_vec();
    let state_n_v: Vec<i64> = state_n.as_slice()?.to_vec();
    let state_mean_arr = state_mean.as_array().to_owned();
    let state_m2_arr = state_m2.as_array().to_owned();
    let r = log_joint_v.len();
    if state_n_v.len() != r || state_mean_arr.shape()[0] != r || state_m2_arr.shape()[0] != r {
        return Err(PyValueError::new_err("inconsistent state shapes"));
    }
    if state_mean_arr.shape()[1] != d || state_m2_arr.shape()[1] != d {
        return Err(PyValueError::new_err("dimension mismatch"));
    }

    // Predictive log-pdf per state.
    let mut pred_logs = Vec::with_capacity(r);
    for i in 0..r {
        let mean_row: Vec<f64> = state_mean_arr.row(i).to_vec();
        let m2_row: Vec<f64> = state_m2_arr.row(i).to_vec();
        let n = state_n_v[i] as usize;
        let var: Vec<f64> = if n < 2 {
            vec![prior_var; d]
        } else {
            m2_row
                .iter()
                .map(|m| (m / (n as f64 - 1.0).max(1.0)).max(1e-8))
                .collect()
        };
        let mean_eff: Vec<f64> = if n == 0 { vec![0.0; d] } else { mean_row };
        pred_logs.push(student_t_logpdf_diag(xv, &mean_eff, &var, n, 3.0));
    }

    let h = hazard.max(1e-12).min(1.0 - 1e-12);
    let log_h = h.ln();
    let log_1mh = (1.0 - h).ln();

    let mut combined = Vec::with_capacity(r);
    for i in 0..r {
        combined.push(log_joint_v[i] + pred_logs[i]);
    }
    let pred_norm = logsumexp(&combined);
    let cp_log = logsumexp(&combined.iter().map(|v| v + log_h).collect::<Vec<f64>>());
    let growth: Vec<f64> = combined.iter().map(|v| v + log_1mh).collect();

    let truncate = (growth.len() + 1).min(max_run + 1);
    let mut new_log_joint = vec![0.0_f64; truncate];
    new_log_joint[0] = cp_log;
    let kept = truncate - 1;
    new_log_joint[1..1 + kept].copy_from_slice(&growth[..kept]);
    let norm = logsumexp(&new_log_joint);
    for v in new_log_joint.iter_mut() {
        *v -= norm;
    }
    let probs: Vec<f64> = new_log_joint.iter().map(|v| v.exp()).collect();
    let prob_sum: f64 = probs.iter().sum();
    let probs: Vec<f64> = probs.iter().map(|p| p / prob_sum).collect();
    let cp_prob = probs[0];
    let run_lengths: Vec<f64> = (0..probs.len()).map(|i| i as f64).collect();
    let run_mean: f64 = run_lengths.iter().zip(probs.iter()).map(|(rl, p)| rl * p).sum();
    let map_run = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as i64)
        .unwrap_or(0);

    // Update states: prepend a fresh prior-update, then update existing states.
    let r_out = probs.len();
    let mut new_state_n = vec![1_i64; r_out];
    let mut new_state_mean = Array2::<f64>::zeros((r_out, d));
    let mut new_state_m2 = Array2::<f64>::zeros((r_out, d));
    // Position 0: prior updated by xv (Welford from n=0 to n=1).
    for j in 0..d {
        new_state_mean[(0, j)] = xv[j];
    }
    // Positions 1..r_out: take old position 0..r_out-1, advance by xv.
    let limit = (r_out - 1).min(r);
    for i in 0..limit {
        let old_n = state_n_v[i];
        let new_n = old_n + 1;
        new_state_n[i + 1] = new_n;
        for j in 0..d {
            let mean_old = state_mean_arr[(i, j)];
            let m2_old = state_m2_arr[(i, j)];
            if old_n == 0 {
                new_state_mean[(i + 1, j)] = xv[j];
                new_state_m2[(i + 1, j)] = 0.0;
            } else {
                let delta = xv[j] - mean_old;
                let mean_new = mean_old + delta / new_n as f64;
                let delta2 = xv[j] - mean_new;
                new_state_mean[(i + 1, j)] = mean_new;
                new_state_m2[(i + 1, j)] = m2_old + delta * delta2;
            }
        }
    }

    let new_log_joint_py: Bound<'py, PyArray1<f64>> = new_log_joint.clone().into_pyarray(py);
    let new_state_n_py: Bound<'py, PyArray1<i64>> = new_state_n.clone().into_pyarray(py);
    let mean_flat: Vec<f64> = new_state_mean.iter().cloned().collect();
    let m2_flat: Vec<f64> = new_state_m2.iter().cloned().collect();
    let new_state_mean_py: Bound<'py, PyArray1<f64>> = mean_flat.into_pyarray(py);
    let new_state_m2_py: Bound<'py, PyArray1<f64>> = m2_flat.into_pyarray(py);

    Ok(PyTuple::new(
        py,
        &[
            new_log_joint_py.into_any(),
            cp_prob.into_pyobject(py)?.into_any(),
            run_mean.into_pyobject(py)?.into_any(),
            map_run.into_pyobject(py)?.into_any(),
            pred_norm.into_pyobject(py)?.into_any(),
            new_state_n_py.into_any(),
            new_state_mean_py.into_any(),
            new_state_m2_py.into_any(),
            (r_out as i64).into_pyobject(py)?.into_any(),
            (d as i64).into_pyobject(py)?.into_any(),
        ],
    )?)
}

/// Log-space Viterbi decode against a precomputed transition cost matrix.
///
/// - `observed_idx` : observed candidate-regime indices, shape (T,)
/// - `cost`         : transition-cost matrix, shape (S, S) where ``cost[i,j]``
///                    is the cost of moving from state ``i`` to state ``j``
/// - `start_costs`  : initial cost vector, shape (S,)
/// - `emission`     : emission-cost matrix, shape (T, S) where
///                    ``emission[t, s]`` is the cost of emitting the observed
///                    label at time ``t`` from state ``s``
///
/// Returns the decoded path as a Vec<u32> of length T.
#[pyfunction]
fn wfst_viterbi_decode<'py>(
    py: Python<'py>,
    cost: PyReadonlyArray2<'py, f64>,
    start_costs: PyReadonlyArray1<'py, f64>,
    emission: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let cost_arr = cost.as_array().to_owned();
    let start = start_costs.as_slice()?.to_vec();
    let emission_arr = emission.as_array().to_owned();
    let s = cost_arr.shape()[0];
    let t = emission_arr.shape()[0];
    if cost_arr.shape()[1] != s || start.len() != s || emission_arr.shape()[1] != s {
        return Err(PyValueError::new_err("inconsistent shapes for viterbi inputs"));
    }
    if t == 0 {
        let empty: Vec<u32> = vec![];
        return Ok(empty.into_pyarray(py));
    }
    let mut dp = Array2::<f64>::from_elem((t, s), f64::INFINITY);
    let mut back = Array2::<i32>::from_elem((t, s), -1);
    for j in 0..s {
        dp[(0, j)] = start[j] + emission_arr[(0, j)];
    }
    for ti in 1..t {
        for j in 0..s {
            let mut best_cost = f64::INFINITY;
            let mut best_src = -1_i32;
            for i in 0..s {
                let c = dp[(ti - 1, i)] + cost_arr[(i, j)];
                if c < best_cost {
                    best_cost = c;
                    best_src = i as i32;
                }
            }
            dp[(ti, j)] = best_cost + emission_arr[(ti, j)];
            back[(ti, j)] = best_src;
        }
    }
    // Recover path by argmin of last layer.
    let mut path: Vec<u32> = Vec::with_capacity(t);
    let mut cur = 0_usize;
    let mut min_cost = f64::INFINITY;
    for j in 0..s {
        if dp[(t - 1, j)] < min_cost {
            min_cost = dp[(t - 1, j)];
            cur = j;
        }
    }
    path.push(cur as u32);
    for ti in (1..t).rev() {
        let prev = back[(ti, cur)];
        if prev < 0 {
            break;
        }
        cur = prev as usize;
        path.push(cur as u32);
    }
    path.reverse();
    Ok(path.into_pyarray(py))
}

/// PSI between two histograms over a shared bin grid.
#[pyfunction]
fn population_stability_index_kernel<'py>(
    expected_pct: PyReadonlyArray1<'py, f64>,
    actual_pct: PyReadonlyArray1<'py, f64>,
) -> PyResult<f64> {
    let e = expected_pct.as_slice()?;
    let a = actual_pct.as_slice()?;
    if e.len() != a.len() {
        return Err(PyValueError::new_err("histograms must have equal length"));
    }
    let mut psi = 0.0;
    for (ev, av) in e.iter().zip(a.iter()) {
        let ee = ev.max(1e-6);
        let aa = av.max(1e-6);
        psi += (aa - ee) * (aa / ee).ln();
    }
    Ok(psi)
}

/// Rolling Mahalanobis distance between an observation and a window mean
/// using a ridge-stabilised covariance.
#[pyfunction]
fn rolling_mahalanobis_distance_kernel<'py>(
    x: PyReadonlyArray1<'py, f64>,
    mean: PyReadonlyArray1<'py, f64>,
    cov: PyReadonlyArray2<'py, f64>,
    ridge: f64,
) -> PyResult<f64> {
    let xv = x.as_slice()?;
    let mu = mean.as_slice()?;
    let cov_arr = cov.as_array().to_owned();
    let d = xv.len();
    if mu.len() != d || cov_arr.shape() != [d, d] {
        return Err(PyValueError::new_err("dimension mismatch"));
    }
    let diff = Array1::<f64>::from(
        xv.iter().zip(mu.iter()).map(|(x, m)| x - m).collect::<Vec<f64>>(),
    );
    let mut cov_ridge = cov_arr.clone();
    for i in 0..d {
        cov_ridge[(i, i)] += ridge;
    }
    // Tiny d (typically d <= 16). Use Gauss elimination.
    let solution = gauss_solve(&cov_ridge, &diff)?;
    let quad: f64 = diff.iter().zip(solution.iter()).map(|(a, b)| a * b).sum();
    Ok(quad.max(0.0).sqrt())
}

fn gauss_solve(matrix: &Array2<f64>, b: &Array1<f64>) -> PyResult<Array1<f64>> {
    let n = matrix.shape()[0];
    let mut aug = Array2::<f64>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[(i, j)] = matrix[(i, j)];
        }
        aug[(i, n)] = b[i];
    }
    for i in 0..n {
        let mut max_row = i;
        for k in i + 1..n {
            if aug[(k, i)].abs() > aug[(max_row, i)].abs() {
                max_row = k;
            }
        }
        if max_row != i {
            for j in 0..=n {
                let tmp = aug[(i, j)];
                aug[(i, j)] = aug[(max_row, j)];
                aug[(max_row, j)] = tmp;
            }
        }
        let pivot = aug[(i, i)];
        if pivot.abs() < 1e-14 {
            return Err(PyValueError::new_err("singular system"));
        }
        for k in i + 1..n {
            let factor = aug[(k, i)] / pivot;
            for j in i..=n {
                aug[(k, j)] -= factor * aug[(i, j)];
            }
        }
    }
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = aug[(i, n)];
        for j in i + 1..n {
            s -= aug[(i, j)] * x[j];
        }
        x[i] = s / aug[(i, i)];
    }
    Ok(x)
}

/// Legacy placeholder retained for backward compatibility with v0.7 callers.
#[pyfunction]
fn bocpd_change_probability(score: f64, hazard: f64) -> PyResult<f64> {
    let h = hazard.clamp(0.0001, 0.95);
    let s = score.abs().min(20.0);
    Ok((h + (1.0 - h) * (1.0 - (-s / 6.0).exp())).clamp(0.0, 1.0))
}

#[pyfunction]
fn transition_cost(base_cost: f64, event_boost: f64, change_point_prob: f64) -> PyResult<f64> {
    let cp = change_point_prob.clamp(0.0, 1.0);
    Ok((base_cost - event_boost * cp).max(0.0))
}

#[pymodule]
fn mre_rust_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = LN_PI; // suppress dead-code warning while keeping the constant available
    m.add_function(wrap_pyfunction!(bocpd_change_probability, m)?)?;
    m.add_function(wrap_pyfunction!(transition_cost, m)?)?;
    m.add_function(wrap_pyfunction!(bocpd_diag_update, m)?)?;
    m.add_function(wrap_pyfunction!(wfst_viterbi_decode, m)?)?;
    m.add_function(wrap_pyfunction!(population_stability_index_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_mahalanobis_distance_kernel, m)?)?;
    Ok(())
}
