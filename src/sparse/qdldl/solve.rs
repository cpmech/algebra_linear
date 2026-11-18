// Derived/transcribed from `osqp/qdldl` v0.1.9 (Apache License 2.0, Copyright OSQP Developers),
// adapted and modified for `algebra_linear` in pure safe Rust (Apache-2.0 §4(b) notice).

//! Triangular and full `L D L^T` solve routines (`QDLDL_Lsolve`, `QDLDL_Ltsolve`, `QDLDL_solve`).

/// Solves `(I + L) x = b` in place, where `L` is strictly lower-triangular CSC.
///
/// Transcribed from `QDLDL_Lsolve` in `osqp/qdldl` `v0.1.9`.
pub fn solve_l(l_col_ptr: &[usize], l_row_idx: &[usize], l_values: &[f64], x: &mut [f64]) {
    let n = x.len();
    debug_assert_eq!(l_col_ptr.len(), n + 1);
    debug_assert_eq!(l_row_idx.len(), l_values.len());
    for j in 0..n {
        let xj = x[j];
        for p in l_col_ptr[j]..l_col_ptr[j + 1] {
            x[l_row_idx[p]] -= l_values[p] * xj;
        }
    }
}

/// Solves `(I + L)^T x = b` in place, where `L` is strictly lower-triangular CSC.
///
/// Transcribed from `QDLDL_Ltsolve` in `osqp/qdldl` `v0.1.9`.
pub fn solve_lt(l_col_ptr: &[usize], l_row_idx: &[usize], l_values: &[f64], x: &mut [f64]) {
    let n = x.len();
    debug_assert_eq!(l_col_ptr.len(), n + 1);
    debug_assert_eq!(l_row_idx.len(), l_values.len());
    for j in (0..n).rev() {
        let mut sum = x[j];
        for p in l_col_ptr[j]..l_col_ptr[j + 1] {
            sum -= l_values[p] * x[l_row_idx[p]];
        }
        x[j] = sum;
    }
}

/// Solves `(I + L) D (I + L)^T x = b` in place using `d_inv = 1 / D`.
///
/// Transcribed from `QDLDL_solve` in `osqp/qdldl` `v0.1.9`.
pub fn solve(l_col_ptr: &[usize], l_row_idx: &[usize], l_values: &[f64], d_inv: &[f64], x: &mut [f64]) {
    let n = x.len();
    debug_assert_eq!(l_col_ptr.len(), n + 1);
    debug_assert_eq!(l_row_idx.len(), l_values.len());
    debug_assert_eq!(d_inv.len(), n);
    solve_l(l_col_ptr, l_row_idx, l_values, x);
    for j in 0..n {
        x[j] *= d_inv[j];
    }
    solve_lt(l_col_ptr, l_row_idx, l_values, x);
}
