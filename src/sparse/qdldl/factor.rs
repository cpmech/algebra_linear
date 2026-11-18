// Derived/transcribed from `osqp/qdldl` v0.1.9 (Apache License 2.0, Copyright OSQP Developers),
// adapted and modified for `algebra_linear` in pure safe Rust (Apache-2.0 §4(b) notice).

//! Numeric `L D L^T` factorization (`QDLDL_factor`).

use super::csc::UpperTriCscView;
use super::error::{Error, Result};
use super::etree::{NONE, SymbolicFactor};

/// Numeric `L` and `D` factors produced by [`factor_in_place`].
#[derive(Clone, Debug)]
pub struct NumericFactor {
    /// Row indices of the strictly lower-triangular CSC factor `L`.
    pub l_row_idx: Vec<usize>,
    /// Values of the strictly lower-triangular CSC factor `L`.
    pub l_values: Vec<f64>,
    /// Diagonal entries `D[k]`.
    pub d: Vec<f64>,
    /// Reciprocal diagonal entries `1.0 / D[k]`.
    pub d_inv: Vec<f64>,
}

impl NumericFactor {
    /// Allocates numeric factor arrays sized for `symbolic`.
    pub fn from_symbolic(symbolic: &SymbolicFactor) -> Self {
        let nnz_l = symbolic.nnz_l();
        Self {
            l_row_idx: vec![0usize; nnz_l],
            l_values: vec![0.0f64; nnz_l],
            d: vec![0.0f64; symbolic.n],
            d_inv: vec![0.0f64; symbolic.n],
        }
    }
}

/// Reusable workspace matching the five scratch arrays of upstream `QDLDL_factor`.
#[derive(Clone, Debug)]
pub struct FactorWorkspace {
    /// Dense right-hand-side accumulator `y` of length `n`.
    pub y: Vec<f64>,
    /// Reachability marker array `yMarkers` of length `n`.
    pub y_markers: Vec<bool>,
    /// Reach stack `yIdx` storing nonzero indices of `y` in topological order.
    pub y_idx: Vec<usize>,
    /// Per-branch reversal buffer `elimBuffer` of length `n`.
    pub elim_buffer: Vec<usize>,
    /// Next write cursor `LNextSpaceInCol` for each column of `L`, length `n`.
    pub l_next: Vec<usize>,
}

impl FactorWorkspace {
    /// Allocates all five workspace vectors for a matrix of order `n`.
    pub fn new(n: usize) -> Self {
        Self {
            y: vec![0.0f64; n],
            y_markers: vec![false; n],
            y_idx: vec![0usize; n],
            elim_buffer: vec![0usize; n],
            l_next: vec![0usize; n],
        }
    }
}

/// Computes the numeric `L D L^T` factorization into pre-allocated `numeric`
/// storage using `work`, returning the positive inertia (number of `D[k] > 0`).
///
/// Transcribed from `QDLDL_factor` in `osqp/qdldl` `v0.1.9`. Uses two-buffer
/// per-branch reversal (`elim_buffer` + `y_idx`) so independent elimination-tree
/// branches merging at an ancestor are visited in exact topological order.
///
/// # Errors
///
/// * [`Error::StructureMismatch`] if `a` disagrees with `symbolic`;
/// * [`Error::NotFinite`] if a computed pivot is `NaN` or infinite;
/// * [`Error::ZeroDiagonal`] if a diagonal pivot `D[k]` is zero or its
///   reciprocal `1.0 / D[k]` overflows to non-finite.
pub fn factor_in_place(a: UpperTriCscView<'_>, symbolic: &SymbolicFactor, numeric: &mut NumericFactor, work: &mut FactorWorkspace) -> Result<usize> {
    let n = symbolic.n;
    if a.n() != n || a.nnz() != symbolic.nnz_a || a.pattern().fingerprint() != symbolic.fingerprint {
        return Err(Error::StructureMismatch);
    }
    if n == 0 {
        return Ok(0);
    }

    debug_assert_eq!(symbolic.etree.len(), n);
    debug_assert_eq!(symbolic.l_col_ptr.len(), n + 1);
    debug_assert_eq!(numeric.l_row_idx.len(), symbolic.nnz_l());
    debug_assert_eq!(numeric.l_values.len(), symbolic.nnz_l());
    debug_assert_eq!(numeric.d.len(), n);
    debug_assert_eq!(numeric.d_inv.len(), n);
    debug_assert_eq!(work.y.len(), n);
    debug_assert_eq!(work.y_markers.len(), n);
    debug_assert_eq!(work.y_idx.len(), n);
    debug_assert_eq!(work.elim_buffer.len(), n);
    debug_assert_eq!(work.l_next.len(), n);

    work.y.fill(0.0);
    work.y_markers.fill(false);
    work.l_next.copy_from_slice(&symbolic.l_col_ptr[..n]);
    let mut positive_inertia = 0usize;

    for k in 0..n {
        // Step 2a: compute reach(k) with two-buffer per-branch reversal.
        work.y_markers[k] = true;
        let mut nnz_y = 0usize;
        for (&i, &val) in a.pattern().row_idx_of_col(k).iter().zip(a.val_of_col(k)) {
            work.y[i] = val;
            let mut curr = i;
            let mut nnz_e = 0usize;
            while curr != NONE && !work.y_markers[curr] {
                work.elim_buffer[nnz_e] = curr;
                nnz_e += 1;
                work.y_markers[curr] = true;
                curr = symbolic.etree[curr];
            }
            while nnz_e > 0 {
                nnz_e -= 1;
                work.y_idx[nnz_y] = work.elim_buffer[nnz_e];
                nnz_y += 1;
            }
        }

        // Step 2b: solve L(0..k, 0..k) * y = a(0..k, k) and update pivot D[k].
        let mut dval = work.y[k];
        work.y[k] = 0.0;
        for idx in (0..nnz_y).rev() {
            let c = work.y_idx[idx];
            let yc = work.y[c];
            work.y[c] = 0.0;
            work.y_markers[c] = false;

            let col_start = symbolic.l_col_ptr[c];
            let col_end = work.l_next[c];
            for p in col_start..col_end {
                work.y[numeric.l_row_idx[p]] -= numeric.l_values[p] * yc;
            }

            let l_val = yc * numeric.d_inv[c];
            numeric.l_row_idx[col_end] = k;
            numeric.l_values[col_end] = l_val;
            work.l_next[c] = col_end + 1;

            dval -= yc * l_val;
        }
        work.y_markers[k] = false;

        // Step 2c: validate pivot and store D[k] and D_inv[k].
        if !dval.is_finite() {
            return Err(Error::NotFinite);
        }
        if dval == 0.0 {
            return Err(Error::ZeroDiagonal { index: k });
        }
        let d_inv_k = 1.0 / dval;
        if !d_inv_k.is_finite() {
            return Err(Error::ZeroDiagonal { index: k });
        }
        numeric.d[k] = dval;
        numeric.d_inv[k] = d_inv_k;
        if dval > 0.0 {
            positive_inertia += 1;
        }
    }

    #[cfg(debug_assertions)]
    for c in 0..n {
        debug_assert_eq!(work.l_next[c] - symbolic.l_col_ptr[c], symbolic.lnz[c]);
    }

    Ok(positive_inertia)
}
