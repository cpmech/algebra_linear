// Derived/transcribed from `osqp/qdldl` v0.1.9 (Apache License 2.0, Copyright OSQP Developers),
// adapted and modified for `algebra_linear` in pure safe Rust (Apache-2.0 §4(b) notice).

#![forbid(unsafe_code)]

//! Native pure-Rust port of `osqp/qdldl` `v0.1.9` for quasi-definite `L D L^T` factorization.
//!
//! [`Qdldl`] computes an unpivoted `A = (I + L) D (I + L)^T` factorization of
//! a symmetric matrix `A` whose upper triangle (`row <= col`) is provided in
//! compressed sparse column (CSC) format with strictly increasing row indices.
//! Non-empty columns with omitted diagonal entries are permitted, supporting
//! symmetric quasi-definite (SQD) KKT saddle-point matrices with zero dual
//! blocks.
//!
//! # Example
//!
//! ```
//! use algebra_linear::sparse::qdldl::Qdldl;
//!
//! // Upper triangle of A = [4 1; 1 3] in CSC form.
//! let col_ptr = [0, 1, 3];
//! let row_idx = [0, 0, 1];
//! let values = [4.0, 1.0, 3.0];
//!
//! let mut solver = Qdldl::analyze(&col_ptr, &row_idx).unwrap();
//! solver.factorize(&col_ptr, &row_idx, &values).unwrap();
//!
//! let mut b = [1.0, 2.0];
//! solver.solve(&mut b).unwrap();
//! assert!((b[0] - 1.0 / 11.0).abs() < 1e-14);
//! assert!((b[1] - 7.0 / 11.0).abs() < 1e-14);
//! ```

mod csc;
mod error;
mod etree;
mod factor;
mod solve;
#[cfg(test)]
mod tests;

pub use error::{Error, Result};

use csc::{UpperTriCscPattern, UpperTriCscView};
use etree::{SymbolicFactor, compute_etree};
use factor::{FactorWorkspace, NumericFactor, factor_in_place};

/// Structural and inertia statistics from the most recent successful QDLDL factorization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QdldlStats {
    /// Order `n` of the `n x n` matrix.
    pub n: usize,
    /// Number of stored entries in the upper-triangular input matrix `A`.
    pub nnz_a: usize,
    /// Number of stored strictly lower-triangular entries in `L`.
    pub nnz_l: usize,
    /// Number of strictly positive diagonal pivots (`D[k] > 0`).
    pub positive_inertia: usize,
}

/// Native QDLDL `L D L^T` factorization handle.
///
/// Symbolic analysis (`QDLDL_etree`) runs once in [`Self::analyze`], while
/// numeric factorizations (`QDLDL_factor`) reuse all allocated factor and
/// workspace buffers in [`Self::factorize`]. Solves (`QDLDL_solve`,
/// `QDLDL_Lsolve`, `QDLDL_Ltsolve`) require `&self` and allocate nothing.
#[derive(Clone, Debug)]
pub struct Qdldl {
    symbolic: SymbolicFactor,
    numeric: NumericFactor,
    work: FactorWorkspace,
    factorized: bool,
    stats: Option<QdldlStats>,
}

impl Qdldl {
    /// Validates an upper-triangular CSC sparsity pattern and performs symbolic
    /// elimination-tree analysis (`QDLDL_etree`).
    ///
    /// Missing diagonal entries in non-empty columns are permitted (e.g., for
    /// SQD KKT matrices with zero trailing blocks). Empty columns are rejected
    /// with [`Error::EmptyColumn`].
    ///
    /// # Errors
    ///
    /// Returns any structural validation error from CSC checking,
    /// [`Error::EmptyColumn`] if a column has zero stored entries, or
    /// [`Error::IndexOverflow`] if `L`'s entry count overflows `usize`.
    pub fn analyze(col_ptr: &[usize], row_idx: &[usize]) -> Result<Self> {
        let pattern = UpperTriCscPattern::new(col_ptr, row_idx)?;
        let symbolic = compute_etree(pattern)?;
        let numeric = NumericFactor::from_symbolic(&symbolic);
        let work = FactorWorkspace::new(symbolic.n);
        Ok(Self {
            symbolic,
            numeric,
            work,
            factorized: false,
            stats: None,
        })
    }

    /// Factorizes numeric values on the sparsity pattern analyzed in
    /// [`Self::analyze`] (`QDLDL_factor`).
    ///
    /// Validation errors ([`Error::BadShape`], CSC structure errors,
    /// [`Error::StructureMismatch`], and non-finite input [`Error::NotFinite`])
    /// leave any previous valid factorization and statistics intact. If numeric
    /// elimination fails ([`Error::ZeroDiagonal`] or [`Error::NotFinite`] from
    /// overflow), any prior factorization is invalidated so stale solves cannot
    /// occur, and subsequent valid calls on the same pattern recover cleanly.
    ///
    /// # Errors
    ///
    /// * CSC validation or [`Error::BadShape`] errors;
    /// * [`Error::StructureMismatch`] if `(n, nnz_a, fingerprint)` differs from
    ///   the analyzed pattern;
    /// * [`Error::NotFinite`] if any input value or computed pivot is non-finite;
    /// * [`Error::ZeroDiagonal`] if a zero pivot or reciprocal overflow occurs.
    pub fn factorize(&mut self, col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> Result<()> {
        let a = UpperTriCscView::new(col_ptr, row_idx, values)?;
        if a.n() != self.symbolic.n || a.nnz() != self.symbolic.nnz_a || a.pattern().fingerprint() != self.symbolic.fingerprint {
            return Err(Error::StructureMismatch);
        }
        if !values.iter().all(|v| v.is_finite()) {
            return Err(Error::NotFinite);
        }

        // Invalidate prior state before modifying numeric/workspace buffers so a
        // mid-factorization failure never leaves stale factors active.
        self.factorized = false;
        self.stats = None;

        let positive_inertia = factor_in_place(a, &self.symbolic, &mut self.numeric, &mut self.work)?;
        self.factorized = true;
        self.stats = Some(QdldlStats {
            n: self.symbolic.n,
            nnz_a: self.symbolic.nnz_a,
            nnz_l: self.symbolic.nnz_l(),
            positive_inertia,
        });
        Ok(())
    }

    /// Solves `A x = (I + L) D (I + L)^T x = b` in place (`QDLDL_solve`) with
    /// zero allocations.
    ///
    /// # Errors
    ///
    /// * [`Error::NotFactorized`] if no numeric factorization is currently active;
    /// * [`Error::BadShape`] if `b.len()` differs from the matrix order `n`.
    pub fn solve(&self, b: &mut [f64]) -> Result<()> {
        self.check_solve_args(b)?;
        solve::solve(
            &self.symbolic.l_col_ptr,
            &self.numeric.l_row_idx,
            &self.numeric.l_values,
            &self.numeric.d_inv,
            b,
        );
        Ok(())
    }

    /// Solves `(I + L) x = b` in place (`QDLDL_Lsolve`) with zero allocations.
    ///
    /// # Errors
    ///
    /// * [`Error::NotFactorized`] if no numeric factorization is currently active;
    /// * [`Error::BadShape`] if `b.len()` differs from the matrix order `n`.
    pub fn solve_l(&self, b: &mut [f64]) -> Result<()> {
        self.check_solve_args(b)?;
        solve::solve_l(&self.symbolic.l_col_ptr, &self.numeric.l_row_idx, &self.numeric.l_values, b);
        Ok(())
    }

    /// Solves `(I + L)^T x = b` in place (`QDLDL_Ltsolve`) with zero allocations.
    ///
    /// # Errors
    ///
    /// * [`Error::NotFactorized`] if no numeric factorization is currently active;
    /// * [`Error::BadShape`] if `b.len()` differs from the matrix order `n`.
    pub fn solve_lt(&self, b: &mut [f64]) -> Result<()> {
        self.check_solve_args(b)?;
        solve::solve_lt(&self.symbolic.l_col_ptr, &self.numeric.l_row_idx, &self.numeric.l_values, b);
        Ok(())
    }

    /// Returns the diagonal factor `D` of the active factorization, or `None`
    /// if the solver is not currently factorized.
    #[inline]
    pub fn d(&self) -> Option<&[f64]> {
        if self.factorized { Some(&self.numeric.d) } else { None }
    }

    /// Returns the statistics of the active factorization, or `None` if the
    /// solver is not currently factorized.
    #[inline]
    pub fn stats(&self) -> Option<&QdldlStats> {
        self.stats.as_ref()
    }

    #[inline]
    fn check_solve_args(&self, b: &[f64]) -> Result<()> {
        if !self.factorized {
            return Err(Error::NotFactorized);
        }
        if b.len() != self.symbolic.n {
            return Err(Error::BadShape);
        }
        Ok(())
    }
}
