//! The opaque sparse LU handle.
//!
//! [`SparseLu`] keeps the symbolic analysis and the numeric factors together.
//! The caller owns the CSC arrays: the same pattern is supplied to `analyze`
//! and to every `factorize`, while only the values change between
//! factorizations.

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::time::Instant;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
use web_time::Instant;

use crate::sparse::SparseError;
use crate::sparse::csc::{CscPattern, CscView};
use crate::sparse::lu::SparseScalar;
use crate::sparse::solver::{LuMethod, LuSolver, LuSymbolic};
use crate::sparse::stats::{SparseLuStats, flops_estimate};

/// An opaque sparse LU factorization.
///
/// The handle owns the symbolic analysis and, after [`Self::factorize`], the
/// numeric factors. The matrix itself stays with the caller as the CSC arrays
/// that were passed in; nothing is copied or converted.
///
/// # Example
///
/// ```
/// use algebra_linear::sparse::{LuMethod, SparseLu};
///
/// // A = [4 1; 1 3] in CSC form.
/// let col_ptr = [0, 2, 4];
/// let row_idx = [0, 1, 0, 1];
/// let values = [4.0, 1.0, 1.0, 3.0];
///
/// let mut lu = SparseLu::analyze(&col_ptr, &row_idx, LuMethod::auto()).unwrap();
/// assert!(lu.stats().is_none());
/// lu.factorize(&col_ptr, &row_idx, &values).unwrap();
///
/// let mut b = [1.0, 2.0];
/// lu.solve(&mut b).unwrap();
/// assert!((b[0] - 1.0 / 11.0).abs() < 1e-14);
/// assert!((b[1] - 7.0 / 11.0).abs() < 1e-14);
/// ```
#[derive(Debug)]
pub struct SparseLu<T: SparseScalar> {
    n: usize,
    symbolic: LuSymbolic,
    solver: Option<LuSolver<T>>,
    stats: Option<SparseLuStats>,
}

impl<T: SparseScalar> SparseLu<T> {
    /// Analyzes the CSC pattern of a square matrix.
    ///
    /// The order is `n = col_ptr.len() - 1`. A row index at or above `n`
    /// describes a matrix with more rows than columns and is rejected; a
    /// structurally empty trailing row is still a valid `n x n` pattern.
    ///
    /// # Errors
    ///
    /// [`SparseError::BadShape`] if `col_ptr` is empty, the array lengths
    /// disagree, or a row index implies a non-square matrix, plus the
    /// structural errors of the CSC validation.
    pub fn analyze(col_ptr: &[usize], row_idx: &[usize], method: LuMethod) -> Result<Self, SparseError> {
        let n = col_ptr.len().checked_sub(1).ok_or(SparseError::BadShape)?;
        if row_idx.iter().any(|&row| row >= n) {
            return Err(SparseError::BadShape);
        }
        let pattern = CscPattern::new(n, n, col_ptr, row_idx)?;
        let symbolic = LuSymbolic::analyze(pattern, method)?;
        Ok(Self {
            n,
            symbolic,
            solver: None,
            stats: None,
        })
    }

    /// Factorizes new values on the pattern given to [`Self::analyze`].
    ///
    /// Only the values may change; the index arrays must describe the same
    /// pattern. A failed factorization leaves the previous successful factors
    /// and statistics in place.
    ///
    /// # Errors
    ///
    /// [`SparseError::StructureMismatch`] if the pattern differs,
    /// [`SparseError::NotFinite`] if a value is NaN or infinite,
    /// [`SparseError::Singular`] if a pivot is zero, and the shape errors of
    /// the CSC view.
    pub fn factorize(&mut self, col_ptr: &[usize], row_idx: &[usize], values: &[T]) -> Result<(), SparseError> {
        let a = CscView::new(self.n, self.n, col_ptr, row_idx, values)?;
        let start = Instant::now();
        let solver = self.symbolic.factorize(a)?;
        let factorization_time = start.elapsed();

        let nnz_l = solver.nnz_l();
        let nnz_u = solver.nnz_u();
        let nnz_a = self.symbolic.nnz();
        let fill_ratio = if nnz_a == 0 { 0.0 } else { (nnz_l + nnz_u) as f64 / nnz_a as f64 };
        let stats = SparseLuStats {
            n: solver.n(),
            nnz_a,
            nnz_l,
            nnz_u,
            fill_ratio,
            flops_estimate: flops_estimate(solver.l_col_ptr(), solver.u_col_ptr()),
            ordering_time: self.symbolic.simplicial().ordering_time(),
            factorization_time,
        };

        self.solver = Some(solver);
        self.stats = Some(stats);
        Ok(())
    }

    /// Solves `A x = b` in place.
    ///
    /// # Errors
    ///
    /// [`SparseError::NotFactorized`] if no factorization has succeeded, and
    /// [`SparseError::BadShape`] if `b.len()` is not the order of the matrix.
    pub fn solve(&self, b: &mut [T]) -> Result<(), SparseError> {
        match &self.solver {
            Some(solver) => solver.solve(b),
            None => Err(SparseError::NotFactorized),
        }
    }

    /// Returns the statistics of the most recent successful factorization.
    ///
    /// `None` before [`Self::factorize`] has succeeded.
    pub fn stats(&self) -> Option<&SparseLuStats> {
        self.stats.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Complex64;

    const COL_PTR: [usize; 3] = [0, 2, 4];
    const ROW_IDX: [usize; 4] = [0, 1, 0, 1];
    const VALUES: [f64; 4] = [4.0, 1.0, 1.0, 3.0];
    const EXACT: [f64; 2] = [1.0 / 11.0, 7.0 / 11.0];

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() <= tol * (1.0 + e.abs()), "entry {i}: {a} vs {e}");
        }
    }

    #[test]
    fn every_method_solves_the_same_real_system() {
        for method in [LuMethod::Simplicial, LuMethod::Supernodal, LuMethod::auto()] {
            let mut lu = SparseLu::analyze(&COL_PTR, &ROW_IDX, method).unwrap();
            lu.factorize(&COL_PTR, &ROW_IDX, &VALUES).unwrap();
            let mut b = [1.0, 2.0];
            lu.solve(&mut b).unwrap();
            assert_close(&b, &EXACT, 1e-14);
        }
    }

    #[test]
    fn solves_a_complex_system() {
        let values = [
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(3.0, 0.0),
        ];
        let mut lu = SparseLu::analyze(&COL_PTR, &ROW_IDX, LuMethod::auto()).unwrap();
        lu.factorize(&COL_PTR, &ROW_IDX, &values).unwrap();
        let mut b = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        lu.solve(&mut b).unwrap();
        for i in 0..2 {
            assert!((b[i] - Complex64::new(EXACT[i], 0.0)).norm() < 1e-14, "entry {i}");
        }
    }

    #[test]
    fn refactorizes_new_values_on_the_same_pattern() {
        let mut lu = SparseLu::analyze(&COL_PTR, &ROW_IDX, LuMethod::auto()).unwrap();

        lu.factorize(&COL_PTR, &ROW_IDX, &VALUES).unwrap();
        let mut b = [1.0, 1.0];
        lu.solve(&mut b).unwrap();
        assert_close(&b, &[2.0 / 11.0, 3.0 / 11.0], 1e-14);

        let new_values = [5.0, 2.0, 2.0, 2.0];
        lu.factorize(&COL_PTR, &ROW_IDX, &new_values).unwrap();
        let mut b = [1.0, 1.0];
        lu.solve(&mut b).unwrap();
        assert_close(&b, &[0.0, 0.5], 1e-14);
    }

    #[test]
    fn rejects_a_changed_pattern() {
        let mut lu = SparseLu::analyze(&COL_PTR, &ROW_IDX, LuMethod::auto()).unwrap();
        lu.factorize(&COL_PTR, &ROW_IDX, &VALUES).unwrap();
        let other_col_ptr = [0, 2, 3];
        let other_row_idx = [0, 1, 1];
        let other_values = [1.0, 1.0, 1.0];
        assert_eq!(
            lu.factorize(&other_col_ptr, &other_row_idx, &other_values).unwrap_err(),
            SparseError::StructureMismatch
        );
    }

    #[test]
    fn solve_requires_a_factorization_and_the_right_length() {
        let mut lu = SparseLu::analyze(&COL_PTR, &ROW_IDX, LuMethod::auto()).unwrap();
        let mut b = [1.0, 2.0];
        assert_eq!(lu.solve(&mut b).unwrap_err(), SparseError::NotFactorized);

        lu.factorize(&COL_PTR, &ROW_IDX, &VALUES).unwrap();
        let mut bad = [1.0, 2.0, 3.0];
        assert_eq!(lu.solve(&mut bad).unwrap_err(), SparseError::BadShape);
    }

    #[test]
    fn rejects_non_finite_values() {
        let mut lu = SparseLu::analyze(&COL_PTR, &ROW_IDX, LuMethod::auto()).unwrap();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let values = [4.0, 1.0, bad, 3.0];
            assert_eq!(lu.factorize(&COL_PTR, &ROW_IDX, &values).unwrap_err(), SparseError::NotFinite);
        }
    }

    #[test]
    fn rejects_non_square_and_malformed_input() {
        // Three rows but only two columns: the row index 2 is out of range for
        // an n x n matrix with n = col_ptr.len() - 1.
        assert_eq!(
            SparseLu::<f64>::analyze(&[0, 2, 3], &[0, 1, 2], LuMethod::auto()).unwrap_err(),
            SparseError::BadShape
        );
        assert_eq!(SparseLu::<f64>::analyze(&[], &[], LuMethod::auto()).unwrap_err(), SparseError::BadShape);
        assert_eq!(
            SparseLu::<f64>::analyze(&[0, 2, 2], &[1, 0], LuMethod::auto()).unwrap_err(),
            SparseError::UnsortedRows
        );
    }

    #[test]
    fn reports_stats_after_each_method() {
        for method in [LuMethod::Simplicial, LuMethod::Supernodal, LuMethod::auto()] {
            let mut lu = SparseLu::analyze(&COL_PTR, &ROW_IDX, method).unwrap();
            assert!(lu.stats().is_none());

            lu.factorize(&COL_PTR, &ROW_IDX, &VALUES).unwrap();
            let s = lu.stats().expect("stats after a successful factorization");
            let solver = lu.solver.as_ref().expect("factors after a successful factorization");
            assert_eq!(s.n, 2);
            assert_eq!(s.nnz_a, 4);
            assert_eq!(s.nnz_l, solver.nnz_l());
            assert_eq!(s.nnz_u, solver.nnz_u());
            assert!(s.nnz_l >= 3);
            assert!(s.nnz_u >= 2);
            assert_eq!(s.nnz_factors(), s.nnz_l + s.nnz_u);
            assert!(s.fill_ratio > 1.0);
            assert!((s.fill_ratio - (s.nnz_l + s.nnz_u) as f64 / s.nnz_a as f64).abs() < f64::EPSILON);
            assert!(s.flops_estimate > 0);
            assert!(s.ordering_time.as_nanos() < 1_000_000_000);
            assert!(s.factorization_time.as_nanos() < 1_000_000_000);
        }
    }

    #[test]
    fn empty_matrix_has_zero_fill_and_flops() {
        let mut lu = SparseLu::<f64>::analyze(&[0], &[], LuMethod::auto()).unwrap();
        lu.factorize(&[0], &[], &[]).unwrap();
        let s = lu.stats().unwrap();
        assert_eq!(s.n, 0);
        assert_eq!(s.nnz_a, 0);
        assert_eq!(s.fill_ratio, 0.0);
        assert_eq!(s.flops_estimate, 0);
        assert_eq!(s.nnz_factors(), 0);
        let mut b: [f64; 0] = [];
        lu.solve(&mut b).unwrap();
    }
}
