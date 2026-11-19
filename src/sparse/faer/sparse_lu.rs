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

use super::csc::{CscPattern, CscView};
use super::error::Error;
use super::lu::SparseScalar;
use super::solver::{EffectiveLuMethod, LuMethod, LuSolver, LuSymbolic};
use super::stats::{SparseLuStats, flops_estimate};

/// An opaque sparse LU factorization.
///
/// The handle owns the symbolic analysis and, after [`Self::factorize`], the
/// numeric factors. The matrix itself stays with the caller as the CSC arrays
/// that were passed in; nothing is copied or converted.
///
/// # Example
///
/// ```
/// use algebra_linear::sparse::faer::{LuMethod, SparseLu};
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
    /// [`Error::BadShape`] if `col_ptr` is empty, the array lengths
    /// disagree, or a row index implies a non-square matrix, plus the
    /// structural errors of the CSC validation.
    pub fn analyze(col_ptr: &[usize], row_idx: &[usize], method: LuMethod) -> Result<Self, Error> {
        let n = col_ptr.len().checked_sub(1).ok_or(Error::BadShape)?;
        if row_idx.iter().any(|&row| row >= n) {
            return Err(Error::BadShape);
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
    /// [`Error::StructureMismatch`] if the pattern differs,
    /// [`Error::NotFinite`] if a value is NaN or infinite,
    /// [`Error::Singular`] if a pivot is zero, and the shape errors of
    /// the CSC view.
    pub fn factorize(&mut self, col_ptr: &[usize], row_idx: &[usize], values: &[T]) -> Result<(), Error> {
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
            method: self.symbolic.effective_method(),
            flop_ratio: self.symbolic.flop_ratio(),
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
    /// [`Error::NotFactorized`] if no factorization has succeeded, and
    /// [`Error::BadShape`] if `b.len()` is not the order of the matrix.
    pub fn solve(&self, b: &mut [T]) -> Result<(), Error> {
        match &self.solver {
            Some(solver) => solver.solve(b),
            None => Err(Error::NotFactorized),
        }
    }

    /// Returns the factorization method selected during symbolic analysis.
    ///
    /// Available immediately after [`Self::analyze`], before [`Self::factorize`]
    /// has been called.
    #[inline]
    pub fn effective_method(&self) -> EffectiveLuMethod {
        self.symbolic.effective_method()
    }

    /// Returns the symbolic flop-ratio proxy `Σ col_counts² / nnz(A)`, if computed.
    ///
    /// Available immediately after [`Self::analyze`]. Returns `None` when
    /// symbolic analysis ran with [`LuMethod::Simplicial`].
    #[inline]
    pub fn flop_ratio(&self) -> Option<f64> {
        self.symbolic.flop_ratio()
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
    /// What one sparse factorization of a small 3D box costs.
    ///
    /// `cargo test --release --lib bench_sparse_lu -- --ignored --nocapture`
    ///
    /// The matrix is the one a driver run factors every step: a 4x4x4 node lattice, three dofs per
    /// node, every node coupled to its six neighbours through a full 3x3 block. 192 dofs, and the
    /// factorization is repeated, as the driver does for each accepted step.
    #[test]
    #[ignore]
    fn bench_sparse_lu() {
        use crate::sparse::faer::{LuMethod, SparseLu};
        use std::time::Instant;
        let m = 4_usize; // nodes per side
        let n = m * m * m * 3;
        // the loops below run k outer, i inner, so the node numbering must match that order
        let node = |i: usize, j: usize, k: usize| (k * m * m + j * m + i) * 3;
        let mut col_ptr = vec![0_usize];
        let mut row_idx: Vec<usize> = Vec::new();
        for k in 0..m {
            for j in 0..m {
                for i in 0..m {
                    // every node owns three columns, one per component, and the block coupling connects
                    // all three components of a node to all three of its neighbours
                    for _d in 0..3 {
                        let mut rows: Vec<usize> = Vec::new();
                        for dk in 0..3_i32 {
                            for dj in 0..3_i32 {
                                for di in 0..3_i32 {
                                    let (ii, jj, kk) = (i as i32 + di - 1, j as i32 + dj - 1, k as i32 + dk - 1);
                                    if ii < 0 || jj < 0 || kk < 0 || ii >= m as i32 || jj >= m as i32 || kk >= m as i32 {
                                        continue;
                                    }
                                    let base = node(ii as usize, jj as usize, kk as usize);
                                    for d in 0..3 {
                                        rows.push(base + d);
                                    }
                                }
                            }
                        }
                        rows.sort_unstable();
                        row_idx.extend_from_slice(&rows);
                        col_ptr.push(row_idx.len());
                    }
                }
            }
        }
        let nnz = row_idx.len();
        let mut values = vec![0.0_f64; nnz];
        for col in 0..n {
            let mut sum = 0.0;
            for p in col_ptr[col]..col_ptr[col + 1] {
                if row_idx[p] != col {
                    values[p] = -1.0;
                    sum += 1.0;
                }
            }
            for p in col_ptr[col]..col_ptr[col + 1] {
                if row_idx[p] == col {
                    values[p] = 10.0 + sum;
                }
            }
        }

        let reps = 160;
        for (label, method) in [("simplicial", LuMethod::Simplicial), ("auto", LuMethod::auto())] {
            let mut lu = SparseLu::<f64>::analyze(&col_ptr, &row_idx, method).unwrap();
            lu.factorize(&col_ptr, &row_idx, &values).unwrap();
            let (nnz_l, nnz_u, fill) = {
                let stats = lu.stats().unwrap();
                (stats.nnz_l, stats.nnz_u, stats.fill_ratio)
            };
            let mut b = vec![1.0; n];
            let t = Instant::now();
            for _ in 0..reps {
                lu.factorize(&col_ptr, &row_idx, &values).unwrap();
            }
            let per_factorize = t.elapsed().as_secs_f64() / reps as f64;
            lu.solve(&mut b).unwrap();
            let t = Instant::now();
            for _ in 0..(reps * 10) {
                b.fill(1.0);
                lu.solve(&mut b).unwrap();
            }
            let per_solve = t.elapsed().as_secs_f64() / (reps * 10) as f64;
            println!(
                "{label}: n = {n}, nnz(A) = {nnz}, nnz(L) = {nnz_l}, nnz(U) = {nnz_u}, fill = {fill:.2} -> factorize {:.3} ms, solve {:.1} us",
                per_factorize * 1e3,
                per_solve * 1e6
            );
        }
    }

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
            Error::StructureMismatch
        );
    }

    #[test]
    fn solve_requires_a_factorization_and_the_right_length() {
        let mut lu = SparseLu::analyze(&COL_PTR, &ROW_IDX, LuMethod::auto()).unwrap();
        let mut b = [1.0, 2.0];
        assert_eq!(lu.solve(&mut b).unwrap_err(), Error::NotFactorized);

        lu.factorize(&COL_PTR, &ROW_IDX, &VALUES).unwrap();
        let mut bad = [1.0, 2.0, 3.0];
        assert_eq!(lu.solve(&mut bad).unwrap_err(), Error::BadShape);
    }

    #[test]
    fn rejects_non_finite_values() {
        let mut lu = SparseLu::analyze(&COL_PTR, &ROW_IDX, LuMethod::auto()).unwrap();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let values = [4.0, 1.0, bad, 3.0];
            assert_eq!(lu.factorize(&COL_PTR, &ROW_IDX, &values).unwrap_err(), Error::NotFinite);
        }
    }

    #[test]
    fn rejects_non_square_and_malformed_input() {
        // Three rows but only two columns: the row index 2 is out of range for
        // an n x n matrix with n = col_ptr.len() - 1.
        assert_eq!(
            SparseLu::<f64>::analyze(&[0, 2, 3], &[0, 1, 2], LuMethod::auto()).unwrap_err(),
            Error::BadShape
        );
        assert_eq!(SparseLu::<f64>::analyze(&[], &[], LuMethod::auto()).unwrap_err(), Error::BadShape);
        assert_eq!(
            SparseLu::<f64>::analyze(&[0, 2, 2], &[1, 0], LuMethod::auto()).unwrap_err(),
            Error::UnsortedRows
        );
    }

    fn build_diagonally_dominant_csc(n: usize, dense: bool) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let mut col_ptr = vec![0];
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        for j in 0..n {
            if dense {
                for i in 0..n {
                    row_idx.push(i);
                    values.push(if i == j { (n as f64) * 2.0 } else { 1.0 });
                }
            } else {
                for i in j.saturating_sub(1)..=(j + 1).min(n.saturating_sub(1)) {
                    row_idx.push(i);
                    values.push(if i == j { 5.0 } else { 1.0 });
                }
            }
            col_ptr.push(row_idx.len());
        }
        (col_ptr, row_idx, values)
    }

    fn verify_method_and_ratio(
        col_ptr: &[usize],
        row_idx: &[usize],
        values: &[f64],
        method: LuMethod,
        expected_method: EffectiveLuMethod,
        check_ratio: impl Fn(Option<f64>),
    ) {
        let n = col_ptr.len() - 1;
        let mut lu = SparseLu::analyze(col_ptr, row_idx, method).unwrap();
        assert!(lu.stats().is_none());
        assert_eq!(lu.effective_method(), expected_method);
        check_ratio(lu.flop_ratio());

        lu.factorize(col_ptr, row_idx, values).unwrap();
        let stats = lu.stats().expect("stats after factorization");
        assert_eq!(stats.method, expected_method);
        assert_eq!(stats.flop_ratio, lu.flop_ratio());
        check_ratio(stats.flop_ratio);

        let mut x = vec![1.0; n];
        lu.solve(&mut x).unwrap();
        let mut ax = vec![0.0; n];
        for j in 0..n {
            for k in col_ptr[j]..col_ptr[j + 1] {
                ax[row_idx[k]] += values[k] * x[j];
            }
        }
        for (i, &val) in ax.iter().enumerate() {
            assert!((val - 1.0).abs() < 1e-11, "residual mismatch at row {i}: {val}");
        }
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
            assert_eq!(s.method, lu.effective_method());
            assert_eq!(s.flop_ratio, lu.flop_ratio());
            assert!(s.ordering_time.as_nanos() < 1_000_000_000);
            assert!(s.factorization_time.as_nanos() < 1_000_000_000);
        }
    }

    #[test]
    fn reports_effective_method_and_flop_ratio_before_and_after_factorize() {
        let (tri_cp, tri_ri, tri_vals) = build_diagonally_dominant_csc(9, false);
        verify_method_and_ratio(
            &tri_cp,
            &tri_ri,
            &tri_vals,
            LuMethod::Simplicial,
            EffectiveLuMethod::Simplicial,
            |ratio| {
                assert_eq!(ratio, None);
            },
        );
        verify_method_and_ratio(
            &tri_cp,
            &tri_ri,
            &tri_vals,
            LuMethod::Supernodal,
            EffectiveLuMethod::Supernodal,
            |ratio| {
                let r = ratio.expect("supernodal computes flop ratio");
                assert!(r > 0.0, "expected positive ratio, got {r}");
            },
        );
        verify_method_and_ratio(&tri_cp, &tri_ri, &tri_vals, LuMethod::auto(), EffectiveLuMethod::Simplicial, |ratio| {
            let r = ratio.expect("auto computes flop ratio");
            assert!(r > 0.0 && r <= 40.0, "expected 0 < ratio <= 40, got {r}");
        });

        let (dense_cp, dense_ri, dense_vals) = build_diagonally_dominant_csc(12, true);
        verify_method_and_ratio(
            &dense_cp,
            &dense_ri,
            &dense_vals,
            LuMethod::Auto { flop_ratio_threshold: 1.0 },
            EffectiveLuMethod::Supernodal,
            |ratio| {
                let r = ratio.expect("auto computes flop ratio");
                assert!(r > 1.0, "expected ratio > 1.0, got {r}");
            },
        );

        verify_method_and_ratio(&[0], &[], &[], LuMethod::auto(), EffectiveLuMethod::Simplicial, |ratio| {
            assert_eq!(ratio, Some(0.0));
        });
    }

    #[test]
    fn empty_matrix_has_zero_fill_and_flops() {
        let mut lu = SparseLu::<f64>::analyze(&[0], &[], LuMethod::auto()).unwrap();
        assert_eq!(lu.effective_method(), EffectiveLuMethod::Simplicial);
        assert_eq!(lu.flop_ratio(), Some(0.0));
        lu.factorize(&[0], &[], &[]).unwrap();
        let s = lu.stats().unwrap();
        assert_eq!(s.n, 0);
        assert_eq!(s.nnz_a, 0);
        assert_eq!(s.fill_ratio, 0.0);
        assert_eq!(s.flops_estimate, 0);
        assert_eq!(s.nnz_factors(), 0);
        assert_eq!(s.method, EffectiveLuMethod::Simplicial);
        assert_eq!(s.flop_ratio, Some(0.0));
        let mut b: [f64; 0] = [];
        lu.solve(&mut b).unwrap();
    }
}
