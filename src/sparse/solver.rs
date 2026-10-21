//! A public choice between the simplicial and supernodal factorizations.
//!
//! [`SymbolicLu`] and [`SymbolicSupernodal`] stay available
//! directly; this module is the convenience layer that analyzes a pattern once
//! and factors it with the requested method, including an automatic choice.
//!
//! The automatic choice uses a **flop proxy**, not a timing measurement:
//!
//! ```text
//! ratio = Σ_j col_counts[j]² / nnz(A)
//! ```
//!
//! The counts are those of the Cholesky factor of `AᵀA`, so `ratio` grows with
//! the average column density. A supernode is profitable once its columns are
//! dense, and the default threshold (`40.0`, matching the upstream default of
//! `1.0 × 40`) is well above what a banded or tridiagonal pattern produces and
//! below a dense one. It is a heuristic; benchmarking remains the deciding
//! evidence.
//!
//! The supernodal factorization is still experimental: it is slower than the
//! simplicial path on the matrices benchmarked so far, so
//! [`LuMethod::Simplicial`] (the [`Default`]) remains the recommended choice.

use crate::sparse::SparseError;
use crate::sparse::csc::CscPattern;
use crate::sparse::csc::CscView;
use crate::sparse::lu::{LuFactors, SparseScalar};
use crate::sparse::supernodal::{SupernodalLu, SymbolicSupernodal};
use crate::sparse::symbolic::SymbolicLu;

/// Maximum number of columns a child supernode may have to be merged into its
/// parent by the relaxation pass.
const RELAX_MAX_CHILD_SIZE: usize = 4;

/// Maximum ratio of introduced zeros to the expanded diagonal block allowed
/// when a child is merged into its parent.
const RELAX_FILL_TOLERANCE: f64 = 0.1;

/// Which factorization the solver should use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LuMethod {
    /// The simplicial factorization.
    Simplicial,
    /// The supernodal factorization.
    Supernodal,
    /// Pick supernodal when the flop proxy exceeds `flop_ratio_threshold`.
    Auto {
        /// Threshold on `Σ_j col_counts[j]² / nnz(A)`; `40.0` is the default.
        flop_ratio_threshold: f64,
    },
}

impl LuMethod {
    /// Returns the automatic method with the default threshold (`40.0`).
    #[inline]
    pub fn auto() -> Self {
        Self::Auto { flop_ratio_threshold: 40.0 }
    }
}

impl Default for LuMethod {
    /// The default is the simplicial path, the one that was proven first.
    fn default() -> Self {
        Self::Simplicial
    }
}

/// A symbolic analysis that can factorize with either method.
#[derive(Clone, Debug)]
pub struct LuSymbolic {
    n: usize,
    nnz: usize,
    requested: LuMethod,
    supernodal: bool,
    flop_ratio: f64,
    simplicial: SymbolicLu,
    supernodal_symbolic: Option<SymbolicSupernodal>,
}

impl LuSymbolic {
    /// Analyzes `pattern` and resolves the method.
    ///
    /// The supernodal symbolic structure is always computed: its column counts
    /// are what the automatic choice uses. If the simplicial method is chosen,
    /// the structure is dropped after the decision.
    ///
    /// # Errors
    ///
    /// [`SparseError::BadShape`] if the matrix is not square, plus the ordering
    /// and symbolic errors of the two analyses.
    pub fn analyze(pattern: CscPattern<'_>, method: LuMethod) -> Result<Self, SparseError> {
        let simplicial = SymbolicLu::analyze(pattern)?;

        // The supernodal structure is only needed when it may run. Building it
        // for an explicit simplicial request is pure overhead: the column
        // counts, the supernode partition and the transposed pattern are all
        // discarded. `Auto` still needs it to evaluate the flop proxy.
        let (supernodal_symbolic, flop_ratio) = match method {
            LuMethod::Simplicial => (None, 0.0),
            LuMethod::Supernodal | LuMethod::Auto { .. } => {
                // The fundamental partition leaves mostly one-column
                // supernodes on the matrices measured so far, so apply the
                // simplified relaxation: merge a small child into its parent
                // while the extra zeros stay within the tolerance.
                let symbolic = SymbolicSupernodal::analyze_with_permutation(pattern, simplicial.col_perm(), simplicial.col_perm_inv())?
                    .with_relaxation(RELAX_MAX_CHILD_SIZE, RELAX_FILL_TOLERANCE);
                let counts = symbolic.col_counts();
                let squares: f64 = counts.iter().map(|&c| (c as f64) * (c as f64)).sum();
                let ratio = if pattern.nnz() == 0 { 0.0 } else { squares / pattern.nnz() as f64 };
                (Some(symbolic), ratio)
            }
        };

        let supernodal = match method {
            LuMethod::Simplicial => false,
            LuMethod::Supernodal => true,
            LuMethod::Auto { flop_ratio_threshold } => flop_ratio > flop_ratio_threshold,
        };
        Ok(Self {
            n: pattern.ncols(),
            nnz: pattern.nnz(),
            requested: method,
            supernodal,
            flop_ratio,
            simplicial,
            supernodal_symbolic: if supernodal { supernodal_symbolic } else { None },
        })
    }

    /// Returns the method that was requested.
    #[inline]
    pub fn requested(&self) -> LuMethod {
        self.requested
    }

    /// Returns `true` if the supernodal factorization was selected.
    #[inline]
    pub fn is_supernodal(&self) -> bool {
        self.supernodal
    }

    /// Returns the flop proxy the automatic choice used.
    #[inline]
    pub fn flop_ratio(&self) -> f64 {
        self.flop_ratio
    }

    /// Returns the order of the matrix.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of stored entries the analysis saw.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// Returns the simplicial symbolic analysis.
    #[inline]
    pub fn simplicial(&self) -> &SymbolicLu {
        &self.simplicial
    }

    /// Returns the supernodal symbolic analysis, if the supernodal method was
    /// selected.
    #[inline]
    pub fn supernodal(&self) -> Option<&SymbolicSupernodal> {
        self.supernodal_symbolic.as_ref()
    }

    /// Factorizes `a` with the selected method.
    ///
    /// # Errors
    ///
    /// [`SparseError::StructureMismatch`] if the pattern changed, and
    /// [`SparseError::Singular`] if a pivot is zero.
    pub fn factorize<T: SparseScalar>(&self, a: CscView<'_, T>) -> Result<LuSolver<T>, SparseError> {
        if self.supernodal {
            let symbolic = self.supernodal_symbolic.as_ref().ok_or(SparseError::InvalidStructure)?;
            Ok(LuSolver {
                inner: LuSolverInner::Supernodal(symbolic.factorize(a)?),
            })
        } else {
            Ok(LuSolver {
                inner: LuSolverInner::Simplicial(self.simplicial.factorize(a)?),
            })
        }
    }
}

/// The factors produced by an [`LuSymbolic`], whichever method ran.
#[derive(Clone, Debug)]
pub struct LuSolver<T> {
    inner: LuSolverInner<T>,
}

#[derive(Clone, Debug)]
enum LuSolverInner<T> {
    Simplicial(LuFactors<T>),
    Supernodal(SupernodalLu<T>),
}

impl<T> LuSolver<T> {
    /// Returns `true` if the supernodal factorization produced these factors.
    #[inline]
    pub fn is_supernodal(&self) -> bool {
        matches!(self.inner, LuSolverInner::Supernodal(_))
    }

    /// Returns the order of the matrix.
    #[inline]
    pub fn n(&self) -> usize {
        match &self.inner {
            LuSolverInner::Simplicial(f) => f.n(),
            LuSolverInner::Supernodal(f) => f.n(),
        }
    }

    /// Returns the number of stored entries in `L`.
    #[inline]
    pub fn nnz_l(&self) -> usize {
        match &self.inner {
            LuSolverInner::Simplicial(f) => f.nnz_l(),
            LuSolverInner::Supernodal(f) => f.nnz_l(),
        }
    }

    /// Returns the number of stored entries in `U`.
    #[inline]
    pub fn nnz_u(&self) -> usize {
        match &self.inner {
            LuSolverInner::Simplicial(f) => f.nnz_u(),
            LuSolverInner::Supernodal(f) => f.nnz_u(),
        }
    }

    /// Returns the simplicial factors, if that is what ran.
    #[inline]
    pub fn as_simplicial(&self) -> Option<&LuFactors<T>> {
        match &self.inner {
            LuSolverInner::Simplicial(f) => Some(f),
            LuSolverInner::Supernodal(_) => None,
        }
    }

    /// Returns the supernodal factors, if that is what ran.
    #[inline]
    pub fn as_supernodal(&self) -> Option<&SupernodalLu<T>> {
        match &self.inner {
            LuSolverInner::Simplicial(_) => None,
            LuSolverInner::Supernodal(f) => Some(f),
        }
    }
}

impl<T: SparseScalar> LuSolver<T> {
    /// Solves `A x = b` in place.
    ///
    /// # Errors
    ///
    /// [`SparseError::BadShape`] if `b.len()` is not the order of the matrix.
    pub fn solve(&self, b: &mut [T]) -> Result<(), SparseError> {
        match &self.inner {
            LuSolverInner::Simplicial(f) => f.solve(b),
            LuSolverInner::Supernodal(f) => f.solve(b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tridiag(n: usize) -> (Vec<usize>, Vec<usize>) {
        let mut col_ptr = vec![0];
        let mut row_idx = Vec::new();
        for j in 0..n {
            if j > 0 {
                row_idx.push(j - 1);
            }
            row_idx.push(j);
            if j + 1 < n {
                row_idx.push(j + 1);
            }
            col_ptr.push(row_idx.len());
        }
        (col_ptr, row_idx)
    }

    fn dense(n: usize) -> (Vec<usize>, Vec<usize>) {
        let mut col_ptr = vec![0];
        let mut row_idx = Vec::new();
        for _j in 0..n {
            for i in 0..n {
                row_idx.push(i);
            }
            col_ptr.push(row_idx.len());
        }
        (col_ptr, row_idx)
    }

    fn values(n: usize, col_ptr: &[usize], row_idx: &[usize]) -> Vec<f64> {
        let mut values = vec![0.0; row_idx.len()];
        for j in 0..n {
            for k in col_ptr[j]..col_ptr[j + 1] {
                values[k] = if row_idx[k] == j { 5.0 } else { 1.0 };
            }
        }
        values
    }

    fn solve(method: LuMethod, n: usize, col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> Vec<f64> {
        let pattern = CscPattern::new(n, n, col_ptr, row_idx).unwrap();
        let a = CscView::new(n, n, col_ptr, row_idx, values).unwrap();
        let symbolic = LuSymbolic::analyze(pattern, method).unwrap();
        let factors = symbolic.factorize(a).unwrap();
        let mut b = vec![1.0; n];
        factors.solve(&mut b).unwrap();
        b
    }

    #[test]
    fn both_methods_solve_the_same_system() {
        let (cp, ri) = tridiag(9);
        let v = values(9, &cp, &ri);
        let s = solve(LuMethod::Simplicial, 9, &cp, &ri, &v);
        let n = solve(LuMethod::Supernodal, 9, &cp, &ri, &v);
        for i in 0..9 {
            assert!((s[i] - n[i]).abs() < 1e-12, "entry {i}");
        }
    }

    #[test]
    fn automatic_selection_picks_by_density() {
        let (cp, ri) = tridiag(9);
        let v = values(9, &cp, &ri);
        let s = LuSymbolic::analyze(CscPattern::new(9, 9, &cp, &ri).unwrap(), LuMethod::auto()).unwrap();
        assert!(!s.is_supernodal(), "tridiagonal should stay simplicial");
        let n = solve(LuMethod::auto(), 9, &cp, &ri, &v);
        assert_eq!(n.len(), 9);

        // A low threshold exercises the dense case; the default (40.0) is
        // tuned for large matrices and keeps a 12x12 pattern simplicial.
        let (cp, ri) = dense(12);
        let v = values(12, &cp, &ri);
        let method = LuMethod::Auto { flop_ratio_threshold: 1.0 };
        let symbolic = LuSymbolic::analyze(CscPattern::new(12, 12, &cp, &ri).unwrap(), method).unwrap();
        assert!(symbolic.is_supernodal(), "dense should be supernodal");
        solve(method, 12, &cp, &ri, &v);
    }

    #[test]
    fn forced_methods_are_respected() {
        let (cp, ri) = dense(6);
        let s = LuSymbolic::analyze(CscPattern::new(6, 6, &cp, &ri).unwrap(), LuMethod::Simplicial).unwrap();
        assert!(!s.is_supernodal());
        let n = LuSymbolic::analyze(CscPattern::new(6, 6, &cp, &ri).unwrap(), LuMethod::Supernodal).unwrap();
        assert!(n.is_supernodal());
        assert!(n.flop_ratio() > 0.0);
    }

    #[test]
    fn rejects_non_finite_values_in_both_methods() {
        let (cp, ri) = tridiag(6);
        for method in [LuMethod::Simplicial, LuMethod::Supernodal] {
            let symbolic = LuSymbolic::analyze(CscPattern::new(6, 6, &cp, &ri).unwrap(), method).unwrap();
            let mut values = values(6, &cp, &ri);
            values[0] = f64::NAN;
            let view = CscView::new(6, 6, &cp, &ri, &values).unwrap();
            assert_eq!(symbolic.factorize(view).unwrap_err(), SparseError::NotFinite);
        }
    }

    #[test]
    fn rejects_a_changed_pattern() {
        let (cp, ri) = tridiag(6);
        let symbolic = LuSymbolic::analyze(CscPattern::new(6, 6, &cp, &ri).unwrap(), LuMethod::Simplicial).unwrap();
        let other = CscView::new(6, 6, &[0, 0, 0, 0, 0, 0, 0], &[], &[] as &[f64]).unwrap();
        assert_eq!(symbolic.factorize(other).unwrap_err(), SparseError::StructureMismatch);
    }
}
