//! The dense coarse-level solve, standing in for AMGCL's backend coarse direct solver.
//!
//! It reuses the crate's column-major `dgetrf`/`dgetrs` pair — the same kernels `dgesv` uses. The
//! factorization is computed once during setup and reused in every cycle, so coarse-pivot semantics
//! follow this crate's LAPACK implementation rather than a second AMG-specific LU. Converting the
//! coarse CSR operator to column-major dense storage is the only dense conversion in the setup.

use crate::lapack::{LapackError, dgetrf_with_pivots, dgetrs};
use crate::sparse::amgcl::matrix::CsrMatrix;
use crate::sparse::amgcl::memory::{MemoryBudget, try_vec_filled, vec_bytes};
use crate::sparse::amgcl::{Error, Result};

/// Dense LU factorization with partial pivoting for the coarsest AMG level.
///
/// The coarse matrix is converted from CSR to the crate's column-major dense
/// layout once during setup. Factorization and solves use the same LAPACK
/// routines as the rest of `algebra_linear`.
#[derive(Clone, Debug)]
pub(crate) struct DenseLu {
    lu: Vec<f64>,
    pivots: Vec<usize>,
}

impl DenseLu {
    #[cfg(test)]
    pub(crate) fn new(matrix: &CsrMatrix) -> Result<Self> {
        Self::new_with_budget(matrix, &MemoryBudget::new(0), None)
    }

    pub(crate) fn new_with_budget(matrix: &CsrMatrix, budget: &MemoryBudget, level: Option<usize>) -> Result<Self> {
        if !matrix.is_square() {
            return Err(Error::InvalidMatrix("coarse direct solver requires a square matrix".into()));
        }
        let n = matrix.nrows();
        let dense_len = n.saturating_mul(n);
        let dense_bytes = vec_bytes::<f64>(dense_len);
        let pivots_bytes = vec_bytes::<usize>(n);
        budget.check_additional(dense_bytes.saturating_add(pivots_bytes), level)?;
        let exact_dense_len = n.checked_mul(n).ok_or_else(|| budget.alloc_error(usize::MAX, level))?;

        let mut lu = try_vec_filled(0.0, exact_dense_len, budget, level)?;
        let mut pivots = try_vec_filled(0usize, n, budget, level)?;
        fill_dense_from_csr(matrix, n, &mut lu);

        // First attempt: the operator as the Galerkin product produced it.
        match dgetrf_with_pivots(&mut lu, n, &mut pivots) {
            Ok(_determinant) => return Ok(Self { lu, pivots }),
            Err(LapackError::Singular) => {}
            Err(error) => return Err(map_lapack_error(error)),
        }

        // A Galerkin coarse operator can come out semi-definite: dependent rows appear when the
        // strength graph splits a connected component, and a coarsest level that only supplies a
        // correction does not need an exact factorization. Shift the diagonal by a fraction of the
        // operator's own scale and retry, escalating while the shift stays negligible. The coarse
        // correction is damped into the cycle, so a relative shift of 1e-12 costs nothing and turns
        // an exactly singular operator into a well-posed one.
        let scale = matrix_scale(matrix);
        for factor in SHIFT_FACTORS {
            fill_dense_from_csr(matrix, n, &mut lu);
            apply_shift(&mut lu, n, factor * scale);
            match dgetrf_with_pivots(&mut lu, n, &mut pivots) {
                Ok(_determinant) => return Ok(Self { lu, pivots }),
                Err(LapackError::Singular) => continue,
                Err(error) => return Err(map_lapack_error(error)),
            }
        }

        fill_dense_from_csr(matrix, n, &mut lu);
        let row = (0..n).find(|&i| lu[i + i * n] == 0.0).unwrap_or(0);
        Err(Error::SingularMatrix { row })
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        vec_bytes::<f64>(self.lu.len()).saturating_add(vec_bytes::<usize>(self.pivots.len()))
    }

    pub(crate) fn solve(&self, rhs: &[f64], x: &mut [f64]) {
        debug_assert_eq!(rhs.len(), x.len());
        debug_assert_eq!(self.pivots.len(), rhs.len());
        x.copy_from_slice(rhs);
        dgetrs(&self.lu, &self.pivots, x);
    }
}

/// Relative diagonal shifts tried, in order, when the coarse operator is singular
const SHIFT_FACTORS: [f64; 4] = [1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6];

/// Converts the coarse CSR matrix to the crate's column-major dense layout inside `lu`
fn fill_dense_from_csr(matrix: &CsrMatrix, n: usize, lu: &mut [f64]) {
    lu.fill(0.0);
    for row in 0..n {
        let entries = matrix.row(row);
        for (&column, &value) in entries.columns.iter().zip(entries.values) {
            lu[row + column * n] += value;
        }
    }
}

/// Largest absolute entry of the matrix, and one for an all-zero operator
fn matrix_scale(matrix: &CsrMatrix) -> f64 {
    let mut scale: f64 = 0.0;
    for row in 0..matrix.nrows() {
        for &value in matrix.row(row).values {
            scale = scale.max(value.abs());
        }
    }
    if scale > 0.0 { scale } else { 1.0 }
}

/// Adds `eps` to every diagonal entry of the column-major dense matrix
fn apply_shift(lu: &mut [f64], n: usize, eps: f64) {
    for i in 0..n {
        lu[i + i * n] += eps;
    }
}

/// Maps a LAPACK failure to the AMG error
fn map_lapack_error(error: LapackError) -> Error {
    Error::SolverBreakdown(match error {
        LapackError::Convergence => "coarse LU factorization did not converge",
        LapackError::NotImplemented(_) => "coarse LU factorization is not implemented",
        LapackError::Singular | LapackError::BadShape => "coarse LU factorization failed",
    })
}
