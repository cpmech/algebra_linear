use crate::lapack::{LapackError, dgetrf, dgetrs};
use crate::sparse::amgcl::matrix::CsrMatrix;
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
    pub(crate) fn new(matrix: &CsrMatrix) -> Result<Self> {
        if !matrix.is_square() {
            return Err(Error::InvalidMatrix("coarse direct solver requires a square matrix".into()));
        }
        let n = matrix.nrows();
        let mut lu = dense_from_csr(matrix, n);

        // First attempt: the operator as the Galerkin product produced it.
        match dgetrf(&mut lu, n) {
            Ok((pivots, _determinant)) => return Ok(Self { lu, pivots }),
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
            let mut lu = dense_from_csr(matrix, n);
            apply_shift(&mut lu, n, factor * scale);
            match dgetrf(&mut lu, n) {
                Ok((pivots, _determinant)) => return Ok(Self { lu, pivots }),
                Err(LapackError::Singular) => continue,
                Err(error) => return Err(map_lapack_error(error)),
            }
        }

        let unshifted = dense_from_csr(matrix, n);
        let row = (0..n).find(|&i| unshifted[i + i * n] == 0.0).unwrap_or(0);
        Err(Error::SingularMatrix { row })
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

/// Converts the coarse CSR matrix to the crate's column-major dense layout
fn dense_from_csr(matrix: &CsrMatrix, n: usize) -> Vec<f64> {
    let mut lu = vec![0.0; n * n];
    for row in 0..n {
        let entries = matrix.row(row);
        for (&column, &value) in entries.columns.iter().zip(entries.values) {
            lu[row + column * n] += value;
        }
    }
    lu
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
