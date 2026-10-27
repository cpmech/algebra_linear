use crate::lapack::{LapackError, dgetrf, dgetrs};
use crate::sparse::amgcl::{CsrMatrix, Error, Result};

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
        let mut lu = vec![0.0; n * n];
        for row in 0..n {
            let entries = matrix.row(row);
            for (&column, &value) in entries.columns.iter().zip(entries.values) {
                lu[row + column * n] += value;
            }
        }
        let pivots = match dgetrf(&mut lu, n) {
            Ok((pivots, _determinant)) => pivots,
            Err(LapackError::Singular) => {
                let row = (0..n).find(|&i| lu[i + i * n] == 0.0).unwrap_or(0);
                return Err(Error::SingularMatrix { row });
            }
            Err(LapackError::BadShape) => {
                return Err(Error::InvalidMatrix("coarse dense matrix has an inconsistent shape".into()));
            }
            Err(error) => {
                return Err(Error::SolverBreakdown(match error {
                    LapackError::Convergence => "coarse LU factorization did not converge",
                    LapackError::NotImplemented(_) => "coarse LU factorization is not implemented",
                    LapackError::Singular | LapackError::BadShape => unreachable!(),
                }));
            }
        };
        Ok(Self { lu, pivots })
    }

    pub(crate) fn solve(&self, rhs: &[f64], x: &mut [f64]) {
        debug_assert_eq!(rhs.len(), x.len());
        debug_assert_eq!(self.pivots.len(), rhs.len());
        x.copy_from_slice(rhs);
        dgetrs(&self.lu, &self.pivots, x);
    }
}
