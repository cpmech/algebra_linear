//! Statistics of a sparse LU factorization.

use core::time::Duration;

/// Structure, cost and timing of a sparse LU factorization.
///
/// The counts describe the factors in their permuted coordinate system. The
/// flop figure is an **estimate**: it counts one multiply-add per `(L, U)` entry
/// pair, which is the dominant term of the simplicial factorization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparseLuStats {
    /// The order of the matrix.
    pub n: usize,
    /// The number of stored entries in the input matrix.
    pub nnz_a: usize,
    /// The number of stored entries in `L`, including its unit diagonal.
    pub nnz_l: usize,
    /// The number of stored entries in `U`.
    pub nnz_u: usize,
    /// `(nnz_l + nnz_u) / nnz_a`, or zero for an empty matrix.
    pub fill_ratio: f64,
    /// An estimate of the multiply-adds in the factorization.
    pub flops_estimate: usize,
    /// The time spent computing the column ordering.
    pub ordering_time: Duration,
    /// The time spent computing the numeric factorization.
    pub factorization_time: Duration,
}

impl SparseLuStats {
    /// Returns the total number of stored entries in both factors.
    #[inline]
    pub fn nnz_factors(&self) -> usize {
        self.nnz_l + self.nnz_u
    }
}

/// Counts one multiply-add per `(L, U)` entry pair, summed over the columns.
///
/// This is the flop estimate shared by the simplicial and supernodal paths.
pub(crate) fn flops_estimate(l_col_ptr: &[usize], u_col_ptr: &[usize]) -> usize {
    debug_assert_eq!(l_col_ptr.len(), u_col_ptr.len());
    let n = l_col_ptr.len().saturating_sub(1);
    let mut total = 0usize;
    for j in 0..n {
        let nl = l_col_ptr[j + 1] - l_col_ptr[j];
        let nu = u_col_ptr[j + 1] - u_col_ptr[j];
        total = total.saturating_add(nl.saturating_mul(nu));
    }
    total
}
