//! The symbolic LU analysis.
//!
//! For the simplicial path the symbolic analysis is small: a COLAMD column
//! ordering plus a fingerprint of the pattern, so a later numeric factorization
//! can prove it is looking at the same structure. The elimination tree and the
//! supernodal analysis are only needed by the supernodal path and are not part
//! of this module yet.

use std::time::Duration;
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::time::Instant;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
use web_time::Instant;

use super::csc::CscPattern;
use super::ordering::colamd::{ColamdControl, colamd_order};
use super::{SparseError, perm};

/// The result of a symbolic LU analysis.
///
/// Build it once with [`SymbolicLu::analyze`], then call it for each set of
/// values with the same pattern; see [`SymbolicLu::factorize`].
#[derive(Clone, Debug)]
pub struct SymbolicLu {
    n: usize,
    nnz: usize,
    fingerprint: u64,
    col_perm: Vec<usize>,
    col_perm_inv: Vec<usize>,
    ordering_time: Duration,
}

impl SymbolicLu {
    /// Analyzes the pattern: computes the COLAMD column ordering and records a
    /// fingerprint of the structure.
    ///
    /// # Errors
    ///
    /// [`SparseError::BadShape`] if the matrix is not square, and the ordering
    /// errors of [`colamd_order`].
    pub fn analyze(pattern: CscPattern<'_>) -> Result<Self, SparseError> {
        if pattern.nrows() != pattern.ncols() {
            return Err(SparseError::BadShape);
        }
        let start = Instant::now();
        let ordering = colamd_order(pattern, ColamdControl::default())?;
        let ordering_time = start.elapsed();
        debug_assert!(perm::is_permutation(ordering.perm()));
        Ok(Self {
            n: pattern.ncols(),
            nnz: pattern.nnz(),
            fingerprint: fingerprint(pattern),
            col_perm: ordering.perm().to_vec(),
            col_perm_inv: ordering.perm_inv().to_vec(),
            ordering_time,
        })
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

    /// Returns the column permutation, mapping a new position to the original
    /// column.
    #[inline]
    pub fn col_perm(&self) -> &[usize] {
        &self.col_perm
    }

    /// Returns the inverse column permutation, mapping an original column to
    /// its new position.
    #[inline]
    pub fn col_perm_inv(&self) -> &[usize] {
        &self.col_perm_inv
    }

    /// Returns the time the ordering took during [`Self::analyze`].
    #[inline]
    pub fn ordering_time(&self) -> Duration {
        self.ordering_time
    }

    /// Returns `true` if `pattern` has the structure this analysis was built
    /// for: same order, same number of entries, same index arrays.
    pub fn matches(&self, pattern: CscPattern<'_>) -> bool {
        pattern.ncols() == self.n && pattern.nrows() == self.n && pattern.nnz() == self.nnz && fingerprint(pattern) == self.fingerprint
    }

    /// Checks the structure before a numeric factorization.
    ///
    /// # Errors
    ///
    /// [`SparseError::StructureMismatch`] if `pattern` differs from the one the
    /// analysis saw. The comparison is a 64-bit fingerprint of the index arrays,
    /// so it is O(`nnz`) and never allocates.
    pub(crate) fn check_pattern(&self, pattern: CscPattern<'_>) -> Result<(), SparseError> {
        if self.matches(pattern) {
            Ok(())
        } else {
            Err(SparseError::StructureMismatch)
        }
    }
}

/// FNV-1a over the column pointers and row indices.
pub(crate) fn fingerprint(pattern: CscPattern<'_>) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in pattern.col_ptr().iter().chain(pattern.row_idx().iter()) {
        for byte in x.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_by_two() -> (Vec<usize>, Vec<usize>) {
        (vec![0, 2, 3], vec![0, 1, 1])
    }

    #[test]
    fn analyzes_a_small_matrix() {
        let (col_ptr, row_idx) = two_by_two();
        let pattern = CscPattern::new(2, 2, &col_ptr, &row_idx).unwrap();
        let symbolic = SymbolicLu::analyze(pattern).unwrap();
        assert_eq!(symbolic.n(), 2);
        assert_eq!(symbolic.nnz(), 3);
        assert!(perm::is_permutation(symbolic.col_perm()));
        assert!(perm::is_permutation(symbolic.col_perm_inv()));
        assert!(symbolic.matches(pattern));
    }

    #[test]
    fn detects_a_pattern_change() {
        let (col_ptr, row_idx) = two_by_two();
        let pattern = CscPattern::new(2, 2, &col_ptr, &row_idx).unwrap();
        let symbolic = SymbolicLu::analyze(pattern).unwrap();

        // Same shape and nnz, different pattern.
        let other = CscPattern::new(2, 2, &[0, 1, 3], &[0, 0, 1]).unwrap();
        assert!(!symbolic.matches(other));
        assert_eq!(symbolic.check_pattern(other).unwrap_err(), SparseError::StructureMismatch);

        // Different nnz.
        let empty = CscPattern::new(2, 2, &[0, 0, 0], &[]).unwrap();
        assert!(!symbolic.matches(empty));
    }

    #[test]
    fn rejects_rectangular() {
        let pattern = CscPattern::new(2, 3, &[0, 0, 0, 0], &[]).unwrap();
        assert_eq!(SymbolicLu::analyze(pattern).unwrap_err(), SparseError::BadShape);
    }

    #[test]
    fn handles_a_zero_by_zero_matrix() {
        let pattern = CscPattern::new(0, 0, &[0], &[]).unwrap();
        let symbolic = SymbolicLu::analyze(pattern).unwrap();
        assert_eq!(symbolic.n(), 0);
        assert!(symbolic.col_perm().is_empty());
    }
}
