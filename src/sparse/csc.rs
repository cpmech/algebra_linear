//! Compressed sparse column (CSC) views.
//!
//! A [`CscPattern`] borrows the two index arrays of a CSC matrix; a [`CscView`]
//! adds the values. Neither owns storage and neither copies the caller's arrays,
//! so a caller can wrap an existing matrix without a conversion pass.
//!
//! Both require the row indices of each column to be **strictly increasing**:
//! duplicates are reported as [`SparseError::DuplicateEntry`] and out-of-order
//! rows as [`SparseError::UnsortedRows`]. Requiring order makes the symbolic and
//! numeric passes single-sweep and lets a later adapter from another crate's CSC
//! type prove its conversion infallible.

use super::SparseError;

/// A borrowed sparsity pattern in compressed sparse column form.
///
/// The storage is described by `col_ptr` (`ncols + 1` entries, starting at zero
/// and non-decreasing) and `row_idx` (`col_ptr[ncols]` entries). This type does
/// not carry values; see [`CscView`] for that.
#[derive(Clone, Copy, Debug)]
pub struct CscPattern<'a> {
    nrows: usize,
    ncols: usize,
    col_ptr: &'a [usize],
    row_idx: &'a [usize],
}

impl<'a> CscPattern<'a> {
    /// Validates the arrays and borrows them.
    ///
    /// # Errors
    ///
    /// * [`SparseError::BadShape`] if `col_ptr` does not have `ncols + 1`
    ///   entries, or if its last entry is not `row_idx.len()`;
    /// * [`SparseError::InvalidStructure`] if `col_ptr` does not start at zero,
    ///   is not non-decreasing, or if a row index is not below `nrows`;
    /// * [`SparseError::DuplicateEntry`] if a column names the same row twice;
    /// * [`SparseError::UnsortedRows`] if a column's rows are out of order.
    pub fn new(nrows: usize, ncols: usize, col_ptr: &'a [usize], row_idx: &'a [usize]) -> Result<Self, SparseError> {
        if col_ptr.len() != ncols + 1 {
            return Err(SparseError::BadShape);
        }
        if col_ptr[0] != 0 {
            return Err(SparseError::InvalidStructure);
        }
        if col_ptr[ncols] != row_idx.len() {
            return Err(SparseError::BadShape);
        }
        for w in 1..=ncols {
            if col_ptr[w] < col_ptr[w - 1] {
                return Err(SparseError::InvalidStructure);
            }
        }
        for &r in row_idx {
            if r >= nrows {
                return Err(SparseError::InvalidStructure);
            }
        }
        for j in 0..ncols {
            let (start, end) = (col_ptr[j], col_ptr[j + 1]);
            for k in start + 1..end {
                if row_idx[k - 1] == row_idx[k] {
                    return Err(SparseError::DuplicateEntry);
                }
                if row_idx[k - 1] > row_idx[k] {
                    return Err(SparseError::UnsortedRows);
                }
            }
        }
        Ok(Self {
            nrows,
            ncols,
            col_ptr,
            row_idx,
        })
    }

    /// Returns the number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Returns the number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Returns the number of stored entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.col_ptr[self.ncols]
    }

    /// Returns the column pointers, with `ncols + 1` entries.
    #[inline]
    pub fn col_ptr(&self) -> &'a [usize] {
        self.col_ptr
    }

    /// Returns the row indices, with `nnz` entries.
    #[inline]
    pub fn row_idx(&self) -> &'a [usize] {
        self.row_idx
    }

    /// Returns the row indices of column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= ncols`, which is a caller bug rather than bad data.
    #[inline]
    pub fn row_idx_of_col(&self, j: usize) -> &'a [usize] {
        assert!(j < self.ncols, "column index out of range");
        &self.row_idx[self.col_ptr[j]..self.col_ptr[j + 1]]
    }
}

/// A borrowed CSC matrix: a [`CscPattern`] plus its values.
#[derive(Clone, Copy, Debug)]
pub struct CscView<'a, T> {
    pattern: CscPattern<'a>,
    values: &'a [T],
}

impl<'a, T> CscView<'a, T> {
    /// Validates the arrays and borrows them.
    ///
    /// # Errors
    ///
    /// The pattern errors of [`CscPattern::new`], plus
    /// [`SparseError::BadShape`] if `values.len()` differs from the pattern's
    /// number of stored entries.
    pub fn new(nrows: usize, ncols: usize, col_ptr: &'a [usize], row_idx: &'a [usize], values: &'a [T]) -> Result<Self, SparseError> {
        let pattern = CscPattern::new(nrows, ncols, col_ptr, row_idx)?;
        if values.len() != pattern.nnz() {
            return Err(SparseError::BadShape);
        }
        Ok(Self { pattern, values })
    }

    /// Returns the borrowed pattern.
    #[inline]
    pub fn pattern(&self) -> CscPattern<'a> {
        self.pattern
    }

    /// Returns the number of stored entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.pattern.nnz()
    }

    /// Returns the values, with `nnz` entries.
    #[inline]
    pub fn values(&self) -> &'a [T] {
        self.values
    }

    /// Returns the values of column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= ncols`, which is a caller bug rather than bad data.
    #[inline]
    pub(crate) fn val_of_col(&self, j: usize) -> &'a [T] {
        let (start, end) = (self.pattern.col_ptr[j], self.pattern.col_ptr[j + 1]);
        &self.values[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_accepts_a_valid_matrix() {
        let col_ptr = [0, 2, 3];
        let row_idx = [0, 1, 1];
        let a = CscPattern::new(2, 2, &col_ptr, &row_idx).unwrap();
        assert_eq!(a.nrows(), 2);
        assert_eq!(a.ncols(), 2);
        assert_eq!(a.nnz(), 3);
        assert_eq!(a.row_idx_of_col(0), &[0, 1]);
        assert_eq!(a.row_idx_of_col(1), &[1]);
    }

    #[test]
    fn pattern_accepts_empty_matrices() {
        let a = CscPattern::new(0, 0, &[0], &[]).unwrap();
        assert_eq!(a.nnz(), 0);
        let b = CscPattern::new(3, 3, &[0, 0, 0, 0], &[]).unwrap();
        assert_eq!(b.nnz(), 0);
        assert_eq!(b.row_idx_of_col(2), &[] as &[usize]);
    }

    #[test]
    fn pattern_rejects_short_col_ptr() {
        assert_eq!(CscPattern::new(2, 2, &[0, 1], &[0]).unwrap_err(), SparseError::BadShape);
    }

    #[test]
    fn pattern_rejects_len_mismatch() {
        assert_eq!(CscPattern::new(2, 2, &[0, 1, 2], &[0]).unwrap_err(), SparseError::BadShape);
    }

    #[test]
    fn pattern_rejects_nonzero_start() {
        assert_eq!(CscPattern::new(2, 2, &[1, 1, 2], &[0, 1]).unwrap_err(), SparseError::InvalidStructure);
    }

    #[test]
    fn pattern_rejects_decreasing_col_ptr() {
        assert_eq!(CscPattern::new(2, 2, &[0, 2, 1], &[0]).unwrap_err(), SparseError::InvalidStructure);
    }

    #[test]
    fn pattern_rejects_row_out_of_range() {
        assert_eq!(CscPattern::new(2, 2, &[0, 1, 2], &[0, 2]).unwrap_err(), SparseError::InvalidStructure);
    }

    #[test]
    fn pattern_rejects_duplicate_and_unsorted_rows() {
        assert_eq!(CscPattern::new(3, 3, &[0, 2, 2, 2], &[1, 1]).unwrap_err(), SparseError::DuplicateEntry);
        assert_eq!(CscPattern::new(3, 3, &[0, 2, 2, 2], &[2, 1]).unwrap_err(), SparseError::UnsortedRows);
    }

    #[test]
    fn view_checks_the_value_length() {
        let col_ptr = [0, 2, 3];
        let row_idx = [0, 1, 1];
        assert_eq!(CscView::new(2, 2, &col_ptr, &row_idx, &[1.0, 2.0]).unwrap_err(), SparseError::BadShape);
        let a = CscView::new(2, 2, &col_ptr, &row_idx, &[4.0, 1.0, 3.0]).unwrap();
        assert_eq!(a.nnz(), 3);
        assert_eq!(a.val_of_col(0), &[4.0, 1.0]);
        assert_eq!(a.pattern().row_idx_of_col(1), &[1]);
    }

    #[test]
    fn complex_values_are_borrowed_unchanged() {
        use crate::Complex64;
        let col_ptr = [0, 1];
        let row_idx = [0];
        let values = [Complex64::new(1.0, -2.0)];
        let a = CscView::new(1, 1, &col_ptr, &row_idx, &values).unwrap();
        assert_eq!(a.values()[0], Complex64::new(1.0, -2.0));
    }
}
