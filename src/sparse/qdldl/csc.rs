// Derived/transcribed from `osqp/qdldl` v0.1.9 (Apache License 2.0, Copyright OSQP Developers),
// adapted and modified for `algebra_linear` in pure safe Rust (Apache-2.0 §4(b) notice).

//! Borrowed upper-triangular compressed sparse column (CSC) validators.
//!
//! QDLDL expects the upper triangle of a symmetric matrix in CSC form with
//! strictly increasing row indices per column (`row <= col`). Non-empty columns
//! with an omitted diagonal entry (`row_idx[end - 1] < col`) are permitted so
//! symmetric quasi-definite (SQD) KKT systems with zero trailing blocks can be
//! represented directly without explicit zero diagonals; empty columns are
//! rejected later by [`super::etree::compute_etree`].

use super::error::{Error, Result};

/// A validated, borrowed upper-triangular sparsity pattern in CSC form.
#[derive(Clone, Copy, Debug)]
pub struct UpperTriCscPattern<'a> {
    n: usize,
    col_ptr: &'a [usize],
    row_idx: &'a [usize],
    fingerprint: u64,
}

impl<'a> UpperTriCscPattern<'a> {
    /// Validates and borrows upper-triangular CSC index arrays.
    ///
    /// Supports `n == 0` (`col_ptr = &[0]`, `row_idx = &[]`). Allows omitted
    /// diagonal entries in non-empty columns.
    ///
    /// # Errors
    ///
    /// * [`Error::BadShape`] if `col_ptr` is empty or `col_ptr[n] != row_idx.len()`;
    /// * [`Error::InvalidStructure`] if `col_ptr[0] != 0`, `col_ptr` is not
    ///   non-decreasing, or any `row >= n`;
    /// * [`Error::LowerTriangularEntry`] if any entry has `row > col`;
    /// * [`Error::DuplicateEntry`] if a column repeats a row index;
    /// * [`Error::UnsortedRows`] if a column's row indices are not strictly
    ///   increasing.
    pub fn new(col_ptr: &'a [usize], row_idx: &'a [usize]) -> Result<Self> {
        let n = col_ptr.len().checked_sub(1).ok_or(Error::BadShape)?;
        if col_ptr[0] != 0 {
            return Err(Error::InvalidStructure);
        }
        if col_ptr[n] != row_idx.len() {
            return Err(Error::BadShape);
        }
        for j in 0..n {
            if col_ptr[j + 1] < col_ptr[j] {
                return Err(Error::InvalidStructure);
            }
        }
        for j in 0..n {
            let start = col_ptr[j];
            let end = col_ptr[j + 1];
            for p in start..end {
                let row = row_idx[p];
                if row >= n {
                    return Err(Error::InvalidStructure);
                }
                if row > j {
                    return Err(Error::LowerTriangularEntry { row, col: j });
                }
                if p > start {
                    let prev = row_idx[p - 1];
                    if prev == row {
                        return Err(Error::DuplicateEntry);
                    }
                    if prev > row {
                        return Err(Error::UnsortedRows);
                    }
                }
            }
        }
        let fingerprint = fnv1a_fingerprint(col_ptr, row_idx);
        Ok(Self {
            n,
            col_ptr,
            row_idx,
            fingerprint,
        })
    }

    /// Returns the order `n` of the `n x n` matrix.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of stored entries in the upper triangle.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.col_ptr[self.n]
    }

    /// Returns the 64-bit FNV-1a fingerprint of `col_ptr` and `row_idx`.
    #[inline]
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// Returns the column pointers (`n + 1` entries).
    #[inline]
    pub fn col_ptr(&self) -> &'a [usize] {
        self.col_ptr
    }

    /// Returns the row indices (`nnz` entries).
    #[cfg(test)]
    #[inline]
    pub fn row_idx(&self) -> &'a [usize] {
        self.row_idx
    }

    /// Returns the row indices of column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= n`.
    #[inline]
    pub fn row_idx_of_col(&self, j: usize) -> &'a [usize] {
        assert!(j < self.n, "column index out of range");
        &self.row_idx[self.col_ptr[j]..self.col_ptr[j + 1]]
    }
}

/// A validated, borrowed upper-triangular CSC matrix with scalar values.
#[derive(Clone, Copy, Debug)]
pub struct UpperTriCscView<'a> {
    pattern: UpperTriCscPattern<'a>,
    values: &'a [f64],
}

impl<'a> UpperTriCscView<'a> {
    /// Validates and borrows an upper-triangular CSC matrix and its values.
    ///
    /// # Errors
    ///
    /// Returns any validation error from [`UpperTriCscPattern::new`], or
    /// [`Error::BadShape`] if `values.len() != pattern.nnz()`.
    pub fn new(col_ptr: &'a [usize], row_idx: &'a [usize], values: &'a [f64]) -> Result<Self> {
        let pattern = UpperTriCscPattern::new(col_ptr, row_idx)?;
        if values.len() != pattern.nnz() {
            return Err(Error::BadShape);
        }
        Ok(Self { pattern, values })
    }

    /// Returns the borrowed sparsity pattern.
    #[inline]
    pub fn pattern(&self) -> UpperTriCscPattern<'a> {
        self.pattern
    }

    /// Returns the order `n` of the `n x n` matrix.
    #[inline]
    pub fn n(&self) -> usize {
        self.pattern.n()
    }

    /// Returns the number of stored entries in the upper triangle.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.pattern.nnz()
    }

    /// Returns the nonzero values (`nnz` entries).
    #[cfg(test)]
    #[inline]
    pub fn values(&self) -> &'a [f64] {
        self.values
    }

    /// Returns the values of column `j`.
    ///
    /// # Panics
    ///
    /// Panics if `j >= n`.
    #[inline]
    pub fn val_of_col(&self, j: usize) -> &'a [f64] {
        assert!(j < self.pattern.n, "column index out of range");
        &self.values[self.pattern.col_ptr[j]..self.pattern.col_ptr[j + 1]]
    }
}

/// Computes the 64-bit FNV-1a fingerprint over `col_ptr` and `row_idx`.
fn fnv1a_fingerprint(col_ptr: &[usize], row_idx: &[usize]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in col_ptr.iter().chain(row_idx.iter()) {
        for byte in x.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}
