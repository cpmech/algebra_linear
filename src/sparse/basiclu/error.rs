//! Error types for [`super::BasicLu`].
//!
//! Translates the integer status codes of `ERGO-Code/HiGHS`
//! (`highs/ipm/basiclu/basiclu.h`, MIT) into strongly typed Rust errors.

use core::fmt;

/// Errors reported by `sparse::basiclu`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Dimension mismatch in matrix arrays, right-hand-side vectors, or update slices.
    BadShape,
    /// `col_ptr` does not start at `0`, is not non-decreasing, or a row index is out of bounds.
    InvalidStructure,
    /// A column's row indices are not strictly increasing.
    UnsortedRows,
    /// A column contains duplicate row indices.
    DuplicateEntry,
    /// An input matrix entry, update entry, or right-hand-side scalar is NaN or infinite.
    NotFinite,
    /// Factorization stopped with `rank < n` while [`BasicLuParams::allow_singular`](super::BasicLuParams::allow_singular) is `false`.
    Singular {
        /// Numerical rank achieved before stopping.
        rank: usize,
        /// Matrix order.
        n: usize,
    },
    /// Rank-1 column update produced a numerically or structurally singular pivot; prior factors remain intact.
    SingularUpdate,
    /// Number of updates since the last factorization reached [`BasicLuParams::max_updates`](super::BasicLuParams::max_updates).
    MaxUpdatesReached,
    /// Replacement column index `col` is outside `0..n`.
    InvalidColumn {
        /// Out-of-range column index.
        col: usize,
        /// Matrix order.
        n: usize,
    },
    /// Solve or update was called before a successful factorization.
    NotFactorized,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadShape => f.write_str("inconsistent shape or length"),
            Self::InvalidStructure => f.write_str("invalid CSC structure or out-of-bounds row index"),
            Self::UnsortedRows => f.write_str("row indices must be strictly increasing within a column"),
            Self::DuplicateEntry => f.write_str("duplicate entry in a column"),
            Self::NotFinite => f.write_str("input contains a NaN or infinite value"),
            Self::Singular { rank, n } => write!(f, "matrix is singular (rank {rank} of {n})"),
            Self::SingularUpdate => f.write_str("column update produced a singular pivot"),
            Self::MaxUpdatesReached => f.write_str("maximum number of updates reached; refactorize"),
            Self::InvalidColumn { col, n } => write!(f, "column index {col} out of bounds for matrix order {n}"),
            Self::NotFactorized => f.write_str("the matrix has not been factorized"),
        }
    }
}

impl std::error::Error for Error {}

/// Result type returned by `sparse::basiclu` operations.
pub type Result<T> = core::result::Result<T, Error>;
