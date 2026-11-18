// Derived/transcribed from `osqp/qdldl` v0.1.9 (Apache License 2.0, Copyright OSQP Developers),
// adapted and modified for `algebra_linear` in pure safe Rust (Apache-2.0 §4(b) notice).

//! Error types for the native QDLDL `L D L^T` solver.

use core::fmt;

/// Errors returned by upper-triangular CSC validation, symbolic analysis,
/// numeric factorization, and triangular solves in [`super::Qdldl`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// An array or vector length disagrees with the matrix order or entry count.
    BadShape,
    /// The CSC arrays are not a valid compressed sparse column structure.
    InvalidStructure,
    /// A column's row indices are not strictly increasing.
    UnsortedRows,
    /// A column contains duplicate row indices.
    DuplicateEntry,
    /// An entry lies strictly below the main diagonal (`row > col`).
    LowerTriangularEntry {
        /// Zero-based row index of the out-of-triangle entry.
        row: usize,
        /// Zero-based column index of the out-of-triangle entry.
        col: usize,
    },
    /// A column in the upper-triangular matrix contains no entries.
    EmptyColumn {
        /// Zero-based index of the empty column.
        col: usize,
    },
    /// An index or nonzero count overflowed `usize`.
    IndexOverflow,
    /// The matrix pattern differs from the pattern seen during symbolic analysis.
    StructureMismatch,
    /// A matrix entry or computed diagonal pivot is `NaN` or infinite.
    NotFinite,
    /// A zero diagonal pivot (or underflow producing a non-finite reciprocal) occurred.
    ZeroDiagonal {
        /// Zero-based index of the zero diagonal pivot.
        index: usize,
    },
    /// A solve was requested before a successful numeric factorization.
    NotFactorized,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadShape => f.write_str("inconsistent shape or length"),
            Self::InvalidStructure => f.write_str("invalid upper-triangular CSC structure"),
            Self::UnsortedRows => f.write_str("row indices must be strictly increasing within a column"),
            Self::DuplicateEntry => f.write_str("duplicate entry in a column"),
            Self::LowerTriangularEntry { row, col } => {
                write!(f, "lower-triangular entry at ({row}, {col}); expected row <= col")
            }
            Self::EmptyColumn { col } => {
                write!(f, "empty column {col} in upper-triangular matrix")
            }
            Self::IndexOverflow => f.write_str("index or nonzero count overflow"),
            Self::StructureMismatch => f.write_str("the matrix pattern differs from the symbolic analysis"),
            Self::NotFinite => f.write_str("the matrix or pivot has a NaN or infinite value"),
            Self::ZeroDiagonal { index } => {
                write!(f, "zero diagonal pivot at index {index}")
            }
            Self::NotFactorized => f.write_str("the matrix has not been factorized"),
        }
    }
}

impl std::error::Error for Error {}

/// Result type returned by the QDLDL routines.
pub type Result<T> = core::result::Result<T, Error>;
