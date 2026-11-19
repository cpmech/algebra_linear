//! The error type of the direct sparse solver.

/// Errors reported by the sparse routines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// A shape or length was inconsistent with another.
    BadShape,
    /// The CSC arrays are not a valid compressed sparse column structure.
    InvalidStructure,
    /// A column's row indices are not strictly increasing.
    UnsortedRows,
    /// A column has two entries in the same row.
    DuplicateEntry,
    /// The matrix is singular: a column has no nonzero pivot.
    Singular,
    /// An index or a workspace size overflowed `usize`.
    IndexOverflow,
    /// The pattern differs from the one the symbolic analysis was built for.
    StructureMismatch,
    /// An entry of the matrix is NaN or infinite.
    NotFinite,
    /// `solve` was called before a successful factorization.
    NotFactorized,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            Self::BadShape => "inconsistent shape or length",
            Self::InvalidStructure => "invalid CSC structure",
            Self::UnsortedRows => "row indices must be strictly increasing within a column",
            Self::DuplicateEntry => "duplicate entry in a column",
            Self::Singular => "the matrix is singular",
            Self::IndexOverflow => "index or workspace size overflow",
            Self::StructureMismatch => "the matrix pattern differs from the symbolic analysis",
            Self::NotFinite => "the matrix has a NaN or infinite entry",
            Self::NotFactorized => "the matrix has not been factorized",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for Error {}

/// The result type of the direct sparse solver.
pub type Result<T> = core::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_are_exact() {
        let cases = [
            (Error::BadShape, "inconsistent shape or length"),
            (Error::InvalidStructure, "invalid CSC structure"),
            (Error::UnsortedRows, "row indices must be strictly increasing within a column"),
            (Error::DuplicateEntry, "duplicate entry in a column"),
            (Error::Singular, "the matrix is singular"),
            (Error::IndexOverflow, "index or workspace size overflow"),
            (Error::StructureMismatch, "the matrix pattern differs from the symbolic analysis"),
            (Error::NotFinite, "the matrix has a NaN or infinite entry"),
            (Error::NotFactorized, "the matrix has not been factorized"),
        ];
        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn usable_through_std_error() {
        let error: &dyn std::error::Error = &Error::BadShape;
        assert!(!error.to_string().is_empty());
    }
}
