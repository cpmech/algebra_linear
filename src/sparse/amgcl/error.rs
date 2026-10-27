use core::fmt;

/// Errors reported by matrix construction, AMG setup, and iterative solvers.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// CSR arrays or matrix entries are inconsistent.
    InvalidMatrix(String),
    /// A parameter is outside its meaningful range.
    InvalidParameter(String),
    /// A matrix/vector operation received incompatible dimensions.
    DimensionMismatch {
        /// Required vector or dimension length.
        expected: usize,
        /// Supplied vector or dimension length.
        actual: usize,
        /// Operation that detected the mismatch.
        context: &'static str,
    },
    /// A diagonal or coarse-level pivot is numerically singular.
    SingularMatrix {
        /// Zero-based row containing the singular diagonal or pivot.
        row: usize,
    },
    /// An iterative method encountered a scalar breakdown.
    SolverBreakdown(&'static str),
    /// A computation produced NaN or infinity.
    NonFinite,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMatrix(message) => write!(f, "invalid matrix: {message}"),
            Self::InvalidParameter(message) => write!(f, "invalid parameter: {message}"),
            Self::DimensionMismatch { expected, actual, context } => write!(f, "dimension mismatch in {context}: expected {expected}, got {actual}"),
            Self::SingularMatrix { row } => {
                write!(f, "matrix is numerically singular at row {row}")
            }
            Self::SolverBreakdown(message) => write!(f, "solver breakdown: {message}"),
            Self::NonFinite => write!(f, "computation produced a non-finite value"),
        }
    }
}

impl std::error::Error for Error {}

/// Result type used throughout the crate.
pub type Result<T> = core::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_are_exact() {
        let cases = [
            (Error::InvalidMatrix(String::from("ragged CSR")), "invalid matrix: ragged CSR"),
            (
                Error::InvalidParameter(String::from("negative tolerance")),
                "invalid parameter: negative tolerance",
            ),
            (
                Error::DimensionMismatch {
                    expected: 7,
                    actual: 5,
                    context: "krylov_solve",
                },
                "dimension mismatch in krylov_solve: expected 7, got 5",
            ),
            (Error::SingularMatrix { row: 4 }, "matrix is numerically singular at row 4"),
            (Error::SolverBreakdown("zero denominator"), "solver breakdown: zero denominator"),
            (Error::NonFinite, "computation produced a non-finite value"),
        ];
        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn usable_through_std_error() {
        let error: &dyn std::error::Error = &Error::NonFinite;
        assert!(!error.to_string().is_empty());
    }
}
