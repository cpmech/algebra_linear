//! Sparse direct and iterative solvers.
//!
//! [`SparseLu`] is the direct solver: it analyzes a compressed sparse column
//! (CSC) pattern once, then factorizes any values on that pattern. The COLAMD
//! ordering reduces fill, [`LuMethod`] chooses a simplicial or supernodal
//! factorization, and [`SparseLuStats`] reports the structure, cost and timing.
//! The [`amgcl`] submodule provides algebraic multigrid and preconditioned
//! Krylov solvers for compressed sparse row (CSR) arrays.
//!
//! # Provenance
//!
//! The algorithms are translated from faer commit `ddbd2ae`
//! (<https://codeberg.org/sarah-quinones/faer>):
//!
//! * `faer/src/sparse/linalg/lu.rs` — the simplicial LU, MIT;
//! * `faer/src/sparse/linalg/colamd.rs` — the COLAMD ordering, faer's
//!   transcription of SuiteSparse COLAMD, BSD 3-Clause;
//! * the column elimination tree and postorder helpers from
//!   `faer/src/sparse/linalg/qr.rs` and `.../cholesky.rs`.
//!
//! The [`amgcl`] submodule follows AMGCL revision
//! `1fbc24e687b404962c31bc4db598f6c73c0f4333` (MIT).
//!
//! See `LICENSE` for the notices.
//!
//! # Conventions
//!
//! The crate's conventions hold here as well: column-major dense storage,
//! 0-based `usize` indices, no workspace arguments, and errors instead of
//! `INFO`. Sparse inputs are not converted between formats: sparse LU callers
//! provide CSC, while AMGCL callers provide CSR as required by the solver.
//!
//! # Example
//!
//! ```
//! use algebra_linear::sparse::{SparseLu, LuMethod};
//!
//! let col_ptr = [0, 2, 4];
//! let row_idx = [0, 1, 0, 1];
//! let values = [4.0, 1.0, 1.0, 3.0];
//! let mut lu = SparseLu::analyze(&col_ptr, &row_idx, LuMethod::auto()).unwrap();
//! lu.factorize(&col_ptr, &row_idx, &values).unwrap();
//! let mut b = [1.0, 2.0];
//! lu.solve(&mut b).unwrap();
//! assert!((b[0] - 1.0 / 11.0).abs() < 1e-14);
//! assert!((b[1] - 7.0 / 11.0).abs() < 1e-14);
//! ```

pub mod amgcl;

mod csc;
mod etree;
mod lu;
mod ordering;
mod perm;
mod solver;
mod sparse_lu;
mod stats;
mod supernodal;
mod symbolic;

#[cfg(test)]
mod accuracy_tests;

pub use lu::SparseScalar;
pub use solver::LuMethod;
pub use sparse_lu::SparseLu;
pub use stats::SparseLuStats;

/// Errors reported by the sparse routines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseError {
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

impl core::fmt::Display for SparseError {
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

impl std::error::Error for SparseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_are_exact() {
        let cases = [
            (SparseError::BadShape, "inconsistent shape or length"),
            (SparseError::InvalidStructure, "invalid CSC structure"),
            (SparseError::UnsortedRows, "row indices must be strictly increasing within a column"),
            (SparseError::DuplicateEntry, "duplicate entry in a column"),
            (SparseError::Singular, "the matrix is singular"),
            (SparseError::IndexOverflow, "index or workspace size overflow"),
            (SparseError::StructureMismatch, "the matrix pattern differs from the symbolic analysis"),
            (SparseError::NotFinite, "the matrix has a NaN or infinite entry"),
            (SparseError::NotFactorized, "the matrix has not been factorized"),
        ];
        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn usable_through_std_error() {
        let error: &dyn std::error::Error = &SparseError::BadShape;
        assert!(!error.to_string().is_empty());
    }
}
