//! Sparse direct and iterative solvers.
//!
//! The module provides a simplicial sparse LU factorization in compressed sparse
//! column (CSC) form, a supernodal one, and the COLAMD column ordering they use
//! to reduce fill. The supernodal path is usable directly (`SymbolicSupernodal`,
//! `SupernodalLu`) or through `LuSymbolic`/`LuMethod`, which chooses between the
//! two with a flop proxy (or a forced method). The [`amgcl`] submodule provides
//! algebraic multigrid and preconditioned Krylov solvers for compressed sparse
//! row (CSR) matrices.
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
//! use algebra_linear::sparse::{CscView, SymbolicLu};
//!
//! // A = [4 1; 1 3] in CSC form.
//! let col_ptr = [0, 2, 4];
//! let row_idx = [0, 1, 0, 1];
//! let values = [4.0, 1.0, 1.0, 3.0];
//!
//! let a = CscView::new(2, 2, &col_ptr, &row_idx, &values).unwrap();
//! let symbolic = SymbolicLu::analyze(a.pattern()).unwrap();
//! let factors = symbolic.factorize(a).unwrap();
//!
//! let mut b = [1.0, 2.0];
//! factors.solve(&mut b).unwrap();
//! assert!((b[0] - 1.0 / 11.0).abs() < 1e-14);
//! assert!((b[1] - 7.0 / 11.0).abs() < 1e-14);
//! ```

pub mod amgcl;
pub mod csc;
mod etree;
pub mod lu;
pub mod ordering;
pub mod perm;
pub mod solver;

#[cfg(test)]
mod accuracy_tests;
mod stats;
pub mod supernodal;
pub mod symbolic;

pub use csc::{CscPattern, CscView};
pub use lu::{LuFactors, SparseScalar};
pub use ordering::colamd::{ColamdControl, ColamdOrdering, colamd_order};
pub use solver::{LuMethod, LuSolver, LuSymbolic};
pub use stats::SparseLuStats;
pub use supernodal::{SupernodalLu, SymbolicSupernodal};
pub use symbolic::SymbolicLu;

/// Errors reported by the sparse routines.
///
/// The enum is non-exhaustive: it is expected to gain variants as the module
/// grows, and a caller should not match it exhaustively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
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
        };
        f.write_str(msg)
    }
}

impl std::error::Error for SparseError {}
