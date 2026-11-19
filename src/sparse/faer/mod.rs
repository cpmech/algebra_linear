//! The sparse direct solver: a faer port plus this crate's supernodal path.
//!
//! This is the crate's general sparse LU. [`SparseLu`] analyzes a compressed
//! sparse column (CSC) pattern once — a COLAMD column ordering, a fingerprinted
//! structure and, for the supernodal path, the column counts and the supernode
//! partition — and then factorizes any values on that pattern. [`LuMethod`]
//! chooses the simplicial or supernodal factorization, [`EffectiveLuMethod`]
//! reports what the analysis selected, [`SparseLuStats`] reports structure,
//! cost and timing, and [`SparseScalar`] is the scalar trait behind the numeric
//! kernels (`f64` and `Complex64`).
//!
//! The translated algorithms come from faer commit `ddbd2ae`
//! (<https://codeberg.org/sarah-quinones/faer>); the supernodal numeric
//! factorization and the Rust handle are this crate's own. Nothing here depends
//! on faer at build time.
//!
//! # Source map
//!
//! | upstream faer file | local file | what changed |
//! |---|---|---|
//! | `src/sparse/linalg/lu.rs` | `lu.rs` | `MatMut`, `MemStack` and `PermRef` are replaced by slices and `Vec` workspaces, and the scalar operations go through [`SparseScalar`] |
//! | `src/sparse/linalg/qr.rs` (`ghost_col_etree`), `.../cholesky.rs` (`ghost_postorder`) | `etree.rs` | the `I: Index` generics are replaced by `usize`/`isize` |
//! | `src/sparse/linalg/qr.rs` (`ghost_column_counts_aat`, `ghost_least_common_ancestor`), `.../cholesky.rs` (fundamental supernode partition) | `supernodal.rs` | only the fundamental partition is translated; `with_relaxation` is a simplified version of faer's relaxation |
//! | `src/sparse/linalg/colamd.rs` | `colamd.rs` | faer's abstractions are replaced by the local CSC pattern, and the jumbled-input repair path is not translated because the pattern is validated on construction |
//! | — | `supernodal_lu.rs` | ours: a correctness-first numeric supernodal LU over the symbolic structure, which keeps no contribution blocks between supernodes and instead updates each front from the stored `L` and `U` columns |
//! | — | `symbolic.rs`, `solver.rs`, `sparse_lu.rs`, `stats.rs` | ours: the symbolic analysis, the method choice behind the handle, the handle itself, and the statistics with their flop estimate |
//! | — | `csc.rs`, `perm.rs` | ours: the borrowed CSC pattern and view, and the permutation checks and inverse |
//! | — | `error.rs` | ours: [`Error`] and [`Result`] |
//! | — | `accuracy_tests.rs` | ours, `#[cfg(test)]`: forward error and scaled residual on LAPACK's random matrices |
//!
//! # Licenses
//!
//! `lu.rs`, `etree.rs` and `supernodal.rs` are translated from MIT-licensed
//! faer. `colamd.rs` is translated from faer's BSD-3-Clause transcription of
//! SuiteSparse COLAMD (Copyright 1998-2022 Timothy A. Davis). See `LICENSE` for
//! both notices. Every other file in this module is ours.
#![forbid(unsafe_code)]

#[cfg(test)]
mod accuracy_tests;
mod colamd;
mod csc;
mod error;
mod etree;
mod lu;
mod perm;
mod solver;
mod sparse_lu;
mod stats;
mod supernodal;
mod supernodal_lu;
mod symbolic;

pub use {
    error::{Error, Result},
    lu::SparseScalar,
    solver::{EffectiveLuMethod, LuMethod},
    sparse_lu::SparseLu,
    stats::SparseLuStats,
};
