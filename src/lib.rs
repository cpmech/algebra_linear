#![warn(missing_docs)]
//! Pure-Rust BLAS, LAPACK, and sparse LU direct solver (`sparse::faer`).
//!
//! # Layout
//!
//! An upstream port is a folder named after its upstream (`lapack/`,
//! `sparse/faer/`) or a single file when all kernels share
//! one preamble (`blas.rs`), and each holds the translated kernels plus the Rust
//! handles, error and statistics types and the tests that expose them. Level
//! roots (`lib.rs`, `sparse/mod.rs`) hold module declarations, crate-level
//! re-exports and level documentation only. The C reference-parity harness
//! (`dev_lapack.rs`, `parity.rs`) is feature-gated at the `src/` root.
//!
//! # Conventions
//!
//! * Matrices are column-major, so element `(i, j)` of an `nrow`-row matrix is at
//!   `data[i + j * nrow]`.
//! * Indices are 0-based and lengths are `usize`, unlike the Fortran the routines
//!   are derived from (`BLAS`/`LAPACK` are 1-based and take workspace arguments
//!   that are an artifact of being written in Fortran, not part of the math).
//! * Nothing takes a workspace argument. The point of a Rust implementation is
//!   that workspace can be allocated internally.
//!
//! # Provenance and verification
//!
//! Each routine names the LAPACK 3.12.1 source it follows, so it can be checked
//! against the reference. Correctness is tested for parity against the reference
//! LAPACK where the environment provides it (the `dev_lapack` feature, a
//! development-only harness), and against closed-form results otherwise; see the
//! tests in each module.
//!
//! The dense routines here are derived from LAPACK and reference BLAS, which
//! are BSD-3-Clause; `sparse::faer` follows the MIT-licensed faer project.
//! The crate is distributed under `BSD-3-Clause`; see `LICENSE` for all
//! copyright notices and license texts.

pub mod blas;
pub mod lapack;
pub mod sparse;

pub(crate) mod constants;

#[cfg(feature = "dev_lapack")]
pub(crate) mod dev_lapack;

#[cfg(all(test, feature = "dev_lapack"))]
mod parity;

pub use num_complex::Complex64;

/// Builds a [`Complex64`] from its real and imaginary parts.
///
/// ```
/// use algebra_linear::{cpx, Complex64};
/// let z = cpx!(1.0, 2.0);
/// assert_eq!(z, Complex64::new(1.0, 2.0));
/// ```
#[macro_export]
macro_rules! cpx {
    ($real:expr, $imag:expr) => {
        $crate::Complex64::new($real, $imag)
    };
}

// run the code blocks in README.md as doctests
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctest;
