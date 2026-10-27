#![warn(missing_docs)]
//! Pure-Rust BLAS, LAPACK, sparse direct solvers, and AMG/Krylov solvers.
//!
//! # Layout and conventions
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
//! The routines here are derived from LAPACK and reference BLAS, which are
//! BSD-3-Clause; see `LICENSE` for their copyright notices and this crate's
//! licence.

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
