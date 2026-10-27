//! Algebraic multigrid and Krylov solvers for user-supplied CSR matrices.
//!
//! This module is a native Rust port of the core CPU functionality in
//! [AMGCL](https://github.com/ddemidov/amgcl), based on upstream revision
//! `1fbc24e687b404962c31bc4db598f6c73c0f4333`. It provides classical
//! Ruge–Stuben and aggregation-based coarsening, Galerkin products, relaxation,
//! dense coarse LU, preconditioned CG/BiCGStab/GMRES, and optional safe
//! shared-memory execution using standard-library scoped threads.
//!
//! AMGCL consumes compressed sparse row (CSR) matrices directly. The module
//! does not convert between CSR and the CSC format used by the sparse LU
//! solvers; callers provide the format required by their selected solver.
//!
//! ```
//! use algebra_linear::sparse::amgcl::{Amg, AmgParams, Cg, CsrMatrix, SolverParams};
//!
//! # fn main() -> Result<(), algebra_linear::sparse::amgcl::Error> {
//! let a = CsrMatrix::poisson_2d(16, 16)?;
//! let b = vec![1.0; a.nrows()];
//! let mut x = vec![0.0; a.nrows()];
//! let mut amg = Amg::new(&a, AmgParams::default())?;
//! let solver = Cg::new(SolverParams {
//!     tolerance: 1e-10,
//!     max_iterations: 100,
//!     ..SolverParams::default()
//! });
//! let report = solver.solve(&a, &mut amg, &b, &mut x)?;
//! assert!(report.converged);
//! # Ok(())
//! # }
//! ```
#![forbid(unsafe_code)]

mod amg;
mod coarse;
mod coarsening;
mod error;
mod matrix;
mod parallel;
mod preconditioner;
mod relaxation;
mod solver;
#[cfg(test)]
mod tests;
mod vector;

pub use amg::{Amg, AmgParams, LevelInfo};
pub use coarsening::{AggregationParams, Coarsening, CoarseningParams, PlainAggregationParams, SmoothedAggregationParams};
pub use error::{Error, Result};
pub use matrix::{CsrMatrix, CsrRow};
pub use parallel::Parallelism;
pub use preconditioner::{Identity, Preconditioner};
pub use relaxation::{ChebyshevParams, Ilu0Params, Smoother};
pub use solver::{BiCgStab, Cg, Gmres, SolveReport, SolverParams};
