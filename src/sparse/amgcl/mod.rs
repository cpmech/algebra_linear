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
//! [`amg_setup`] takes the CSR arrays and returns an opaque [`Amg`]; passing
//! `None` for its parameters stores the matrix only and applies as the identity
//! preconditioner. [`krylov_solve`] solves the stored system.
//!
//! ```
//! use algebra_linear::sparse::amgcl::{amg_setup, krylov_solve, AmgParams, KrylovMethod, SolverParams};
//!
//! // The 1-D Laplacian on three unknowns, A = tridiag(-1, 4, -1), in CSR form.
//! let row_ptr = [0, 2, 5, 7];
//! let col_idx = [0, 1, 0, 1, 2, 1, 2];
//! let values = [4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
//!
//! let mut amg = amg_setup(&row_ptr, &col_idx, &values, Some(AmgParams::default())).unwrap();
//! let b = [1.0, 2.0, 3.0];
//! let mut x = [0.0; 3];
//! let report = krylov_solve(
//!     KrylovMethod::Cg,
//!     &mut amg,
//!     &b,
//!     &mut x,
//!     SolverParams {
//!         tolerance: 1e-12,
//!         max_iterations: 20,
//!         ..SolverParams::default()
//!     },
//! )
//! .unwrap();
//! assert!(report.converged);
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

pub use amg::{Amg, AmgParams, LevelInfo, amg_setup};
pub use coarsening::{AggregationParams, Coarsening, CoarseningParams, PlainAggregationParams, SmoothedAggregationParams};
pub use error::{Error, Result};
pub use parallel::Parallelism;
pub use relaxation::{ChebyshevParams, Ilu0Params, Smoother};
pub use solver::{KrylovMethod, SolveReport, SolverParams, krylov_solve};
