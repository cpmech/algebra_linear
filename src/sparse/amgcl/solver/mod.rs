mod bicgstab;
mod cg;
mod gmres;

pub use bicgstab::BiCgStab;
pub use cg::Cg;
pub use gmres::Gmres;

use crate::sparse::amgcl::parallel::Parallelism;
use crate::sparse::amgcl::vector::{check_finite_with, norm_with};
use crate::sparse::amgcl::{CsrMatrix, Error, Result};

/// Common stopping parameters for iterative solvers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverParams {
    /// Maximum number of Krylov iterations.
    pub max_iterations: usize,
    /// Relative residual target: `‖b-Ax‖₂ ≤ tol * ‖b‖₂`.
    pub tolerance: f64,
    /// Absolute target; the solver stops at the larger of this and the relative target.
    pub absolute_tolerance: f64,
    /// Shared-memory execution for matrix and vector kernels.
    pub parallelism: Parallelism,
}

impl Default for SolverParams {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-8,
            absolute_tolerance: 0.0,
            parallelism: Parallelism::default(),
        }
    }
}

/// Iteration count and achieved true/recursive relative residual.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveReport {
    /// Number of completed Krylov iterations.
    pub iterations: usize,
    /// Achieved residual norm divided by right-hand-side norm.
    pub relative_residual: f64,
    /// Whether the configured absolute or relative target was reached.
    pub converged: bool,
}

pub(crate) struct Problem {
    pub norm_rhs: f64,
    pub target: f64,
}

pub(crate) fn prepare(a: &CsrMatrix, rhs: &[f64], x: &[f64], params: SolverParams) -> Result<Problem> {
    if !a.is_square() {
        return Err(Error::InvalidMatrix("Krylov solver requires a square matrix".into()));
    }
    let n = a.nrows();
    if rhs.len() != n || x.len() != n {
        return Err(Error::DimensionMismatch {
            expected: n,
            actual: if rhs.len() != n { rhs.len() } else { x.len() },
            context: "Krylov solve",
        });
    }
    params.parallelism.validate()?;
    if !params.tolerance.is_finite() || params.tolerance < 0.0 || !params.absolute_tolerance.is_finite() || params.absolute_tolerance < 0.0 {
        return Err(Error::InvalidParameter("solver tolerances must be finite and nonnegative".into()));
    }
    check_finite_with(rhs, params.parallelism)?;
    check_finite_with(x, params.parallelism)?;
    let norm_rhs = norm_with(rhs, params.parallelism);
    if !norm_rhs.is_finite() {
        return Err(Error::NonFinite);
    }
    Ok(Problem {
        norm_rhs,
        target: (params.tolerance * norm_rhs).max(params.absolute_tolerance),
    })
}

pub(crate) fn report(iterations: usize, residual: f64, norm_rhs: f64, target: f64) -> SolveReport {
    SolveReport {
        iterations,
        relative_residual: residual / norm_rhs,
        converged: residual <= target,
    }
}
