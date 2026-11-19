use super::{Problem, SolveReport, SolverParams, prepare, report};
use crate::sparse::amgcl::matrix::CsrMatrix;
use crate::sparse::amgcl::memory::{MemoryBudget, try_vec_filled, vec_bytes};
use crate::sparse::amgcl::parallel::for_each_mut;
use crate::sparse::amgcl::preconditioner::Preconditioner;
use crate::sparse::amgcl::vector::{axpy_with, check_finite_with, dot_with, norm_with};
use crate::sparse::amgcl::{Error, Result};

/// Preconditioned conjugate gradients for symmetric positive-definite systems.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cg {
    /// Iteration and stopping parameters.
    pub params: SolverParams,
}

impl Cg {
    /// Creates a CG solver with the supplied stopping parameters.
    pub fn new(params: SolverParams) -> Self {
        Self { params }
    }

    /// Solves `A x = rhs`, using `x` as the initial guess and output.
    #[cfg(test)]
    pub fn solve<P: Preconditioner + ?Sized>(&self, a: &CsrMatrix, preconditioner: &mut P, rhs: &[f64], x: &mut [f64]) -> Result<SolveReport> {
        let budget = MemoryBudget::new(self.params.max_memory_mb);
        self.solve_with_budget(a, preconditioner, rhs, x, &budget)
    }

    /// Solves `A x = rhs` while enforcing `budget`.
    pub(crate) fn solve_with_budget<P: Preconditioner + ?Sized>(
        &self,
        a: &CsrMatrix,
        preconditioner: &mut P,
        rhs: &[f64],
        x: &mut [f64],
        budget: &MemoryBudget,
    ) -> Result<SolveReport> {
        let Problem { norm_rhs, target } = prepare(a, rhs, x, self.params)?;
        let parallel = self.params.parallelism;
        if norm_rhs == 0.0 {
            x.fill(0.0);
            return Ok(SolveReport {
                iterations: 0,
                relative_residual: 0.0,
                converged: true,
            });
        }
        let n = a.nrows();
        budget.check_additional(vec_bytes::<f64>(n).saturating_mul(4), None)?;
        let mut r = try_vec_filled(0.0, n, budget, None)?;
        let mut z = try_vec_filled(0.0, n, budget, None)?;
        let mut p = try_vec_filled(0.0, n, budget, None)?;
        let mut q = try_vec_filled(0.0, n, budget, None)?;
        a.residual_unchecked_with(rhs, x, &mut r, parallel);
        let mut residual = norm_with(&r, parallel);
        if residual <= target {
            return Ok(report(0, residual, norm_rhs, target));
        }

        let mut rho_previous = 0.0;
        for iteration in 0..self.params.max_iterations {
            preconditioner.apply(&r, &mut z)?;
            let rho = dot_with(&r, &z, parallel);
            if !rho.is_finite() || rho <= 0.0 {
                return Err(Error::SolverBreakdown("CG preconditioned residual is not positive"));
            }
            if iteration == 0 {
                p.copy_from_slice(&z);
            } else {
                let beta = rho / rho_previous;
                for_each_mut(&mut p, parallel, |offset, chunk| {
                    for (local, value) in chunk.iter_mut().enumerate() {
                        let i = offset + local;
                        *value = z[i] + beta * *value;
                    }
                });
            }
            a.spmv_unchecked_with(&p, &mut q, parallel);
            let curvature = dot_with(&p, &q, parallel);
            if !curvature.is_finite() || curvature <= 0.0 {
                return Err(Error::SolverBreakdown(
                    "CG search direction has nonpositive curvature; matrix may not be SPD",
                ));
            }
            let alpha = rho / curvature;
            axpy_with(alpha, &p, x, parallel);
            axpy_with(-alpha, &q, &mut r, parallel);
            residual = norm_with(&r, parallel);
            if !residual.is_finite() {
                return Err(Error::NonFinite);
            }
            if residual <= target {
                return Ok(report(iteration + 1, residual, norm_rhs, target));
            }
            rho_previous = rho;
        }
        check_finite_with(x, parallel)?;
        Ok(report(self.params.max_iterations, residual, norm_rhs, target))
    }
}
