use super::{Problem, SolveReport, SolverParams, prepare, report};
use crate::sparse::amgcl::parallel::for_each_mut;
use crate::sparse::amgcl::preconditioner::Preconditioner;
use crate::sparse::amgcl::vector::{axpy_with, check_finite_with, dot_with, norm_with};
use crate::sparse::amgcl::{CsrMatrix, Error, Result};

/// Right-preconditioned BiCGStab for general square systems.
#[derive(Clone, Copy, Debug, Default)]
pub struct BiCgStab {
    /// Iteration and stopping parameters.
    pub params: SolverParams,
}

impl BiCgStab {
    /// Creates a BiCGStab solver with the supplied stopping parameters.
    pub fn new(params: SolverParams) -> Self {
        Self { params }
    }

    /// Solves `A x = rhs`, using `x` as the initial guess and output.
    pub fn solve<P: Preconditioner + ?Sized>(&self, a: &CsrMatrix, preconditioner: &mut P, rhs: &[f64], x: &mut [f64]) -> Result<SolveReport> {
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
        let mut r = vec![0.0; n];
        let mut p = vec![0.0; n];
        let mut v = vec![0.0; n];
        let mut s = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut phat = vec![0.0; n];
        let mut shat = vec![0.0; n];
        a.residual_unchecked_with(rhs, x, &mut r, parallel);
        let r_hat = r.clone();
        let mut residual = norm_with(&r, parallel);
        if residual <= target {
            return Ok(report(0, residual, norm_rhs, target));
        }

        let mut rho_previous = 1.0;
        let mut alpha = 1.0;
        let mut omega = 1.0;
        for iteration in 0..self.params.max_iterations {
            let rho = dot_with(&r_hat, &r, parallel);
            if !rho.is_finite() || rho == 0.0 {
                return Err(Error::SolverBreakdown("zero rho in BiCGStab"));
            }
            if iteration == 0 {
                p.copy_from_slice(&r);
            } else {
                if omega == 0.0 {
                    return Err(Error::SolverBreakdown("zero omega in BiCGStab"));
                }
                let beta = (rho / rho_previous) * (alpha / omega);
                for_each_mut(&mut p, parallel, |offset, chunk| {
                    for (local, value) in chunk.iter_mut().enumerate() {
                        let i = offset + local;
                        *value = r[i] + beta * (*value - omega * v[i]);
                    }
                });
            }
            preconditioner.apply(&p, &mut phat)?;
            a.spmv_unchecked_with(&phat, &mut v, parallel);
            let denominator = dot_with(&r_hat, &v, parallel);
            if !denominator.is_finite() || denominator == 0.0 {
                return Err(Error::SolverBreakdown("zero alpha denominator in BiCGStab"));
            }
            alpha = rho / denominator;
            axpy_with(alpha, &phat, x, parallel);
            for_each_mut(&mut s, parallel, |offset, chunk| {
                for (local, value) in chunk.iter_mut().enumerate() {
                    let i = offset + local;
                    *value = r[i] - alpha * v[i];
                }
            });
            residual = norm_with(&s, parallel);
            if !residual.is_finite() {
                return Err(Error::NonFinite);
            }
            if residual <= target {
                r.copy_from_slice(&s);
                return Ok(report(iteration + 1, residual, norm_rhs, target));
            }

            preconditioner.apply(&s, &mut shat)?;
            a.spmv_unchecked_with(&shat, &mut t, parallel);
            let tt = dot_with(&t, &t, parallel);
            if !tt.is_finite() || tt == 0.0 {
                return Err(Error::SolverBreakdown("zero omega denominator in BiCGStab"));
            }
            omega = dot_with(&t, &s, parallel) / tt;
            if !omega.is_finite() || omega == 0.0 {
                return Err(Error::SolverBreakdown("zero omega in BiCGStab"));
            }
            axpy_with(omega, &shat, x, parallel);
            for_each_mut(&mut r, parallel, |offset, chunk| {
                for (local, value) in chunk.iter_mut().enumerate() {
                    let i = offset + local;
                    *value = s[i] - omega * t[i];
                }
            });
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
