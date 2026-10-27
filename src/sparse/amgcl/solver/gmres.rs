use super::{Problem, SolveReport, SolverParams, prepare, report};
use crate::sparse::amgcl::preconditioner::Preconditioner;
use crate::sparse::amgcl::vector::{axpy_with, check_finite_with, dot_with, norm_with, set_scaled_with};
use crate::sparse::amgcl::{CsrMatrix, Error, Result};

/// Right-preconditioned restarted GMRES with modified Gram–Schmidt orthogonalization.
#[derive(Clone, Copy, Debug)]
pub struct Gmres {
    /// Iteration and stopping parameters.
    pub params: SolverParams,
    /// Maximum Krylov dimension between restarts.
    pub restart: usize,
}

impl Default for Gmres {
    fn default() -> Self {
        Self {
            params: SolverParams::default(),
            restart: 30,
        }
    }
}

impl Gmres {
    /// Creates restarted GMRES; returns an error when `restart == 0`.
    pub fn new(params: SolverParams, restart: usize) -> Result<Self> {
        if restart == 0 {
            return Err(Error::InvalidParameter("GMRES restart must be positive".into()));
        }
        Ok(Self { params, restart })
    }

    /// Solves `A x = rhs`, using `x` as the initial guess and output.
    pub fn solve<P: Preconditioner + ?Sized>(&self, a: &CsrMatrix, preconditioner: &mut P, rhs: &[f64], x: &mut [f64]) -> Result<SolveReport> {
        if self.restart == 0 {
            return Err(Error::InvalidParameter("GMRES restart must be positive".into()));
        }
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
        let m = self.restart.min(self.params.max_iterations.max(1));
        let mut r = vec![0.0; n];
        a.residual_unchecked_with(rhs, x, &mut r, parallel);
        let mut beta = norm_with(&r, parallel);
        if beta <= target {
            return Ok(report(0, beta, norm_rhs, target));
        }

        let mut total_iterations = 0;
        while total_iterations < self.params.max_iterations {
            let mut basis = Vec::with_capacity(m + 1);
            let mut preconditioned_basis = Vec::with_capacity(m);
            let mut normalized = vec![0.0; n];
            set_scaled_with(&r, 1.0 / beta, &mut normalized, parallel);
            basis.push(normalized);
            let mut h = vec![0.0; (m + 1) * m];
            let mut cosines = vec![0.0; m];
            let mut sines = vec![0.0; m];
            let mut g = vec![0.0; m + 1];
            g[0] = beta;
            let mut used = 0;

            for j in 0..m {
                if total_iterations >= self.params.max_iterations {
                    break;
                }
                let mut z = vec![0.0; n];
                preconditioner.apply(&basis[j], &mut z)?;
                let mut w = vec![0.0; n];
                a.spmv_unchecked_with(&z, &mut w, parallel);
                for i in 0..=j {
                    h[i * m + j] = dot_with(&basis[i], &w, parallel);
                    axpy_with(-h[i * m + j], &basis[i], &mut w, parallel);
                }
                let next_norm = norm_with(&w, parallel);
                h[(j + 1) * m + j] = next_norm;
                if next_norm > 0.0 {
                    let mut normalized = vec![0.0; n];
                    set_scaled_with(&w, 1.0 / next_norm, &mut normalized, parallel);
                    basis.push(normalized);
                }

                for i in 0..j {
                    let upper = h[i * m + j];
                    let lower = h[(i + 1) * m + j];
                    h[i * m + j] = cosines[i] * upper + sines[i] * lower;
                    h[(i + 1) * m + j] = -sines[i] * upper + cosines[i] * lower;
                }
                let upper = h[j * m + j];
                let lower = h[(j + 1) * m + j];
                let denominator = upper.hypot(lower);
                if denominator == 0.0 || !denominator.is_finite() {
                    return Err(Error::SolverBreakdown("singular Hessenberg rotation in GMRES"));
                }
                cosines[j] = upper / denominator;
                sines[j] = lower / denominator;
                h[j * m + j] = denominator;
                h[(j + 1) * m + j] = 0.0;
                g[j + 1] = -sines[j] * g[j];
                g[j] *= cosines[j];

                preconditioned_basis.push(z);
                used = j + 1;
                total_iterations += 1;
                let estimated_residual = g[j + 1].abs();
                if estimated_residual <= target || next_norm == 0.0 {
                    break;
                }
            }

            // Backsolve the least-squares problem in the rotated Hessenberg system.
            let mut coefficients = vec![0.0; used];
            for i in (0..used).rev() {
                let mut value = g[i];
                for j in i + 1..used {
                    value -= h[i * m + j] * coefficients[j];
                }
                let diagonal = h[i * m + i];
                if diagonal == 0.0 || !diagonal.is_finite() {
                    return Err(Error::SolverBreakdown("singular Hessenberg solve in GMRES"));
                }
                coefficients[i] = value / diagonal;
            }
            for i in 0..used {
                axpy_with(coefficients[i], &preconditioned_basis[i], x, parallel);
            }
            check_finite_with(x, parallel)?;

            // Restart from the true residual, not the Givens estimate.
            a.residual_unchecked_with(rhs, x, &mut r, parallel);
            beta = norm_with(&r, parallel);
            if !beta.is_finite() {
                return Err(Error::NonFinite);
            }
            if beta <= target {
                return Ok(report(total_iterations, beta, norm_rhs, target));
            }
            if used == 0 {
                break;
            }
        }
        Ok(report(total_iterations, beta, norm_rhs, target))
    }
}
