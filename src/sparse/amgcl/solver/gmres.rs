use super::{Problem, SolveReport, SolverParams, prepare, report};
use crate::sparse::amgcl::matrix::CsrMatrix;
use crate::sparse::amgcl::memory::{MemoryBudget, try_vec_filled, try_vec_with_capacity, vec_bytes};
use crate::sparse::amgcl::preconditioner::Preconditioner;
use crate::sparse::amgcl::vector::{axpy_with, check_finite_with, dot_with, norm_with, set_scaled_with};
use crate::sparse::amgcl::{Error, Result};

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
        let krylov_bytes = m.saturating_add(3).saturating_mul(vec_bytes::<f64>(n));
        let hessenberg_scalars = m
            .saturating_add(1)
            .saturating_mul(m)
            .saturating_add(m.saturating_mul(4))
            .saturating_add(1);
        let hessenberg_bytes = vec_bytes::<f64>(hessenberg_scalars).saturating_add(vec_bytes::<Vec<f64>>(m));
        budget.check_additional(krylov_bytes.saturating_add(hessenberg_bytes), None)?;

        let mut r = try_vec_filled(0.0, n, budget, None)?;
        a.residual_unchecked_with(rhs, x, &mut r, parallel);
        let mut beta = norm_with(&r, parallel);
        if beta <= target {
            return Ok(report(0, beta, norm_rhs, target));
        }

        let mut z = try_vec_filled(0.0, n, budget, None)?;
        let mut w = try_vec_filled(0.0, n, budget, None)?;
        let mut basis: Vec<Vec<f64>> = try_vec_with_capacity(m, budget, None)?;
        let h_len = m
            .checked_add(1)
            .and_then(|rows| rows.checked_mul(m))
            .ok_or_else(|| budget.alloc_error(usize::MAX, None))?;
        let mut h = try_vec_filled(0.0, h_len, budget, None)?;
        let mut cosines = try_vec_filled(0.0, m, budget, None)?;
        let mut sines = try_vec_filled(0.0, m, budget, None)?;
        let mut g = try_vec_filled(0.0, m.saturating_add(1), budget, None)?;
        let mut coefficients = try_vec_filled(0.0, m, budget, None)?;

        let mut total_iterations = 0;
        while total_iterations < self.params.max_iterations {
            if basis.is_empty() {
                basis.push(try_vec_filled(0.0, n, budget, None)?);
            }
            set_scaled_with(&r, 1.0 / beta, &mut basis[0], parallel);
            h.fill(0.0);
            cosines.fill(0.0);
            sines.fill(0.0);
            g.fill(0.0);
            coefficients.fill(0.0);
            g[0] = beta;
            let mut used = 0;

            for j in 0..m {
                preconditioner.apply(&basis[j], &mut z)?;
                a.spmv_unchecked_with(&z, &mut w, parallel);
                for i in 0..=j {
                    let hij = dot_with(&basis[i], &w, parallel);
                    h[i * m + j] = hij;
                    axpy_with(-hij, &basis[i], &mut w, parallel);
                }
                let next_norm = norm_with(&w, parallel);
                h[(j + 1) * m + j] = next_norm;

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

                used = j + 1;
                total_iterations += 1;
                let estimated_residual = g[j + 1].abs();
                if j + 1 == m || estimated_residual <= target || next_norm == 0.0 || total_iterations >= self.params.max_iterations {
                    break;
                }
                if basis.len() <= j + 1 {
                    basis.push(try_vec_filled(0.0, n, budget, None)?);
                }
                set_scaled_with(&w, 1.0 / next_norm, &mut basis[j + 1], parallel);
            }

            // Backsolve the least-squares problem in the rotated Hessenberg system.
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
            w.fill(0.0);
            for i in 0..used {
                axpy_with(coefficients[i], &basis[i], &mut w, parallel);
            }
            preconditioner.apply(&w, &mut z)?;
            axpy_with(1.0, &z, x, parallel);
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
        }
        Ok(report(total_iterations, beta, norm_rhs, target))
    }
}
