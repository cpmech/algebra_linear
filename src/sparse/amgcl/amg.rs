use std::sync::Arc;

use crate::sparse::amgcl::coarse::DenseLu;
use crate::sparse::amgcl::coarsening::{Coarsening, transfer};
use crate::sparse::amgcl::matrix::CsrMatrix;
use crate::sparse::amgcl::parallel::Parallelism;
use crate::sparse::amgcl::preconditioner::Preconditioner;
use crate::sparse::amgcl::relaxation::{Relaxation, Smoother};
use crate::sparse::amgcl::vector::check_finite_with;
use crate::sparse::amgcl::{Error, Result};

/// AMG hierarchy and cycle parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmgParams {
    /// Coarsening and interpolation strategy.
    pub coarsening: Coarsening,
    /// Relaxation scheme used before and after coarse-grid correction.
    pub smoother: Smoother,
    /// Directly solve levels with at most this many rows.
    pub coarse_enough: usize,
    /// Use dense partial-pivoted LU on a sufficiently small coarsest level.
    pub direct_coarse: bool,
    /// Maximum number of levels, including the finest and coarsest.
    pub max_levels: usize,
    /// Relaxation sweeps before restriction on each level.
    pub pre_smoothing: usize,
    /// Relaxation sweeps after prolongation on each level.
    pub post_smoothing: usize,
    /// One for a V-cycle, two for a W-cycle, and so on.
    pub cycle_iterations: usize,
    /// Number of multigrid cycles per preconditioner application.
    pub preconditioner_cycles: usize,
    /// Shared-memory execution for setup, cycle, and smoother kernels.
    pub parallelism: Parallelism,
}

impl Default for AmgParams {
    fn default() -> Self {
        Self {
            coarsening: Coarsening::default(),
            smoother: Smoother::GaussSeidel,
            coarse_enough: 64,
            direct_coarse: true,
            max_levels: 32,
            pre_smoothing: 1,
            post_smoothing: 1,
            cycle_iterations: 1,
            preconditioner_cycles: 1,
            parallelism: Parallelism::default(),
        }
    }
}

impl AmgParams {
    fn validate(&self) -> Result<()> {
        self.parallelism.validate()?;
        self.coarsening.validate()?;
        if self.coarse_enough == 0 {
            return Err(Error::InvalidParameter("coarse_enough must be positive".into()));
        }
        if self.max_levels == 0 {
            return Err(Error::InvalidParameter("max_levels must be positive".into()));
        }
        if self.cycle_iterations == 0 {
            return Err(Error::InvalidParameter("cycle_iterations must be positive".into()));
        }
        self.smoother.validate()
    }
}

/// Public information about one AMG level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LevelInfo {
    /// Number of rows on this level.
    pub unknowns: usize,
    /// Number of stored matrix entries on this level.
    pub nonzeros: usize,
    /// Whether this terminal level has a dense LU factorization.
    pub direct_solver: bool,
}

#[derive(Debug)]
struct Level {
    a: Arc<CsrMatrix>,
    p: Option<CsrMatrix>,
    r: Option<CsrMatrix>,
    relaxation: Option<Relaxation>,
    coarse: Option<DenseLu>,
    residual: Vec<f64>,
    coarse_rhs: Vec<f64>,
    coarse_solution: Vec<f64>,
}

impl Level {
    fn new(
        a: CsrMatrix,
        p: Option<CsrMatrix>,
        r: Option<CsrMatrix>,
        coarse: Option<DenseLu>,
        smoother: Smoother,
        parallel: Parallelism,
    ) -> Result<Self> {
        let n = a.nrows();
        let needs_smoother = coarse.is_none() || p.is_some();
        let relaxation = if needs_smoother {
            Some(Relaxation::new_with(&a, smoother, parallel)?)
        } else {
            None
        };
        let coarse_size = p.as_ref().map_or(0, CsrMatrix::ncols);
        Ok(Self {
            a: Arc::new(a),
            p,
            r,
            relaxation,
            coarse,
            residual: vec![0.0; n],
            coarse_rhs: vec![0.0; coarse_size],
            coarse_solution: vec![0.0; coarse_size],
        })
    }

    /// A single level with no relaxation and no coarse factorization.
    fn matrix_only(a: CsrMatrix) -> Self {
        Self {
            a: Arc::new(a),
            p: None,
            r: None,
            relaxation: None,
            coarse: None,
            residual: Vec::new(),
            coarse_rhs: Vec::new(),
            coarse_solution: Vec::new(),
        }
    }

    fn relax(&mut self, forward: bool, rhs: &[f64], x: &mut [f64], parallel: Parallelism) {
        self.relaxation
            .as_mut()
            .expect("relaxed level has relaxation data")
            .apply_with(&self.a, rhs, x, &mut self.residual, forward, parallel);
    }
}

/// Algebraic multigrid hierarchy usable directly or as a preconditioner.
///
/// Build the state with [`amg_setup`] and solve the stored finest-level system
/// with [`krylov_solve`](crate::sparse::amgcl::krylov_solve). The type owns all
/// matrices and reusable work vectors, so [`Amg::apply`] and Krylov solves take
/// `&mut self`. Serial cycles perform no allocation; parallel execution only
/// adds scoped-worker bookkeeping.
#[derive(Debug)]
pub struct Amg {
    /// `None` is matrix-only state: the finest matrix with no relaxation.
    params: Option<AmgParams>,
    levels: Vec<Level>,
}

/// Builds AMG state from raw compressed sparse row (CSR) arrays.
///
/// The matrix is `n` by `n` with `n = row_ptr.len() - 1`. The arrays are
/// validated as they are copied into the private matrix: the row pointers must
/// be monotone and in range, the column indices in `0..n`, and every value
/// finite.
///
/// `params = None` builds matrix-only state: one level holding the finest
/// matrix, no relaxation and no coarse factorization. [`Amg::apply`] on that
/// state copies the right-hand side to the solution, which is the identity
/// preconditioner, and [`krylov_solve`](crate::sparse::amgcl::krylov_solve)
/// works with it.
///
/// # Errors
///
/// [`Error::InvalidMatrix`] if the CSR arrays are malformed or the matrix is
/// empty, and the parameter errors of the AMG hierarchy otherwise.
pub fn amg_setup(row_ptr: &[usize], col_idx: &[usize], values: &[f64], params: Option<AmgParams>) -> Result<Amg> {
    let n = row_ptr
        .len()
        .checked_sub(1)
        .ok_or_else(|| Error::InvalidMatrix("row_ptr must hold at least one entry".into()))?;
    let matrix = CsrMatrix::from_parts(n, n, row_ptr.to_vec(), col_idx.to_vec(), values.to_vec())?;
    if matrix.nrows() == 0 {
        return Err(Error::InvalidMatrix("AMG matrix is empty".into()));
    }
    match params {
        Some(params) => Amg::build(matrix, params),
        None => Ok(Amg::matrix_only(matrix)),
    }
}

impl Amg {
    /// Builds the selected AMG hierarchy and all Galerkin coarse operators.
    fn build(matrix: CsrMatrix, params: AmgParams) -> Result<Self> {
        params.validate()?;

        let mut levels = Vec::new();
        let mut current = matrix;
        let mut coarsening = params.coarsening;
        loop {
            let n = current.nrows();
            let may_coarsen = n > params.coarse_enough && levels.len() + 1 < params.max_levels;
            if may_coarsen {
                if let Some(transfer) = transfer(&current, coarsening)? {
                    let p = transfer.prolongation;
                    let r = p.transpose();
                    let ap = current.matmul_with(&p, params.parallelism)?;
                    let mut coarse_matrix = r.matmul_with(&ap, params.parallelism)?;
                    coarse_matrix.scale(transfer.coarse_scale);
                    if coarse_matrix.values().iter().any(|v| !v.is_finite()) {
                        return Err(Error::NonFinite);
                    }
                    levels.push(Level::new(current, Some(p), Some(r), None, params.smoother, params.parallelism)?);
                    // AMGCL weakens the strength threshold on successive
                    // smoothed-aggregation levels to control operator growth.
                    if let Coarsening::SmoothedAggregation(mut aggregation) = coarsening {
                        aggregation.aggregation.eps_strong *= 0.5;
                        coarsening = Coarsening::SmoothedAggregation(aggregation);
                    }
                    current = coarse_matrix;
                    continue;
                }
            }

            let use_direct = params.direct_coarse && n <= params.coarse_enough;
            let coarse = if use_direct { Some(DenseLu::new(&current)?) } else { None };
            levels.push(Level::new(current, None, None, coarse, params.smoother, params.parallelism)?);
            break;
        }
        Ok(Self {
            params: Some(params),
            levels,
        })
    }

    /// Matrix-only state: the finest matrix and nothing else.
    fn matrix_only(matrix: CsrMatrix) -> Self {
        Self {
            params: None,
            levels: vec![Level::matrix_only(matrix)],
        }
    }

    /// Returns a handle to the level-0 matrix.
    pub(crate) fn matrix_handle(&self) -> Arc<CsrMatrix> {
        Arc::clone(&self.levels[0].a)
    }

    /// Applies the configured number of cycles after clearing `x`.
    ///
    /// Matrix-only state copies `rhs` to `x`: it is the identity preconditioner.
    pub fn apply(&mut self, rhs: &[f64], x: &mut [f64]) -> Result<()> {
        self.check_vectors(rhs, x)?;
        let Some(params) = self.params else {
            x.copy_from_slice(rhs);
            return Ok(());
        };
        if params.preconditioner_cycles == 0 {
            x.copy_from_slice(rhs);
            return Ok(());
        }
        x.fill(0.0);
        for _ in 0..params.preconditioner_cycles {
            Self::cycle_impl(&mut self.levels, params, rhs, x);
        }
        check_finite_with(x, params.parallelism)
    }

    /// Adds one AMG cycle to the existing approximation `x`.
    ///
    /// Matrix-only state has no cycle to add, so `x` is left unchanged.
    pub fn cycle(&mut self, rhs: &[f64], x: &mut [f64]) -> Result<()> {
        self.check_vectors(rhs, x)?;
        let Some(params) = self.params else {
            return Ok(());
        };
        Self::cycle_impl(&mut self.levels, params, rhs, x);
        check_finite_with(x, params.parallelism)
    }

    fn cycle_impl(levels: &mut [Level], params: AmgParams, rhs: &[f64], x: &mut [f64]) {
        let (level, rest) = levels.split_first_mut().expect("nonempty AMG hierarchy");
        if rest.is_empty() {
            if let Some(coarse) = &level.coarse {
                coarse.solve(rhs, x);
            } else {
                for _ in 0..params.pre_smoothing {
                    level.relax(true, rhs, x, params.parallelism);
                }
                for _ in 0..params.post_smoothing {
                    level.relax(false, rhs, x, params.parallelism);
                }
            }
            return;
        }

        for _ in 0..params.cycle_iterations {
            for _ in 0..params.pre_smoothing {
                level.relax(true, rhs, x, params.parallelism);
            }
            level.a.residual_unchecked_with(rhs, x, &mut level.residual, params.parallelism);
            level.r.as_ref().expect("non-coarse level has restriction").spmv_unchecked_with(
                &level.residual,
                &mut level.coarse_rhs,
                params.parallelism,
            );
            level.coarse_solution.fill(0.0);
            Self::cycle_impl(rest, params, &level.coarse_rhs, &mut level.coarse_solution);
            level
                .p
                .as_ref()
                .expect("non-coarse level has prolongation")
                .spmv_add_unchecked_with(&level.coarse_solution, x, params.parallelism);
            for _ in 0..params.post_smoothing {
                level.relax(false, rhs, x, params.parallelism);
            }
        }
    }

    /// Iterates over level dimensions from finest to coarsest.
    pub fn levels(&self) -> impl ExactSizeIterator<Item = LevelInfo> + '_ {
        self.levels.iter().map(|level| LevelInfo {
            unknowns: level.a.nrows(),
            nonzeros: level.a.nnz(),
            direct_solver: level.coarse.is_some(),
        })
    }

    /// Sum of level sizes divided by the finest-level size.
    pub fn grid_complexity(&self) -> f64 {
        self.levels.iter().map(|l| l.a.nrows()).sum::<usize>() as f64 / self.levels[0].a.nrows() as f64
    }

    /// Sum of level nonzeros divided by finest-level nonzeros.
    pub fn operator_complexity(&self) -> f64 {
        self.levels.iter().map(|l| l.a.nnz()).sum::<usize>() as f64 / self.levels[0].a.nnz().max(1) as f64
    }

    fn check_vectors(&self, rhs: &[f64], x: &[f64]) -> Result<()> {
        let n = self.levels[0].a.nrows();
        if rhs.len() != n || x.len() != n {
            return Err(Error::DimensionMismatch {
                expected: n,
                actual: if rhs.len() != n { rhs.len() } else { x.len() },
                context: "AMG preconditioner application",
            });
        }
        Ok(())
    }
}

impl Preconditioner for Amg {
    fn apply(&mut self, rhs: &[f64], x: &mut [f64]) -> Result<()> {
        Amg::apply(self, rhs, x)
    }
}
