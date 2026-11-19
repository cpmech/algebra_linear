use std::sync::Arc;

use crate::sparse::amgcl::coarse::DenseLu;
use crate::sparse::amgcl::coarsening::{Coarsening, transfer_with_budget};
use crate::sparse::amgcl::matrix::CsrMatrix;
use crate::sparse::amgcl::memory::{MemoryBudget, csr_bytes, try_clone_slice, try_vec_filled, vec_bytes};
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
    /// Use dense partial-pivoted LU on a sufficiently small coarsest level
    ///
    /// When the coarsest operator is singular, the factorization is retried with a small diagonal
    /// shift; if it still does not factor, the level relaxes instead of solving.
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
    /// Maximum working memory budget in MiB (`0` disables the limit).
    pub max_memory_mb: usize,
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
            max_memory_mb: 0,
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
    #[allow(clippy::too_many_arguments)]
    fn new_with_budget(
        a: Arc<CsrMatrix>,
        p: Option<CsrMatrix>,
        r: Option<CsrMatrix>,
        coarse: Option<DenseLu>,
        smoother: Smoother,
        parallel: Parallelism,
        budget: &MemoryBudget,
        level_idx: usize,
    ) -> Result<Self> {
        let n = a.nrows();
        let needs_smoother = coarse.is_none() || p.is_some();
        let relaxation = if needs_smoother {
            let rel = Relaxation::new_with_budget(a.as_ref(), smoother, parallel, budget, Some(level_idx))?;
            budget.record_retained(rel.memory_bytes());
            Some(rel)
        } else {
            None
        };
        let coarse_size = p.as_ref().map_or(0, CsrMatrix::ncols);
        let work_bytes = vec_bytes::<f64>(n).saturating_add(vec_bytes::<f64>(coarse_size).saturating_mul(2));
        budget.check_additional(work_bytes, Some(level_idx))?;
        let residual = try_vec_filled(0.0, n, budget, Some(level_idx))?;
        let coarse_rhs = try_vec_filled(0.0, coarse_size, budget, Some(level_idx))?;
        let coarse_solution = try_vec_filled(0.0, coarse_size, budget, Some(level_idx))?;
        budget.record_retained(vec_bytes::<f64>(
            residual.len().saturating_add(coarse_rhs.len()).saturating_add(coarse_solution.len()),
        ));
        Ok(Self {
            a,
            p,
            r,
            relaxation,
            coarse,
            residual,
            coarse_rhs,
            coarse_solution,
        })
    }

    /// A single level with no relaxation and no coarse factorization.
    fn matrix_only(a: Arc<CsrMatrix>) -> Self {
        Self {
            a,
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
/// Build the state with [`amg_setup`] or [`amg_setup_owned`] and solve the stored
/// finest-level system with [`krylov_solve`](crate::sparse::amgcl::krylov_solve).
/// The type owns all matrices and reusable work vectors, so [`Amg::apply`] and
/// Krylov solves take `&mut self`. Serial cycles perform no allocation; parallel
/// execution only adds scoped-worker bookkeeping.
#[derive(Debug)]
pub struct Amg {
    /// `None` is matrix-only state: the finest matrix with no relaxation.
    params: Option<AmgParams>,
    /// The hierarchy, from the finest level to the coarsest.
    levels: Vec<Level>,
    /// The matrix the caller handed in
    ///
    /// The levels may carry a copy with a null diagonal filled in, because a smoother cannot divide
    /// by zero; the system that is solved and whose residual the Krylov method measures is this one.
    original: Arc<CsrMatrix>,
    /// Retained heap memory footprint in bytes.
    retained_bytes: usize,
}

/// Builds AMG state from borrowed compressed sparse row (CSR) arrays.
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
/// empty, [`Error::OutOfMemory`] if `max_memory_mb` is exceeded or allocation
/// fails, and the parameter errors of the AMG hierarchy otherwise.
pub fn amg_setup(row_ptr: &[usize], col_idx: &[usize], values: &[f64], params: Option<AmgParams>) -> Result<Amg> {
    let n = validate_csr_dims(row_ptr, params)?;
    let budget = MemoryBudget::new(params.map_or(0, |p| p.max_memory_mb));
    budget.check_additional(csr_bytes(n, values.len()), None)?;
    let row_ptr_owned = try_clone_slice(row_ptr, &budget, None)?;
    budget.record_retained(vec_bytes::<usize>(row_ptr_owned.len()));
    let col_idx_owned = try_clone_slice(col_idx, &budget, None)?;
    budget.record_retained(vec_bytes::<usize>(col_idx_owned.len()));
    let values_owned = try_clone_slice(values, &budget, None)?;
    budget.record_retained(vec_bytes::<f64>(values_owned.len()));
    let matrix = CsrMatrix::from_parts(n, n, row_ptr_owned, col_idx_owned, values_owned)?;
    amg_setup_with_budget(matrix, params, budget)
}

/// Builds AMG state by taking ownership of compressed sparse row (CSR) vectors.
///
/// Avoids copying the finest-level CSR buffers compared to [`amg_setup`], while
/// enforcing the same validation and [`AmgParams::max_memory_mb`] budget.
///
/// # Errors
///
/// [`Error::InvalidMatrix`] if the CSR arrays are malformed or the matrix is
/// empty, [`Error::OutOfMemory`] if `max_memory_mb` is exceeded or allocation
/// fails, and the parameter errors of the AMG hierarchy otherwise.
pub fn amg_setup_owned(row_ptr: Vec<usize>, col_idx: Vec<usize>, values: Vec<f64>, params: Option<AmgParams>) -> Result<Amg> {
    let n = validate_csr_dims(&row_ptr, params)?;
    let matrix = CsrMatrix::from_parts(n, n, row_ptr, col_idx, values)?;
    let budget = MemoryBudget::new(params.map_or(0, |p| p.max_memory_mb));
    let matrix_bytes = matrix.memory_bytes();
    budget.check_additional(matrix_bytes, None)?;
    budget.record_retained(matrix_bytes);
    amg_setup_with_budget(matrix, params, budget)
}

fn validate_csr_dims(row_ptr: &[usize], params: Option<AmgParams>) -> Result<usize> {
    let n = row_ptr
        .len()
        .checked_sub(1)
        .ok_or_else(|| Error::InvalidMatrix("row_ptr must hold at least one entry".into()))?;
    if n == 0 {
        return Err(Error::InvalidMatrix("AMG matrix is empty".into()));
    }
    if let Some(p) = params {
        p.validate()?;
    }
    Ok(n)
}

fn amg_setup_with_budget(matrix: CsrMatrix, params: Option<AmgParams>, budget: MemoryBudget) -> Result<Amg> {
    match params {
        Some(params) => Amg::build(matrix, params, budget),
        None => Ok(Amg::matrix_only(matrix, budget)),
    }
}

impl Amg {
    /// Builds the selected AMG hierarchy and all Galerkin coarse operators.
    fn build(matrix: CsrMatrix, params: AmgParams, budget: MemoryBudget) -> Result<Self> {
        let mut levels = Vec::new();
        // The caller's matrix is the system the Krylov method solves and measures; the hierarchy below
        // may carry a copy with a null diagonal filled in, because a smoother cannot divide by zero.
        let original = Arc::new(matrix);
        let mut current = match Self::ensure_nonzero_diagonal(&original, &budget)? {
            Some(filled) => {
                budget.record_retained(filled.memory_bytes());
                Arc::new(filled)
            }
            None => Arc::clone(&original),
        };
        let mut coarsening = params.coarsening;
        loop {
            let level_idx = levels.len();
            let n = current.nrows();
            let may_coarsen = n > params.coarse_enough && level_idx + 1 < params.max_levels;
            if may_coarsen {
                if let Some(transfer) = transfer_with_budget(&current, coarsening, &budget, level_idx)? {
                    let p = transfer.prolongation;
                    budget.record_retained(p.memory_bytes());
                    let r = p.transpose_with_budget(&budget, Some(level_idx))?;
                    budget.record_retained(r.memory_bytes());
                    let ap = current.matmul_with_budget(&p, params.parallelism, &budget, Some(level_idx))?;
                    let ap_bytes = ap.memory_bytes();
                    budget.record_retained(ap_bytes);
                    let mut coarse_matrix = r.matmul_with_budget(&ap, params.parallelism, &budget, Some(level_idx))?;
                    drop(ap);
                    budget.release_retained(ap_bytes);
                    coarse_matrix.scale(transfer.coarse_scale);
                    if coarse_matrix.values().iter().any(|v| !v.is_finite()) {
                        return Err(Error::NonFinite);
                    }
                    budget.record_retained(coarse_matrix.memory_bytes());
                    levels.push(Level::new_with_budget(
                        Arc::clone(&current),
                        Some(p),
                        Some(r),
                        None,
                        params.smoother,
                        params.parallelism,
                        &budget,
                        level_idx,
                    )?);
                    // AMGCL weakens the strength threshold on successive
                    // smoothed-aggregation levels to control operator growth.
                    if let Coarsening::SmoothedAggregation(mut aggregation) = coarsening {
                        aggregation.aggregation.eps_strong *= 0.5;
                        coarsening = Coarsening::SmoothedAggregation(aggregation);
                    }
                    current = Arc::new(coarse_matrix);
                    continue;
                }
            }

            let use_direct = params.direct_coarse && n <= params.coarse_enough;
            // A coarsest operator that cannot be factorized is not fatal: the level then relaxes
            // instead of solving, which is the standard alternative in AMG and keeps a hierarchy
            // whose Galerkin product came out semi-definite usable. `LevelInfo::direct_solver`
            // reports which one a level got.
            let coarse = if use_direct {
                match DenseLu::new_with_budget(&current, &budget, Some(level_idx)) {
                    Ok(lu) => {
                        budget.record_retained(lu.memory_bytes());
                        Some(lu)
                    }
                    Err(err @ Error::OutOfMemory { .. }) => return Err(err),
                    Err(_) => None,
                }
            } else {
                None
            };
            levels.push(Level::new_with_budget(
                current,
                None,
                None,
                coarse,
                params.smoother,
                params.parallelism,
                &budget,
                level_idx,
            )?);
            break;
        }
        Ok(Self {
            params: Some(params),
            levels,
            original,
            retained_bytes: budget.retained_bytes(),
        })
    }

    /// Returns the matrix with a unit diagonal wherever the diagonal is exactly zero
    ///
    /// The hierarchy never solves the system itself -- the Krylov method does, on the caller's matrix --
    /// so replacing a zero diagonal by one only changes the preconditioner, and it turns a hard setup
    /// failure into an unknown the smoother simply cannot help with. Matrices that already have a full
    /// diagonal are returned untouched.
    fn ensure_nonzero_diagonal(a: &CsrMatrix, budget: &MemoryBudget) -> Result<Option<CsrMatrix>> {
        let n = a.nrows();
        let diagonal = a.diagonal_with_budget(budget, None)?;
        let _diag_guard = budget.guard(vec_bytes::<f64>(n));
        let missing_count = diagonal.iter().filter(|&&d| d == 0.0).count();
        if missing_count == 0 {
            return Ok(None);
        }
        let nnz_cap = a.nnz().saturating_add(missing_count);
        let row_cap = a.max_row_nnz().saturating_add(1);
        CsrMatrix::from_row_producer_with_budget(n, n, nnz_cap, row_cap, budget, None, |row, row_buf| {
            let entries = a.row(row);
            for (&column, &value) in entries.columns.iter().zip(entries.values) {
                if value != 0.0 {
                    row_buf.push((column, value));
                }
            }
            if diagonal[row] == 0.0 {
                row_buf.push((row, 1.0));
            }
            Ok(())
        })
        .map(Some)
    }

    /// Matrix-only state: the finest matrix and nothing else.
    fn matrix_only(matrix: CsrMatrix, budget: MemoryBudget) -> Self {
        let original = Arc::new(matrix);
        Self {
            params: None,
            levels: vec![Level::matrix_only(Arc::clone(&original))],
            original,
            retained_bytes: budget.retained_bytes(),
        }
    }

    /// Retained heap memory footprint in bytes across the stored matrix and AMG hierarchy.
    pub fn memory_bytes(&self) -> usize {
        self.retained_bytes
    }

    #[cfg(test)]
    pub(crate) fn computed_memory_bytes(&self) -> usize {
        let mut total = self.original.memory_bytes();
        for level in &self.levels {
            if !Arc::ptr_eq(&level.a, &self.original) {
                total = total.saturating_add(level.a.memory_bytes());
            }
            total = total.saturating_add(level.p.as_ref().map_or(0, CsrMatrix::memory_bytes));
            total = total.saturating_add(level.r.as_ref().map_or(0, CsrMatrix::memory_bytes));
            total = total.saturating_add(level.relaxation.as_ref().map_or(0, Relaxation::memory_bytes));
            total = total.saturating_add(level.coarse.as_ref().map_or(0, DenseLu::memory_bytes));
            total = total.saturating_add(vec_bytes::<f64>(
                level.residual.len() + level.coarse_rhs.len() + level.coarse_solution.len(),
            ));
        }
        total
    }

    /// Configured memory budget in MiB (`0` when unlimited or matrix-only).
    pub(crate) fn max_memory_mb(&self) -> usize {
        self.params.map_or(0, |params| params.max_memory_mb)
    }

    /// Returns a handle to the caller's matrix, which is the one the solve measures
    pub(crate) fn matrix_handle(&self) -> Arc<CsrMatrix> {
        Arc::clone(&self.original)
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
