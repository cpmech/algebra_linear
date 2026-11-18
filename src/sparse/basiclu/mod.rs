//! Pure-Rust port of BasicLU for general unsymmetric sparse systems with rank-1 updates.
//!
//! This module is a safe Rust port of BasicLU from `ERGO-Code/HiGHS`
//! (`highs/ipm/basiclu/`, MIT License, Lukas Schork & Jacek Gondzio). It is
//! designed for LP simplex bases and interior-point crossover, maintaining a
//! factorization of the form:
//!
//! ```text
//! P_r^(0) A P_c^(0) = L R_1^(-1) ... R_t^(-1) U
//! ```
//!
//! where column and row singletons are peeled directly from CSC before
//! column-oriented Markowitz threshold pivoting, `L` and `LT` are stored in
//! dual CSC/CSR form, `U` and `UT` are stored in dynamic `LuFileVal`
//! pools, solves support both dense and iterative Gilbert–Peierls hyper-sparse
//! FTRAN/BTRAN with automatic `sparse_threshold` switching, and rank-1 column
//! replacements use Forrest–Tomlin and pure-permutation updates.
//!
//! All indices are 0-based `usize` and all scalars are `f64`. No workspace
//! arguments or reallocation retry loops appear in the public API.

#![forbid(unsafe_code)]

mod degree_list;
mod eliminate;
mod error;
mod factors;
mod file;
mod markowitz;
mod params;
mod singletons;
mod solve;
mod update;

#[cfg(test)]
mod tests;

pub use error::{Error, Result};
pub use params::{BasicLuParams, BasicLuStats};

use eliminate::factorize_bump;
use factors::{LuFactors, build_factors};
use markowitz::ActiveBump;
use singletons::peel_singletons;
use solve::{SolveWorkspace, solve_dense, solve_sparse, solve_sparse_transpose, solve_transpose_dense};
use update::apply_column_update;

/// Sparse LU factorization handle with hyper-sparse solves and Forrest–Tomlin rank-1 updates.
#[derive(Clone, Debug)]
pub struct BasicLu {
    params: BasicLuParams,
    factors: Option<LuFactors>,
    stats: Option<BasicLuStats>,
    work: SolveWorkspace,
}

impl Default for BasicLu {
    fn default() -> Self {
        Self::new(BasicLuParams::default())
    }
}

impl BasicLu {
    /// Creates a new solver handle configured with `params`.
    pub fn new(params: BasicLuParams) -> Self {
        Self {
            params,
            factors: None,
            stats: None,
            work: SolveWorkspace::default(),
        }
    }

    /// Factorizes an `n x n` matrix supplied in compressed sparse column (CSC) form.
    ///
    /// Row indices within each column must be strictly increasing, and all values
    /// must be finite. If factorization fails, any prior successful factors and
    /// statistics remain intact.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadShape`], [`Error::InvalidStructure`],
    /// [`Error::DuplicateEntry`], [`Error::UnsortedRows`], or [`Error::NotFinite`]
    /// on malformed CSC inputs, and [`Error::Singular`] when `rank < n` while
    /// [`BasicLuParams::allow_singular`] is `false`.
    pub fn factorize(&mut self, col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> Result<()> {
        let peeling = peel_singletons(col_ptr, row_idx, values, &self.params)?;
        let n = peeling.n;
        let nnz_a = peeling.nnz_a;

        let mut bump = ActiveBump::from_peeling(col_ptr, row_idx, values, &peeling, &self.params);
        let bump_size = bump.n;
        let bump_data = factorize_bump(&mut bump, &self.params);
        let factors = build_factors(peeling, bump_data, &self.params)?;

        let stats = BasicLuStats {
            n,
            rank: factors.rank,
            nnz_a,
            nnz_l: factors.l_row_idx.len(),
            nnz_u: factors.u_col.live_nnz,
            nnz_r: 0,
            bump_size,
            updates: 0,
            forrest_tomlin_updates: 0,
            pivot_error: 0.0,
        };

        self.work.ensure_len(n);
        self.factors = Some(factors);
        self.stats = Some(stats);
        Ok(())
    }

    /// Solves `A x = b` in place using dense FTRAN (`lu_solve_dense.c`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFactorized`], [`Error::BadShape`], or [`Error::NotFinite`].
    pub fn solve(&mut self, b: &mut [f64]) -> Result<()> {
        solve_dense(self.factors.as_ref(), &mut self.work, b)
    }

    /// Solves `A^T x = b` in place using dense BTRAN (`lu_solve_dense.c`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFactorized`], [`Error::BadShape`], or [`Error::NotFinite`].
    pub fn solve_transpose(&mut self, b: &mut [f64]) -> Result<()> {
        solve_transpose_dense(self.factors.as_ref(), &mut self.work, b)
    }

    /// Solves `A x = b` for a sparse right-hand side (`lu_solve_sparse.c`),
    /// switching to dense mode automatically when the reachable set exceeds
    /// [`BasicLuParams::sparse_threshold`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFactorized`], [`Error::BadShape`],
    /// [`Error::InvalidStructure`], or [`Error::NotFinite`].
    pub fn solve_sparse(&mut self, rhs_idx: &[usize], rhs_val: &[f64], out_idx: &mut Vec<usize>, out_val: &mut Vec<f64>) -> Result<()> {
        solve_sparse(self.factors.as_ref(), &mut self.work, &self.params, rhs_idx, rhs_val, out_idx, out_val)
    }

    /// Solves `A^T x = b` for a sparse right-hand side (`lu_solve_sparse.c`),
    /// switching to dense mode automatically when the reachable set exceeds
    /// [`BasicLuParams::sparse_threshold`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFactorized`], [`Error::BadShape`],
    /// [`Error::InvalidStructure`], or [`Error::NotFinite`].
    pub fn solve_sparse_transpose(&mut self, rhs_idx: &[usize], rhs_val: &[f64], out_idx: &mut Vec<usize>, out_val: &mut Vec<f64>) -> Result<()> {
        solve_sparse_transpose(self.factors.as_ref(), &mut self.work, &self.params, rhs_idx, rhs_val, out_idx, out_val)
    }

    /// Replaces column `col` of the active factorization with `(new_row_idx, new_val)`
    /// via Forrest–Tomlin or pure-permutation rank-1 update (`lu_update.c`).
    ///
    /// If the update fails validation or produces a numerically singular pivot,
    /// the prior factors and statistics remain intact.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFactorized`], [`Error::InvalidColumn`],
    /// [`Error::MaxUpdatesReached`], [`Error::BadShape`],
    /// [`Error::InvalidStructure`], [`Error::NotFinite`], or
    /// [`Error::SingularUpdate`].
    pub fn update(&mut self, col: usize, new_row_idx: &[usize], new_val: &[f64]) -> Result<()> {
        apply_column_update(
            self.factors.as_mut(),
            self.stats.as_mut(),
            &mut self.work,
            &self.params,
            col,
            new_row_idx,
            new_val,
        )
    }

    /// Returns statistics of the active factorization, or `None` before [`Self::factorize`].
    #[inline]
    pub fn stats(&self) -> Option<&BasicLuStats> {
        self.stats.as_ref()
    }

    /// Returns the solver tuning parameters.
    #[inline]
    pub fn params(&self) -> &BasicLuParams {
        &self.params
    }
}
