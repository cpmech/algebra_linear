//! Tuning parameters and factorization statistics for [`super::BasicLu`].
//!
//! Mirrors the control and diagnostic parameters from `ERGO-Code/HiGHS`
//! (`highs/ipm/basiclu/basiclu.h` and `lu_initialize.c`, MIT).

/// Tuning parameters for [`super::BasicLu`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BasicLuParams {
    /// Absolute minimum pivot magnitude (`BASICLU_ABS_PIVOT_TOLERANCE`). Default: `1e-14`.
    pub abs_pivot_tol: f64,
    /// Column-wise threshold partial pivoting factor in `(0.0, 1.0]` (`BASICLU_REL_PIVOT_TOLERANCE`). Default: `0.1`.
    pub rel_pivot_tol: f64,
    /// Drop tolerance for near-zero or canceled entries (`BASICLU_DROP_TOLERANCE`). Default: `1e-20`.
    pub drop_tol: f64,
    /// Fraction of `n` above which hyper-sparse solves switch to dense loops (`0.0..=1.0`). Default: `0.05`.
    pub sparse_threshold: f64,
    /// Maximum number of rows inspected during Markowitz bucket search once a valid pivot is found. Default: `4`.
    pub search_rows: usize,
    /// Maximum number of column-replacement updates allowed before refactorization is required. Default: `100`.
    pub max_updates: usize,
    /// Dead-space fraction triggering `LuFile` compaction during relocation. Default: `0.5`.
    pub compress_threshold: f64,
    /// If `true`, rank-deficient matrices (`rank < n`) are completed with unit columns instead of failing with [`Error::Singular`](super::Error::Singular). Default: `false`.
    pub allow_singular: bool,
}

impl Default for BasicLuParams {
    fn default() -> Self {
        Self {
            abs_pivot_tol: 1e-14,
            rel_pivot_tol: 0.1,
            drop_tol: 1e-20,
            sparse_threshold: 0.05,
            search_rows: 4,
            max_updates: 100,
            compress_threshold: 0.5,
            allow_singular: false,
        }
    }
}

/// Structural, rank, and update statistics of the active factorization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BasicLuStats {
    /// Order of the square matrix.
    pub n: usize,
    /// Numerical rank achieved during factorization.
    pub rank: usize,
    /// Number of stored nonzero entries in the input matrix `A`.
    pub nnz_a: usize,
    /// Number of strictly subdiagonal stored entries in `L`.
    pub nnz_l: usize,
    /// Number of strictly off-diagonal stored entries in `U`.
    pub nnz_u: usize,
    /// Total number of stored weights across all Forrest–Tomlin row etas `R_1, ..., R_t`.
    pub nnz_r: usize,
    /// Order of the active bump remaining after singleton peeling.
    pub bump_size: usize,
    /// Total number of column-replacement updates applied since factorization.
    pub updates: usize,
    /// Number of updates that appended a Forrest–Tomlin row eta.
    pub forrest_tomlin_updates: usize,
    /// Estimated relative pivot discrepancy recorded during updates (`0.0` for standalone [`BasicLu::update`](super::BasicLu::update)).
    pub pivot_error: f64,
}
