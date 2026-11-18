//! Finalized LU factor assembly and Forrest–Tomlin eta storage.
//!
//! Ported from `ERGO-Code/HiGHS` (`highs/ipm/basiclu/lu_build_factors.c`, MIT).
//!
//! # Unified Coordinate Indexing
//! - **`r_id in 0..n`:** fixed factorization-step row identifier defined by
//!   `p_row_l[r_id] = orig_row` and `p_row_l_inv[orig_row] = r_id`.
//! - **`c_id in 0..n`:** original matrix column identifier `orig_col`.
//! - `L`, `LT`, `pivot`, `u_row` line keys, `u_col` row indices, and `EtaStorage`
//!   all use stable `r_id`, whereas `u_col` line keys and `u_row` column indices
//!   use original `c_id`. Dynamic triangular ordering is tracked separately via
//!   `u_row_order`/`u_row_pos` and `u_col_order`/`u_col_pos`.

use super::eliminate::BumpFactorData;
use super::error::{Error, Result};
use super::file::{LuFileVal, NONE};
use super::params::BasicLuParams;
use super::singletons::SingletonPeeling;

/// Packed Forrest–Tomlin row-eta factors `R_1, ..., R_t`.
#[derive(Clone, Debug)]
pub(crate) struct EtaStorage {
    /// Fixed `r_id` of the pivot row updated by `R_s`.
    pub eta_pivot_row: Vec<usize>,
    /// CSR pointers into `eta_idx`/`eta_val`, length `t + 1`.
    pub eta_ptr: Vec<usize>,
    /// Fixed `r_id` indices of the row elimination multipliers.
    pub eta_idx: Vec<usize>,
    /// Row elimination weights `w_k`.
    pub eta_val: Vec<f64>,
}

impl EtaStorage {
    pub(crate) fn new() -> Self {
        Self {
            eta_pivot_row: Vec::new(),
            eta_ptr: vec![0],
            eta_idx: Vec::new(),
            eta_val: Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.eta_pivot_row.len()
    }

    pub(crate) fn push_eta(&mut self, p_row: usize, rows: &[usize], vals: &[f64]) {
        debug_assert_eq!(rows.len(), vals.len());
        self.eta_pivot_row.push(p_row);
        self.eta_idx.extend_from_slice(rows);
        self.eta_val.extend_from_slice(vals);
        self.eta_ptr.push(self.eta_idx.len());
    }
}

/// Complete LU factors supporting dense/hyper-sparse solves and rank-1 updates.
#[derive(Clone, Debug)]
pub(crate) struct LuFactors {
    pub n: usize,
    pub rank: usize,

    /// Initial factorization row permutation: `r_id -> orig_row`.
    pub p_row_l: Vec<usize>,
    /// Inverse initial row permutation: `orig_row -> r_id`.
    pub p_row_l_inv: Vec<usize>,

    /// Immutable unit lower-triangular `L` in CSC (`r_id` coords, unit diag implicit).
    pub l_col_ptr: Vec<usize>,
    pub l_row_idx: Vec<usize>,
    pub l_val: Vec<f64>,

    /// Immutable transpose `LT` in CSR (`r_id` coords, unit diag implicit).
    pub lt_row_ptr: Vec<usize>,
    pub lt_col_idx: Vec<usize>,
    pub lt_val: Vec<f64>,

    /// Mutable strictly off-diagonal `U` indexed by `c_id`, storing `(r_id, val)`.
    pub u_col: LuFileVal,
    /// Mutable strictly off-diagonal `U` indexed by `r_id`, storing `(c_id, val)`.
    pub u_row: LuFileVal,
    /// Diagonal pivots indexed by fixed `r_id`.
    pub pivot: Vec<f64>,

    /// Dynamic triangular row order: `step -> r_id`.
    pub u_row_order: Vec<usize>,
    /// Dynamic triangular row inverse order: `r_id -> step`.
    pub u_row_pos: Vec<usize>,
    /// Dynamic triangular column order: `step -> c_id` (original column).
    pub u_col_order: Vec<usize>,
    /// Dynamic triangular column inverse order: `c_id -> step`.
    pub u_col_pos: Vec<usize>,

    /// Accumulated Forrest–Tomlin row etas `R_1, ..., R_t`.
    pub etas: EtaStorage,
}

/// Assembles [`LuFactors`] from singleton peeling and bump factorization data.
pub(crate) fn build_factors(peeling: SingletonPeeling, bump_data: BumpFactorData, params: &BasicLuParams) -> Result<LuFactors> {
    let n = peeling.n;
    let peeled_n = peeling.p_row.len();
    let bump_rank = bump_data.pivots.len();
    let rank = peeled_n + bump_rank;

    if rank < n && !params.allow_singular {
        return Err(Error::Singular { rank, n });
    }

    let mut p_row_l = Vec::with_capacity(n);
    p_row_l.extend_from_slice(&peeling.p_row);
    p_row_l.extend_from_slice(&bump_data.p_row);

    let mut u_col_order = Vec::with_capacity(n);
    u_col_order.extend_from_slice(&peeling.p_col);
    u_col_order.extend_from_slice(&bump_data.p_col);

    let mut pivot = Vec::with_capacity(n);
    pivot.extend_from_slice(&peeling.pivots);
    pivot.extend_from_slice(&bump_data.pivots);

    let mut p_row_l_inv = vec![NONE; n];
    for (r_id, &orig_row) in p_row_l.iter().enumerate() {
        p_row_l_inv[orig_row] = r_id;
    }

    let mut u_col_pos = vec![NONE; n];
    for (step, &orig_col) in u_col_order.iter().enumerate() {
        u_col_pos[orig_col] = step;
    }

    // Complete rank-deficient factors with unit columns (`pivot = 1.0`).
    if rank < n {
        for orig_row in 0..n {
            if p_row_l_inv[orig_row] == NONE {
                p_row_l_inv[orig_row] = p_row_l.len();
                p_row_l.push(orig_row);
            }
        }
        for orig_col in 0..n {
            if u_col_pos[orig_col] == NONE {
                u_col_pos[orig_col] = u_col_order.len();
                u_col_order.push(orig_col);
            }
        }
        pivot.resize(n, 1.0);
    }

    let u_row_order: Vec<usize> = (0..n).collect();
    let u_row_pos: Vec<usize> = (0..n).collect();

    // Build immutable `L` (CSC) and `LT` (CSR) in `r_id` coordinates.
    let mut l_counts = vec![0usize; n];
    for &(_, col_step, _) in &peeling.l_entries {
        l_counts[col_step] += 1;
    }
    for &(_, bump_step, _) in &bump_data.l_entries {
        l_counts[peeled_n + bump_step] += 1;
    }

    let mut l_col_ptr = vec![0usize; n + 1];
    for j in 0..n {
        l_col_ptr[j + 1] = l_col_ptr[j] + l_counts[j];
    }
    let nnz_l = l_col_ptr[n];
    let mut l_cursor = l_col_ptr[..n].to_vec();
    let mut l_row_idx_unsorted = vec![0usize; nnz_l];
    let mut l_val_unsorted = vec![0.0_f64; nnz_l];

    for &(orig_row, col_step, val) in &peeling.l_entries {
        let r_id = p_row_l_inv[orig_row];
        debug_assert!(r_id > col_step);
        let dst = l_cursor[col_step];
        l_row_idx_unsorted[dst] = r_id;
        l_val_unsorted[dst] = val;
        l_cursor[col_step] = dst + 1;
    }
    for &(orig_row, bump_step, val) in &bump_data.l_entries {
        let col_step = peeled_n + bump_step;
        let r_id = p_row_l_inv[orig_row];
        debug_assert!(r_id > col_step);
        let dst = l_cursor[col_step];
        l_row_idx_unsorted[dst] = r_id;
        l_val_unsorted[dst] = val;
        l_cursor[col_step] = dst + 1;
    }

    // Build `LT` (CSR) via counting sort over `L` (gives sorted columns in `LT`),
    // then transpose once more into `L` so `L`'s rows are also strictly ordered.
    let mut lt_counts = vec![0usize; n];
    for &r in &l_row_idx_unsorted {
        lt_counts[r] += 1;
    }
    let mut lt_row_ptr = vec![0usize; n + 1];
    for i in 0..n {
        lt_row_ptr[i + 1] = lt_row_ptr[i] + lt_counts[i];
    }
    let mut lt_cursor = lt_row_ptr[..n].to_vec();
    let mut lt_col_idx = vec![0usize; nnz_l];
    let mut lt_val = vec![0.0_f64; nnz_l];
    for j in 0..n {
        for p in l_col_ptr[j]..l_col_ptr[j + 1] {
            let r = l_row_idx_unsorted[p];
            let dst = lt_cursor[r];
            lt_col_idx[dst] = j;
            lt_val[dst] = l_val_unsorted[p];
            lt_cursor[r] = dst + 1;
        }
    }

    // Reconstruct sorted `L` from `LT` in linear time.
    l_cursor.copy_from_slice(&l_col_ptr[..n]);
    let mut l_row_idx = vec![0usize; nnz_l];
    let mut l_val = vec![0.0_f64; nnz_l];
    for r in 0..n {
        for p in lt_row_ptr[r]..lt_row_ptr[r + 1] {
            let c = lt_col_idx[p];
            let dst = l_cursor[c];
            l_row_idx[dst] = r;
            l_val[dst] = lt_val[p];
            l_cursor[c] = dst + 1;
        }
    }

    // Build mutable dual `u_col` (keyed by `c_id`, row indices `r_id`) and
    // `u_row` (keyed by `r_id`, col indices `c_id`).
    let mut u_col_counts = vec![0usize; n];
    let mut u_row_counts = vec![0usize; n];
    for &(r_id, c_id, _) in &peeling.u_entries {
        u_row_counts[r_id] += 1;
        u_col_counts[c_id] += 1;
    }
    for &(bump_step, c_id, _) in &bump_data.u_entries {
        let r_id = peeled_n + bump_step;
        u_row_counts[r_id] += 1;
        u_col_counts[c_id] += 1;
    }

    let mut u_col = LuFileVal::new(n, 0, params.compress_threshold);
    let mut u_row = LuFileVal::new(n, 0, params.compress_threshold);
    let mut u_col_cursor = u_col.allocate_sequential(&u_col_counts, 2);
    let mut u_row_cursor = u_row.allocate_sequential(&u_row_counts, 2);

    for &(r_id, c_id, val) in &peeling.u_entries {
        debug_assert!(u_row_pos[r_id] < u_col_pos[c_id]);
        let cp = u_col_cursor[c_id];
        u_col.idx[cp] = r_id;
        u_col.val[cp] = val;
        u_col_cursor[c_id] = cp + 1;

        let rp = u_row_cursor[r_id];
        u_row.idx[rp] = c_id;
        u_row.val[rp] = val;
        u_row_cursor[r_id] = rp + 1;
    }
    for &(bump_step, c_id, val) in &bump_data.u_entries {
        let r_id = peeled_n + bump_step;
        debug_assert!(u_row_pos[r_id] < u_col_pos[c_id]);
        let cp = u_col_cursor[c_id];
        u_col.idx[cp] = r_id;
        u_col.val[cp] = val;
        u_col_cursor[c_id] = cp + 1;

        let rp = u_row_cursor[r_id];
        u_row.idx[rp] = c_id;
        u_row.val[rp] = val;
        u_row_cursor[r_id] = rp + 1;
    }

    Ok(LuFactors {
        n,
        rank,
        p_row_l,
        p_row_l_inv,
        l_col_ptr,
        l_row_idx,
        l_val,
        lt_row_ptr,
        lt_col_idx,
        lt_val,
        u_col,
        u_row,
        pivot,
        u_row_order,
        u_row_pos,
        u_col_order,
        u_col_pos,
        etas: EtaStorage::new(),
    })
}
