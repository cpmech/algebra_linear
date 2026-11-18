//! Active bump setup and Zlatev/Markowitz threshold pivot search.
//!
//! Ported from `ERGO-Code/HiGHS` (`highs/ipm/basiclu/lu_setup_bump.c` and
//! `lu_markowitz.c`, MIT). Numerical values live in columns (`cols: LuFileVal`),
//! while rows (`rows: LuFileIdx`) hold column indices only.

use super::degree_list::DegreeLists;
use super::file::{LuFileIdx, LuFileVal, NONE};
use super::params::BasicLuParams;
use super::singletons::SingletonPeeling;

/// Column-oriented active Schur-complement submatrix after singleton peeling.
#[derive(Clone, Debug)]
pub(crate) struct ActiveBump {
    pub n: usize,
    pub cols: LuFileVal,
    pub rows: LuFileIdx,
    pub col_max: Vec<f64>,
    pub bump_orig_rows: Vec<usize>,
    pub bump_orig_cols: Vec<usize>,
    pub col_deg_list: DegreeLists,
    pub row_deg_list: DegreeLists,
}

/// Selected pivot in bump-local coordinates.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PivotCandidate {
    pub row: usize,
    pub col: usize,
    pub val: f64,
}

impl ActiveBump {
    /// Copies the uneliminated `bump_n x bump_n` active submatrix into
    /// column-oriented [`LuFileVal`] and row-oriented [`LuFileIdx`] storage.
    pub(crate) fn from_peeling(col_ptr: &[usize], row_idx: &[usize], values: &[f64], peeling: &SingletonPeeling, params: &BasicLuParams) -> Self {
        let orig_n = peeling.n;
        let mut orig_to_bump_row = vec![NONE; orig_n];
        let mut orig_to_bump_col = vec![NONE; orig_n];
        let mut bump_orig_rows = Vec::with_capacity(orig_n - peeling.p_row.len());
        let mut bump_orig_cols = Vec::with_capacity(orig_n - peeling.p_col.len());

        for i in 0..orig_n {
            if !peeling.row_elim[i] {
                orig_to_bump_row[i] = bump_orig_rows.len();
                bump_orig_rows.push(i);
            }
            if !peeling.col_elim[i] {
                orig_to_bump_col[i] = bump_orig_cols.len();
                bump_orig_cols.push(i);
            }
        }

        let bump_n = bump_orig_rows.len();
        debug_assert_eq!(bump_n, bump_orig_cols.len());

        if bump_n == 0 {
            return Self {
                n: 0,
                cols: LuFileVal::new(0, 0, params.compress_threshold),
                rows: LuFileIdx::new(0, 0, params.compress_threshold),
                col_max: Vec::new(),
                bump_orig_rows,
                bump_orig_cols,
                col_deg_list: DegreeLists::new(0),
                row_deg_list: DegreeLists::new(0),
            };
        }

        let drop_tol = params.drop_tol;
        let mut col_counts = vec![0usize; bump_n];
        let mut row_counts = vec![0usize; bump_n];
        let mut col_max = vec![0.0_f64; bump_n];

        for (bc, &orig_c) in bump_orig_cols.iter().enumerate() {
            let mut cnt = 0usize;
            let mut max_v = 0.0_f64;
            for pos in col_ptr[orig_c]..col_ptr[orig_c + 1] {
                let br = orig_to_bump_row[row_idx[pos]];
                if br == NONE {
                    continue;
                }
                let v = values[pos];
                if v.abs() > drop_tol {
                    cnt += 1;
                    row_counts[br] += 1;
                    max_v = max_v.max(v.abs());
                }
            }
            col_counts[bc] = cnt;
            col_max[bc] = max_v;
        }

        let mut cols = LuFileVal::new(bump_n, 0, params.compress_threshold);
        let mut rows = LuFileIdx::new(bump_n, 0, params.compress_threshold);
        let mut col_cursors = cols.allocate_sequential(&col_counts, 4);
        let mut row_cursors = rows.allocate_sequential(&row_counts, 4);

        for (bc, &orig_c) in bump_orig_cols.iter().enumerate() {
            for pos in col_ptr[orig_c]..col_ptr[orig_c + 1] {
                let br = orig_to_bump_row[row_idx[pos]];
                if br == NONE {
                    continue;
                }
                let v = values[pos];
                if v.abs() > drop_tol {
                    let cp = col_cursors[bc];
                    cols.idx[cp] = br;
                    cols.val[cp] = v;
                    col_cursors[bc] = cp + 1;

                    let rp = row_cursors[br];
                    rows.idx[rp] = bc;
                    row_cursors[br] = rp + 1;
                }
            }
        }

        let mut col_deg_list = DegreeLists::new(bump_n);
        let mut row_deg_list = DegreeLists::new(bump_n);
        for bc in 0..bump_n {
            col_deg_list.insert(bc, col_counts[bc]);
            row_deg_list.insert(bc, row_counts[bc]);
        }

        Self {
            n: bump_n,
            cols,
            rows,
            col_max,
            bump_orig_rows,
            bump_orig_cols,
            col_deg_list,
            row_deg_list,
        }
    }

    /// Evicts a dead column (`col_max < abs_pivot_tol`) cleanly from both `cols`
    /// and `rows`, updating both degree lists and pruning any zero-degree rows.
    fn evict_dead_col(&mut self, c: usize) {
        let len = self.cols.len(c);
        for pos in 0..len {
            let r = self.cols.idx_slice(c)[pos];
            self.rows.remove_idx(r, c);
            self.row_deg_list.decrement(r);
        }
        self.cols.clear_line(c);
        self.col_deg_list.remove(c);
        self.col_max[c] = 0.0;
        self.prune_degree_zero();
    }

    /// Removes all degree-0 rows and columns from active search.
    pub(crate) fn prune_degree_zero(&mut self) {
        while self.row_deg_list.pop_bucket(0).is_some() {}
        while let Some(c) = self.col_deg_list.pop_bucket(0) {
            self.col_max[c] = 0.0;
        }
    }

    /// Searches for a numerically stable pivot minimizing Markowitz fill-in cost
    /// (`(row_deg - 1) * (col_deg - 1)`), returning `None` if no valid pivot remains.
    pub(crate) fn find_pivot(&mut self, params: &BasicLuParams) -> Option<PivotCandidate> {
        let abs_tol = params.abs_pivot_tol;
        let rel_tol = params.rel_pivot_tol.clamp(0.0, 1.0);

        'search: loop {
            self.prune_degree_zero();

            // Fast path: column singletons in the bump (`col_deg == 1`).
            while self.col_deg_list.head[1] != NONE {
                let c = self.col_deg_list.head[1];
                let r = self.cols.idx_slice(c)[0];
                let val = self.cols.val_slice(c)[0];
                if val != 0.0 && val.abs() >= abs_tol {
                    return Some(PivotCandidate { row: r, col: c, val });
                }
                self.evict_dead_col(c);
            }

            // Fast path: row singletons in the bump (`row_deg == 1`).
            let mut r = self.row_deg_list.head[1];
            while r != NONE {
                let next_r = self.row_deg_list.next[r];
                let c = self.rows.idx_slice(r)[0];
                if self.col_max[c] == 0.0 || self.col_max[c] < abs_tol {
                    self.evict_dead_col(c);
                    continue 'search;
                }
                if let Some(val) = self.cols.find_val(c, r) {
                    let thresh = (rel_tol * self.col_max[c]).max(abs_tol);
                    if val != 0.0 && val.abs() >= thresh {
                        return Some(PivotCandidate { row: r, col: c, val });
                    }
                }
                r = next_r;
            }

            // Zlatev/Markowitz search across degrees `d = 2..=n`.
            let mut best_cost = usize::MAX;
            let mut best_rel = 0.0_f64;
            let mut best_pivot: Option<PivotCandidate> = None;
            let mut inspected_rows = 0usize;

            for d in 2..=self.n {
                // Scan columns of degree `d`.
                let mut c = self.col_deg_list.head[d];
                while c != NONE {
                    let next_c = self.col_deg_list.next[c];
                    let cmax = self.col_max[c];
                    if cmax == 0.0 || cmax < abs_tol {
                        self.evict_dead_col(c);
                        continue 'search;
                    }
                    let thresh = (rel_tol * cmax).max(abs_tol);
                    let idxs = self.cols.idx_slice(c);
                    let vals = self.cols.val_slice(c);
                    for (&row, &val) in idxs.iter().zip(vals.iter()) {
                        if val != 0.0 && val.abs() >= thresh {
                            let r_deg = self.row_deg_list.deg[row];
                            let cost = (r_deg - 1) * (d - 1);
                            let rel = val.abs() / cmax;
                            if cost < best_cost || (cost == best_cost && rel > best_rel) {
                                best_cost = cost;
                                best_rel = rel;
                                best_pivot = Some(PivotCandidate { row, col: c, val });
                            }
                        }
                    }
                    c = next_c;
                }

                // Provably optimal after scanning columns of degree `d` and rows `< d`.
                if best_cost <= d * (d - 1) {
                    break;
                }

                // Scan rows of degree `d`.
                let mut row = self.row_deg_list.head[d];
                while row != NONE {
                    if best_pivot.is_some() && inspected_rows >= params.search_rows {
                        break;
                    }
                    let next_row = self.row_deg_list.next[row];
                    for pos in 0..self.rows.len(row) {
                        let col = self.rows.idx_slice(row)[pos];
                        let cmax = self.col_max[col];
                        if cmax == 0.0 || cmax < abs_tol {
                            self.evict_dead_col(col);
                            continue 'search;
                        }
                        let c_deg = self.col_deg_list.deg[col];
                        let cost = (d - 1) * (c_deg - 1);
                        if cost <= best_cost {
                            if let Some(val) = self.cols.find_val(col, row) {
                                let thresh = (rel_tol * cmax).max(abs_tol);
                                if val != 0.0 && val.abs() >= thresh {
                                    let rel = val.abs() / cmax;
                                    if cost < best_cost || (cost == best_cost && rel > best_rel) {
                                        best_cost = cost;
                                        best_rel = rel;
                                        best_pivot = Some(PivotCandidate { row, col, val });
                                    }
                                }
                            }
                        }
                    }
                    inspected_rows += 1;
                    if best_pivot.is_some() && inspected_rows >= params.search_rows {
                        break;
                    }
                    row = next_row;
                }

                // Provably optimal after scanning both columns and rows of degree `d`.
                if best_cost <= d * d {
                    break;
                }
            }

            return best_pivot;
        }
    }
}
