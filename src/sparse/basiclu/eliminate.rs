//! Column-oriented Schur-complement elimination for the active bump.
//!
//! Ported from `ERGO-Code/HiGHS` (`highs/ipm/basiclu/lu_pivot.c` and
//! `lu_factorize_bump.c`, MIT). Copies the pivot column and pivot row into
//! scratch before updating target columns so line compaction cannot invalidate
//! slices, and removes numerical cancellations symmetrically from both `cols`
//! and `rows`.

use super::markowitz::{ActiveBump, PivotCandidate};
use super::params::BasicLuParams;

/// Recorded bump elimination output mapped back to original row/column indices.
#[derive(Clone, Debug)]
pub(crate) struct BumpFactorData {
    pub p_row: Vec<usize>,
    pub p_col: Vec<usize>,
    pub pivots: Vec<f64>,
    /// Strictly subdiagonal bump `L` entries: `(orig_row, step_offset_in_bump, val)`.
    pub l_entries: Vec<(usize, usize, f64)>,
    /// Strictly superdiagonal bump `U` entries: `(step_offset_in_bump, orig_col, val)`.
    pub u_entries: Vec<(usize, usize, f64)>,
}

/// Reusable scratch buffers for bump Gaussian elimination.
struct ElimWorkspace {
    piv_col_row: Vec<usize>,
    piv_col_lval: Vec<f64>,
    piv_row_col: Vec<usize>,
    piv_row_uval: Vec<f64>,
    work: Vec<f64>,
    mark: Vec<u32>,
    stamp: u32,
    fill_rows: Vec<usize>,
}

impl ElimWorkspace {
    fn new(bump_n: usize) -> Self {
        Self {
            piv_col_row: Vec::new(),
            piv_col_lval: Vec::new(),
            piv_row_col: Vec::new(),
            piv_row_uval: Vec::new(),
            work: vec![0.0; bump_n],
            mark: vec![0; bump_n],
            stamp: 0,
            fill_rows: Vec::new(),
        }
    }

    #[inline]
    fn next_stamp(&mut self) -> u32 {
        if self.stamp == u32::MAX {
            self.mark.fill(0);
            self.stamp = 1;
        } else {
            self.stamp += 1;
        }
        self.stamp
    }
}

/// Eliminates one selected pivot `(p_row, p_col)` from `bump`.
fn eliminate_pivot(
    bump: &mut ActiveBump,
    cand: PivotCandidate,
    step: usize,
    params: &BasicLuParams,
    ws: &mut ElimWorkspace,
    out: &mut BumpFactorData,
) {
    let p_row = cand.row;
    let p_col = cand.col;
    let piv = cand.val;
    let drop_tol = params.drop_tol;

    out.p_row.push(bump.bump_orig_rows[p_row]);
    out.p_col.push(bump.bump_orig_cols[p_col]);
    out.pivots.push(piv);

    // Copy pivot column multipliers into scratch and unlink `p_col` from `rows`.
    ws.piv_col_row.clear();
    ws.piv_col_lval.clear();
    for (&r, &v) in bump.cols.idx_slice(p_col).iter().zip(bump.cols.val_slice(p_col).iter()) {
        if r == p_row {
            continue;
        }
        bump.rows.remove_idx(r, p_col);
        bump.row_deg_list.decrement(r);
        let l_val = v / piv;
        if l_val.abs() > drop_tol {
            ws.piv_col_row.push(r);
            ws.piv_col_lval.push(l_val);
            out.l_entries.push((bump.bump_orig_rows[r], step, l_val));
        }
    }

    // Copy pivot row entries into scratch and unlink `p_row` from `cols`.
    ws.piv_row_col.clear();
    ws.piv_row_uval.clear();
    for &c in bump.rows.idx_slice(p_row) {
        if c == p_col {
            continue;
        }
        if let Some(u_val) = bump.cols.remove_idx(c, p_row) {
            bump.col_deg_list.decrement(c);
            if u_val.abs() > drop_tol {
                ws.piv_row_col.push(c);
                ws.piv_row_uval.push(u_val);
                out.u_entries.push((step, bump.bump_orig_cols[c], u_val));
            }
        }
    }

    // Remove the eliminated pivot lines from `ActiveBump`.
    bump.cols.clear_line(p_col);
    bump.rows.clear_line(p_row);
    bump.col_deg_list.remove(p_col);
    bump.row_deg_list.remove(p_row);
    bump.col_max[p_col] = 0.0;

    // Rank-1 Schur complement update on each target column `c` in `piv_row`.
    for idx_c in 0..ws.piv_row_col.len() {
        let c = ws.piv_row_col[idx_c];
        let u_val = ws.piv_row_uval[idx_c];
        let stamp = ws.next_stamp();
        ws.fill_rows.clear();

        // Scatter existing column `c` into dense workspace.
        for (&r, &v) in bump.cols.idx_slice(c).iter().zip(bump.cols.val_slice(c).iter()) {
            ws.work[r] = v;
            ws.mark[r] = stamp;
        }

        // Subtract `u_val * piv_col` and record fill-in rows.
        for (&r, &l_val) in ws.piv_col_row.iter().zip(ws.piv_col_lval.iter()) {
            let delta = -l_val * u_val;
            if ws.mark[r] == stamp {
                ws.work[r] += delta;
            } else if delta.abs() > drop_tol {
                ws.work[r] = delta;
                ws.mark[r] = stamp;
                ws.fill_rows.push(r);
            }
        }

        // Gather existing slots, removing cancellations symmetrically from `cols` and `rows`.
        let mut new_max = 0.0_f64;
        let mut pos = 0usize;
        while pos < bump.cols.len(c) {
            let r = bump.cols.idx_slice(c)[pos];
            let new_v = ws.work[r];
            ws.work[r] = 0.0;
            if new_v.abs() <= drop_tol {
                bump.cols.swap_remove_at(c, pos);
                bump.col_deg_list.decrement(c);
                bump.rows.remove_idx(r, c);
                bump.row_deg_list.decrement(r);
            } else {
                bump.cols.val_slice_mut(c)[pos] = new_v;
                new_max = new_max.max(new_v.abs());
                pos += 1;
            }
        }

        // Append newly created fill-in entries into both `cols[c]` and `rows[r]`.
        let fill_len = ws.fill_rows.len();
        for (k, &r) in ws.fill_rows.iter().enumerate() {
            let fill_v = ws.work[r];
            ws.work[r] = 0.0;
            bump.cols.push_with_slack(c, r, fill_v, fill_len - k - 1);
            bump.rows.push_with_slack(r, c, 0);
            bump.col_deg_list.increment(c);
            bump.row_deg_list.increment(r);
            new_max = new_max.max(fill_v.abs());
        }

        bump.col_max[c] = new_max;
    }
}

/// Factorizes the entire active bump using Markowitz pivot selection and
/// column-oriented Gaussian elimination (`lu_factorize_bump.c`).
pub(crate) fn factorize_bump(bump: &mut ActiveBump, params: &BasicLuParams) -> BumpFactorData {
    let mut out = BumpFactorData {
        p_row: Vec::with_capacity(bump.n),
        p_col: Vec::with_capacity(bump.n),
        pivots: Vec::with_capacity(bump.n),
        l_entries: Vec::new(),
        u_entries: Vec::new(),
    };
    if bump.n == 0 {
        return out;
    }

    let mut ws = ElimWorkspace::new(bump.n);
    while let Some(cand) = bump.find_pivot(params) {
        let step = out.pivots.len();
        eliminate_pivot(bump, cand, step, params, &mut ws, &mut out);
    }
    out
}
