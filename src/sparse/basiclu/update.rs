//! Forrest–Tomlin and pure-permutation rank-1 column replacement updates.
//!
//! Ported from `ERGO-Code/HiGHS` (`highs/ipm/basiclu/lu_solve_for_update.c` and
//! `lu_update.c`, MIT). Preserves prior factors untouched if an update fails
//! validation or produces a singular pivot (`|alpha| < abs_pivot_tol`).

use super::error::{Error, Result};
use super::factors::LuFactors;
use super::params::{BasicLuParams, BasicLuStats};
use super::solve::{SolveWorkspace, dfs_reach, ftran_through_l_and_etas, max_reach_for};

fn clear_spike_workspace(ws: &mut SolveWorkspace) {
    for &r_id in &ws.spike_rows {
        ws.w[r_id] = 0.0;
    }
    ws.spike_rows.clear();
}

/// Replaces `u_col[col]` with the non-diagonal spike entries in `ws.spike_rows`
/// and updates `u_row` symmetrically.
fn replace_u_column(factors: &mut LuFactors, col: usize, p_row: usize, ws: &SolveWorkspace) {
    for &r_old in factors.u_col.idx_slice(col) {
        factors.u_row.remove_idx(r_old, col);
    }
    factors.u_col.clear_line(col);

    let extra = ws.spike_rows.len();
    for (k, &r_id) in ws.spike_rows.iter().enumerate() {
        if r_id == p_row {
            continue;
        }
        let v_r = ws.w[r_id];
        factors.u_col.push_with_slack(col, r_id, v_r, extra - k - 1);
        factors.u_row.push_with_slack(r_id, col, v_r, 0);
    }
}

/// Applies cyclic shift `p -> m` to the triangular ordering arrays without
/// renumbering any entries inside `L`, `LT`, `U`, or prior etas.
fn cyclic_shift_order(factors: &mut LuFactors, p: usize, m: usize, p_row: usize, col: usize) {
    for k in p..m {
        let r_next = factors.u_row_order[k + 1];
        factors.u_row_order[k] = r_next;
        factors.u_row_pos[r_next] = k;

        let c_next = factors.u_col_order[k + 1];
        factors.u_col_order[k] = c_next;
        factors.u_col_pos[c_next] = k;
    }
    factors.u_row_order[m] = p_row;
    factors.u_row_pos[p_row] = m;
    factors.u_col_order[m] = col;
    factors.u_col_pos[col] = m;
}

/// Computes the sparse Forrest–Tomlin row eta `w` solving `U^T w = u_{p_row, *}`
/// across all triangular positions strictly above `p`.
fn compute_row_eta(factors: &LuFactors, p: usize, p_row: usize, params: &BasicLuParams, ws: &mut SolveWorkspace) {
    let n = factors.n;
    ws.eta_rows.clear();
    ws.eta_vals.clear();
    ws.seeds.clear();

    for (&c, &u_val) in factors.u_row.idx_slice(p_row).iter().zip(factors.u_row.val_slice(p_row).iter()) {
        if u_val != 0.0 {
            ws.w2[c] = u_val;
            ws.seeds.push(c);
        }
    }
    if ws.seeds.is_empty() {
        return;
    }

    let max_reach = max_reach_for(n, params.sparse_threshold);
    let gen_ut = ws.next_gen();
    let ut_ok = dfs_reach(&ws.seeds, &mut ws.visited, gen_ut, &mut ws.dfs_stack, &mut ws.reach, max_reach, |c_id| {
        factors.u_row.idx_slice(factors.u_row_order[factors.u_col_pos[c_id]])
    });

    if ut_ok {
        for &c_id in ws.reach.iter().rev() {
            let wc = ws.w2[c_id];
            ws.w2[c_id] = 0.0;
            if wc != 0.0 {
                let r_k = factors.u_row_order[factors.u_col_pos[c_id]];
                let wk = wc / factors.pivot[r_k];
                if wk != 0.0 {
                    ws.eta_rows.push(r_k);
                    ws.eta_vals.push(wk);
                    for (&c2, &u_val) in factors.u_row.idx_slice(r_k).iter().zip(factors.u_row.val_slice(r_k).iter()) {
                        ws.w2[c2] -= u_val * wk;
                    }
                }
            }
        }
    } else {
        for step in (p + 1)..n {
            let c_id = factors.u_col_order[step];
            let wc = ws.w2[c_id];
            ws.w2[c_id] = 0.0;
            if wc != 0.0 {
                let r_k = factors.u_row_order[step];
                let wk = wc / factors.pivot[r_k];
                if wk != 0.0 {
                    ws.eta_rows.push(r_k);
                    ws.eta_vals.push(wk);
                    for (&c2, &u_val) in factors.u_row.idx_slice(r_k).iter().zip(factors.u_row.val_slice(r_k).iter()) {
                        ws.w2[c2] -= u_val * wk;
                    }
                }
            }
        }
    }
}

/// Replaces column `col` of the factorized matrix with `(new_row_idx, new_val)`.
pub(crate) fn apply_column_update(
    factors: Option<&mut LuFactors>,
    stats: Option<&mut BasicLuStats>,
    ws: &mut SolveWorkspace,
    params: &BasicLuParams,
    col: usize,
    new_row_idx: &[usize],
    new_val: &[f64],
) -> Result<()> {
    let factors = factors.ok_or(Error::NotFactorized)?;
    let stats = stats.ok_or(Error::NotFactorized)?;
    let n = factors.n;

    if col >= n {
        return Err(Error::InvalidColumn { col, n });
    }
    if stats.updates >= params.max_updates {
        return Err(Error::MaxUpdatesReached);
    }
    if new_row_idx.len() != new_val.len() {
        return Err(Error::BadShape);
    }
    for &r in new_row_idx {
        if r >= n {
            return Err(Error::InvalidStructure);
        }
    }
    for &v in new_val {
        if !v.is_finite() {
            return Err(Error::NotFinite);
        }
    }

    // Compute spike `v = R_t ... R_1 L^{-1} P_r^(0) a_new` in `r_id` coordinates.
    let reach_known = ftran_through_l_and_etas(factors, ws, new_row_idx, new_val, params);
    ws.spike_rows.clear();
    let drop_tol = params.drop_tol;
    if reach_known {
        for i in 0..ws.reach.len() {
            let r_id = ws.reach[i];
            let v_r = ws.w[r_id];
            if !v_r.is_finite() {
                for &rk in &ws.reach {
                    ws.w[rk] = 0.0;
                }
                ws.spike_rows.clear();
                return Err(Error::SingularUpdate);
            }
            if v_r.abs() > drop_tol {
                ws.spike_rows.push(r_id);
            } else {
                ws.w[r_id] = 0.0;
            }
        }
    } else {
        for r_id in 0..n {
            let v_r = ws.w[r_id];
            if !v_r.is_finite() {
                ws.w.fill(0.0);
                ws.spike_rows.clear();
                return Err(Error::SingularUpdate);
            }
            if v_r.abs() > drop_tol {
                ws.spike_rows.push(r_id);
            } else {
                ws.w[r_id] = 0.0;
            }
        }
    }

    let p = factors.u_col_pos[col];
    let p_row = factors.u_row_order[p];
    let max_pos = ws.spike_rows.iter().map(|&r_id| factors.u_row_pos[r_id]).max();
    let Some(m) = max_pos else {
        clear_spike_workspace(ws);
        return Err(Error::SingularUpdate);
    };
    if m < p {
        clear_spike_workspace(ws);
        return Err(Error::SingularUpdate);
    }

    // Pure-permutation fast path (`m == p` or row `p_row` has no entries in `(p+1)..=m`).
    let needs_eta = m > p && factors.u_row.idx_slice(p_row).iter().any(|&c| factors.u_col_pos[c] <= m);
    if !needs_eta {
        let alpha = ws.w[p_row];
        if !alpha.is_finite() || alpha == 0.0 || alpha.abs() < params.abs_pivot_tol {
            clear_spike_workspace(ws);
            return Err(Error::SingularUpdate);
        }

        replace_u_column(factors, col, p_row, ws);
        factors.pivot[p_row] = alpha;
        if m > p {
            cyclic_shift_order(factors, p, m, p_row, col);
        }
        stats.updates += 1;
        stats.nnz_u = factors.u_col.live_nnz;
        clear_spike_workspace(ws);
        return Ok(());
    }

    // General Forrest–Tomlin update: compute row eta `w`, verify pivot `alpha` before mutating.
    compute_row_eta(factors, p, p_row, params, ws);

    let mut alpha = ws.w[p_row];
    for (&r_k, &wk) in ws.eta_rows.iter().zip(ws.eta_vals.iter()) {
        alpha -= wk * ws.w[r_k];
    }
    if !alpha.is_finite() || alpha == 0.0 || alpha.abs() < params.abs_pivot_tol {
        clear_spike_workspace(ws);
        return Err(Error::SingularUpdate);
    }

    // Commit: delete all off-diagonal entries in row `p_row` from `u_col` and `u_row`.
    for &c in factors.u_row.idx_slice(p_row) {
        factors.u_col.remove_idx(c, p_row);
    }
    factors.u_row.clear_line(p_row);

    replace_u_column(factors, col, p_row, ws);
    factors.pivot[p_row] = alpha;
    cyclic_shift_order(factors, p, m, p_row, col);
    factors.etas.push_eta(p_row, &ws.eta_rows, &ws.eta_vals);

    stats.updates += 1;
    stats.forrest_tomlin_updates += 1;
    stats.nnz_u = factors.u_col.live_nnz;
    stats.nnz_r = factors.etas.eta_val.len();
    clear_spike_workspace(ws);
    Ok(())
}
