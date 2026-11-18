//! Dense and iterative Gilbert–Peierls hyper-sparse FTRAN/BTRAN solves.
//!
//! Ported from `ERGO-Code/HiGHS` (`highs/ipm/basiclu/lu_dfs.c`,
//! `lu_solve_symbolic.c`, `lu_solve_triangular.c`, `lu_solve_dense.c`, and
//! `lu_solve_sparse.c`, MIT). All graph DFS traversals use an explicit stack in
//! [`SolveWorkspace`] (never recursion), making deep triangular solves safe on
//! `wasm32` targets.

use super::error::{Error, Result};
use super::factors::{EtaStorage, LuFactors};
use super::params::BasicLuParams;

/// Reusable scratch workspace owned by [`super::BasicLu`] for solves and updates.
#[derive(Clone, Debug, Default)]
pub(crate) struct SolveWorkspace {
    pub w: Vec<f64>,
    pub w2: Vec<f64>,
    pub visited: Vec<u32>,
    pub visit_gen: u32,
    pub dfs_stack: Vec<(usize, usize)>,
    pub reach: Vec<usize>,
    pub seeds: Vec<usize>,
    pub spike_rows: Vec<usize>,
    pub eta_rows: Vec<usize>,
    pub eta_vals: Vec<f64>,
}

impl SolveWorkspace {
    pub(crate) fn ensure_len(&mut self, n: usize) {
        self.w.resize(n, 0.0);
        self.w2.resize(n, 0.0);
        self.visited.resize(n, 0);
        self.w.fill(0.0);
        self.w2.fill(0.0);
    }

    #[inline]
    pub(crate) fn next_gen(&mut self) -> u32 {
        if self.visit_gen == u32::MAX {
            self.visited.fill(0);
            self.visit_gen = 1;
        } else {
            self.visit_gen += 1;
        }
        self.visit_gen
    }
}

#[inline]
pub(crate) fn max_reach_for(n: usize, sparse_threshold: f64) -> usize {
    let frac = if sparse_threshold.is_finite() {
        sparse_threshold.clamp(0.0, 1.0)
    } else {
        0.0
    };
    (frac * (n as f64)).floor() as usize
}

/// Iterative post-order DFS over a DAG (`lu_dfs.c`).
///
/// Traverses outgoing edges `u -> v` from each unvisited node in `seeds`,
/// recording finished nodes in post-order into `reach`. Iterating `reach` in
/// reverse yields topological order (every `u` precedes its out-neighbors `v`).
/// Returns `false` if the reachable set exceeds `max_reach`.
pub(crate) fn dfs_reach<'a, F>(
    seeds: &[usize],
    visited: &mut [u32],
    stamp: u32,
    dfs_stack: &mut Vec<(usize, usize)>,
    reach: &mut Vec<usize>,
    max_reach: usize,
    neighbors: F,
) -> bool
where
    F: Fn(usize) -> &'a [usize],
{
    reach.clear();
    dfs_stack.clear();
    let mut count = 0usize;

    for &seed in seeds {
        if visited[seed] == stamp {
            continue;
        }
        visited[seed] = stamp;
        count += 1;
        if count > max_reach {
            dfs_stack.clear();
            return false;
        }
        dfs_stack.push((seed, 0));

        while let Some((u, cursor)) = dfs_stack.last_mut() {
            let nbrs = neighbors(*u);
            if *cursor < nbrs.len() {
                let v = nbrs[*cursor];
                *cursor += 1;
                if visited[v] != stamp {
                    visited[v] = stamp;
                    count += 1;
                    if count > max_reach {
                        dfs_stack.clear();
                        return false;
                    }
                    dfs_stack.push((v, 0));
                }
            } else {
                let finished = *u;
                dfs_stack.pop();
                reach.push(finished);
            }
        }
    }
    true
}

#[inline]
fn apply_etas_forward_dense(etas: &EtaStorage, w: &mut [f64]) {
    for s in 0..etas.len() {
        let p_r = etas.eta_pivot_row[s];
        let start = etas.eta_ptr[s];
        let end = etas.eta_ptr[s + 1];
        let mut sum = 0.0;
        for p in start..end {
            sum += etas.eta_val[p] * w[etas.eta_idx[p]];
        }
        w[p_r] -= sum;
    }
}

#[inline]
fn apply_etas_transpose_dense(etas: &EtaStorage, w: &mut [f64]) {
    for s in (0..etas.len()).rev() {
        let p_r = etas.eta_pivot_row[s];
        let s_val = w[p_r];
        if s_val != 0.0 {
            let start = etas.eta_ptr[s];
            let end = etas.eta_ptr[s + 1];
            for p in start..end {
                w[etas.eta_idx[p]] -= etas.eta_val[p] * s_val;
            }
        }
    }
}

#[inline]
fn solve_l_dense(factors: &LuFactors, w: &mut [f64]) {
    for j in 0..factors.n {
        let wj = w[j];
        if wj != 0.0 {
            for p in factors.l_col_ptr[j]..factors.l_col_ptr[j + 1] {
                w[factors.l_row_idx[p]] -= factors.l_val[p] * wj;
            }
        }
    }
}

#[inline]
fn solve_lt_dense_into_slice(factors: &LuFactors, w: &mut [f64], out: &mut [f64]) {
    for r_id in (0..factors.n).rev() {
        let wi = w[r_id];
        w[r_id] = 0.0;
        out[factors.p_row_l[r_id]] = wi;
        if wi != 0.0 {
            for p in factors.lt_row_ptr[r_id]..factors.lt_row_ptr[r_id + 1] {
                w[factors.lt_col_idx[p]] -= factors.lt_val[p] * wi;
            }
        }
    }
}

#[inline]
fn solve_lt_dense_to_sparse(factors: &LuFactors, w: &mut [f64], out_idx: &mut Vec<usize>, out_val: &mut Vec<f64>) {
    for r_id in (0..factors.n).rev() {
        let wi = w[r_id];
        w[r_id] = 0.0;
        if wi != 0.0 {
            out_idx.push(factors.p_row_l[r_id]);
            out_val.push(wi);
            for p in factors.lt_row_ptr[r_id]..factors.lt_row_ptr[r_id + 1] {
                w[factors.lt_col_idx[p]] -= factors.lt_val[p] * wi;
            }
        }
    }
}

#[inline]
fn solve_u_dense_to_sparse(factors: &LuFactors, w: &mut [f64], out_idx: &mut Vec<usize>, out_val: &mut Vec<f64>) {
    for step in (0..factors.n).rev() {
        let r_id = factors.u_row_order[step];
        let c_id = factors.u_col_order[step];
        let wr = w[r_id];
        w[r_id] = 0.0;
        if wr != 0.0 {
            let xj = wr / factors.pivot[r_id];
            out_idx.push(c_id);
            out_val.push(xj);
            for (&r_above, &u_val) in factors.u_col.idx_slice(c_id).iter().zip(factors.u_col.val_slice(c_id).iter()) {
                w[r_above] -= u_val * xj;
            }
        }
    }
}

fn validate_dense_rhs(factors: Option<&LuFactors>, b: &[f64]) -> Result<usize> {
    let factors = factors.ok_or(Error::NotFactorized)?;
    if b.len() != factors.n {
        return Err(Error::BadShape);
    }
    for &v in b {
        if !v.is_finite() {
            return Err(Error::NotFinite);
        }
    }
    Ok(factors.n)
}

fn validate_sparse_rhs(factors: Option<&LuFactors>, rhs_idx: &[usize], rhs_val: &[f64]) -> Result<usize> {
    let factors = factors.ok_or(Error::NotFactorized)?;
    if rhs_idx.len() != rhs_val.len() {
        return Err(Error::BadShape);
    }
    for &r in rhs_idx {
        if r >= factors.n {
            return Err(Error::InvalidStructure);
        }
    }
    for &v in rhs_val {
        if !v.is_finite() {
            return Err(Error::NotFinite);
        }
    }
    Ok(factors.n)
}

/// Computes `w = R_t ... R_1 L^{-1} P_r^(0) rhs` using hyper-sparse DFS when
/// the reachable set stays below `sparse_threshold`, falling back to dense if needed.
/// Returns `true` when `ws.reach` is guaranteed to contain all nonzeros of `ws.w`.
pub(crate) fn ftran_through_l_and_etas(
    factors: &LuFactors,
    ws: &mut SolveWorkspace,
    rhs_idx: &[usize],
    rhs_val: &[f64],
    params: &BasicLuParams,
) -> bool {
    let n = factors.n;
    let gen_seed = ws.next_gen();
    ws.seeds.clear();
    for (&orig_r, &v) in rhs_idx.iter().zip(rhs_val.iter()) {
        let r_id = factors.p_row_l_inv[orig_r];
        ws.w[r_id] += v;
        if ws.visited[r_id] != gen_seed {
            ws.visited[r_id] = gen_seed;
            ws.seeds.push(r_id);
        }
    }
    ws.seeds.retain(|&r_id| ws.w[r_id] != 0.0);
    if ws.seeds.is_empty() {
        ws.reach.clear();
        return true;
    }

    let max_reach = max_reach_for(n, params.sparse_threshold);
    let gen_l = ws.next_gen();
    let dfs_ok = dfs_reach(&ws.seeds, &mut ws.visited, gen_l, &mut ws.dfs_stack, &mut ws.reach, max_reach, |j| {
        &factors.l_row_idx[factors.l_col_ptr[j]..factors.l_col_ptr[j + 1]]
    });

    if !dfs_ok {
        solve_l_dense(factors, &mut ws.w);
        apply_etas_forward_dense(&factors.etas, &mut ws.w);
        return false;
    }

    for &j in ws.reach.iter().rev() {
        let wj = ws.w[j];
        if wj != 0.0 {
            for p in factors.l_col_ptr[j]..factors.l_col_ptr[j + 1] {
                ws.w[factors.l_row_idx[p]] -= factors.l_val[p] * wj;
            }
        }
    }

    for s in 0..factors.etas.len() {
        let p_r = factors.etas.eta_pivot_row[s];
        let start = factors.etas.eta_ptr[s];
        let end = factors.etas.eta_ptr[s + 1];
        let mut sum = 0.0;
        for p in start..end {
            sum += factors.etas.eta_val[p] * ws.w[factors.etas.eta_idx[p]];
        }
        if sum != 0.0 {
            ws.w[p_r] -= sum;
            if ws.visited[p_r] != gen_l {
                ws.visited[p_r] = gen_l;
                ws.reach.push(p_r);
            }
        }
    }

    true
}

/// Solves `A x = b` in place (`lu_solve_dense.c`).
pub(crate) fn solve_dense(factors: Option<&LuFactors>, ws: &mut SolveWorkspace, b: &mut [f64]) -> Result<()> {
    let n = validate_dense_rhs(factors, b)?;
    if n == 0 {
        return Ok(());
    }
    let factors = factors.expect("validated above");

    for r_id in 0..n {
        ws.w[r_id] = b[factors.p_row_l[r_id]];
    }
    solve_l_dense(factors, &mut ws.w);
    apply_etas_forward_dense(&factors.etas, &mut ws.w);

    for step in (0..n).rev() {
        let r_id = factors.u_row_order[step];
        let c_id = factors.u_col_order[step];
        let xj = ws.w[r_id] / factors.pivot[r_id];
        ws.w[r_id] = 0.0;
        b[c_id] = xj;
        if xj != 0.0 {
            for (&r_above, &u_val) in factors.u_col.idx_slice(c_id).iter().zip(factors.u_col.val_slice(c_id).iter()) {
                ws.w[r_above] -= u_val * xj;
            }
        }
    }
    Ok(())
}

/// Solves `A^T x = b` in place (`lu_solve_dense.c`).
pub(crate) fn solve_transpose_dense(factors: Option<&LuFactors>, ws: &mut SolveWorkspace, b: &mut [f64]) -> Result<()> {
    let n = validate_dense_rhs(factors, b)?;
    if n == 0 {
        return Ok(());
    }
    let factors = factors.expect("validated above");

    ws.w2.copy_from_slice(b);
    for step in 0..n {
        let c_id = factors.u_col_order[step];
        let r_id = factors.u_row_order[step];
        let ui = ws.w2[c_id] / factors.pivot[r_id];
        ws.w2[c_id] = 0.0;
        ws.w[r_id] = ui;
        if ui != 0.0 {
            for (&c2, &u_val) in factors.u_row.idx_slice(r_id).iter().zip(factors.u_row.val_slice(r_id).iter()) {
                ws.w2[c2] -= u_val * ui;
            }
        }
    }

    apply_etas_transpose_dense(&factors.etas, &mut ws.w);
    solve_lt_dense_into_slice(factors, &mut ws.w, b);
    Ok(())
}

/// Hyper-sparse FTRAN `A x = b` with automatic switch to dense mode (`lu_solve_sparse.c`).
pub(crate) fn solve_sparse(
    factors: Option<&LuFactors>,
    ws: &mut SolveWorkspace,
    params: &BasicLuParams,
    rhs_idx: &[usize],
    rhs_val: &[f64],
    out_idx: &mut Vec<usize>,
    out_val: &mut Vec<f64>,
) -> Result<()> {
    let n = validate_sparse_rhs(factors, rhs_idx, rhs_val)?;
    out_idx.clear();
    out_val.clear();
    if n == 0 || rhs_idx.is_empty() {
        return Ok(());
    }
    let factors = factors.expect("validated above");
    let max_reach = max_reach_for(n, params.sparse_threshold);

    let reach_known = ftran_through_l_and_etas(factors, ws, rhs_idx, rhs_val, params);
    if !reach_known || ws.reach.len() > max_reach {
        solve_u_dense_to_sparse(factors, &mut ws.w, out_idx, out_val);
        return Ok(());
    }

    ws.seeds.clear();
    for &r_id in &ws.reach {
        if ws.w[r_id] != 0.0 {
            ws.seeds.push(r_id);
        }
    }
    if ws.seeds.is_empty() {
        return Ok(());
    }

    let gen_u = ws.next_gen();
    let u_ok = dfs_reach(&ws.seeds, &mut ws.visited, gen_u, &mut ws.dfs_stack, &mut ws.reach, max_reach, |r_id| {
        factors.u_col.idx_slice(factors.u_col_order[factors.u_row_pos[r_id]])
    });

    if !u_ok {
        solve_u_dense_to_sparse(factors, &mut ws.w, out_idx, out_val);
        return Ok(());
    }

    for &r_id in ws.reach.iter().rev() {
        let wr = ws.w[r_id];
        ws.w[r_id] = 0.0;
        if wr != 0.0 {
            let c_id = factors.u_col_order[factors.u_row_pos[r_id]];
            let xj = wr / factors.pivot[r_id];
            out_idx.push(c_id);
            out_val.push(xj);
            for (&r_above, &u_val) in factors.u_col.idx_slice(c_id).iter().zip(factors.u_col.val_slice(c_id).iter()) {
                ws.w[r_above] -= u_val * xj;
            }
        }
    }
    Ok(())
}

/// Hyper-sparse BTRAN `A^T x = b` with automatic switch to dense mode (`lu_solve_sparse.c`).
pub(crate) fn solve_sparse_transpose(
    factors: Option<&LuFactors>,
    ws: &mut SolveWorkspace,
    params: &BasicLuParams,
    rhs_idx: &[usize],
    rhs_val: &[f64],
    out_idx: &mut Vec<usize>,
    out_val: &mut Vec<f64>,
) -> Result<()> {
    let n = validate_sparse_rhs(factors, rhs_idx, rhs_val)?;
    out_idx.clear();
    out_val.clear();
    if n == 0 || rhs_idx.is_empty() {
        return Ok(());
    }
    let factors = factors.expect("validated above");

    let gen_seed = ws.next_gen();
    ws.seeds.clear();
    for (&c_id, &v) in rhs_idx.iter().zip(rhs_val.iter()) {
        ws.w2[c_id] += v;
        if ws.visited[c_id] != gen_seed {
            ws.visited[c_id] = gen_seed;
            ws.seeds.push(c_id);
        }
    }
    ws.seeds.retain(|&c_id| ws.w2[c_id] != 0.0);
    if ws.seeds.is_empty() {
        return Ok(());
    }

    let max_reach = max_reach_for(n, params.sparse_threshold);
    let gen_ut = ws.next_gen();
    let ut_ok = dfs_reach(&ws.seeds, &mut ws.visited, gen_ut, &mut ws.dfs_stack, &mut ws.reach, max_reach, |c_id| {
        factors.u_row.idx_slice(factors.u_row_order[factors.u_col_pos[c_id]])
    });

    if !ut_ok {
        for step in 0..n {
            let c_id = factors.u_col_order[step];
            let r_id = factors.u_row_order[step];
            let wc = ws.w2[c_id];
            ws.w2[c_id] = 0.0;
            if wc != 0.0 {
                let ui = wc / factors.pivot[r_id];
                ws.w[r_id] = ui;
                for (&c2, &u_val) in factors.u_row.idx_slice(r_id).iter().zip(factors.u_row.val_slice(r_id).iter()) {
                    ws.w2[c2] -= u_val * ui;
                }
            }
        }
        apply_etas_transpose_dense(&factors.etas, &mut ws.w);
        solve_lt_dense_to_sparse(factors, &mut ws.w, out_idx, out_val);
        return Ok(());
    }

    let gen_eta = ws.next_gen();
    ws.seeds.clear();
    for &c_id in ws.reach.iter().rev() {
        let wc = ws.w2[c_id];
        ws.w2[c_id] = 0.0;
        if wc != 0.0 {
            let r_id = factors.u_row_order[factors.u_col_pos[c_id]];
            let ui = wc / factors.pivot[r_id];
            ws.w[r_id] = ui;
            if ws.visited[r_id] != gen_eta {
                ws.visited[r_id] = gen_eta;
                ws.seeds.push(r_id);
            }
            for (&c2, &u_val) in factors.u_row.idx_slice(r_id).iter().zip(factors.u_row.val_slice(r_id).iter()) {
                ws.w2[c2] -= u_val * ui;
            }
        }
    }

    for s in (0..factors.etas.len()).rev() {
        let p_r = factors.etas.eta_pivot_row[s];
        let s_val = ws.w[p_r];
        if s_val != 0.0 {
            let start = factors.etas.eta_ptr[s];
            let end = factors.etas.eta_ptr[s + 1];
            for p in start..end {
                let k_r = factors.etas.eta_idx[p];
                ws.w[k_r] -= factors.etas.eta_val[p] * s_val;
                if ws.visited[k_r] != gen_eta {
                    ws.visited[k_r] = gen_eta;
                    ws.seeds.push(k_r);
                }
            }
        }
    }

    ws.seeds.retain(|&r_id| ws.w[r_id] != 0.0);
    if ws.seeds.is_empty() {
        return Ok(());
    }
    if ws.seeds.len() > max_reach {
        solve_lt_dense_to_sparse(factors, &mut ws.w, out_idx, out_val);
        return Ok(());
    }

    let gen_lt = ws.next_gen();
    let lt_ok = dfs_reach(&ws.seeds, &mut ws.visited, gen_lt, &mut ws.dfs_stack, &mut ws.reach, max_reach, |r_id| {
        &factors.lt_col_idx[factors.lt_row_ptr[r_id]..factors.lt_row_ptr[r_id + 1]]
    });

    if !lt_ok {
        solve_lt_dense_to_sparse(factors, &mut ws.w, out_idx, out_val);
        return Ok(());
    }

    for &r_id in ws.reach.iter().rev() {
        let wr = ws.w[r_id];
        ws.w[r_id] = 0.0;
        if wr != 0.0 {
            out_idx.push(factors.p_row_l[r_id]);
            out_val.push(wr);
            for p in factors.lt_row_ptr[r_id]..factors.lt_row_ptr[r_id + 1] {
                ws.w[factors.lt_col_idx[p]] -= factors.lt_val[p] * wr;
            }
        }
    }
    Ok(())
}
