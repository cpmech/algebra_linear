//! Input CSC validation and triangular singleton peeling.
//!
//! Ported from `ERGO-Code/HiGHS` (`highs/ipm/basiclu/lu_singletons.c`, MIT).
//! Column singletons (`col_deg == 1`) are peeled first into front positions
//! `0..n_cs`, followed by row singletons (`row_deg == 1`) into `n_cs..n_cs+n_rs`.
//! Remaining uneliminated rows and columns form the square active bump.

use std::collections::VecDeque;

use super::error::{Error, Result};
use super::params::BasicLuParams;

/// Output of the CSC validation and column/row singleton peeling phase.
#[derive(Clone, Debug)]
pub(crate) struct SingletonPeeling {
    pub n: usize,
    pub nnz_a: usize,
    pub p_row: Vec<usize>,
    pub p_col: Vec<usize>,
    pub pivots: Vec<f64>,
    /// Strictly subdiagonal singleton `L` entries: `(orig_row, l_step_col, val)`.
    pub l_entries: Vec<(usize, usize, f64)>,
    /// Strictly superdiagonal singleton `U` entries: `(u_step_row, orig_col, val)`.
    pub u_entries: Vec<(usize, usize, f64)>,
    pub row_elim: Vec<bool>,
    pub col_elim: Vec<bool>,
}

/// Validates a square CSC matrix using the same structural checks as `CscView`.
pub(crate) fn validate_csc(col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> Result<usize> {
    let n = col_ptr.len().checked_sub(1).ok_or(Error::BadShape)?;
    if col_ptr[0] != 0 {
        return Err(Error::InvalidStructure);
    }
    if col_ptr[n] != row_idx.len() {
        return Err(Error::BadShape);
    }
    for w in 1..=n {
        if col_ptr[w] < col_ptr[w - 1] {
            return Err(Error::InvalidStructure);
        }
    }
    for &r in row_idx {
        if r >= n {
            return Err(Error::InvalidStructure);
        }
    }
    for j in 0..n {
        let (start, end) = (col_ptr[j], col_ptr[j + 1]);
        for k in (start + 1)..end {
            if row_idx[k - 1] == row_idx[k] {
                return Err(Error::DuplicateEntry);
            }
            if row_idx[k - 1] > row_idx[k] {
                return Err(Error::UnsortedRows);
            }
        }
    }
    if values.len() != row_idx.len() {
        return Err(Error::BadShape);
    }
    for &v in values {
        if !v.is_finite() {
            return Err(Error::NotFinite);
        }
    }
    Ok(n)
}

/// Validates CSC input and peels column and row singletons directly without
/// building the bump workspace for triangular entries.
pub(crate) fn peel_singletons(col_ptr: &[usize], row_idx: &[usize], values: &[f64], params: &BasicLuParams) -> Result<SingletonPeeling> {
    let n = validate_csc(col_ptr, row_idx, values)?;
    let nnz_a = values.len();

    if n == 0 {
        return Ok(SingletonPeeling {
            n: 0,
            nnz_a: 0,
            p_row: Vec::new(),
            p_col: Vec::new(),
            pivots: Vec::new(),
            l_entries: Vec::new(),
            u_entries: Vec::new(),
            row_elim: Vec::new(),
            col_elim: Vec::new(),
        });
    }

    let drop_tol = params.drop_tol;
    let abs_tol = params.abs_pivot_tol;

    let mut row_deg = vec![0usize; n];
    let mut col_deg = vec![0usize; n];
    let mut active_nnz = 0usize;

    for j in 0..n {
        let mut cd = 0usize;
        for pos in col_ptr[j]..col_ptr[j + 1] {
            if values[pos].abs() > drop_tol {
                row_deg[row_idx[pos]] += 1;
                cd += 1;
            }
        }
        col_deg[j] = cd;
        active_nnz += cd;
    }

    // Build lightweight CSR index storing `(orig_col, csc_pos)` for active entries.
    let mut row_ptr = vec![0usize; n + 1];
    for i in 0..n {
        row_ptr[i + 1] = row_ptr[i] + row_deg[i];
    }
    let mut row_cursor = row_ptr[..n].to_vec();
    let mut row_col = vec![0usize; active_nnz];
    let mut row_csc_pos = vec![0usize; active_nnz];
    for j in 0..n {
        for pos in col_ptr[j]..col_ptr[j + 1] {
            if values[pos].abs() > drop_tol {
                let r = row_idx[pos];
                let dst = row_cursor[r];
                row_col[dst] = j;
                row_csc_pos[dst] = pos;
                row_cursor[r] = dst + 1;
            }
        }
    }

    let mut row_elim = vec![false; n];
    let mut col_elim = vec![false; n];
    let mut p_row = Vec::new();
    let mut p_col = Vec::new();
    let mut pivots = Vec::new();
    let mut l_entries = Vec::new();
    let mut u_entries = Vec::new();

    // Step 1a: Peel column singletons first (`0..n_cs`).
    let mut col_queue = VecDeque::new();
    for (j, &deg) in col_deg.iter().enumerate() {
        if deg == 1 {
            col_queue.push_back(j);
        }
    }

    while let Some(j) = col_queue.pop_front() {
        if col_elim[j] || col_deg[j] != 1 {
            continue;
        }
        let mut piv_row = None;
        let mut piv_val = 0.0;
        for pos in col_ptr[j]..col_ptr[j + 1] {
            let v = values[pos];
            let r = row_idx[pos];
            if !row_elim[r] && v.abs() > drop_tol {
                piv_row = Some(r);
                piv_val = v;
                break;
            }
        }
        let Some(i) = piv_row else {
            continue;
        };
        if piv_val == 0.0 || piv_val.abs() < abs_tol {
            // Defer numerically unsafe singleton out of singleton peeling.
            continue;
        }

        let step = p_row.len();
        p_row.push(i);
        p_col.push(j);
        pivots.push(piv_val);
        row_elim[i] = true;
        col_elim[j] = true;

        // Remaining active entries in pivot row `i` enter `U` directly without fill-in.
        for k in row_ptr[i]..row_ptr[i + 1] {
            let c = row_col[k];
            if col_elim[c] {
                continue;
            }
            let u_val = values[row_csc_pos[k]];
            u_entries.push((step, c, u_val));
            col_deg[c] -= 1;
            if col_deg[c] == 1 {
                col_queue.push_back(c);
            }
        }
    }

    // Step 1b: Peel row singletons second (`n_cs..n_cs + n_rs`).
    let mut row_queue = VecDeque::new();
    for i in 0..n {
        if !row_elim[i] && row_deg[i] == 1 {
            row_queue.push_back(i);
        }
    }

    while let Some(i) = row_queue.pop_front() {
        if row_elim[i] || row_deg[i] != 1 {
            continue;
        }
        let mut piv_col = None;
        let mut piv_val = 0.0;
        for k in row_ptr[i]..row_ptr[i + 1] {
            let c = row_col[k];
            if !col_elim[c] {
                piv_col = Some(c);
                piv_val = values[row_csc_pos[k]];
                break;
            }
        }
        let Some(j) = piv_col else {
            continue;
        };
        if piv_val == 0.0 || piv_val.abs() < abs_tol {
            continue;
        }

        let step = p_row.len();
        p_row.push(i);
        p_col.push(j);
        pivots.push(piv_val);
        row_elim[i] = true;
        col_elim[j] = true;

        // Remaining active entries in column `j` enter `L` directly (`v / piv`).
        for pos in col_ptr[j]..col_ptr[j + 1] {
            let r = row_idx[pos];
            if row_elim[r] {
                continue;
            }
            let v = values[pos];
            if v.abs() <= drop_tol {
                continue;
            }
            row_deg[r] -= 1;
            if row_deg[r] == 1 {
                row_queue.push_back(r);
            }
            let l_val = v / piv_val;
            if l_val.abs() > drop_tol {
                l_entries.push((r, step, l_val));
            }
        }
    }

    Ok(SingletonPeeling {
        n,
        nnz_a,
        p_row,
        p_col,
        pivots,
        l_entries,
        u_entries,
        row_elim,
        col_elim,
    })
}
