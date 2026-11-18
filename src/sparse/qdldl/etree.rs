// Derived/transcribed from `osqp/qdldl` v0.1.9 (Apache License 2.0, Copyright OSQP Developers),
// adapted and modified for `algebra_linear` in pure safe Rust (Apache-2.0 §4(b) notice).

//! Elimination tree and column nonzero counts (`QDLDL_etree`).

use super::csc::UpperTriCscPattern;
use super::error::{Error, Result};

/// Sentinel representing no parent or unvisited node in the elimination tree.
pub(crate) const NONE: usize = usize::MAX;

/// Result of symbolic analysis on an upper-triangular CSC pattern.
#[derive(Clone, Debug)]
pub struct SymbolicFactor {
    /// Order `n` of the `n x n` matrix.
    pub n: usize,
    /// Number of stored entries in the upper-triangular input matrix `A`.
    pub nnz_a: usize,
    /// 64-bit FNV-1a fingerprint of the `col_ptr` and `row_idx` arrays of `A`.
    pub fingerprint: u64,
    /// Elimination tree parent of each column, or [`NONE`] for root columns.
    pub etree: Vec<usize>,
    /// Number of strictly lower-triangular nonzeros in each column of `L`.
    #[allow(dead_code)]
    pub lnz: Vec<usize>,
    /// CSC column pointers for the strictly lower-triangular factor `L` (`n + 1` entries).
    pub l_col_ptr: Vec<usize>,
}

impl SymbolicFactor {
    /// Returns the total number of stored entries in `L`.
    #[inline]
    pub fn nnz_l(&self) -> usize {
        self.l_col_ptr[self.n]
    }
}

/// Computes the elimination tree, per-column fill counts `lnz`, and CSC column
/// pointers `l_col_ptr` for `L`.
///
/// Transcribed from `QDLDL_etree` in `osqp/qdldl` `v0.1.9`. Disconnected
/// components with multiple roots (`etree[r] == NONE`) and non-empty columns
/// with omitted diagonal entries are valid.
///
/// # Errors
///
/// * [`Error::EmptyColumn`] if any column `j` has no stored entries;
/// * [`Error::IndexOverflow`] if the prefix sum of `lnz` overflows `usize`.
pub fn compute_etree(pattern: UpperTriCscPattern<'_>) -> Result<SymbolicFactor> {
    let n = pattern.n();
    let nnz_a = pattern.nnz();
    let fingerprint = pattern.fingerprint();
    if n == 0 {
        return Ok(SymbolicFactor {
            n: 0,
            nnz_a,
            fingerprint,
            etree: Vec::new(),
            lnz: Vec::new(),
            l_col_ptr: vec![0],
        });
    }

    let col_ptr = pattern.col_ptr();
    let mut etree = vec![NONE; n];
    let mut lnz = vec![0usize; n];
    let mut work = vec![NONE; n];

    for j in 0..n {
        if col_ptr[j] == col_ptr[j + 1] {
            return Err(Error::EmptyColumn { col: j });
        }
        work[j] = j;
        for &i in pattern.row_idx_of_col(j) {
            let mut curr = i;
            while work[curr] != j {
                if etree[curr] == NONE {
                    etree[curr] = j;
                }
                lnz[curr] += 1;
                work[curr] = j;
                curr = etree[curr];
            }
        }
    }

    let l_col_ptr = prefix_sum_l_col_ptr(&lnz)?;
    Ok(SymbolicFactor {
        n,
        nnz_a,
        fingerprint,
        etree,
        lnz,
        l_col_ptr,
    })
}

/// Computes the overflow-checked prefix sum `l_col_ptr` from per-column counts `lnz`.
pub(crate) fn prefix_sum_l_col_ptr(lnz: &[usize]) -> Result<Vec<usize>> {
    let n = lnz.len();
    let mut l_col_ptr = vec![0usize; n + 1];
    for j in 0..n {
        l_col_ptr[j + 1] = l_col_ptr[j].checked_add(lnz[j]).ok_or(Error::IndexOverflow)?;
    }
    Ok(l_col_ptr)
}
