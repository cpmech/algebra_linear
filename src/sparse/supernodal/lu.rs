//! The numeric supernodal LU factorization.
//!
//! This is a correctness-first supernodal factorization built on the structure
//! computed by [`SymbolicSupernodal`]. For each supernode it assembles a dense
//! front from `A` and from the updates of the previous columns, factors the
//! panel with partial pivoting, and appends the resulting `L` and `U` columns.
//!
//! It differs from faer's implementation in one deliberate way: it does not keep
//! contribution blocks between supernodes. Instead, each front is updated by
//! multiplying the stored `L` and `U` columns directly, which is slower for
//! large fronts but removes the block bookkeeping and its failure modes. The
//! output is the same CSC `L`/`U` layout the simplicial path uses, so the two
//! paths can be compared directly.

use crate::sparse::SparseError;
use crate::sparse::csc::CscView;
use crate::sparse::lu::{SparseScalar, unit_lower_solve, upper_solve};
use crate::sparse::supernodal::SymbolicSupernodal;

/// The numeric factors of a supernodal LU.
///
/// `L` is unit lower triangular and `U` upper triangular, both in CSC form and
/// expressed in the permuted coordinate system: row indices are positions, not
/// original row numbers. The diagonal entry of every `L` column is stored first
/// and the diagonal of every `U` column last, which is what the sparse solves
/// assume.
#[derive(Clone, Debug)]
pub struct SupernodalLu<T> {
    n: usize,
    l_col_ptr: Vec<usize>,
    l_row_idx: Vec<usize>,
    l_val: Vec<T>,
    u_col_ptr: Vec<usize>,
    u_row_idx: Vec<usize>,
    u_val: Vec<T>,
    row_perm: Vec<usize>,
    col_perm_inv: Vec<usize>,
}

impl<T> SupernodalLu<T> {
    /// Returns the order of the matrix.
    #[inline]
    pub(crate) fn n(&self) -> usize {
        self.n
    }

    /// Returns the `L` column pointers.
    #[inline]
    pub fn l_col_ptr(&self) -> &[usize] {
        &self.l_col_ptr
    }

    /// Returns the `L` row indices (positions).
    // Test-only: exposed so the tests can inspect the factors.
    #[cfg(test)]
    #[inline]
    pub fn l_row_idx(&self) -> &[usize] {
        &self.l_row_idx
    }

    /// Returns the `L` values.
    // Test-only: exposed so the tests can inspect the factors.
    #[cfg(test)]
    #[inline]
    pub fn l_values(&self) -> &[T] {
        &self.l_val
    }

    /// Returns the `U` column pointers.
    #[inline]
    pub fn u_col_ptr(&self) -> &[usize] {
        &self.u_col_ptr
    }

    /// Returns the `U` row indices (positions).
    // Test-only: exposed so the tests can inspect the factors.
    #[cfg(test)]
    #[inline]
    pub fn u_row_idx(&self) -> &[usize] {
        &self.u_row_idx
    }

    /// Returns the `U` values.
    // Test-only: exposed so the tests can inspect the factors.
    #[cfg(test)]
    #[inline]
    pub fn u_values(&self) -> &[T] {
        &self.u_val
    }

    /// Returns the number of stored entries in `L`.
    #[inline]
    pub fn nnz_l(&self) -> usize {
        self.l_col_ptr[self.n]
    }

    /// Returns the number of stored entries in `U`.
    #[inline]
    pub fn nnz_u(&self) -> usize {
        self.u_col_ptr[self.n]
    }

    /// Returns the row permutation (position to original row).
    // Test-only: exposed so the tests can inspect the factors.
    #[cfg(test)]
    #[inline]
    pub fn row_perm(&self) -> &[usize] {
        &self.row_perm
    }
}

impl<T: SparseScalar> SupernodalLu<T> {
    /// Solves `A x = b` in place.
    ///
    /// # Errors
    ///
    /// [`SparseError::BadShape`] if `b.len()` is not the order of the matrix.
    pub fn solve(&self, b: &mut [T]) -> Result<(), SparseError> {
        let n = self.n;
        if b.len() != n {
            return Err(SparseError::BadShape);
        }
        let mut tmp = vec![T::zero(); n];
        for i in 0..n {
            tmp[i] = b[self.row_perm[i]];
        }
        unit_lower_solve(&self.l_col_ptr, &self.l_row_idx, &self.l_val, &mut tmp);
        upper_solve(&self.u_col_ptr, &self.u_row_idx, &self.u_val, &mut tmp);
        for i in 0..n {
            b[i] = tmp[self.col_perm_inv[i]];
        }
        Ok(())
    }
}

impl SymbolicSupernodal {
    /// Computes the numeric supernodal factors of `a`.
    ///
    /// The pattern must be the one the analysis saw; only the values may change.
    ///
    /// # Errors
    ///
    /// [`SparseError::StructureMismatch`] if the structure changed, and
    /// [`SparseError::Singular`] if a pivot is zero.
    pub fn factorize<T: SparseScalar>(&self, a: CscView<'_, T>) -> Result<SupernodalLu<T>, SparseError> {
        if !self.matches(a.pattern()) {
            return Err(SparseError::StructureMismatch);
        }
        if a.values().iter().any(|v| !v.is_finite()) {
            return Err(SparseError::NotFinite);
        }
        let n = self.n;
        let ns = self.n_supernodes();
        let ptr = &self.supernode_ptr;

        // Transpose of A with each column replaced by its new position.
        let mut at_counts = vec![0usize; n];
        for &row in a.pattern().row_idx() {
            at_counts[row] += 1;
        }
        let mut at_ptr = vec![0usize; n + 1];
        for r in 0..n {
            at_ptr[r + 1] = at_ptr[r] + at_counts[r];
        }
        let mut at_next = at_ptr.clone();
        let mut at_col = vec![0usize; a.nnz()];
        let mut at_val = vec![T::zero(); a.nnz()];
        for c in 0..n {
            let new_c = self.col_perm_inv[c];
            for k in a.pattern().col_ptr()[c]..a.pattern().col_ptr()[c + 1] {
                let row = a.pattern().row_idx()[k];
                let idx = at_next[row];
                at_next[row] += 1;
                at_col[idx] = new_c;
                at_val[idx] = a.values()[k];
            }
        }

        let mut row_perm: Vec<usize> = (0..n).collect();
        let mut row_perm_inv: Vec<usize> = (0..n).collect();
        let mut l_cols: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
        let mut u_cols: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
        // Original row -> supernodes whose L panel has an entry in that row.
        let mut l_compress_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
        // Supernode -> earlier supernodes whose U columns reach it.
        let mut u_compress_cols: Vec<Vec<usize>> = vec![Vec::new(); ns];
        // Columns of each supernode's U panel, for the C_s union.
        let mut u_panel_cols: Vec<Vec<usize>> = vec![Vec::new(); ns];

        let mut row_mark = vec![0usize; n];
        let mut col_mark = vec![0usize; n];
        let mut row_local = vec![usize::MAX; n];
        let mut col_local = vec![usize::MAX; n];
        let mut row_tag = 0usize;
        let mut col_tag = 0usize;

        let mut r_rows: Vec<usize> = Vec::new();
        let mut c_cols: Vec<usize> = Vec::new();

        for s in 0..ns {
            let s_begin = ptr[s];
            let s_end = ptr[s + 1];
            let s_size = s_end - s_begin;

            // R_s: rows of A's columns in s plus the L rows of the descendants
            // whose U columns reach s, all with a current position >= s_begin.
            row_tag += 1;
            r_rows.clear();
            for j in s_begin..s_end {
                let pj = self.col_perm[j];
                for &row in a.pattern().row_idx_of_col(pj) {
                    if row_perm_inv[row] >= s_begin && row_mark[row] != row_tag {
                        row_mark[row] = row_tag;
                        r_rows.push(row);
                    }
                }
            }
            for &d in &u_compress_cols[s] {
                for j in ptr[d]..ptr[d + 1] {
                    for &(row, _) in &l_cols[j] {
                        if row_perm_inv[row] >= s_begin && row_mark[row] != row_tag {
                            row_mark[row] = row_tag;
                            r_rows.push(row);
                        }
                    }
                }
            }
            // Every position in the supernode must be a pivot candidate, even
            // if its row has no structural entry yet; fill may reach it.
            for pos in s_begin..s_end {
                let row = row_perm[pos];
                if row_mark[row] != row_tag {
                    row_mark[row] = row_tag;
                    r_rows.push(row);
                }
            }
            r_rows.sort_by_key(|&row| row_perm_inv[row]);
            debug_assert!(r_rows.len() >= s_size);

            // C_s: columns >= s_end that touch the rows of s, plus the U columns
            // of the descendants whose L panels cover those rows.
            col_tag += 1;
            c_cols.clear();
            for i in s_begin..s_end {
                let row = row_perm[i];
                for k in at_ptr[row]..at_ptr[row + 1] {
                    let c = at_col[k];
                    if c >= s_end && col_mark[c] != col_tag {
                        col_mark[c] = col_tag;
                        c_cols.push(c);
                    }
                }
                for &d in &l_compress_rows[row] {
                    for &c in &u_panel_cols[d] {
                        if c >= s_end && col_mark[c] != col_tag {
                            col_mark[c] = col_tag;
                            c_cols.push(c);
                        }
                    }
                }
            }
            c_cols.sort_unstable();
            let nrows = r_rows.len();
            for (idx, &row) in r_rows.iter().enumerate() {
                row_local[row] = idx;
            }

            // The L part of the front: the diagonal columns, assembled from A
            // and updated by every previous column.
            let mut front = vec![T::zero(); nrows * s_size];
            for (jj, j) in (s_begin..s_end).enumerate() {
                let pj = self.col_perm[j];
                let (start, end) = (a.pattern().col_ptr()[pj], a.pattern().col_ptr()[pj + 1]);
                for k in start..end {
                    let row = a.pattern().row_idx()[k];
                    let li = row_local[row];
                    if li != usize::MAX {
                        front[li + jj * nrows] += a.values()[k];
                    }
                }
                for &(k, u_val) in &u_cols[j] {
                    if k >= s_begin {
                        continue;
                    }
                    for &(row, l_val) in &l_cols[k] {
                        let li = row_local[row];
                        if li != usize::MAX {
                            front[li + jj * nrows] -= l_val * u_val;
                        }
                    }
                }
            }

            // Factor the panel and place each pivot row at its step position.
            let mut ipiv = vec![0usize; s_size];
            lu_front(&mut front, nrows, s_size, s_size, &mut ipiv)?;
            for k in 0..s_size {
                let pivot = ipiv[k];
                let pivot_row = r_rows[pivot];
                let new_pos = s_begin + k;
                let old_pos = row_perm_inv[pivot_row];
                let displaced = row_perm[new_pos];
                if old_pos != new_pos {
                    row_perm.swap(new_pos, old_pos);
                    row_perm_inv[pivot_row] = new_pos;
                    row_perm_inv[displaced] = old_pos;
                }
                if pivot != k {
                    r_rows.swap(k, pivot);
                    row_local[pivot_row] = k;
                    if displaced != pivot_row {
                        row_local[displaced] = pivot;
                    }
                }
            }

            // C_s is built after the pivots: the pivot rows decide which
            // columns to the right receive a non-zero row.
            col_tag += 1;
            c_cols.clear();
            for i in s_begin..s_end {
                let row = row_perm[i];
                for k in at_ptr[row]..at_ptr[row + 1] {
                    let c = at_col[k];
                    if c >= s_end && col_mark[c] != col_tag {
                        col_mark[c] = col_tag;
                        c_cols.push(c);
                    }
                }
                for &d in &l_compress_rows[row] {
                    for &c in &u_panel_cols[d] {
                        if c >= s_end && col_mark[c] != col_tag {
                            col_mark[c] = col_tag;
                            c_cols.push(c);
                        }
                    }
                }
            }
            c_cols.sort_unstable();
            for (idx, &c) in c_cols.iter().enumerate() {
                col_local[c] = idx;
            }

            // The U part of the front, then the forward solve L11 \ U12.
            let ncols = c_cols.len();
            let mut u_front = vec![T::zero(); nrows * ncols];
            for (li, &row) in r_rows.iter().enumerate() {
                for k in at_ptr[row]..at_ptr[row + 1] {
                    let c = at_col[k];
                    if c >= s_end {
                        let ci = col_local[c];
                        if ci != usize::MAX {
                            u_front[li + ci * nrows] += at_val[k];
                        }
                    }
                }
            }
            for (ci, &c) in c_cols.iter().enumerate() {
                for &(k, u_val) in &u_cols[c] {
                    if k >= s_begin {
                        continue;
                    }
                    for &(row, l_val) in &l_cols[k] {
                        let li = row_local[row];
                        if li != usize::MAX {
                            u_front[li + ci * nrows] -= l_val * u_val;
                        }
                    }
                }
            }
            for jj in 0..s_size {
                for ii in jj + 1..s_size {
                    let l = front[ii + jj * nrows];
                    for ci in 0..ncols {
                        let pivot_val = u_front[jj + ci * nrows];
                        u_front[ii + ci * nrows] -= l * pivot_val;
                    }
                }
            }

            // Store the L columns and the U panels. Exact zeros are skipped:
            // the front is dense in the row union `r_rows`, but many of those
            // entries never become nonzero, and carrying them would make every
            // later update walk them.
            for (jj, j) in (s_begin..s_end).enumerate() {
                for li in jj..nrows {
                    let row = r_rows[li];
                    let val = if li == jj { T::one() } else { front[li + jj * nrows] };
                    if li == jj || val.abs() != 0.0 {
                        l_cols[j].push((row, val));
                    }
                }
            }
            for (jj, j) in (s_begin..s_end).enumerate() {
                for ii in 0..=jj {
                    let val = front[ii + jj * nrows];
                    if ii == jj || val.abs() != 0.0 {
                        u_cols[j].push((s_begin + ii, val));
                    }
                }
            }
            for (ci, &c) in c_cols.iter().enumerate() {
                for ii in 0..s_size {
                    let val = u_front[ii + ci * nrows];
                    if val.abs() != 0.0 {
                        u_cols[c].push((s_begin + ii, val));
                    }
                }
                let node = supernode_of(ptr, c);
                if node > s && u_compress_cols[node].last() != Some(&s) {
                    u_compress_cols[node].push(s);
                }
            }
            // Record the rows whose position is below this supernode.
            for &row in &r_rows {
                if row_perm_inv[row] >= s_end {
                    l_compress_rows[row].push(s);
                }
            }
            for &row in &r_rows {
                row_local[row] = usize::MAX;
            }
            for &c in &c_cols {
                col_local[c] = usize::MAX;
            }
            // Keep the column list for this supernode's C_s union without
            // cloning it; `take` is restored by the next iteration's `clear`.
            u_panel_cols[s] = core::mem::take(&mut c_cols);
        }

        // Convert L's original row indices to positions and sort both factors.
        for j in 0..n {
            for (row, _) in &mut l_cols[j] {
                *row = row_perm_inv[*row];
            }
            l_cols[j].sort_by_key(|(row, _)| *row);
            if l_cols[j].first().map(|(row, _)| *row) != Some(j) {
                return Err(SparseError::InvalidStructure);
            }
        }
        for c in 0..n {
            u_cols[c].sort_by_key(|(row, _)| *row);
            if u_cols[c].last().map(|(row, _)| *row) != Some(c) {
                return Err(SparseError::InvalidStructure);
            }
        }

        let mut l_col_ptr = vec![0usize; n + 1];
        let mut l_row_idx = Vec::with_capacity(l_cols.iter().map(|c| c.len()).sum());
        let mut l_val = Vec::with_capacity(l_row_idx.capacity());
        for j in 0..n {
            l_col_ptr[j + 1] = l_col_ptr[j] + l_cols[j].len();
            for &(row, val) in &l_cols[j] {
                l_row_idx.push(row);
                l_val.push(val);
            }
        }
        let mut u_col_ptr = vec![0usize; n + 1];
        let mut u_row_idx = Vec::with_capacity(u_cols.iter().map(|c| c.len()).sum());
        let mut u_val = Vec::with_capacity(u_row_idx.capacity());
        for c in 0..n {
            u_col_ptr[c + 1] = u_col_ptr[c] + u_cols[c].len();
            for &(row, val) in &u_cols[c] {
                u_row_idx.push(row);
                u_val.push(val);
            }
        }

        Ok(SupernodalLu {
            n,
            l_col_ptr,
            l_row_idx,
            l_val,
            u_col_ptr,
            u_row_idx,
            u_val,
            row_perm,
            col_perm_inv: self.col_perm_inv.clone(),
        })
    }
}

/// Index of the supernode that owns `col`.
fn supernode_of(ptr: &[usize], col: usize) -> usize {
    ptr.partition_point(|&p| p <= col) - 1
}

/// Partial-pivoting LU of the first `nfactor` columns of an `nrows x ncols`
/// column-major front, swapping rows across all `ncols` columns.
fn lu_front<T: SparseScalar>(a: &mut [T], nrows: usize, nfactor: usize, ncols: usize, ipiv: &mut [usize]) -> Result<(), SparseError> {
    for k in 0..nfactor {
        let mut pivot = k;
        let mut best = a[k + k * nrows].abs();
        for i in k + 1..nrows {
            let cand = a[i + k * nrows].abs();
            if cand > best {
                best = cand;
                pivot = i;
            }
        }
        if best == 0.0 {
            return Err(SparseError::Singular);
        }
        if pivot != k {
            for j in 0..ncols {
                a.swap(j * nrows + k, j * nrows + pivot);
            }
        }
        ipiv[k] = pivot;
        let d = a[k + k * nrows];
        let d_inv = d.recip();
        // The front is column-major, so a column is contiguous and a row is not: the elimination must
        // run over the columns outermost and down each column innermost, which keeps both operands of
        // every update in one sequential run (and lets the vectorizer see it). Row k and column k are
        // fixed for this step, so swapping the loops is exact -- the same products, in the same order
        // per element, with no reassociation of any sum.
        for i in k + 1..nrows {
            let factor = a[i + k * nrows] * d_inv;
            a[i + k * nrows] = factor;
        }
        for j in k + 1..ncols {
            let akj = a[k + j * nrows];
            // column j comes after column k, so one split gives both halves without unsafe
            let (head, tail) = a.split_at_mut(j * nrows);
            let col = &mut tail[..nrows];
            let pivot_col = &head[k * nrows..k * nrows + nrows];
            for i in k + 1..nrows {
                col[i] -= akj * pivot_col[i];
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::csc::CscPattern;
    use crate::sparse::symbolic::SymbolicLu;

    fn from_cols(_n: usize, cols: &[&[usize]]) -> (Vec<usize>, Vec<usize>) {
        let mut col_ptr = vec![0];
        let mut row_idx = Vec::new();
        for c in cols {
            let mut c = c.to_vec();
            c.sort_unstable();
            c.dedup();
            row_idx.extend_from_slice(&c);
            col_ptr.push(row_idx.len());
        }
        (col_ptr, row_idx)
    }

    fn tridiag(n: usize) -> (Vec<usize>, Vec<usize>) {
        let cols: Vec<Vec<usize>> = (0..n)
            .map(|j| {
                let mut v = vec![j];
                if j > 0 {
                    v.push(j - 1);
                }
                if j + 1 < n {
                    v.push(j + 1);
                }
                v
            })
            .collect();
        let refs: Vec<&[usize]> = cols.iter().map(|v| v.as_slice()).collect();
        from_cols(n, &refs)
    }

    fn arrow(n: usize) -> (Vec<usize>, Vec<usize>) {
        let mut cols = Vec::new();
        for j in 0..n {
            if j == 0 {
                cols.push((0..n).collect::<Vec<_>>());
            } else {
                cols.push(vec![0, j]);
            }
        }
        let refs: Vec<&[usize]> = cols.iter().map(|v| v.as_slice()).collect();
        from_cols(n, &refs)
    }

    fn grid(rows: usize, cols_n: usize) -> (Vec<usize>, Vec<usize>) {
        let n = rows * cols_n;
        let mut cols = Vec::new();
        for i in 0..n {
            let r = i / cols_n;
            let c = i % cols_n;
            let mut v = vec![i];
            if r > 0 {
                v.push((r - 1) * cols_n + c);
            }
            if r + 1 < rows {
                v.push((r + 1) * cols_n + c);
            }
            if c > 0 {
                v.push(r * cols_n + c - 1);
            }
            if c + 1 < cols_n {
                v.push(r * cols_n + c + 1);
            }
            cols.push(v);
        }
        let refs: Vec<&[usize]> = cols.iter().map(|v| v.as_slice()).collect();
        from_cols(n, &refs)
    }

    fn diagonally_dominant(n: usize, seed: u64) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / 2147483648.0
        };
        let mut cols = vec![Vec::new(); n];
        for j in 0..n {
            cols[j].push(j);
            for _ in 0..2 {
                let i = ((next() * 1e6) as usize) % n;
                if i != j {
                    cols[j].push(i);
                }
            }
        }
        let mut col_ptr = vec![0];
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        for j in 0..n {
            cols[j].sort_unstable();
            cols[j].dedup();
            for &i in &cols[j] {
                row_idx.push(i);
                if i == j {
                    values.push(8.0 + next());
                } else {
                    values.push(0.5 - next());
                }
            }
            col_ptr.push(row_idx.len());
        }
        (col_ptr, row_idx, values)
    }

    fn diagonal_values(n: usize, col_ptr: &[usize], row_idx: &[usize], diag: f64, off: f64) -> Vec<f64> {
        let mut values = Vec::with_capacity(row_idx.len());
        for j in 0..n {
            for k in col_ptr[j]..col_ptr[j + 1] {
                values.push(if row_idx[k] == j { diag } else { off });
            }
        }
        values
    }

    fn dense(n: usize, col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> Vec<f64> {
        let mut a = vec![0.0; n * n];
        for j in 0..n {
            for k in col_ptr[j]..col_ptr[j + 1] {
                a[row_idx[k] + j * n] = values[k];
            }
        }
        a
    }

    fn assert_close(got: &[f64], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() <= tol * (1.0 + w.abs()), "entry {i}: {g} vs {w}");
        }
    }

    fn solve_both(n: usize, col_ptr: &[usize], row_idx: &[usize], values: &[f64], b: &[f64]) {
        let pattern = CscPattern::new(n, n, col_ptr, row_idx).unwrap();
        let a = CscView::new(n, n, col_ptr, row_idx, values).unwrap();

        let simplicial = SymbolicLu::analyze(pattern).unwrap().factorize(a).unwrap();
        let supernodal = SymbolicSupernodal::analyze(pattern).unwrap().factorize(a).unwrap();

        let mut bs = b.to_vec();
        simplicial.solve(&mut bs).unwrap();
        let mut bn = b.to_vec();
        supernodal.solve(&mut bn).unwrap();
        assert_close(&bn, &bs, 1e-9);
    }

    #[test]
    fn solves_a_two_by_two() {
        solve_both(2, &[0, 2, 4], &[0, 1, 0, 1], &[4.0, 1.0, 1.0, 3.0], &[1.0, 2.0]);
    }

    #[test]
    fn solves_structured_patterns() {
        for (n, (cp, ri)) in [
            (1, tridiag(1)),
            (4, grid(2, 2)),
            (6, grid(2, 3)),
            (9, grid(3, 3)),
            (5, tridiag(5)),
            (8, arrow(8)),
            (12, grid(3, 4)),
        ] {
            let values = diagonal_values(n, &cp, &ri, 5.0, 1.0);
            let b: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
            solve_both(n, &cp, &ri, &values, &b);
        }
    }

    #[test]
    fn reconstruction_paq_eq_lu() {
        for (n, (cp, ri)) in [(6, tridiag(6)), (7, arrow(7)), (12, grid(3, 4))] {
            let values = diagonal_values(n, &cp, &ri, 5.0, 1.0);
            let a = dense(n, &cp, &ri, &values);
            let pattern = CscPattern::new(n, n, &cp, &ri).unwrap();
            let symbolic = SymbolicSupernodal::analyze(pattern).unwrap();
            let view = CscView::new(n, n, &cp, &ri, &values).unwrap();
            let f = symbolic.factorize(view).unwrap();

            let mut l = vec![0.0; n * n];
            for j in 0..n {
                for k in f.l_col_ptr()[j]..f.l_col_ptr()[j + 1] {
                    l[f.l_row_idx()[k] + j * n] = f.l_values()[k];
                }
            }
            let mut u = vec![0.0; n * n];
            for j in 0..n {
                for k in f.u_col_ptr()[j]..f.u_col_ptr()[j + 1] {
                    u[f.u_row_idx()[k] + j * n] = f.u_values()[k];
                }
            }
            for j in 0..n {
                for i in 0..n {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += l[i + k * n] * u[k + j * n];
                    }
                    let r = f.row_perm()[i];
                    let c = symbolic.col_perm[j];
                    assert!((sum - a[r + c * n]).abs() < 1e-10, "({i},{j})");
                }
            }
        }
    }

    #[test]
    fn solves_a_random_diagonally_dominant_system() {
        let n = 16;
        let (cp, ri, values) = diagonally_dominant(n, 0x1234_5678);
        let x: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
        let a = dense(n, &cp, &ri, &values);
        let mut b = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                b[i] += a[i + j * n] * x[j];
            }
        }
        solve_both(n, &cp, &ri, &values, &b);

        // Exactness: the supernodal solve recovers the chosen solution.
        let pattern = CscPattern::new(n, n, &cp, &ri).unwrap();
        let view = CscView::new(n, n, &cp, &ri, &values).unwrap();
        let f = SymbolicSupernodal::analyze(pattern).unwrap().factorize(view).unwrap();
        f.solve(&mut b).unwrap();
        assert_close(&b, &x, 1e-9);
    }

    #[test]
    fn solves_a_complex_system() {
        use crate::cpx;
        let n = 4;
        let mut cols: Vec<Vec<usize>> = vec![Vec::new(); n];
        for j in 0..n {
            cols[j].push(j);
            if j > 0 {
                cols[j].push(j - 1);
            }
            if j + 1 < n {
                cols[j].push(j + 1);
            }
        }
        let refs: Vec<&[usize]> = cols.iter().map(|v| v.as_slice()).collect();
        let (cp, ri) = from_cols(n, &refs);
        let mut values = Vec::new();
        for j in 0..n {
            for k in cp[j]..cp[j + 1] {
                let i = ri[k];
                if i == j {
                    values.push(cpx!(6.0, 1.0));
                } else {
                    values.push(cpx!(0.5, -0.25));
                }
            }
        }
        let pattern = CscPattern::new(n, n, &cp, &ri).unwrap();
        let view = CscView::new(n, n, &cp, &ri, &values).unwrap();
        let f = SymbolicSupernodal::analyze(pattern).unwrap().factorize(view).unwrap();
        let x: Vec<crate::Complex64> = (0..n).map(|i| cpx!(i as f64 + 1.0, -0.5)).collect();
        let mut b = vec![crate::Complex64::new(0.0, 0.0); n];
        for j in 0..n {
            for k in cp[j]..cp[j + 1] {
                b[ri[k]] += values[k] * x[j];
            }
        }
        f.solve(&mut b).unwrap();
        for i in 0..n {
            assert!((b[i] - x[i]).norm() < 1e-12);
        }
    }

    #[test]
    fn rejects_a_wrong_shape_and_a_singular_matrix() {
        let pattern = CscPattern::new(2, 2, &[0, 2, 4], &[0, 1, 0, 1]).unwrap();
        let symbolic = SymbolicSupernodal::analyze(pattern).unwrap();
        let view = CscView::new(2, 2, &[0, 2, 4], &[0, 1, 0, 1], &[1.0, 2.0, 2.0, 4.0]).unwrap();
        assert_eq!(symbolic.factorize(view).unwrap_err(), SparseError::Singular);

        let other = CscView::new(2, 2, &[0, 2, 3], &[0, 1, 1], &[1.0, 1.0, 1.0]).unwrap();
        assert_eq!(symbolic.factorize(other).unwrap_err(), SparseError::StructureMismatch);

        let mut b = [1.0, 2.0, 3.0];
        let f = {
            let pattern = CscPattern::new(2, 2, &[0, 2, 4], &[0, 1, 0, 1]).unwrap();
            let view = CscView::new(2, 2, &[0, 2, 4], &[0, 1, 0, 1], &[4.0, 1.0, 1.0, 3.0]).unwrap();
            SymbolicSupernodal::analyze(pattern).unwrap().factorize(view).unwrap()
        };
        assert_eq!(f.solve(&mut b).unwrap_err(), SparseError::BadShape);
    }

    #[test]
    fn rejects_nan_and_infinite_entries() {
        let pattern = CscPattern::new(2, 2, &[0, 2, 4], &[0, 1, 0, 1]).unwrap();
        let symbolic = SymbolicSupernodal::analyze(pattern).unwrap();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let values = [4.0, 1.0, bad, 3.0];
            let a = CscView::new(2, 2, &[0, 2, 4], &[0, 1, 0, 1], &values).unwrap();
            assert_eq!(symbolic.factorize(a).unwrap_err(), SparseError::NotFinite);
        }
    }

    #[test]
    fn handles_zero_and_one() {
        let pattern = CscPattern::new(0, 0, &[0], &[]).unwrap();
        let view = CscView::new(0, 0, &[0], &[], &[] as &[f64]).unwrap();
        let f = SymbolicSupernodal::analyze(pattern).unwrap().factorize(view).unwrap();
        let mut b: [f64; 0] = [];
        f.solve(&mut b).unwrap();

        let pattern = CscPattern::new(1, 1, &[0, 1], &[0]).unwrap();
        let view = CscView::new(1, 1, &[0, 1], &[0], &[2.0]).unwrap();
        let f = SymbolicSupernodal::analyze(pattern).unwrap().factorize(view).unwrap();
        let mut b = [6.0];
        f.solve(&mut b).unwrap();
        assert!((b[0] - 3.0).abs() < 1e-14);
    }
}
