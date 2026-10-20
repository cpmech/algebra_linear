//! Supernodal symbolic analysis.
//!
//! The symbolic structure of a supernodal factorization is the column
//! elimination tree, the column counts of the Cholesky factor of `AᵀA`, and the
//! partition of the columns into supernodes. This module computes those three
//! from a CSC pattern and a column permutation; the numeric factorization that
//! consumes them is not part of this module yet.
//!
//! Translated from faer commit `ddbd2ae`: `ghost_column_counts_aat` and
//! `ghost_least_common_ancestor` from `sparse/linalg/qr.rs`, and the fundamental
//! supernode partition from `sparse/linalg/cholesky.rs`
//! (`ghost_factorize_supernodal_symbolic` without the relaxation pass).

use crate::sparse::csc::CscPattern;
use crate::sparse::etree;
use crate::sparse::ordering::colamd::{ColamdControl, colamd_order};

pub mod lu;

use crate::sparse::symbolic::fingerprint;
use crate::sparse::{SparseError, perm};
pub use lu::SupernodalLu;

/// The symbolic structure of a supernodal LU factorization.
///
/// The partition is the **fundamental** one: faer's optional relaxation
/// (amalgamating a child into its parent to trade fill for larger panels) is not
/// applied yet.
#[derive(Clone, Debug)]
pub struct SymbolicSupernodal {
    n: usize,
    nnz: usize,
    fingerprint: u64,
    col_counts: Vec<usize>,
    etree: Vec<isize>,
    col_perm: Vec<usize>,
    col_perm_inv: Vec<usize>,
    supernode_ptr: Vec<usize>,
    super_etree: Vec<isize>,
}

impl SymbolicSupernodal {
    /// Analyzes `pattern` with a COLAMD column ordering.
    ///
    /// # Errors
    ///
    /// [`SparseError::BadShape`] if the matrix is not square, plus the ordering
    /// errors of [`colamd_order`].
    pub fn analyze(pattern: CscPattern<'_>) -> Result<Self, SparseError> {
        if pattern.nrows() != pattern.ncols() {
            return Err(SparseError::BadShape);
        }
        let ordering = colamd_order(pattern, ColamdControl::default())?;
        Self::analyze_with_permutation(pattern, ordering.perm(), ordering.perm_inv())
    }

    /// Analyzes `pattern` with the given column permutation.
    ///
    /// `col_perm` maps a new position to an original column and `col_perm_inv`
    /// is its inverse.
    ///
    /// # Errors
    ///
    /// [`SparseError::BadShape`] if the matrix is not square or either argument
    /// is not a permutation of the columns.
    pub fn analyze_with_permutation(pattern: CscPattern<'_>, col_perm: &[usize], col_perm_inv: &[usize]) -> Result<Self, SparseError> {
        if pattern.nrows() != pattern.ncols() {
            return Err(SparseError::BadShape);
        }
        let n = pattern.ncols();
        if col_perm.len() != n || col_perm_inv.len() != n || !perm::is_permutation(col_perm) || !perm::is_permutation(col_perm_inv) {
            return Err(SparseError::BadShape);
        }
        let (at_col_ptr, at_row_idx) = transpose_pattern(pattern);
        let at = CscPattern::new(n, n, &at_col_ptr, &at_row_idx)?;

        let mut etree = vec![0isize; n];
        etree::col_etree(pattern, Some(col_perm), &mut etree);
        let mut post = vec![0usize; n];
        etree::postorder(&mut post, &etree);
        let col_counts = column_counts_aat(at, col_perm_inv, &etree, &post);
        let (supernode_ptr, super_etree) = fundamental_supernodes(n, &etree, &col_counts);
        Ok(Self {
            n,
            nnz: pattern.nnz(),
            fingerprint: fingerprint(pattern),
            col_counts,
            etree,
            col_perm: col_perm.to_vec(),
            col_perm_inv: col_perm_inv.to_vec(),
            supernode_ptr,
            super_etree,
        })
    }

    /// Returns the order of the matrix.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of stored entries the analysis saw.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// Returns `true` if `pattern` is the one this analysis was built for.
    pub fn matches(&self, pattern: CscPattern<'_>) -> bool {
        pattern.nrows() == self.n && pattern.ncols() == self.n && pattern.nnz() == self.nnz && fingerprint(pattern) == self.fingerprint
    }

    /// Returns the number of supernodes.
    #[inline]
    pub fn n_supernodes(&self) -> usize {
        self.supernode_ptr.len() - 1
    }

    /// Returns the column boundaries: supernode `s` owns columns
    /// `supernode_ptr[s]..supernode_ptr[s + 1]`.
    #[inline]
    pub fn supernode_ptr(&self) -> &[usize] {
        &self.supernode_ptr
    }

    /// Returns the supernode elimination tree; `-1` marks a root.
    #[inline]
    pub fn super_etree(&self) -> &[isize] {
        &self.super_etree
    }

    /// Returns the column counts of the Cholesky factor of `AᵀA`, including the
    /// diagonal.
    #[inline]
    pub fn col_counts(&self) -> &[usize] {
        &self.col_counts
    }

    /// Returns the column elimination tree; `-1` marks a root.
    #[inline]
    pub fn etree(&self) -> &[isize] {
        &self.etree
    }

    /// Returns the column permutation, mapping a new position to an original
    /// column.
    #[inline]
    pub fn col_perm(&self) -> &[usize] {
        &self.col_perm
    }

    /// Returns the inverse column permutation.
    #[inline]
    pub fn col_perm_inv(&self) -> &[usize] {
        &self.col_perm_inv
    }

    /// Returns the number of columns in supernode `s`.
    ///
    /// # Panics
    ///
    /// Panics if `s >= n_supernodes()`.
    #[inline]
    pub fn supernode_size(&self, s: usize) -> usize {
        self.supernode_ptr[s + 1] - self.supernode_ptr[s]
    }

    /// Returns a copy with adjacent supernodes amalgamated.
    ///
    /// A child is merged into its parent when the two are adjacent in the
    /// ordering, the child has at most `max_child_size` columns, and the extra
    /// zeros the merge introduces are at most `fill_tolerance` times the
    /// expanded diagonal-block size. The pass is a simplified version of faer's
    /// relaxation: it keeps faer's fill estimate but drops the child-list
    /// search, so it considers only the immediately preceding supernode.
    ///
    /// The column counts and the elimination tree are unchanged; only the
    /// partition and the supernode tree are rebuilt.
    pub fn with_relaxation(mut self, max_child_size: usize, fill_tolerance: f64) -> Self {
        let n = self.n;
        if n == 0 {
            return self;
        }
        let ns = self.n_supernodes();
        let mut sizes: Vec<usize> = (0..ns).map(|s| self.supernode_ptr[s + 1] - self.supernode_ptr[s]).collect();
        let degrees: Vec<usize> = (0..ns)
            .map(|s| self.col_counts[self.supernode_ptr[s + 1] - 1].saturating_sub(1))
            .collect();
        let mut alive = vec![true; ns];
        for parent in 1..ns {
            let child = parent - 1;
            if !alive[child] || sizes[child] > max_child_size {
                continue;
            }
            let combined = sizes[parent] + sizes[child];
            let degree_gain = degrees[parent].saturating_sub(degrees[child]);
            let extra_zeros = (sizes[parent] + degree_gain) * sizes[child];
            let expanded = combined * (combined + 1) / 2 + degrees[parent] * combined;
            if extra_zeros as f64 <= fill_tolerance * expanded as f64 {
                sizes[parent] = combined;
                alive[child] = false;
            }
        }
        let mut supernode_ptr = Vec::with_capacity(ns + 1);
        supernode_ptr.push(0);
        for s in 0..ns {
            if alive[s] {
                supernode_ptr.push(supernode_ptr.last().copied().unwrap_or(0) + sizes[s]);
            }
        }
        let n_supernodes = supernode_ptr.len() - 1;
        let mut index_to_super = vec![0usize; n];
        for s in 0..n_supernodes {
            for c in supernode_ptr[s]..supernode_ptr[s + 1] {
                index_to_super[c] = s;
            }
        }
        let mut super_etree = vec![-1isize; n_supernodes];
        for s in 0..n_supernodes {
            let last = supernode_ptr[s + 1] - 1;
            if self.etree[last] >= 0 {
                super_etree[s] = index_to_super[self.etree[last] as usize] as isize;
            }
        }
        self.supernode_ptr = supernode_ptr;
        self.super_etree = super_etree;
        self
    }
}

/// Transposes a CSC pattern with a counting sort, producing sorted row indices.
fn transpose_pattern(pattern: CscPattern<'_>) -> (Vec<usize>, Vec<usize>) {
    let n = pattern.ncols();
    let m = pattern.nrows();
    let mut counts = vec![0usize; m];
    for &row in pattern.row_idx() {
        counts[row] += 1;
    }
    let mut col_ptr = vec![0usize; m + 1];
    for i in 0..m {
        col_ptr[i + 1] = col_ptr[i] + counts[i];
    }
    let mut next = col_ptr.clone();
    let mut row_idx = vec![0usize; pattern.nnz()];
    for j in 0..n {
        for &row in pattern.row_idx_of_col(j) {
            row_idx[next[row]] = j;
            next[row] += 1;
        }
    }
    (col_ptr, row_idx)
}

/// The least common ancestor step of faer's column-count algorithm.
fn least_common_ancestor(i: usize, j: usize, first: &[isize], max_first: &mut [isize], prev_leaf: &mut [isize], ancestor: &mut [usize]) -> isize {
    if i <= j || first[j] <= max_first[i] {
        return -2;
    }
    max_first[i] = first[j];
    let j_prev = prev_leaf[i];
    prev_leaf[i] = j as isize;
    if j_prev < 0 {
        return -1;
    }
    let mut lca = j_prev as usize;
    while lca != ancestor[lca] {
        lca = ancestor[lca];
    }
    let mut node = j_prev as usize;
    while node != lca {
        let next = ancestor[node];
        ancestor[node] = lca;
        node = next;
    }
    lca as isize
}

/// Column counts of the Cholesky factor of `AᵀA`, where `at` is the pattern of
/// `Aᵀ`. Translated from faer's `ghost_column_counts_aat`.
fn column_counts_aat(at: CscPattern<'_>, perm_inv: &[usize], etree: &[isize], post: &[usize]) -> Vec<usize> {
    let m = at.nrows();
    let n = at.ncols();
    debug_assert_eq!(m, n);
    let mut delta = vec![0isize; m];
    let mut first = vec![-1isize; m];
    let mut max_first = vec![-1isize; m];
    let mut prev_leaf = vec![-1isize; m];
    let mut ancestor = (0..m).collect::<Vec<usize>>();
    let mut next = vec![-1isize; n];
    let mut head = vec![-1isize; m];

    let mut post_inv = vec![0usize; m];
    for (i, &p) in post.iter().enumerate() {
        post_inv[p] = i;
    }

    // Bucket the columns by the postorder position of their first row.
    for j in 0..n {
        let mut min_row = usize::MAX;
        for &row in at.row_idx_of_col(j) {
            let row = perm_inv[row];
            let p = post_inv[row];
            if p < min_row {
                min_row = p;
            }
        }
        if min_row != usize::MAX {
            next[j] = head[min_row];
            head[min_row] = j as isize;
        }
    }

    // Initialize the counts and the `first` array along each etree root path.
    for k in 0..m {
        let mut pk = post[k];
        delta[pk] = if first[pk] < 0 { 1 } else { 0 };
        loop {
            if first[pk] >= 0 {
                break;
            }
            first[pk] = k as isize;
            if etree[pk] >= 0 {
                pk = etree[pk] as usize;
            } else {
                break;
            }
        }
    }

    // The LCA pass.
    for k in 0..m {
        let pk = post[k];
        if etree[pk] >= 0 {
            delta[etree[pk] as usize] -= 1;
        }
        let mut j = head[k];
        head[k] = -1;
        while j >= 0 {
            let jj = j as usize;
            for &row in at.row_idx_of_col(jj) {
                let i = perm_inv[row];
                let lca = least_common_ancestor(i, pk, &first, &mut max_first, &mut prev_leaf, &mut ancestor);
                if lca != -2 {
                    delta[pk] += 1;
                    if lca >= 0 {
                        delta[lca as usize] -= 1;
                    }
                }
            }
            j = next[jj];
        }
        if etree[pk] >= 0 {
            ancestor[pk] = etree[pk] as usize;
        }
    }

    // Accumulate the deltas up the tree.
    for k in 0..m {
        if etree[k] >= 0 {
            delta[etree[k] as usize] += delta[k];
        }
    }

    delta
        .into_iter()
        .map(|d| {
            debug_assert!(d >= 0, "column count became negative");
            d as usize
        })
        .collect()
}

/// Partitions the columns into fundamental supernodes.
///
/// A column joins the previous supernode when it is the only child of its
/// parent in the etree and its column count is one more than the next column's.
fn fundamental_supernodes(n: usize, etree: &[isize], col_counts: &[usize]) -> (Vec<usize>, Vec<isize>) {
    if n == 0 {
        return (vec![0], Vec::new());
    }
    let mut child_count = vec![0usize; n];
    for &parent in etree {
        if parent >= 0 {
            child_count[parent as usize] += 1;
        }
    }
    let mut sizes = vec![0usize; n];
    sizes[0] = 1;
    let mut current = 0usize;
    for j in 1..n {
        let prev = j - 1;
        let is_parent_of_prev = etree[prev] == j as isize;
        let is_parent_of_only_prev = child_count[j] == 1;
        let same_pattern_as_prev = col_counts[prev] == col_counts[j] + 1;
        if !(is_parent_of_prev && is_parent_of_only_prev && same_pattern_as_prev) {
            current += 1;
        }
        sizes[current] += 1;
    }
    let n_supernodes = current + 1;
    sizes.truncate(n_supernodes);
    let mut supernode_ptr = vec![0usize; n_supernodes + 1];
    for s in 0..n_supernodes {
        supernode_ptr[s + 1] = supernode_ptr[s] + sizes[s];
    }
    let mut index_to_super = vec![0usize; n];
    for s in 0..n_supernodes {
        for c in supernode_ptr[s]..supernode_ptr[s + 1] {
            index_to_super[c] = s;
        }
    }
    let mut super_etree = vec![-1isize; n_supernodes];
    for s in 0..n_supernodes {
        let last = supernode_ptr[s + 1] - 1;
        if etree[last] >= 0 {
            super_etree[s] = index_to_super[etree[last] as usize] as isize;
        }
    }
    (supernode_ptr, super_etree)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_cols(_n: usize, cols: &[&[usize]]) -> (Vec<usize>, Vec<usize>) {
        let mut col_ptr = vec![0usize];
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

    fn tridiag5() -> (Vec<usize>, Vec<usize>) {
        from_cols(5, &[&[0, 1], &[0, 1, 2], &[1, 2, 3], &[2, 3, 4], &[3, 4]])
    }

    fn arrow8() -> (Vec<usize>, Vec<usize>) {
        let n = 8;
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

    fn grid4x5() -> (Vec<usize>, Vec<usize>) {
        let (rows, cols_n) = (4, 5);
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

    /// A dense boolean pattern of `(A Q)ᵀ (A Q)`.
    fn permuted_ata(pattern: CscPattern<'_>, perm: &[usize]) -> Vec<bool> {
        let n = pattern.ncols();
        let mut a = vec![false; n * n];
        for j in 0..n {
            for &row in pattern.row_idx_of_col(j) {
                a[row + j * n] = true;
            }
        }
        let mut b = vec![false; n * n];
        for j in 0..n {
            for i in 0..n {
                let mut any = i == j;
                for k in 0..n {
                    if a[k + perm[i] * n] && a[k + perm[j] * n] {
                        any = true;
                        break;
                    }
                }
                b[i + j * n] = any;
            }
        }
        b
    }

    /// Symbolic Cholesky column counts by dense elimination.
    fn brute_force_counts(mut b: Vec<bool>, n: usize) -> Vec<usize> {
        let mut counts = vec![0usize; n];
        for j in 0..n {
            let col: Vec<usize> = (j..n).filter(|&i| b[i + j * n]).collect();
            counts[j] = col.len();
            for &i in &col {
                if i > j {
                    for &k in &col {
                        if k >= i {
                            b[k + i * n] = true;
                        }
                    }
                }
            }
        }
        counts
    }

    fn check_counts(n: usize, col_ptr: &[usize], row_idx: &[usize], perm: &[usize]) {
        let pattern = CscPattern::new(n, n, col_ptr, row_idx).unwrap();
        let perm_inv = perm::invert(perm).unwrap();
        let s = SymbolicSupernodal::analyze_with_permutation(pattern, perm, &perm_inv).unwrap();
        let expected = brute_force_counts(permuted_ata(pattern, perm), n);
        assert_eq!(s.col_counts(), expected.as_slice(), "counts for perm {perm:?}");
    }

    #[test]
    fn counts_match_dense_symbolic_elimination() {
        let (cp, ri) = tridiag5();
        check_counts(5, &cp, &ri, &[0, 1, 2, 3, 4]);
        check_counts(5, &cp, &ri, &[4, 3, 2, 1, 0]);
        check_counts(5, &cp, &ri, &[2, 0, 4, 1, 3]);

        let (cp, ri) = arrow8();
        check_counts(8, &cp, &ri, &[0, 1, 2, 3, 4, 5, 6, 7]);

        let (cp, ri) = grid4x5();
        check_counts(20, &cp, &ri, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
        // The COLAMD permutation the analyzer will actually use.
        let pattern = CscPattern::new(20, 20, &cp, &ri).unwrap();
        let ordering = colamd_order(pattern, ColamdControl::default()).unwrap();
        check_counts(20, &cp, &ri, ordering.perm());
    }

    #[test]
    fn supernodes_partition_the_columns() {
        let cases = [tridiag5(), arrow8(), grid4x5()];
        for (cp, ri) in cases {
            let n = cp.len() - 1;
            let pattern = CscPattern::new(n, n, &cp, &ri).unwrap();
            let s = SymbolicSupernodal::analyze(pattern).unwrap();
            assert_eq!(s.supernode_ptr()[0], 0);
            assert_eq!(*s.supernode_ptr().last().unwrap(), n);
            let mut sizes = Vec::new();
            for i in 0..s.n_supernodes() {
                let size = s.supernode_size(i);
                assert!(size > 0);
                sizes.push(size);
            }
            assert_eq!(sizes.iter().sum::<usize>(), n);
            for (i, &p) in s.super_etree().iter().enumerate() {
                assert!(p == -1 || p > i as isize, "supernode parent after");
            }
            // The supernode tree is the etree of the last column of each node.
            for i in 0..s.n_supernodes() {
                let last = s.supernode_ptr()[i + 1] - 1;
                let parent = s.etree()[last];
                if parent < 0 {
                    assert_eq!(s.super_etree()[i], -1);
                } else {
                    assert_eq!(s.super_etree()[i], supernode_of(s.supernode_ptr(), parent as usize) as isize);
                }
            }
        }
    }

    fn supernode_of(ptr: &[usize], col: usize) -> usize {
        ptr.partition_point(|&p| p <= col) - 1
    }

    #[test]
    fn diagonal_matrix_has_singleton_supernodes() {
        let n = 6;
        let col_ptr: Vec<usize> = (0..=n).collect();
        let row_idx: Vec<usize> = (0..n).collect();
        let pattern = CscPattern::new(n, n, &col_ptr, &row_idx).unwrap();
        let s = SymbolicSupernodal::analyze(pattern).unwrap();
        assert_eq!(s.n_supernodes(), n);
        for i in 0..n {
            assert_eq!(s.supernode_size(i), 1);
            assert_eq!(s.super_etree()[i], -1);
            assert_eq!(s.col_counts()[i], 1);
        }
    }

    #[test]
    fn empty_matrix_has_no_supernodes() {
        let pattern = CscPattern::new(0, 0, &[0], &[]).unwrap();
        let s = SymbolicSupernodal::analyze(pattern).unwrap();
        assert_eq!(s.n_supernodes(), 0);
        assert_eq!(s.supernode_ptr(), &[0]);
    }

    #[test]
    fn relaxation_merges_adjacent_supernodes() {
        let (cp, ri) = grid4x5();
        let n = cp.len() - 1;
        let pattern = CscPattern::new(n, n, &cp, &ri).unwrap();
        let base = SymbolicSupernodal::analyze(pattern).unwrap();
        let relaxed = base.clone().with_relaxation(2, 0.5);
        assert!(relaxed.n_supernodes() <= base.n_supernodes());
        assert_eq!(relaxed.supernode_ptr()[0], 0);
        assert_eq!(*relaxed.supernode_ptr().last().unwrap(), n);
        assert_eq!(relaxed.col_counts(), base.col_counts());
        assert_eq!(relaxed.etree(), base.etree());
        assert_eq!(relaxed.supernode_ptr()[relaxed.n_supernodes()], n);
        for i in 0..relaxed.n_supernodes() {
            assert!(relaxed.supernode_size(i) > 0);
            let parent = relaxed.super_etree()[i];
            assert!(parent == -1 || parent > i as isize);
        }
        // At least one merge has to happen for a cutoff this generous.
        assert!(relaxed.n_supernodes() < base.n_supernodes());
    }

    #[test]
    fn relaxed_supernodes_solve_like_the_simplicial_path() {
        let (cp, ri) = grid4x5();
        let n = cp.len() - 1;
        // Use the diagonal check per column instead of the row index heuristic.
        let mut values = vec![0.0; ri.len()];
        for j in 0..n {
            for k in cp[j]..cp[j + 1] {
                values[k] = if ri[k] == j { 5.0 } else { 1.0 };
            }
        }
        let pattern = CscPattern::new(n, n, &cp, &ri).unwrap();
        let view = crate::sparse::CscView::new(n, n, &cp, &ri, &values).unwrap();
        let simplicial = crate::sparse::SymbolicLu::analyze(pattern).unwrap();
        let mut b0 = vec![1.0; n];
        simplicial.factorize(view).unwrap().solve(&mut b0).unwrap();

        let relaxed = SymbolicSupernodal::analyze(pattern).unwrap().with_relaxation(2, 0.5);
        let mut b1 = vec![1.0; n];
        relaxed.factorize(view).unwrap().solve(&mut b1).unwrap();
        for i in 0..n {
            assert!((b0[i] - b1[i]).abs() < 1e-10, "entry {i}");
        }
    }

    #[test]
    fn rejects_bad_shapes_and_permutations() {
        let pattern = CscPattern::new(2, 3, &[0, 0, 0, 0], &[]).unwrap();
        assert_eq!(SymbolicSupernodal::analyze(pattern).unwrap_err(), SparseError::BadShape);

        let pattern = CscPattern::new(2, 2, &[0, 1, 2], &[0, 1]).unwrap();
        assert_eq!(
            SymbolicSupernodal::analyze_with_permutation(pattern, &[0, 0], &[0, 1]).unwrap_err(),
            SparseError::BadShape
        );
        assert_eq!(
            SymbolicSupernodal::analyze_with_permutation(pattern, &[0, 1], &[0, 1, 2]).unwrap_err(),
            SparseError::BadShape
        );
    }
}
