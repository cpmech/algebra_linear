//! The COLAMD column ordering.
//!
//! Translated from `faer/src/sparse/linalg/colamd.rs` at faer commit `ddbd2ae`,
//! which is itself a transcription of SuiteSparse COLAMD (BSD 3-Clause,
//! Copyright 1998-2022 Timothy A. Davis). See `LICENSE` for both notices.
//!
//! The translation keeps faer's algorithm and data layout — signed sentinels,
//! the dense row/column counts, aggressive absorption, super-column detection
//! and garbage collection — and replaces faer's abstractions:
//!
//! | faer | here |
//! |---|---|
//! | `I: Index`, `I::Signed` | `usize`, `isize` |
//! | `SymbolicSparseColMatRef` | [`CscPattern`] |
//! | `MemStack`/`StackReq` | `Vec` allocated here |
//! | `FaerError` | [`SparseError`] |
//!
//! The returned permutation is **postordered** through the column elimination
//! tree, exactly as faer's `colamd::order` does; that postorder is what the
//! golden tests compare against.
//!
//! faer's "jumbled" input path (unsorted rows within a column) is not translated:
//! it is unreachable here because [`CscPattern`] rejects unsorted rows, and the
//! faer path itself indexes out of bounds on such input. The inputs this port
//! accepts are exactly the ones that path would have repaired.

use crate::sparse::SparseError;
use crate::sparse::csc::CscPattern;
use crate::sparse::etree::{col_etree, postorder};

/// The sentinel faer uses for "no such index".
///
/// As an `isize` it is `-1`, which is what the internal state compares against.
const NONE: usize = usize::MAX;

/// `usize` to `isize`, the translation of faer's `I::Signed::truncate`.
#[inline]
fn sx(x: usize) -> isize {
    x as isize
}

/// `isize` to `usize`, the translation of faer's `zx`.
#[inline]
fn zx(x: isize) -> usize {
    x as usize
}

/// Tuning parameters for the ordering.
///
/// The defaults match faer's `Control::default` and SuiteSparse's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColamdControl {
    /// A row is "dense" if its degree exceeds `dense_row * ncols`.
    ///
    /// A negative value disables the test.
    pub dense_row: f64,
    /// A column is "dense" if its degree exceeds `dense_col * min(nrows, ncols)`.
    ///
    /// A negative value disables the test.
    pub dense_col: f64,
    /// Whether to absorb a row that is fully covered by the pivot column.
    pub aggressive: bool,
}

impl Default for ColamdControl {
    fn default() -> Self {
        Self {
            dense_row: 0.5,
            dense_col: 0.5,
            aggressive: true,
        }
    }
}

/// A fill-reducing column ordering.
///
/// [`Self::perm`] maps a new position to the original column
/// (`perm[new] = old`); [`Self::perm_inv`] maps an original column to its new
/// position (`perm_inv[old] = new`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColamdOrdering {
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
}

impl ColamdOrdering {
    /// Returns the permutation, mapping new position to original column.
    #[inline]
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// Returns the inverse permutation, mapping original column to new position.
    #[inline]
    pub fn perm_inv(&self) -> &[usize] {
        &self.perm_inv
    }

    /// Returns the length of the permutation.
    #[inline]
    pub fn len(&self) -> usize {
        self.perm.len()
    }

    /// Returns `true` if the ordering is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.perm.is_empty()
    }
}

/// One column's state, following faer's `ColamdCol`.
#[derive(Clone, Copy, Debug)]
struct ColamdCol {
    start: isize,
    length: isize,
    shared1: isize,
    shared2: isize,
    shared3: isize,
    shared4: isize,
}

impl ColamdCol {
    #[inline]
    fn is_dead_principal(&self) -> bool {
        self.start == -1
    }

    #[inline]
    fn is_dead(&self) -> bool {
        self.start < 0
    }

    #[inline]
    fn is_alive(&self) -> bool {
        !self.is_dead()
    }

    #[inline]
    fn kill_principal(&mut self) {
        self.start = -1;
    }

    #[inline]
    fn kill_non_principal(&mut self) {
        self.start = -2;
    }
}

/// One row's state, following faer's `ColamdRow`.
#[derive(Clone, Copy, Debug)]
struct ColamdRow {
    start: isize,
    length: isize,
    shared1: isize,
    shared2: isize,
}

impl ColamdRow {
    #[inline]
    fn is_dead(&self) -> bool {
        self.shared2 < 0
    }

    #[inline]
    fn is_alive(&self) -> bool {
        !self.is_dead()
    }

    #[inline]
    fn kill(&mut self) {
        self.shared2 = -1;
    }
}

#[inline]
fn clear_mark(tag_mark: usize, max_mark: usize, row: &mut [ColamdRow]) -> usize {
    if tag_mark == 0 || tag_mark >= max_mark {
        for r in row {
            if r.is_alive() {
                r.shared2 = 0;
            }
        }
        1
    } else {
        tag_mark
    }
}

/// Computes a COLAMD ordering of `pattern`.
///
/// The matrix must be square: the postordering step walks the column elimination
/// tree over the row indices, which is only defined here for `nrows == ncols`.
///
/// # Errors
///
/// * [`SparseError::BadShape`] if the matrix is not square;
/// * [`SparseError::IndexOverflow`] if the work array size overflows `usize`.
///
/// The pattern itself is validated by [`CscPattern::new`], so malformed CSC
/// arrays are rejected before this routine runs.
pub fn colamd_order(pattern: CscPattern<'_>, control: ColamdControl) -> Result<ColamdOrdering, SparseError> {
    let m = pattern.nrows();
    let n = pattern.ncols();
    if m != n {
        return Err(SparseError::BadShape);
    }
    if n == 0 {
        return Ok(ColamdOrdering {
            perm: Vec::new(),
            perm_inv: Vec::new(),
        });
    }

    let nnz = pattern.nnz();
    let mut col = vec![
        ColamdCol {
            start: 0,
            length: 0,
            shared1: 0,
            shared2: 0,
            shared3: 0,
            shared4: 0,
        };
        n + 1
    ];
    let mut row = vec![
        ColamdRow {
            start: 0,
            length: 0,
            shared1: 0,
            shared2: 0,
        };
        m + 1
    ];
    let mut p = vec![0usize; n + 1];

    // `2*nnz + nnz/5 + n`, as in faer's `order_scratch`.
    let size = 2usize
        .checked_mul(nnz)
        .and_then(|x| x.checked_add(nnz / 5))
        .and_then(|x| x.checked_add(n))
        .ok_or(SparseError::IndexOverflow)?;
    let mut a = vec![0usize; size];

    p[0] = 0;
    for j in 0..n {
        let rows = pattern.row_idx_of_col(j);
        p[j + 1] = p[j] + rows.len();
        a[p[j]..p[j + 1]].copy_from_slice(rows);
    }

    for c in 0..n {
        col[c] = ColamdCol {
            start: sx(p[c]),
            length: sx(p[c + 1] - p[c]),
            shared1: 1,
            shared2: 0,
            shared3: -1,
            shared4: -1,
        };
    }
    col[n] = ColamdCol {
        start: -1,
        length: -1,
        shared1: -1,
        shared2: -1,
        shared3: -1,
        shared4: -1,
    };
    for r in 0..m {
        row[r] = ColamdRow {
            start: 0,
            length: 0,
            shared1: -1,
            shared2: -1,
        };
    }
    row[m] = ColamdRow {
        start: -1,
        length: -1,
        shared1: -1,
        shared2: -1,
    };

    // Row counts. The input is sorted and duplicate-free, so faer's jumbled
    // detection and repair are not needed.
    for c in 0..n {
        for k in p[c]..p[c + 1] {
            let r = a[k];
            row[r].length += 1;
            row[r].shared2 = sx(c);
        }
    }
    row[0].start = sx(p[n]);
    row[0].shared1 = row[0].start;
    row[0].shared2 = -1;
    for r in 1..m {
        row[r].start = row[r - 1].start + row[r - 1].length;
        row[r].shared1 = row[r].start;
        row[r].shared2 = -1;
    }

    // Transpose the pattern into row form. Column `c` is written to every row
    // that contains it, in increasing column order.
    for c in 0..n {
        for k in p[c]..p[c + 1] {
            let r = a[k];
            a[zx(row[r].shared1)] = c;
            row[r].shared1 += 1;
        }
    }
    for r in 0..m {
        row[r].shared2 = 0;
        row[r].shared1 = row[r].length;
    }

    let dense_row_count = if control.dense_row < 0.0 {
        n - 1
    } else {
        Ord::max(16, (control.dense_row * n as f64) as usize)
    };
    let dense_col_count = if control.dense_col < 0.0 {
        m - 1
    } else {
        Ord::max(16, (control.dense_col * Ord::min(m, n) as f64) as usize)
    };

    let mut max_deg = 0usize;
    let mut ncol2 = n;
    let head: &mut [usize] = &mut p;

    // Degenerate columns first: an empty column is placed last.
    for c in (0..n).rev() {
        let deg = col[c].length;
        if deg == 0 {
            ncol2 -= 1;
            col[c].shared2 = sx(ncol2);
            col[c].kill_principal();
        }
    }

    // Dense columns next; their rows lose one entry each.
    for c in (0..n).rev() {
        if col[c].is_dead() {
            continue;
        }
        let deg = col[c].length;
        if zx(deg) > dense_col_count {
            ncol2 -= 1;
            col[c].shared2 = sx(ncol2);
            let mut cp = zx(col[c].start);
            let cp_end = cp + zx(col[c].length);
            while cp < cp_end {
                row[a[cp]].shared1 -= 1;
                cp += 1;
            }
            col[c].kill_principal();
        }
    }

    // Dense rows are removed from the graph.
    for r in 0..m {
        let deg = zx(row[r].shared1);
        debug_assert!(deg <= n);
        if deg > dense_row_count || deg == 0 {
            row[r].kill();
        } else {
            max_deg = Ord::max(deg, max_deg);
        }
    }

    // Initial approximate degrees: score = sum of row degrees, minus one each.
    for c in (0..n).rev() {
        if col[c].is_dead() {
            continue;
        }
        let mut score = 0usize;
        let mut cp = zx(col[c].start);
        let mut new_cp = cp;
        let cp_end = cp + zx(col[c].length);
        while cp < cp_end {
            let r = a[cp];
            cp += 1;
            if row[r].is_dead() {
                continue;
            }
            a[new_cp] = r;
            new_cp += 1;
            score += zx(row[r].shared1) - 1;
            score = Ord::min(score, n);
        }
        let col_length = new_cp - zx(col[c].start);
        if col_length == 0 {
            ncol2 -= 1;
            col[c].shared2 = sx(ncol2);
            col[c].kill_principal();
        } else {
            debug_assert!(score <= n);
            col[c].length = sx(col_length);
            col[c].shared2 = sx(score);
        }
    }

    // Bucket the columns by score.
    head.fill(NONE);
    let mut min_score = n;
    for c in (0..n).rev() {
        if col[c].is_alive() {
            let score = zx(col[c].shared2);
            debug_assert!(min_score <= n);
            debug_assert!(score <= n);
            debug_assert!(sx(head[score]) >= -1);
            let next_col = head[score];
            col[c].shared3 = -1;
            col[c].shared4 = sx(next_col);
            if next_col != NONE {
                col[next_col].shared3 = sx(c);
            }
            head[score] = c;
            min_score = Ord::min(score, min_score);
        }
    }

    let max_mark = zx(isize::MAX) - n;
    let mut tag_mark = clear_mark(0, max_mark, &mut row);
    let mut min_score = 0usize;
    let mut pfree = 2 * nnz;
    let mut k = 0usize;

    while k < ncol2 {
        debug_assert!(min_score <= n);
        debug_assert!(sx(head[min_score]) >= -1);
        while head[min_score] == NONE && min_score < n {
            min_score += 1;
        }
        let pivot_col = head[min_score];
        let mut next_col = col[pivot_col].shared4;
        head[min_score] = zx(next_col);
        if next_col != -1 {
            col[zx(next_col)].shared3 = -1;
        }
        debug_assert!(!col[pivot_col].is_dead());
        let pivot_col_score = col[pivot_col].shared2;
        col[pivot_col].shared2 = sx(k);
        let pivot_col_thickness = col[pivot_col].shared1;
        debug_assert!(pivot_col_thickness > 0);
        k += zx(pivot_col_thickness);
        let needed_memory = Ord::min(pivot_col_score, sx(n - k));
        if sx(pfree) + needed_memory >= sx(a.len()) {
            pfree = garbage_collection(&mut row, &mut col, &mut a, pfree);
            debug_assert!(sx(pfree) + needed_memory < sx(a.len()));
            tag_mark = clear_mark(0, max_mark, &mut row);
        }

        // Build the pivot row: the union of the columns adjacent to the pivot.
        let pivot_row_start = pfree;
        let mut pivot_row_degree = 0usize;
        col[pivot_col].shared1 = -pivot_col_thickness;
        let mut cp = zx(col[pivot_col].start);
        let cp_end = cp + zx(col[pivot_col].length);
        while cp < cp_end {
            let r = a[cp];
            cp += 1;
            if row[r].is_alive() {
                let mut rp = zx(row[r].start);
                let rp_end = rp + zx(row[r].length);
                while rp < rp_end {
                    let c = a[rp];
                    rp += 1;
                    let col_thickness = col[c].shared1;
                    if col_thickness > 0 && col[c].is_alive() {
                        col[c].shared1 = -col_thickness;
                        a[pfree] = c;
                        pfree += 1;
                        pivot_row_degree += zx(col_thickness);
                    }
                }
            }
        }
        col[pivot_col].shared1 = pivot_col_thickness;
        max_deg = Ord::max(max_deg, pivot_row_degree);

        // The pivot column's rows die.
        let mut cp = zx(col[pivot_col].start);
        let cp_end = cp + zx(col[pivot_col].length);
        while cp < cp_end {
            let r = a[cp];
            cp += 1;
            row[r].kill();
        }

        let pivot_row_length = pfree - pivot_row_start;
        let pivot_row = if pivot_row_length > 0 { a[zx(col[pivot_col].start)] } else { NONE };

        // Update the row marks for the columns in the pivot row.
        let mut rp = pivot_row_start;
        let rp_end = rp + pivot_row_length;
        while rp < rp_end {
            let c = a[rp];
            rp += 1;
            debug_assert!(col[c].is_alive());
            let col_thickness = -col[c].shared1;
            debug_assert!(col_thickness > 0);
            col[c].shared1 = col_thickness;
            let cur_score = zx(col[c].shared2);
            let prev_col = col[c].shared3;
            let next_col = col[c].shared4;
            debug_assert!(cur_score <= n);
            if prev_col == -1 {
                head[cur_score] = zx(next_col);
            } else {
                col[zx(prev_col)].shared4 = next_col;
            }
            if next_col != -1 {
                col[zx(next_col)].shared3 = prev_col;
            }
            let mut cp = zx(col[c].start);
            let cp_end = cp + zx(col[c].length);
            while cp < cp_end {
                let r = a[cp];
                cp += 1;
                let row_mark = row[r].shared2;
                if row[r].is_dead() {
                    continue;
                }
                debug_assert!(r != pivot_row);
                let mut set_difference = row_mark - sx(tag_mark);
                if set_difference < 0 {
                    debug_assert!(row[r].shared1 <= sx(max_deg));
                    set_difference = row[r].shared1;
                }
                set_difference -= col_thickness;
                debug_assert!(set_difference >= 0);
                if set_difference == 0 && control.aggressive {
                    row[r].kill();
                } else {
                    row[r].shared2 = set_difference + sx(tag_mark);
                }
            }
        }

        // Recompute the scores of the surviving columns and re-bucket them.
        let mut rp = pivot_row_start;
        let rp_end = rp + pivot_row_length;
        while rp < rp_end {
            let c = a[rp];
            rp += 1;
            debug_assert!(col[c].is_alive());
            let mut hash = 0usize;
            let mut cur_score = 0usize;
            let mut cp = zx(col[c].start);
            let mut new_cp = cp;
            let cp_end = cp + zx(col[c].length);
            while cp < cp_end {
                let r = a[cp];
                cp += 1;
                let row_mark = row[r].shared2;
                if row[r].is_dead() {
                    continue;
                }
                debug_assert!(row_mark >= sx(tag_mark));
                a[new_cp] = r;
                new_cp += 1;
                hash += r;
                cur_score += zx(row_mark - sx(tag_mark));
                cur_score = Ord::min(cur_score, n);
            }
            col[c].length = sx(new_cp - zx(col[c].start));
            if zx(col[c].length) == 0 {
                col[c].kill_principal();
                pivot_row_degree -= zx(col[c].shared1);
                col[c].shared2 = sx(k);
                k += zx(col[c].shared1);
            } else {
                col[c].shared2 = sx(cur_score);
                hash %= n + 1;
                let head_column = head[hash];
                let first_col;
                if sx(head_column) > -1 {
                    first_col = col[head_column].shared3;
                    col[head_column].shared3 = sx(c);
                } else {
                    first_col = -(sx(head_column) + 2);
                    head[hash] = zx(-sx(c + 2));
                }
                col[c].shared4 = first_col;
                col[c].shared3 = sx(hash);
                debug_assert!(col[c].is_alive());
            }
        }
        detect_super_cols(&mut col, &mut a, head, pivot_row_start, pivot_row_length);
        col[pivot_col].kill_principal();
        tag_mark = clear_mark(tag_mark + max_deg + 1, max_mark, &mut row);

        // Append the pivot row to the surviving columns' lists.
        let mut rp = pivot_row_start;
        let mut new_rp = rp;
        let rp_end = rp + pivot_row_length;
        while rp < rp_end {
            let c = a[rp];
            rp += 1;
            if col[c].is_dead() {
                continue;
            }
            a[new_rp] = c;
            new_rp += 1;
            a[zx(col[c].start + col[c].length)] = pivot_row;
            col[c].length += 1;
            let mut cur_score = zx(col[c].shared2) + pivot_row_degree;
            let max_score = n - k - zx(col[c].shared1);
            cur_score -= zx(col[c].shared1);
            cur_score = Ord::min(cur_score, max_score);
            col[c].shared2 = sx(cur_score);
            next_col = sx(head[cur_score]);
            col[c].shared4 = next_col;
            col[c].shared3 = -1;
            if next_col != -1 {
                col[zx(next_col)].shared3 = sx(c);
            }
            head[cur_score] = c;
            min_score = Ord::min(min_score, cur_score);
        }
        if pivot_row_degree > 0 {
            row[pivot_row].start = sx(pivot_row_start);
            row[pivot_row].length = sx(new_rp - pivot_row_start);
            row[pivot_row].shared1 = sx(pivot_row_degree);
            row[pivot_row].shared2 = 0;
        }
    }

    // Place the columns that were merged into a super-column, then read the
    // ordering out of `shared2`.
    for i in 0..n {
        debug_assert!(col[i].is_dead());
        if !col[i].is_dead_principal() && col[i].shared2 == -1 {
            let mut parent = i;
            loop {
                parent = zx(col[parent].shared1);
                if col[parent].is_dead_principal() {
                    break;
                }
            }
            let mut c = i;
            let mut order = zx(col[parent].shared2);
            loop {
                debug_assert!(col[c].shared2 == -1);
                col[c].shared2 = sx(order);
                order += 1;
                col[c].shared1 = sx(parent);
                c = zx(col[c].shared1);
                if col[c].shared2 != -1 {
                    break;
                }
            }
            col[parent].shared2 = sx(order);
        }
    }
    let mut perm = vec![0usize; n];
    let mut perm_inv = vec![0usize; n];
    for c in 0..n {
        perm[zx(col[c].shared2)] = c;
    }
    for c in 0..n {
        perm_inv[perm[c]] = c;
    }

    // Postorder through the column elimination tree, as faer does.
    let mut etree = vec![0isize; n];
    let mut post = vec![0usize; n];
    col_etree(pattern, Some(&perm), &mut etree);
    postorder(&mut post, &etree);
    for i in 0..n {
        perm[post[i]] = i;
    }
    for i in 0..n {
        perm_inv[i] = perm[perm_inv[i]];
    }
    for i in 0..n {
        perm[perm_inv[i]] = i;
    }

    Ok(ColamdOrdering { perm, perm_inv })
}

/// Merges columns with identical structure into super-columns.
fn detect_super_cols(col: &mut [ColamdCol], a: &mut [usize], head: &mut [usize], row_start: usize, row_length: usize) {
    let mut rp = row_start;
    let rp_end = rp + row_length;
    while rp < rp_end {
        let c = a[rp];
        rp += 1;
        if col[c].is_dead() {
            continue;
        }
        let hash = zx(col[c].shared3);
        let head_column = sx(head[hash]);
        let first_col = if head_column > -1 {
            col[zx(head_column)].shared3
        } else {
            -(head_column + 2)
        };
        let mut super_c_ = first_col;
        while super_c_ != -1 {
            let super_c = zx(super_c_);
            let length = col[super_c].length;
            let mut prev_c = super_c;
            let mut c_ = col[super_c].shared4;
            while c_ != -1 {
                let c = zx(c_);
                if col[c].length != length || col[c].shared2 != col[super_c].shared2 {
                    prev_c = c;
                    c_ = col[c].shared4;
                    continue;
                }
                let mut cp1 = zx(col[super_c].start);
                let mut cp2 = zx(col[c].start);
                let mut i = 0usize;
                while i < zx(length) {
                    if a[cp1] != a[cp2] {
                        break;
                    }
                    cp1 += 1;
                    cp2 += 1;
                    i += 1;
                }
                if i != zx(length) {
                    prev_c = c;
                    c_ = col[c].shared4;
                    continue;
                }
                col[super_c].shared1 += col[c].shared1;
                col[c].shared1 = sx(super_c);
                col[c].kill_non_principal();
                col[c].shared2 = -1;
                col[prev_c].shared4 = col[c].shared4;
                c_ = col[c].shared4;
            }
            super_c_ = col[super_c].shared4;
        }
        if head_column > -1 {
            col[zx(head_column)].shared3 = -1;
        } else {
            head[hash] = NONE;
        }
    }
}

/// Compacts the columns and rows of `a`, dropping dead entries.
fn garbage_collection(row: &mut [ColamdRow], col: &mut [ColamdCol], a: &mut [usize], pfree: usize) -> usize {
    let mut pdest = 0usize;
    let m = row.len() - 1;
    let n = col.len() - 1;
    for c in 0..n {
        if !col[c].is_dead() {
            let start = zx(col[c].start);
            col[c].start = sx(pdest);
            let length = zx(col[c].length);
            for psrc in start..start + length {
                let r = a[psrc];
                if !row[r].is_dead() {
                    a[pdest] = r;
                    pdest += 1;
                }
            }
            col[c].length = sx(pdest) - col[c].start;
        }
    }
    for r in 0..m {
        if row[r].is_dead() || row[r].length == 0 {
            row[r].kill();
        } else {
            let psrc = zx(row[r].start);
            row[r].shared2 = sx(a[psrc]);
            a[psrc] = !r;
        }
    }
    let mut psrc = pdest;
    while psrc < pfree {
        let psrc_ = psrc;
        psrc += 1;
        if sx(a[psrc_]) < 0 {
            psrc -= 1;
            let r = !a[psrc];
            a[psrc] = zx(row[r].shared2);
            row[r].start = sx(pdest);
            let length = zx(row[r].length);
            for _ in 0..length {
                let c = a[psrc];
                psrc += 1;
                if !col[c].is_dead() {
                    a[pdest] = c;
                    pdest += 1;
                }
            }
            row[r].length = sx(pdest) - row[r].start;
        }
    }
    pdest
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::perm::is_permutation;

    fn from_cols(n: usize, cols: &[&[usize]]) -> (Vec<usize>, Vec<usize>) {
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::new();
        col_ptr.push(0);
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

    fn ring12() -> (Vec<usize>, Vec<usize>) {
        let n = 12;
        let mut cols = Vec::new();
        for j in 0..n {
            cols.push(vec![j, (j + 1) % n, (j + 3) % n, (j + 7) % n]);
        }
        let refs: Vec<&[usize]> = cols.iter().map(|v| v.as_slice()).collect();
        from_cols(n, &refs)
    }

    fn lcg30() -> (Vec<usize>, Vec<usize>) {
        let n = 30;
        let mut state: u64 = 0x2545F4914F6CDD1D;
        let mut cols = Vec::new();
        for j in 0..n {
            let mut v = vec![j];
            for _ in 0..4 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let r = ((state >> 33) as usize) % n;
                v.push(r);
            }
            cols.push(v);
        }
        let refs: Vec<&[usize]> = cols.iter().map(|v| v.as_slice()).collect();
        from_cols(n, &refs)
    }

    fn order(n: usize, col_ptr: &[usize], row_idx: &[usize]) -> ColamdOrdering {
        let pattern = CscPattern::new(n, n, col_ptr, row_idx).unwrap();
        colamd_order(pattern, ColamdControl::default()).unwrap()
    }

    /// The expected permutations were produced by faer's `colamd::order` at
    /// commit `ddbd2ae` through a throwaway harness; see the phase-1 log.
    #[test]
    fn matches_faer_on_fixtures() {
        let cases = [
            (tridiag5(), vec![0, 1, 3, 2, 4], vec![0, 1, 3, 2, 4]),
            (arrow8(), vec![0, 1, 2, 3, 4, 5, 6, 7], vec![0, 1, 2, 3, 4, 5, 6, 7]),
            (
                grid4x5(),
                vec![0, 5, 1, 4, 9, 19, 14, 18, 13, 8, 2, 7, 3, 6, 11, 10, 15, 12, 17, 16],
                vec![0, 2, 10, 12, 3, 1, 13, 11, 9, 4, 15, 14, 17, 8, 6, 16, 19, 18, 7, 5],
            ),
            (
                ring12(),
                vec![0, 5, 9, 11, 1, 6, 10, 2, 3, 8, 4, 7],
                vec![0, 4, 7, 8, 10, 1, 5, 11, 9, 2, 6, 3],
            ),
            (
                lcg30(),
                vec![
                    9, 5, 18, 7, 4, 2, 17, 0, 12, 24, 14, 15, 16, 29, 8, 11, 13, 22, 1, 25, 27, 21, 28, 23, 3, 10, 26, 20, 6, 19,
                ],
                vec![
                    7, 18, 5, 24, 4, 1, 28, 3, 14, 0, 25, 15, 8, 16, 10, 11, 12, 6, 2, 29, 27, 21, 17, 23, 9, 19, 26, 20, 22, 13,
                ],
            ),
        ];
        for (i, ((col_ptr, row_idx), expected, expected_inv)) in cases.iter().enumerate() {
            let o = order(expected.len(), col_ptr, row_idx);
            assert_eq!(o.perm(), expected.as_slice(), "perm of case {i}");
            assert_eq!(o.perm_inv(), expected_inv.as_slice(), "perm_inv of case {i}");
        }
    }

    #[test]
    fn returns_valid_permutations() {
        for (n, (col_ptr, row_idx)) in [tridiag5(), arrow8(), grid4x5(), ring12(), lcg30()]
            .into_iter()
            .enumerate()
            .map(|(i, p)| ([5, 8, 20, 12, 30][i], p))
        {
            let o = order(n, &col_ptr, &row_idx);
            assert!(is_permutation(o.perm()));
            assert!(is_permutation(o.perm_inv()));
            for i in 0..n {
                assert_eq!(o.perm_inv()[o.perm()[i]], i);
            }
        }
    }

    #[test]
    fn empty_and_singleton() {
        let o = order(0, &[0], &[]);
        assert!(o.is_empty());

        let o = order(1, &[0, 1], &[0]);
        assert_eq!(o.perm(), &[0]);
        assert_eq!(o.perm_inv(), &[0]);
    }

    #[test]
    fn empty_columns_sort_last() {
        // Column 0 is empty; it must end up last.
        let (col_ptr, row_idx) = from_cols(3, &[&[], &[1, 2], &[2]]);
        let o = order(3, &col_ptr, &row_idx);
        assert_eq!(o.perm()[2], 0);
    }

    #[test]
    fn dense_weights_are_accepted() {
        let (col_ptr, row_idx) = grid4x5();
        let pattern = CscPattern::new(20, 20, &col_ptr, &row_idx).unwrap();
        for control in [
            ColamdControl::default(),
            ColamdControl {
                dense_row: -1.0,
                dense_col: -1.0,
                aggressive: false,
            },
            ColamdControl {
                dense_row: 0.1,
                dense_col: 0.1,
                aggressive: true,
            },
        ] {
            let o = colamd_order(pattern, control).unwrap();
            assert!(is_permutation(o.perm()));
        }
    }

    #[test]
    fn rejects_rectangular() {
        let pattern = CscPattern::new(3, 4, &[0, 0, 0, 0, 0], &[]).unwrap();
        assert_eq!(colamd_order(pattern, ColamdControl::default()).unwrap_err(), SparseError::BadShape);
    }
}
