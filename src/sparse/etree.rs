//! Elimination-tree helpers shared by the orderings and the supernodal analysis.
//!
//! Translated from faer commit `ddbd2ae`: `ghost_col_etree` from
//! `sparse/linalg/qr.rs` and `ghost_postorder` from
//! `sparse/linalg/cholesky.rs`.

use crate::sparse::csc::CscPattern;

/// `usize` to `isize`, the translation of faer's `I::Signed::truncate`.
#[inline]
fn zx(x: isize) -> usize {
    x as usize
}

/// `usize` to `isize`, the translation of faer's `I::Signed::truncate`.
#[inline]
fn sx(x: usize) -> isize {
    x as isize
}

/// Computes the column elimination tree of `pattern`, optionally permuted.
///
/// Translated from faer's `ghost_col_etree` in `sparse/linalg/qr.rs`.
pub(crate) fn col_etree(pattern: CscPattern<'_>, col_perm: Option<&[usize]>, etree: &mut [isize]) {
    let n = pattern.ncols();
    let m = pattern.nrows();
    let mut ancestor = vec![-1isize; n];
    let mut prev = vec![-1isize; m];
    etree.fill(-1);
    for j in 0..n {
        let pj = col_perm.map_or(j, |perm| perm[j]);
        for &row in pattern.row_idx_of_col(pj) {
            let mut i = prev[row];
            while i >= 0 {
                let i_ = zx(i);
                if i_ == j {
                    break;
                }
                let next_i = ancestor[i_];
                ancestor[i_] = sx(j);
                if next_i < 0 {
                    etree[i_] = sx(j);
                    break;
                }
                i = next_i;
            }
            prev[row] = sx(j);
        }
    }
}

/// Computes a postorder of the elimination tree in `etree`.
///
/// Translated from faer's `ghost_postorder` in `sparse/linalg/cholesky.rs`.
pub(crate) fn postorder(post: &mut [usize], etree: &[isize]) {
    let n = post.len();
    if n == 0 {
        return;
    }
    let mut stack = vec![0usize; n];
    let mut first_child = vec![-1isize; n];
    let mut next_child = vec![-1isize; n];
    for j in (0..n).rev() {
        let parent = etree[j];
        if parent >= 0 {
            let parent = zx(parent);
            next_child[j] = first_child[parent];
            first_child[parent] = sx(j);
        }
    }
    let mut start = 0usize;
    for root in 0..n {
        if etree[root] < 0 {
            start = postorder_depth_first_search(post, root, start, &mut stack, &mut first_child, &next_child);
        }
    }
}

/// The depth-first walk behind [`postorder`].
fn postorder_depth_first_search(
    post: &mut [usize],
    root: usize,
    mut start: usize,
    stack: &mut [usize],
    first_child: &mut [isize],
    next_child: &[isize],
) -> usize {
    let mut top = 1usize;
    stack[0] = root;
    while top != 0 {
        let current = stack[top - 1];
        let child = first_child[current];
        if child >= 0 {
            let child = zx(child);
            stack[top] = child;
            top += 1;
            first_child[current] = next_child[child];
        } else {
            post[start] = current;
            start += 1;
            top -= 1;
        }
    }
    start
}
