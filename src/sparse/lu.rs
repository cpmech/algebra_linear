//! The simplicial sparse LU factorization and solve.
//!
//! Translated from `faer/src/sparse/linalg/lu.rs` at faer commit `ddbd2ae` (MIT;
//! see `LICENSE`). The algorithm is faer's: build `L` column by column with a
//! sparse forward solve, choose the pivot of each column by partial pivoting,
//! and store the row permutation that results. The dense abstractions
//! (`MatMut`, `MemStack`, `PermRef`) are replaced with slices and `Vec`
//! workspaces, which the crate allocates itself.
//!
//! The factorization is `P·A·Q = L·U`, where `P` sends row `row_perm[i]` to row
//! `i` and `Q` sends column `col_perm[j]` to column `j`. [`LuFactors::solve`]
//! applies those permutations around the two triangular solves.

use core::ops::{AddAssign, Mul, SubAssign};

use crate::Complex64;
use crate::sparse::SparseError;
use crate::sparse::csc::CscView;
use crate::sparse::symbolic::SymbolicLu;

/// The scalar types the sparse LU supports.
///
/// The trait is sealed in practice by the crate: it is implemented for `f64`
/// and [`Complex64`], the two types the kernels handle.
pub trait SparseScalar: Copy + AddAssign + SubAssign + Mul<Output = Self> {
    /// Returns the additive identity.
    fn zero() -> Self;
    /// Returns the multiplicative identity.
    fn one() -> Self;
    /// Returns the magnitude, used to choose a pivot.
    fn abs(self) -> f64;
    /// Returns the reciprocal.
    fn recip(self) -> Self;
}

impl SparseScalar for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    #[inline]
    fn recip(self) -> Self {
        1.0 / self
    }
}

impl SparseScalar for Complex64 {
    #[inline]
    fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
    #[inline]
    fn one() -> Self {
        Self::new(1.0, 0.0)
    }
    #[inline]
    fn abs(self) -> f64 {
        self.norm()
    }
    #[inline]
    fn recip(self) -> Self {
        self.inv()
    }
}

/// The numeric `L` and `U` factors of a sparse LU, plus the permutations.
///
/// Both factors are in CSC form and live in the **permuted** coordinate system:
/// `L` is unit lower triangular and `U` upper triangular, with row indices being
/// positions, not original row numbers. The first stored entry of every `L`
/// column is its unit diagonal; the last stored entry of every `U` column is its
/// diagonal.
#[derive(Clone, Debug)]
pub struct LuFactors<T> {
    n: usize,
    l_col_ptr: Vec<usize>,
    l_row_idx: Vec<usize>,
    l_val: Vec<T>,
    u_col_ptr: Vec<usize>,
    u_row_idx: Vec<usize>,
    u_val: Vec<T>,
    row_perm: Vec<usize>,
    row_perm_inv: Vec<usize>,
    col_perm: Vec<usize>,
    col_perm_inv: Vec<usize>,
}

impl<T> LuFactors<T> {
    /// Returns the order of the matrix.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the `L` column pointers.
    #[inline]
    pub fn l_col_ptr(&self) -> &[usize] {
        &self.l_col_ptr
    }

    /// Returns the `L` row indices (positions).
    #[inline]
    pub fn l_row_idx(&self) -> &[usize] {
        &self.l_row_idx
    }

    /// Returns the `L` values.
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
    #[inline]
    pub fn u_row_idx(&self) -> &[usize] {
        &self.u_row_idx
    }

    /// Returns the `U` values.
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

    /// Returns the row permutation, mapping a position to an original row
    /// (`row_perm[position] = original_row`).
    #[inline]
    pub fn row_perm(&self) -> &[usize] {
        &self.row_perm
    }

    /// Returns the inverse row permutation.
    #[inline]
    pub fn row_perm_inv(&self) -> &[usize] {
        &self.row_perm_inv
    }

    /// Returns the column permutation.
    #[inline]
    pub fn col_perm(&self) -> &[usize] {
        &self.col_perm
    }

    /// Returns the inverse column permutation.
    #[inline]
    pub fn col_perm_inv(&self) -> &[usize] {
        &self.col_perm_inv
    }
}

impl<T: SparseScalar> LuFactors<T> {
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

impl SymbolicLu {
    /// Computes the numeric `L`/`U` factors for `a`.
    ///
    /// The pattern must be the one the analysis saw; only the values may change.
    ///
    /// # Errors
    ///
    /// [`SparseError::StructureMismatch`] if the structure changed, and
    /// [`SparseError::Singular`] if a column has no nonzero pivot.
    pub fn factorize<T: SparseScalar>(&self, a: CscView<'_, T>) -> Result<LuFactors<T>, SparseError> {
        self.check_pattern(a.pattern())?;
        let n = self.n();
        let mut l_col_ptr = vec![0usize; n + 1];
        let mut u_col_ptr = vec![0usize; n + 1];
        let mut l_row_idx: Vec<usize> = Vec::with_capacity(a.nnz() + n);
        let mut l_val: Vec<T> = Vec::with_capacity(a.nnz() + n);
        let mut u_row_idx: Vec<usize> = Vec::with_capacity(a.nnz() + n);
        let mut u_val: Vec<T> = Vec::with_capacity(a.nnz() + n);

        let mut x = vec![T::zero(); n];
        let mut marked = vec![0usize; n];
        let mut xj = vec![0usize; n];
        let mut stack = vec![0usize; n];
        let mut row_perm_inv = vec![n; n];
        let mut l_pos = 0usize;
        let mut u_pos = 0usize;

        for j in 0..n {
            let pj = self.col_perm()[j];
            let lower = LowerView {
                ncols: j,
                col_ptr: &l_col_ptr[..j + 1],
                row_idx: &l_row_idx[..l_pos],
                val: &l_val[..l_pos],
            };
            let tail_start = l_incomplete_solve_sparse(
                &mut marked,
                j + 1,
                &mut xj,
                &mut x,
                &lower,
                &row_perm_inv,
                a.pattern().row_idx_of_col(pj),
                a.val_of_col(pj),
                &mut stack,
            );

            let touched = tail_start..n;
            let extra = touched.len();
            l_val.resize(l_pos + extra + 1, T::zero());
            l_row_idx.resize(l_pos + extra + 1, 0);
            u_val.resize(u_pos + extra + 1, T::zero());
            u_row_idx.resize(u_pos + extra + 1, 0);

            let mut pivot_idx = n;
            let mut pivot_val = -1.0f64;
            for &i in &xj[tail_start..n] {
                let xi = x[i];
                if row_perm_inv[i] == n {
                    let val = xi.abs();
                    if matches!(val.partial_cmp(&pivot_val), None | Some(core::cmp::Ordering::Greater)) {
                        pivot_idx = i;
                        pivot_val = val;
                    }
                } else {
                    u_row_idx[u_pos] = row_perm_inv[i];
                    u_val[u_pos] = xi;
                    u_pos += 1;
                }
            }
            // faer only reports a pivot that is structurally absent; an exact
            // zero pivot is selected there and produces infinities on solve.
            // A dense LU calls such a matrix singular, so this port does too.
            if pivot_idx == n || pivot_val == 0.0 {
                return Err(SparseError::Singular);
            }
            let x_piv = x[pivot_idx];
            let x_piv_inv = x_piv.recip();
            row_perm_inv[pivot_idx] = j;
            u_row_idx[u_pos] = j;
            u_val[u_pos] = x_piv;
            u_pos += 1;
            u_col_ptr[j + 1] = u_pos;

            l_row_idx[l_pos] = pivot_idx;
            l_val[l_pos] = T::one();
            l_pos += 1;
            for &i in &xj[tail_start..n] {
                let xi = x[i];
                if row_perm_inv[i] == n {
                    l_row_idx[l_pos] = i;
                    l_val[l_pos] = xi * x_piv_inv;
                    l_pos += 1;
                }
                x[i] = T::zero();
            }
            l_col_ptr[j + 1] = l_pos;
        }

        // `L`'s row indices are original rows during the factorization; convert
        // them to positions now that every row is claimed.
        for i in 0..l_pos {
            debug_assert!(row_perm_inv[l_row_idx[i]] < n);
            l_row_idx[i] = row_perm_inv[l_row_idx[i]];
        }
        let mut row_perm = vec![0usize; n];
        for (idx, &p) in row_perm_inv.iter().enumerate() {
            debug_assert!(p < n);
            row_perm[p] = idx;
        }

        Ok(LuFactors {
            n,
            l_col_ptr,
            l_row_idx,
            l_val,
            u_col_ptr,
            u_row_idx,
            u_val,
            row_perm,
            row_perm_inv,
            col_perm: self.col_perm().to_vec(),
            col_perm_inv: self.col_perm_inv().to_vec(),
        })
    }
}

/// A view of the `L` factor as it is being built.
struct LowerView<'a, T> {
    ncols: usize,
    col_ptr: &'a [usize],
    row_idx: &'a [usize],
    val: &'a [T],
}

impl<T> LowerView<'_, T> {
    #[inline]
    fn row_idx_of_col_raw(&self, j: usize) -> &[usize] {
        &self.row_idx[self.col_ptr[j]..self.col_ptr[j + 1]]
    }

    #[inline]
    fn val_of_col(&self, j: usize) -> &[T] {
        &self.val[self.col_ptr[j]..self.col_ptr[j + 1]]
    }
}

/// Computes `L \ A(:, j)` sparsely, returning the start of the touched suffix.
fn l_incomplete_solve_sparse<T: SparseScalar>(
    marked: &mut [usize],
    mark: usize,
    xi: &mut [usize],
    x: &mut [T],
    l: &LowerView<'_, T>,
    row_perm_inv: &[usize],
    bi: &[usize],
    bx: &[T],
    stack: &mut [usize],
) -> usize {
    let tail_start = reach(marked, mark, xi, l, row_perm_inv, bi, stack);
    for (&i, &b) in bi.iter().zip(bx.iter()) {
        x[i] += b;
    }
    for &i in &xi[tail_start..] {
        let pi = row_perm_inv[i];
        if pi >= l.ncols {
            continue;
        }
        let li = l.row_idx_of_col_raw(pi);
        let lx = l.val_of_col(pi);
        let xi_val = x[i];
        for (&row, &val) in li[1..].iter().zip(lx[1..].iter()) {
            x[row] -= val * xi_val;
        }
    }
    tail_start
}

/// Marks the rows reachable from `bi` through `L`, in a topological order.
fn reach<T>(
    marked: &mut [usize],
    mark: usize,
    xi: &mut [usize],
    l: &LowerView<'_, T>,
    row_perm_inv: &[usize],
    bi: &[usize],
    stack: &mut [usize],
) -> usize {
    let mut tail_start = marked.len();
    for &b in bi {
        if marked[b] < mark {
            tail_start = depth_first_search(marked, mark, &mut xi[..tail_start], l, row_perm_inv, b, stack);
        }
    }
    tail_start
}

/// Depth-first search from `b` over the columns of `L`.
fn depth_first_search<T>(
    marked: &mut [usize],
    mark: usize,
    xi: &mut [usize],
    l: &LowerView<'_, T>,
    row_perm_inv: &[usize],
    b: usize,
    stack: &mut [usize],
) -> usize {
    let mut tail_start = xi.len();
    let mut head_len = 1usize;
    xi[0] = b;
    'dfs: while head_len > 0 {
        let b = xi[head_len - 1];
        let pb = row_perm_inv[b];
        let range = if pb < l.ncols { l.col_ptr[pb]..l.col_ptr[pb + 1] } else { 0..0 };
        if marked[b] < mark {
            marked[b] = mark;
            stack[head_len - 1] = range.start;
        }
        let start = stack[head_len - 1];
        let end = range.end;
        for ptr in start..end {
            let i = l.row_idx[ptr];
            if marked[i] == mark {
                continue;
            }
            stack[head_len - 1] = ptr;
            xi[head_len] = i;
            head_len += 1;
            continue 'dfs;
        }
        head_len -= 1;
        tail_start -= 1;
        xi[tail_start] = b;
    }
    tail_start
}

/// Solves `L y = b` for unit lower triangular `L`, in place.
fn unit_lower_solve<T: SparseScalar>(col_ptr: &[usize], row_idx: &[usize], val: &[T], x: &mut [T]) {
    let n = col_ptr.len() - 1;
    for j in 0..n {
        let xj = x[j];
        for k in col_ptr[j] + 1..col_ptr[j + 1] {
            let i = row_idx[k];
            x[i] -= val[k] * xj;
        }
    }
}

/// Solves `U x = b` for upper triangular `U` with stored diagonal, in place.
fn upper_solve<T: SparseScalar>(col_ptr: &[usize], row_idx: &[usize], val: &[T], x: &mut [T]) {
    let n = col_ptr.len() - 1;
    for j in (0..n).rev() {
        let end = col_ptr[j + 1];
        let diag = val[end - 1];
        let xj = x[j] * diag.recip();
        x[j] = xj;
        for k in col_ptr[j]..end - 1 {
            let i = row_idx[k];
            x[i] -= val[k] * xj;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::CscPattern;

    fn analyze(n: usize, col_ptr: &[usize], row_idx: &[usize]) -> SymbolicLu {
        let pattern = CscPattern::new(n, n, col_ptr, row_idx).unwrap();
        SymbolicLu::analyze(pattern).unwrap()
    }

    fn factors(n: usize, col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> LuFactors<f64> {
        let symbolic = analyze(n, col_ptr, row_idx);
        let a = CscView::new(n, n, col_ptr, row_idx, values).unwrap();
        symbolic.factorize(a).unwrap()
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

    fn lu_product(f: &LuFactors<f64>) -> Vec<f64> {
        let n = f.n();
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
        let mut c = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += l[i + k * n] * u[k + j * n];
                }
                c[i + j * n] = sum;
            }
        }
        c
    }

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() <= tol * (1.0 + e.abs()), "entry {i}: {a} vs {e}");
        }
    }

    #[test]
    fn solves_a_two_by_two() {
        // A = [4 1; 1 3]
        let f = factors(2, &[0, 2, 4], &[0, 1, 0, 1], &[4.0, 1.0, 1.0, 3.0]);
        let mut b = [1.0, 2.0];
        f.solve(&mut b).unwrap();
        assert_close(&b, &[1.0 / 11.0, 7.0 / 11.0], 1e-14);
    }

    #[test]
    fn reconstructs_paq_eq_lu() {
        // A 4x4 with a fill-producing pattern.
        // [ 4  1  0  1]
        // [ 1  3  1  0]
        // [ 0  1  5  1]
        // [ 1  0  1  4]
        let col_ptr = [0, 2, 5, 8, 10];
        let row_idx = [0, 1, 0, 1, 2, 1, 2, 3, 0, 3];
        let values = [4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 5.0, 1.0, 1.0, 4.0];
        let f = factors(4, &col_ptr, &row_idx, &values);
        let a = dense(4, &col_ptr, &row_idx, &values);
        let lu = lu_product(&f);

        // B[i][j] = A[row_perm[i], col_perm[j]]
        for j in 0..4 {
            for i in 0..4 {
                let r = f.row_perm()[i];
                let c = f.col_perm()[j];
                assert_close(&[lu[i + j * 4]], &[a[r + c * 4]], 1e-12);
            }
        }
    }

    #[test]
    fn solves_a_banded_system() {
        let n = 8;
        let mut col_ptr = vec![0usize];
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        for j in 0..n {
            if j > 0 {
                row_idx.push(j - 1);
                values.push(-1.0);
            }
            row_idx.push(j);
            values.push(4.0);
            if j + 1 < n {
                row_idx.push(j + 1);
                values.push(-1.0);
            }
            col_ptr.push(row_idx.len());
        }
        let f = factors(n, &col_ptr, &row_idx, &values);
        let a = dense(n, &col_ptr, &row_idx, &values);
        let x_exact: Vec<f64> = (0..n).map(|i| (i as f64) + 0.5).collect();
        let mut b = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                b[i] += a[i + j * n] * x_exact[j];
            }
        }
        f.solve(&mut b).unwrap();
        assert_close(&b, &x_exact, 1e-12);
    }

    #[test]
    fn solves_diagonal_and_arrow_systems() {
        // Diagonal: 2, 3, 4, 5, 6.
        let f = factors(5, &[0, 1, 2, 3, 4, 5], &[0, 1, 2, 3, 4], &[2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut b = [2.0, 6.0, 12.0, 20.0, 30.0];
        f.solve(&mut b).unwrap();
        assert_close(&b, &[1.0, 2.0, 3.0, 4.0, 5.0], 1e-15);

        // Arrow: a dense first row and column, diagonal elsewhere.
        let n = 6;
        let mut col_ptr = vec![0usize];
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        for j in 0..n {
            if j == 0 {
                for i in 0..n {
                    row_idx.push(i);
                    values.push(if i == 0 { 5.0 } else { 1.0 });
                }
            } else {
                row_idx.push(0);
                values.push(1.0);
                row_idx.push(j);
                values.push(4.0);
            }
            col_ptr.push(row_idx.len());
        }
        let f = factors(n, &col_ptr, &row_idx, &values);
        let a = dense(n, &col_ptr, &row_idx, &values);
        let x_exact: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
        let mut b = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                b[i] += a[i + j * n] * x_exact[j];
            }
        }
        f.solve(&mut b).unwrap();
        assert_close(&b, &x_exact, 1e-13);
    }

    #[test]
    fn detects_a_singular_matrix() {
        // [1 2; 2 4] is rank one.
        let symbolic = analyze(2, &[0, 2, 4], &[0, 1, 0, 1]);
        let a = CscView::new(2, 2, &[0, 2, 4], &[0, 1, 0, 1], &[1.0, 2.0, 2.0, 4.0]).unwrap();
        assert_eq!(symbolic.factorize(a).unwrap_err(), SparseError::Singular);
    }

    #[test]
    fn handles_zero_by_zero_and_one_by_one() {
        let symbolic = analyze(0, &[0], &[]);
        let a = CscView::new(0, 0, &[0], &[], &[] as &[f64]).unwrap();
        let f = symbolic.factorize(a).unwrap();
        let mut b: [f64; 0] = [];
        f.solve(&mut b).unwrap();

        let f = factors(1, &[0, 1], &[0], &[2.5]);
        let mut b = [5.0];
        f.solve(&mut b).unwrap();
        assert_close(&b, &[2.0], 1e-15);
    }

    #[test]
    fn reuses_the_symbolic_analysis() {
        let col_ptr = [0, 2, 4];
        let row_idx = [0, 1, 0, 1];
        let symbolic = analyze(2, &col_ptr, &row_idx);

        let a1 = CscView::new(2, 2, &col_ptr, &row_idx, &[4.0, 1.0, 1.0, 3.0]).unwrap();
        let f1 = symbolic.factorize(a1).unwrap();
        let a2 = CscView::new(2, 2, &col_ptr, &row_idx, &[5.0, 2.0, 2.0, 2.0]).unwrap();
        let f2 = symbolic.factorize(a2).unwrap();

        let mut b = [1.0, 1.0];
        f1.solve(&mut b).unwrap();
        assert_close(&b, &[2.0 / 11.0, 3.0 / 11.0], 1e-14);
        let mut b = [1.0, 1.0];
        f2.solve(&mut b).unwrap();
        assert_close(&b, &[0.0, 0.5], 1e-14);

        // A different pattern is refused.
        let other = CscView::new(2, 2, &[0, 2, 3], &[0, 1, 1], &[1.0, 1.0, 1.0]).unwrap();
        assert_eq!(symbolic.factorize(other).unwrap_err(), SparseError::StructureMismatch);
    }

    #[test]
    fn solves_a_complex_system() {
        use crate::cpx;
        // A = [1+2i, 1; 1, 1-i]
        let col_ptr = [0, 2, 4];
        let row_idx = [0, 1, 0, 1];
        let values = [cpx!(1.0, 2.0), cpx!(1.0, 0.0), cpx!(1.0, 0.0), cpx!(1.0, -1.0)];
        let pattern = CscPattern::new(2, 2, &col_ptr, &row_idx).unwrap();
        let symbolic = SymbolicLu::analyze(pattern).unwrap();
        let a = CscView::new(2, 2, &col_ptr, &row_idx, &values).unwrap();
        let f = symbolic.factorize(a).unwrap();

        let x_exact = [cpx!(1.0, 1.0), cpx!(-2.0, 0.5)];
        let mut b = [cpx!(0.0, 0.0), cpx!(0.0, 0.0)];
        for i in 0..2 {
            for j in 0..2 {
                b[i] += values[i + j * 2] * x_exact[j];
            }
        }
        f.solve(&mut b).unwrap();
        for i in 0..2 {
            assert!((b[i] - x_exact[i]).norm() < 1e-13);
        }
    }

    #[test]
    fn solves_random_diagonally_dominant_systems() {
        let n = 24;
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut cols = vec![Vec::new(); n];
        for j in 0..n {
            cols[j].push(j);
            for _ in 0..2 {
                let i = (next() as usize) % n;
                if i != j {
                    cols[j].push(i);
                }
            }
        }
        let mut col_ptr = vec![0usize];
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        for j in 0..n {
            cols[j].sort_unstable();
            cols[j].dedup();
            for &i in &cols[j] {
                row_idx.push(i);
                values.push(if i == j {
                    10.0 + f64::from(next() % 100) / 100.0
                } else {
                    -1.0 + f64::from(next() % 100) / 50.0
                });
            }
            col_ptr.push(row_idx.len());
        }
        let f = factors(n, &col_ptr, &row_idx, &values);
        let a = dense(n, &col_ptr, &row_idx, &values);
        let x_exact: Vec<f64> = (0..n).map(|i| ((i % 5) as f64) - 2.0).collect();
        let mut b = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                b[i] += a[i + j * n] * x_exact[j];
            }
        }
        f.solve(&mut b).unwrap();
        assert_close(&b, &x_exact, 1e-12);

        // Reconstruction on the same pattern.
        let lu = lu_product(&f);
        for j in 0..n {
            for i in 0..n {
                let r = f.row_perm()[i];
                let c = f.col_perm()[j];
                assert!((lu[i + j * n] - a[r + c * n]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn solves_a_complex_random_system() {
        use crate::cpx;
        let n = 10;
        let mut state: u64 = 0x0123_4567_89AB_CDEF;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 32) as f64) / 4294967296.0
        };
        let mut cols = vec![Vec::new(); n];
        for j in 0..n {
            cols[j].push(j);
            let i = ((next() * 1e6) as usize) % n;
            if i != j {
                cols[j].push(i);
            }
        }
        let mut col_ptr = vec![0usize];
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        for j in 0..n {
            cols[j].sort_unstable();
            cols[j].dedup();
            for &i in &cols[j] {
                let re = if i == j { 6.0 + next() } else { 0.5 - next() };
                let im = 0.5 - next();
                row_idx.push(i);
                values.push(cpx!(re, im));
            }
            col_ptr.push(row_idx.len());
        }
        let pattern = CscPattern::new(n, n, &col_ptr, &row_idx).unwrap();
        let symbolic = SymbolicLu::analyze(pattern).unwrap();
        let a = CscView::new(n, n, &col_ptr, &row_idx, &values).unwrap();
        let f = symbolic.factorize(a).unwrap();

        let x_exact: Vec<Complex64> = (0..n).map(|i| cpx!(f64::from(i as u32) - 3.0, 0.25 * f64::from(i as u32))).collect();
        let mut b = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..col_ptr[j + 1] - col_ptr[j] {
                    if row_idx[col_ptr[j] + k] == i {
                        b[i] += values[col_ptr[j] + k] * x_exact[j];
                    }
                }
            }
        }
        f.solve(&mut b).unwrap();
        for i in 0..n {
            assert!((b[i] - x_exact[i]).norm() < 1e-11, "row {i}");
        }
    }

    #[test]
    fn rejects_a_wrong_right_hand_side_length() {
        let f = factors(2, &[0, 2, 4], &[0, 1, 0, 1], &[4.0, 1.0, 1.0, 3.0]);
        let mut b = [1.0, 2.0, 3.0];
        assert_eq!(f.solve(&mut b).unwrap_err(), SparseError::BadShape);
    }
}
