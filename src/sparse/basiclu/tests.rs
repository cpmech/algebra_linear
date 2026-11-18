//! Comprehensive unit, regression, accuracy, hyper-sparse, and update tests
//! for `sparse::basiclu`.

use super::degree_list::DegreeLists;
use super::file::{LuFileIdx, LuFileVal, NONE};
use super::{BasicLu, BasicLuParams, Error};
use crate::lapack::dgesv;
use crate::lapack::families::{Dist, Dlatms, DlatmsPack, DlatmsSym, dlatms};

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::time::Instant;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
use web_time::Instant;

/// Deterministic SplitMix64 PRNG (zero external crates).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn next_f64_signed(&mut self) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
        2.0 * u - 1.0
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() % (bound as u64)) as usize
    }
}

fn csc_to_dense(n: usize, col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> Vec<f64> {
    let mut a = vec![0.0_f64; n * n];
    for j in 0..n {
        for p in col_ptr[j]..col_ptr[j + 1] {
            a[row_idx[p] + j * n] += values[p];
        }
    }
    a
}

fn dense_to_csc(n: usize, a: &[f64], drop_zero: bool) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::new();
    let mut values = Vec::new();
    col_ptr.push(0);
    for j in 0..n {
        for i in 0..n {
            let v = a[i + j * n];
            if !drop_zero || v != 0.0 {
                row_idx.push(i);
                values.push(v);
            }
        }
        col_ptr.push(row_idx.len());
    }
    (col_ptr, row_idx, values)
}

fn transpose_dense(n: usize, a: &[f64]) -> Vec<f64> {
    let mut at = vec![0.0_f64; n * n];
    for j in 0..n {
        for i in 0..n {
            at[j + i * n] = a[i + j * n];
        }
    }
    at
}

fn matvec(n: usize, a: &[f64], x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0_f64; n];
    for j in 0..n {
        for i in 0..n {
            y[i] += a[i + j * n] * x[j];
        }
    }
    y
}

fn matvec_t(n: usize, a: &[f64], x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0_f64; n];
    for j in 0..n {
        let mut s = 0.0;
        for i in 0..n {
            s += a[i + j * n] * x[i];
        }
        y[j] = s;
    }
    y
}

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn matrix_norm_inf(n: usize, a: &[f64]) -> f64 {
    (0..n).map(|i| (0..n).map(|j| a[i + j * n].abs()).sum::<f64>()).fold(0.0_f64, f64::max)
}

fn scaled_residual(n: usize, a: &[f64], x: &[f64], b: &[f64]) -> f64 {
    let ax = matvec(n, a, x);
    let r: Vec<f64> = ax.iter().zip(b.iter()).map(|(u, v)| u - v).collect();
    let denom = matrix_norm_inf(n, a) * norm_inf(x) + norm_inf(b);
    if denom == 0.0 { norm_inf(&r) } else { norm_inf(&r) / denom }
}

fn scaled_residual_t(n: usize, a: &[f64], x: &[f64], b: &[f64]) -> f64 {
    let atx = matvec_t(n, a, x);
    let r: Vec<f64> = atx.iter().zip(b.iter()).map(|(u, v)| u - v).collect();
    let at = transpose_dense(n, a);
    let denom = matrix_norm_inf(n, &at) * norm_inf(x) + norm_inf(b);
    if denom == 0.0 { norm_inf(&r) } else { norm_inf(&r) / denom }
}

fn rel_diff(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let diff: Vec<f64> = actual.iter().zip(expected.iter()).map(|(u, v)| u - v).collect();
    let denom = norm_inf(expected).max(1.0);
    norm_inf(&diff) / denom
}

fn assert_lu_file_val_invariants(file: &LuFileVal) {
    let n = file.begin.len() - 1;
    let mut sum_len = 0usize;
    let mut prev_end = 0usize;
    let mut cur = file.flink[n];
    let mut seen = vec![false; n];
    while cur != n {
        assert!(!seen[cur], "cycle in LuFileVal flink");
        seen[cur] = true;
        let b = file.begin[cur];
        let e = file.end[cur];
        assert!(e > b, "empty line {cur} must be unlinked");
        assert!(b >= prev_end, "lines overlap or out of order: prev_end={prev_end}, begin={b}");
        assert!(e <= file.tail_used, "end {e} exceeds tail_used {}", file.tail_used);
        sum_len += e - b;
        prev_end = e;
        let next = file.flink[cur];
        assert_eq!(file.blink[next], cur, "broken blink at {cur}");
        cur = next;
    }
    for (k, &is_linked) in seen.iter().enumerate() {
        if !is_linked {
            assert_eq!(file.flink[k], NONE);
            assert_eq!(file.blink[k], NONE);
            assert_eq!(file.begin[k], file.end[k]);
        }
    }
    assert_eq!(sum_len, file.live_nnz);
    assert!(file.tail_used <= file.idx.len());
}

#[test]
fn error_display_and_std_error_are_exact() {
    let cases = [
        (Error::BadShape, "inconsistent shape or length"),
        (Error::InvalidStructure, "invalid CSC structure or out-of-bounds row index"),
        (Error::UnsortedRows, "row indices must be strictly increasing within a column"),
        (Error::DuplicateEntry, "duplicate entry in a column"),
        (Error::NotFinite, "input contains a NaN or infinite value"),
        (Error::Singular { rank: 3, n: 5 }, "matrix is singular (rank 3 of 5)"),
        (Error::SingularUpdate, "column update produced a singular pivot"),
        (Error::MaxUpdatesReached, "maximum number of updates reached; refactorize"),
        (Error::InvalidColumn { col: 7, n: 4 }, "column index 7 out of bounds for matrix order 4"),
        (Error::NotFactorized, "the matrix has not been factorized"),
    ];
    for (err, expected) in cases {
        assert_eq!(err.to_string(), expected);
        let dyn_err: &dyn std::error::Error = &err;
        assert_eq!(dyn_err.to_string(), expected);
    }
}

#[test]
fn params_defaults_match_specification() {
    let p = BasicLuParams::default();
    assert_eq!(p.abs_pivot_tol, 1e-14);
    assert_eq!(p.rel_pivot_tol, 0.1);
    assert_eq!(p.drop_tol, 1e-20);
    assert_eq!(p.sparse_threshold, 0.05);
    assert_eq!(p.search_rows, 4);
    assert_eq!(p.max_updates, 100);
    assert_eq!(p.compress_threshold, 0.5);
    assert!(!p.allow_singular);

    let lu = BasicLu::default();
    assert_eq!(*lu.params(), p);
    assert!(lu.stats().is_none());
}

#[test]
fn empty_and_one_by_one_matrices() {
    let mut lu = BasicLu::default();
    assert_eq!(lu.solve(&mut []), Err(Error::NotFactorized));
    assert_eq!(lu.solve_transpose(&mut []), Err(Error::NotFactorized));
    let mut oi = Vec::new();
    let mut ov = Vec::new();
    assert_eq!(lu.solve_sparse(&[], &[], &mut oi, &mut ov), Err(Error::NotFactorized));
    assert_eq!(lu.solve_sparse_transpose(&[], &[], &mut oi, &mut ov), Err(Error::NotFactorized));
    assert_eq!(lu.update(0, &[], &[]), Err(Error::NotFactorized));

    // n == 0
    lu.factorize(&[0], &[], &[]).unwrap();
    let stats0 = lu.stats().unwrap();
    assert_eq!(stats0.n, 0);
    assert_eq!(stats0.rank, 0);
    assert_eq!(stats0.nnz_a, 0);
    lu.solve(&mut []).unwrap();
    lu.solve_transpose(&mut []).unwrap();
    oi.push(99);
    ov.push(99.0);
    lu.solve_sparse(&[], &[], &mut oi, &mut ov).unwrap();
    assert!(oi.is_empty() && ov.is_empty());
    lu.solve_sparse_transpose(&[], &[], &mut oi, &mut ov).unwrap();
    assert!(oi.is_empty() && ov.is_empty());

    // n == 1 nonsingular
    lu.factorize(&[0, 1], &[0], &[4.0]).unwrap();
    let stats1 = lu.stats().unwrap();
    assert_eq!(stats1.n, 1);
    assert_eq!(stats1.rank, 1);
    assert_eq!(stats1.bump_size, 0);
    let mut b = [12.0];
    lu.solve(&mut b).unwrap();
    assert!((b[0] - 3.0).abs() < 1e-15);

    // n == 1 singular under both modes
    let err = lu.factorize(&[0, 1], &[0], &[0.0]).unwrap_err();
    assert_eq!(err, Error::Singular { rank: 0, n: 1 });
    // Previous valid 1x1 factorization must remain intact after failed factorize.
    let mut b2 = [8.0];
    lu.solve(&mut b2).unwrap();
    assert!((b2[0] - 2.0).abs() < 1e-15);

    let mut lu_allow = BasicLu::new(BasicLuParams {
        allow_singular: true,
        ..BasicLuParams::default()
    });
    lu_allow.factorize(&[0, 1], &[0], &[1e-18]).unwrap();
    assert_eq!(lu_allow.stats().unwrap().rank, 0);
    let mut b3 = [5.0];
    lu_allow.solve(&mut b3).unwrap();
    assert_eq!(b3[0], 5.0);
}

#[test]
fn lu_file_reappend_swap_remove_and_compress_invariants() {
    // Tight initial capacity forces multiple reappend() and compress() cycles.
    let mut f = LuFileVal::new(4, 1, 0.35);
    assert_lu_file_val_invariants(&f);

    let mut expected: Vec<Vec<(usize, f64)>> = vec![Vec::new(); 4];
    let mut rng = SplitMix64(42);
    let mut compress_checks = 0usize;

    for step in 0..160 {
        let line = rng.next_usize(4);
        if expected[line].len() > 4 && (step % 3 == 0) {
            let pos = rng.next_usize(expected[line].len());
            f.swap_remove_at(line, pos);
            expected[line].swap_remove(pos);
        } else if step % 17 == 15 {
            f.clear_line(line);
            expected[line].clear();
        } else {
            let idx = step;
            let val = rng.next_f64_signed();
            f.push_with_slack(line, idx, val, 1);
            expected[line].push((idx, val));
        }
        assert_lu_file_val_invariants(&f);

        for (k, exp_line) in expected.iter().enumerate() {
            assert_eq!(f.len(k), exp_line.len());
            let mut got: Vec<(usize, f64)> = f.idx_slice(k).iter().copied().zip(f.val_slice(k).iter().copied()).collect();
            let mut exp_sorted = exp_line.clone();
            got.sort_by_key(|x| x.0);
            exp_sorted.sort_by_key(|x| x.0);
            assert_eq!(got, exp_sorted);
        }

        if step % 23 == 0 {
            f.compress();
            compress_checks += 1;
            assert_eq!(f.tail_used, f.live_nnz);
            assert_lu_file_val_invariants(&f);
        }
    }
    assert!(compress_checks > 0);

    // Exercise LuFileVal Case A (already-last physical line growing capacity in place).
    let mut f_last = LuFileVal::new(2, 2, 0.9);
    f_last.push_with_slack(0, 10, 1.0, 0);
    f_last.push_with_slack(0, 11, 2.0, 0);
    f_last.push_with_slack(0, 12, 3.0, 0);
    assert_eq!(f_last.idx_slice(0), &[10, 11, 12]);
    assert_eq!(f_last.val_slice(0), &[1.0, 2.0, 3.0]);
    assert_lu_file_val_invariants(&f_last);

    // Exercise LuFileIdx tight capacity, swap_remove, clear_line, and compress.
    let mut fi = LuFileIdx::new(3, 1, 0.25);
    for r in 0..12 {
        fi.push_with_slack(r % 3, r, 0);
    }
    assert!(fi.remove_idx(0, 3));
    assert!(!fi.remove_idx(0, 999));
    fi.clear_line(1);
    fi.push_with_slack(2, 99, 2);
    fi.compress();
    assert_eq!(fi.tail_used, fi.live_nnz);
    assert_eq!(fi.len(1), 0);
    assert_eq!(fi.flink[1], NONE);
    let mut line0 = fi.idx_slice(0).to_vec();
    line0.sort_unstable();
    assert_eq!(line0, vec![0, 6, 9]);
}

#[test]
fn degree_list_handles_bucket_zero_and_removals() {
    let mut dl = DegreeLists::new(5);
    dl.insert(0, 2);
    dl.insert(1, 2);
    dl.insert(2, 1);
    dl.insert(3, 0);
    dl.insert(4, 5);

    assert_eq!(dl.deg, vec![2, 2, 1, 0, 5]);
    assert_eq!(dl.pop_bucket(0), Some(3));
    assert_eq!(dl.pop_bucket(0), None);
    assert_eq!(dl.deg[3], NONE);

    // Decrement node 2 down to bucket 0, then increment back to 1.
    dl.decrement(2);
    assert_eq!(dl.deg[2], 0);
    assert_eq!(dl.head[1], NONE);
    assert_eq!(dl.head[0], 2);
    dl.increment(2);
    assert_eq!(dl.deg[2], 1);
    assert_eq!(dl.head[0], NONE);

    // Remove middle/head from bucket 2 cleanly.
    dl.remove(0);
    assert_eq!(dl.deg[0], NONE);
    assert_eq!(dl.head[2], 1);
    assert_eq!(dl.prev[1], NONE);
    dl.remove(1);
    assert_eq!(dl.head[2], NONE);
}

#[test]
fn csc_and_solve_validation_rejects_bad_inputs() {
    let mut lu = BasicLu::default();
    assert_eq!(lu.factorize(&[], &[], &[]), Err(Error::BadShape));
    assert_eq!(lu.factorize(&[1, 1], &[0], &[1.0]), Err(Error::InvalidStructure));
    assert_eq!(lu.factorize(&[0, 2], &[0], &[1.0]), Err(Error::BadShape));
    assert_eq!(lu.factorize(&[0, 2, 1], &[0], &[1.0]), Err(Error::InvalidStructure));
    assert_eq!(lu.factorize(&[0, 1, 1], &[2], &[1.0]), Err(Error::InvalidStructure));
    assert_eq!(lu.factorize(&[0, 2, 2], &[0, 0], &[1.0, 2.0]), Err(Error::DuplicateEntry));
    assert_eq!(lu.factorize(&[0, 2, 2], &[1, 0], &[1.0, 2.0]), Err(Error::UnsortedRows));
    assert_eq!(lu.factorize(&[0, 1, 2], &[0, 1], &[1.0]), Err(Error::BadShape));
    assert_eq!(lu.factorize(&[0, 1, 2], &[0, 1], &[1.0, f64::NAN]), Err(Error::NotFinite));
    assert_eq!(lu.factorize(&[0, 1, 2], &[0, 1], &[f64::INFINITY, 1.0]), Err(Error::NotFinite));

    lu.factorize(&[0, 1, 2], &[0, 1], &[2.0, 3.0]).unwrap();
    assert_eq!(lu.solve(&mut [1.0]), Err(Error::BadShape));
    assert_eq!(lu.solve(&mut [1.0, f64::NAN]), Err(Error::NotFinite));
    assert_eq!(lu.solve_transpose(&mut [1.0, 2.0, 3.0]), Err(Error::BadShape));
    assert_eq!(lu.solve_transpose(&mut [f64::NEG_INFINITY, 2.0]), Err(Error::NotFinite));

    let mut oi = Vec::new();
    let mut ov = Vec::new();
    assert_eq!(lu.solve_sparse(&[0, 1], &[1.0], &mut oi, &mut ov), Err(Error::BadShape));
    assert_eq!(lu.solve_sparse(&[2], &[1.0], &mut oi, &mut ov), Err(Error::InvalidStructure));
    assert_eq!(lu.solve_sparse(&[0], &[f64::NAN], &mut oi, &mut ov), Err(Error::NotFinite));
    assert_eq!(lu.solve_sparse_transpose(&[2], &[1.0], &mut oi, &mut ov), Err(Error::InvalidStructure));
    assert_eq!(lu.solve_sparse_transpose(&[0], &[f64::INFINITY], &mut oi, &mut ov), Err(Error::NotFinite));
}

#[test]
fn singleton_peeling_handles_triangular_permuted_and_abstol_deferral() {
    // 1) Permuted upper triangular: purely column singletons (`bump_size == 0`).
    let mut rng = SplitMix64(101);
    let n = 12;
    let mut u = vec![0.0_f64; n * n];
    for j in 0..n {
        u[j + j * n] = 3.0 + j as f64 * 0.25;
        for i in 0..j {
            if (i + j) % 3 == 0 {
                u[i + j * n] = rng.next_f64_signed();
            }
        }
    }
    let pr = [4, 0, 11, 2, 7, 1, 9, 5, 10, 3, 8, 6];
    let pc = [9, 3, 6, 0, 11, 1, 8, 4, 2, 10, 5, 7];
    let mut a_perm = vec![0.0_f64; n * n];
    for j in 0..n {
        for i in 0..n {
            a_perm[pr[i] + pc[j] * n] = u[i + j * n];
        }
    }
    let (cp, ri, va) = dense_to_csc(n, &a_perm, true);
    let mut lu = BasicLu::default();
    lu.factorize(&cp, &ri, &va).unwrap();
    assert_eq!(lu.stats().unwrap().bump_size, 0);
    assert_eq!(lu.stats().unwrap().nnz_l, 0);

    let x_true: Vec<f64> = (0..n).map(|k| (k + 1) as f64).collect();
    let mut b = matvec(n, &a_perm, &x_true);
    lu.solve(&mut b).unwrap();
    assert!(rel_diff(&b, &x_true) < 1e-13);

    // 2) Permuted lower triangular with large off-diagonal entries: purely row singletons (`bump_size == 0`).
    let mut l_mat = vec![0.0_f64; n * n];
    for j in 0..n {
        l_mat[j + j * n] = 1.0;
        for i in (j + 1)..n {
            if (i * 5 + j) % 4 == 0 {
                l_mat[i + j * n] = 250.0 * rng.next_f64_signed();
            }
        }
    }
    let mut l_perm = vec![0.0_f64; n * n];
    for j in 0..n {
        for i in 0..n {
            l_perm[pr[i] + pc[j] * n] = l_mat[i + j * n];
        }
    }
    let (cp2, ri2, va2) = dense_to_csc(n, &l_perm, true);
    lu.factorize(&cp2, &ri2, &va2).unwrap();
    assert_eq!(lu.stats().unwrap().bump_size, 0);
    let mut b2 = matvec(n, &l_perm, &x_true);
    lu.solve(&mut b2).unwrap();
    assert!(rel_diff(&b2, &x_true) < 1e-12);

    // 3) Mixed column + row singletons (`bump_size == 0`).
    let mut mixed = vec![0.0_f64; 4 * 4];
    // Row 0, col 0 -> col singleton (0,0) = 2, U_{0,1}=3, U_{0,2}=-1
    mixed[0] = 2.0;
    mixed[4] = 3.0;
    mixed[8] = -1.0;
    // Row 1 has col 1 only -> row singleton after (0,0), L_{2,1}, L_{3,1}
    mixed[1 + 4] = 4.0;
    mixed[2 + 4] = 8.0;
    mixed[3 + 4] = -4.0;
    // Remaining 2x2 lower triangular: row 2 has col 2 only, row 3 has col 2 and col 3
    mixed[2 + 8] = 5.0;
    mixed[3 + 8] = 10.0;
    mixed[3 + 12] = 6.0;
    let (cpm, rim, vam) = dense_to_csc(4, &mixed, true);
    lu.factorize(&cpm, &rim, &vam).unwrap();
    assert_eq!(lu.stats().unwrap().bump_size, 0);
    let mut bm = matvec(4, &mixed, &[1.0, -2.0, 3.0, 0.5]);
    lu.solve(&mut bm).unwrap();
    assert!(rel_diff(&bm, &[1.0, -2.0, 3.0, 0.5]) < 1e-14);

    // 4) abs_pivot_tol singleton deferral: tiny singleton pivot must be deferred out of singletons.
    let col_ptr = [0, 2, 4];
    let row_idx = [0, 1, 0, 1];
    let values = [1e-18, 1.0, 1.0, 0.0];
    lu.factorize(&col_ptr, &row_idx, &values).unwrap();
    let mut b_def = [1.0, 1.0];
    lu.solve(&mut b_def).unwrap();
    assert!(rel_diff(&b_def, &[1.0, 1.0 - 1e-18]) < 1e-14);

    let tiny_vals = [1e-18, 0.0, 1.0, 1.0]; // col 0 is singleton (0,0)=1e-18 < abs_pivot_tol
    let err2 = lu.factorize(&col_ptr, &row_idx, &tiny_vals).unwrap_err();
    assert_eq!(err2, Error::Singular { rank: 1, n: 2 });
}

#[test]
fn bump_elimination_handles_shared_pivot_row_cancellation_and_dead_cols() {
    // 1) 5x5 bump where (0,0) has unique minimum Markowitz cost (2-1)*(2-1)=1,
    //    forcing elimination of (0,0) first so (1,1) cancels numerically to 0.0
    //    and MUST be deleted symmetrically from both `cols` and `rows`.
    let a = [
        2.0, 2.0, 0.0, 0.0, 0.0, // col 0 (deg 2)
        2.0, 2.0, 3.0, 1.0, 0.0, // col 1 (deg 4)
        0.0, 1.0, 4.0, 1.0, 2.0, // col 2 (deg 4)
        0.0, 1.0, 2.0, 5.0, 1.0, // col 3 (deg 4)
        0.0, 0.0, 1.0, 2.0, 6.0, // col 4 (deg 3)
    ];
    let (cp, ri, va) = dense_to_csc(5, &a, true);
    assert_eq!(csc_to_dense(5, &cp, &ri, &va), a);
    let mut lu = BasicLu::default();
    lu.factorize(&cp, &ri, &va).unwrap();
    assert_eq!(lu.stats().unwrap().bump_size, 5);
    assert_eq!(lu.stats().unwrap().rank, 5);

    let x_true = [1.5, -2.0, 0.75, 3.25, -1.0];
    let mut b = matvec(5, &a, &x_true);
    let b_copy = b.clone();
    lu.solve(&mut b).unwrap();
    assert!(scaled_residual(5, &a, &b, &b_copy) < 1e-14);
    assert!(rel_diff(&b, &x_true) < 1e-13);

    let mut bt = matvec_t(5, &a, &x_true);
    let bt_copy = bt.clone();
    lu.solve_transpose(&mut bt).unwrap();
    assert!(scaled_residual_t(5, &a, &bt, &bt_copy) < 1e-14);
    assert!(rel_diff(&bt, &x_true) < 1e-13);

    // 2) Dead-column eviction in degree-2 column scan and degree-1 row singleton scan.
    let dead = [
        2.0, 1.0, 1.0, 0.0, // col 0 (deg 3)
        1.0, 3.0, 1.0, 1.0, // col 1 (deg 4)
        1e-18, 1e-18, 0.0, 0.0, // col 2 (deg 2, col_max < abs_pivot_tol!)
        0.0, 1.0, 2.0, 4.0, // col 3 (deg 3)
    ];
    let (cpd, rid, vad) = dense_to_csc(4, &dead, true);
    let err = lu.factorize(&cpd, &rid, &vad).unwrap_err();
    assert_eq!(err, Error::Singular { rank: 3, n: 4 });

    // Trigger bump degree-1 row singleton pointing to a dead column (1e-18) via deferred singleton.
    let dead_row_s = [
        1e-18, 1e-18, 0.0, // col 0 (deg 2, dead)
        0.0, 2.0, 1.0, // col 1 (deg 2)
        0.0, 1.0, 3.0, // col 2 (deg 2)
    ];
    let (cpr, rir, var) = dense_to_csc(3, &dead_row_s, true);
    assert_eq!(lu.factorize(&cpr, &rir, &var).unwrap_err(), Error::Singular { rank: 2, n: 3 });

    // Under allow_singular=true, unit-column completion yields rank 3 and solves any RHS
    // in the range of the non-dead columns accurately (`x[2] == 0.0`).
    let mut lu_allow = BasicLu::new(BasicLuParams {
        allow_singular: true,
        ..BasicLuParams::default()
    });
    lu_allow.factorize(&cpd, &rid, &vad).unwrap();
    assert_eq!(lu_allow.stats().unwrap().rank, 3);
    let x_range = [1.5, -0.5, 0.0, 2.0];
    let mut rhs = matvec(4, &dead, &x_range);
    lu_allow.solve(&mut rhs).unwrap();
    assert!(rel_diff(&rhs, &x_range) < 1e-13);
}

#[test]
fn dlatms_dense_and_random_sparse_match_dgesv() {
    let mut rng = SplitMix64(2026);
    for (case, &(n, cond)) in [(6usize, 1e2), (12, 1e4), (24, 1e3), (36, 1e5)].iter().enumerate() {
        let params = Dlatms {
            dist: Dist::Uniform,
            sym: DlatmsSym::NonSymmetric,
            mode: 3,
            cond,
            dmax: 1.0,
            pack: DlatmsPack::Full,
        };
        let mut a = vec![0.0; n * n];
        let mut d = vec![0.0; n];
        let mut seed = [11 + case as u32 * 7, 29, 47 + case as u32, 83];
        dlatms(&params, n, n, &mut a, &mut d, &mut seed).unwrap();

        let (cp, ri, va) = dense_to_csc(n, &a, false);
        let mut lu = BasicLu::default();
        lu.factorize(&cp, &ri, &va).unwrap();
        assert_eq!(lu.stats().unwrap().rank, n);

        let x_true: Vec<f64> = (0..n).map(|_| rng.next_f64_signed()).collect();
        let b_orig = matvec(n, &a, &x_true);
        let mut b_lu = b_orig.clone();
        let mut b_lapack = b_orig.clone();
        let mut a_lapack = a.clone();

        lu.solve(&mut b_lu).unwrap();
        dgesv(&mut a_lapack, &mut b_lapack).unwrap();

        let tol = 1e-12 * (cond / 1e2).max(1.0);
        assert!(scaled_residual(n, &a, &b_lu, &b_orig) < 1e-12);
        assert!(rel_diff(&b_lu, &b_lapack) < tol);

        // Verify transpose solve against dgesv(A^T).
        let bt_orig = matvec_t(n, &a, &x_true);
        let mut bt_lu = bt_orig.clone();
        let mut bt_lapack = bt_orig.clone();
        let mut at_lapack = transpose_dense(n, &a);

        lu.solve_transpose(&mut bt_lu).unwrap();
        dgesv(&mut at_lapack, &mut bt_lapack).unwrap();
        assert!(scaled_residual_t(n, &a, &bt_lu, &bt_orig) < 1e-12);
        assert!(rel_diff(&bt_lu, &bt_lapack) < tol);
    }

    // Random sparse nonsymmetric matrices (density ~15% + diagonal dominance perturbations).
    for &(n, seed_val) in &[(30usize, 701u64), (50, 907)] {
        let mut prng = SplitMix64(seed_val);
        let mut a = vec![0.0_f64; n * n];
        for j in 0..n {
            a[j + j * n] = 3.0 + prng.next_f64_signed();
            for _ in 0..5 {
                let i = prng.next_usize(n);
                a[i + j * n] += 0.5 * prng.next_f64_signed();
            }
        }
        let (cp, ri, va) = dense_to_csc(n, &a, true);
        let mut lu = BasicLu::default();
        lu.factorize(&cp, &ri, &va).unwrap();

        let b_orig: Vec<f64> = (0..n).map(|_| prng.next_f64_signed()).collect();
        let mut x_lu = b_orig.clone();
        let mut x_lapack = b_orig.clone();
        let mut a_lapack = a.clone();
        lu.solve(&mut x_lu).unwrap();
        dgesv(&mut a_lapack, &mut x_lapack).unwrap();
        assert!(rel_diff(&x_lu, &x_lapack) < 1e-12);

        let mut xt_lu = b_orig.clone();
        let mut xt_lapack = b_orig.clone();
        let mut at_lapack = transpose_dense(n, &a);
        lu.solve_transpose(&mut xt_lu).unwrap();
        dgesv(&mut at_lapack, &mut xt_lapack).unwrap();
        assert!(rel_diff(&xt_lu, &xt_lapack) < 1e-12);
    }
}

#[test]
fn hypersparse_solves_match_dense_across_threshold_transitions() {
    let n = 80;
    let mut rng = SplitMix64(5555);
    let mut a = vec![0.0_f64; n * n];
    // Block-sparse banded/coupled structure so small RHS reaches small subset,
    // while dense RHS crosses `sparse_threshold = 0.05`.
    for j in 0..n {
        a[j + j * n] = 5.0 + 0.3 * rng.next_f64_signed();
        if j + 1 < n && (j % 8 != 7) {
            a[(j + 1) + j * n] = -1.2;
        }
        if j >= 1 && (j % 8 != 0) {
            a[(j - 1) + j * n] = 0.8;
        }
        if j + 16 < n && j % 11 == 0 {
            a[(j + 16) + j * n] = 0.4;
        }
    }
    let (cp, ri, va) = dense_to_csc(n, &a, true);
    let mut lu = BasicLu::default();
    lu.factorize(&cp, &ri, &va).unwrap();

    let mut out_idx = Vec::new();
    let mut out_val = Vec::new();

    // Test 1-nnz, 5-nnz, canceling duplicates, and threshold-crossing dense RHS.
    let cases = vec![
        (vec![3], vec![2.5]),
        (vec![1, 17, 33, 49, 65], vec![1.0, -2.0, 0.5, 3.0, -1.5]),
        (vec![4, 4, 12], vec![2.0, -2.0, 0.0]), // all cancel to zero
        ((0..20).collect::<Vec<usize>>(), vec![1.25; 20]),
        ((0..n).collect::<Vec<usize>>(), (0..n).map(|k| ((k % 9) as f64) - 4.0).collect()),
    ];

    for (rhs_idx, rhs_val) in cases {
        let mut dense_b = vec![0.0_f64; n];
        for (&r, &v) in rhs_idx.iter().zip(rhs_val.iter()) {
            dense_b[r] += v;
        }

        // FTRAN sparse vs dense.
        let mut x_dense = dense_b.clone();
        lu.solve(&mut x_dense).unwrap();
        lu.solve_sparse(&rhs_idx, &rhs_val, &mut out_idx, &mut out_val).unwrap();

        let mut x_sparse = vec![0.0_f64; n];
        for (&idx, &v) in out_idx.iter().zip(out_val.iter()) {
            x_sparse[idx] += v;
        }
        assert!(rel_diff(&x_sparse, &x_dense) < 1e-14);

        // BTRAN sparse vs dense.
        let mut xt_dense = dense_b.clone();
        lu.solve_transpose(&mut xt_dense).unwrap();
        lu.solve_sparse_transpose(&rhs_idx, &rhs_val, &mut out_idx, &mut out_val).unwrap();

        let mut xt_sparse = vec![0.0_f64; n];
        for (&idx, &v) in out_idx.iter().zip(out_val.iter()) {
            xt_sparse[idx] += v;
        }
        assert!(rel_diff(&xt_sparse, &xt_dense) < 1e-14);
    }

    // Exercise sparse_threshold = 0.0 (immediate dense fallback) and 1.0 (always hyper-sparse).
    for thresh in [0.0, 1.0] {
        let mut lu_t = BasicLu::new(BasicLuParams {
            sparse_threshold: thresh,
            ..BasicLuParams::default()
        });
        lu_t.factorize(&cp, &ri, &va).unwrap();
        let rhs_idx = vec![2, 10, 25];
        let rhs_val = vec![1.0, -3.0, 2.0];
        let mut dense_b = vec![0.0; n];
        dense_b[2] = 1.0;
        dense_b[10] = -3.0;
        dense_b[25] = 2.0;

        let mut x_d = dense_b.clone();
        lu_t.solve(&mut x_d).unwrap();
        lu_t.solve_sparse(&rhs_idx, &rhs_val, &mut out_idx, &mut out_val).unwrap();
        let mut x_s = vec![0.0; n];
        for (&i, &v) in out_idx.iter().zip(out_val.iter()) {
            x_s[i] = v;
        }
        assert!(rel_diff(&x_s, &x_d) < 1e-14);
    }

    // Exercise mid-solve sparse->dense fallback in U (after L succeeds) and L^T (after U^T succeeds):
    // Build a 30x30 matrix where L is trivial/sparse and U row/col 29 connects to all 0..29,
    // plus L has a chain below row 0, with sparse_threshold = 0.2 (max_reach = 6).
    let m = 30;
    let mut mat = vec![0.0_f64; m * m];
    for j in 0..m {
        mat[j + j * m] = 4.0;
        if j < m - 1 {
            // Arrow column at m-1 couples U upward across all rows!
            mat[j + (m - 1) * m] = 0.25;
            // Subdiagonal row 0..m-1 in col 0 couples L^T upward!
            if j > 0 {
                mat[j] = 0.25;
            }
        }
    }
    let (cpm, rim, vam) = dense_to_csc(m, &mat, true);
    let mut lu_mid = BasicLu::new(BasicLuParams {
        sparse_threshold: 0.2,
        ..BasicLuParams::default()
    });
    lu_mid.factorize(&cpm, &rim, &vam).unwrap();

    // FTRAN from row m-1: L reach is {m-1} (1 <= 6), then U reach hits all 30 (> 6)!
    let mut x_dense = vec![0.0; m];
    x_dense[m - 1] = 1.0;
    lu_mid.solve(&mut x_dense).unwrap();
    lu_mid.solve_sparse(&[m - 1], &[1.0], &mut out_idx, &mut out_val).unwrap();
    let mut x_sp = vec![0.0; m];
    for (&i, &v) in out_idx.iter().zip(out_val.iter()) {
        x_sp[i] = v;
    }
    assert!(rel_diff(&x_sp, &x_dense) < 1e-14);

    // BTRAN from col 0: U^T reach is small, then L^T reach hits many rows (> 6)!
    let mut xt_dense = vec![0.0; m];
    xt_dense[0] = 1.0;
    lu_mid.solve_transpose(&mut xt_dense).unwrap();
    lu_mid.solve_sparse_transpose(&[0], &[1.0], &mut out_idx, &mut out_val).unwrap();
    let mut xt_sp = vec![0.0; m];
    for (&i, &v) in out_idx.iter().zip(out_val.iter()) {
        xt_sp[i] = v;
    }
    assert!(rel_diff(&xt_sp, &xt_dense) < 1e-14);
}

#[test]
fn deep_triangular_chain_iterative_dfs_4096_does_not_overflow_stack() {
    // 4096-node bidiagonal chain (with coupling into bump via a tiny 2x2 cap)
    // verifying iterative DFS stack handles deep paths cleanly on all targets including wasm32.
    let n = 4096;
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::with_capacity(2 * n);
    let mut values = Vec::with_capacity(2 * n);
    col_ptr.push(0);
    for j in 0..n {
        row_idx.push(j);
        values.push(2.0);
        if j + 1 < n {
            row_idx.push(j + 1);
            values.push(-0.5);
        }
        col_ptr.push(row_idx.len());
    }

    let mut lu = BasicLu::new(BasicLuParams {
        sparse_threshold: 1.0,
        ..BasicLuParams::default()
    });
    lu.factorize(&col_ptr, &row_idx, &values).unwrap();

    let mut out_idx = Vec::new();
    let mut out_val = Vec::new();
    lu.solve_sparse(&[0], &[1.0], &mut out_idx, &mut out_val).unwrap();
    // Exact solution: x[k] = 0.5 * 4^{-k}
    let mut x = vec![0.0; n];
    for (&i, &v) in out_idx.iter().zip(out_val.iter()) {
        x[i] = v;
    }
    assert!((x[0] - 0.5).abs() < 1e-14);
    assert!((x[10] - 0.5 * 0.25_f64.powi(10)).abs() < 1e-14);

    // Transpose chain from index `n - 1` walks all 4096 steps backward!
    lu.solve_sparse_transpose(&[n - 1], &[1.0], &mut out_idx, &mut out_val).unwrap();
    let mut xt = vec![0.0; n];
    for (&i, &v) in out_idx.iter().zip(out_val.iter()) {
        xt[i] = v;
    }
    assert!((xt[n - 1] - 0.5).abs() < 1e-14);
    assert!((xt[n - 11] - 0.5 * 0.25_f64.powi(10)).abs() < 1e-14);
}

#[test]
fn fifty_consecutive_updates_match_fresh_factorization_and_dgesv() {
    // Build a sparse 40x40 LP-basis-like matrix, then perform 50 consecutive
    // column replacements exercising:
    //   1) `m == p` fast path (no eta),
    //   2) Schork–Gondzio pure-permutation shortcut (empty/post-m `u_row[p_row]`, no eta),
    //   3) General Forrest–Tomlin updates (non-empty `u_row[p_row]`, including entries past `m`).
    let n = 40;
    let mut rng = SplitMix64(9999);
    let mut a = vec![0.0_f64; n * n];
    for j in 0..n {
        a[j + j * n] = 4.5 + 0.4 * rng.next_f64_signed();
        // Add sparse upper-triangular and coupled bump entries, keeping several
        // isolated columns/rows to trigger `m == p` and pure-permutation paths.
        if j + 1 < n && j % 5 != 0 {
            a[(j + 1) + j * n] = 0.6 * rng.next_f64_signed();
        }
        if j >= 2 && j % 4 != 0 {
            a[(j - 2) + j * n] = 0.7 * rng.next_f64_signed();
        }
        if j + 7 < n && j % 3 == 0 {
            a[(j + 7) + j * n] = 0.5 * rng.next_f64_signed();
        }
        if j >= 9 && j % 6 == 0 {
            a[(j - 9) + j * n] = -0.4 * rng.next_f64_signed();
        }
    }

    let (cp0, ri0, va0) = dense_to_csc(n, &a, true);
    let mut lu = BasicLu::default();
    lu.factorize(&cp0, &ri0, &va0).unwrap();

    // First verify explicit m == p fast path and pure-permutation shortcut on known columns.
    let mut prev_ft = lu.stats().unwrap().forrest_tomlin_updates;
    assert_eq!(prev_ft, 0);

    // Explicitly craft an `m == p` replacement on column 0 (if row 0 is at p=0).
    let mut out_idx = Vec::new();
    let mut out_val = Vec::new();

    for step in 1..=50 {
        let col = (step * 7 + 3) % n;
        let mut new_rows = Vec::new();
        let mut new_vals = Vec::new();

        // Mix sparse single-entry, upper-only, and full-coupling columns across the 50 updates.
        if step <= 3 {
            // Early steps: replace diagonal-only column to hit fast paths too.
            new_rows.push(col);
            new_vals.push(3.5 + 0.1 * (step as f64));
        } else {
            let mut chosen = vec![col];
            for _ in 0..4 {
                let r = rng.next_usize(n);
                if !chosen.contains(&r) {
                    chosen.push(r);
                }
            }
            chosen.sort_unstable();
            for &r in &chosen {
                let v = if r == col {
                    4.0 + 0.5 * rng.next_f64_signed()
                } else {
                    0.45 * rng.next_f64_signed()
                };
                new_rows.push(r);
                new_vals.push(v);
            }
        }

        // Update reference dense matrix `a` column `col`.
        for i in 0..n {
            a[i + col * n] = 0.0;
        }
        for (&r, &v) in new_rows.iter().zip(new_vals.iter()) {
            a[r + col * n] = v;
        }

        lu.update(col, &new_rows, &new_vals).unwrap();
        let st = *lu.stats().unwrap();
        assert_eq!(st.updates, step);
        if step == 50 {
            assert!(st.forrest_tomlin_updates > 0, "expected general Forrest–Tomlin updates");
            assert!(
                st.forrest_tomlin_updates < st.updates,
                "expected some pure-permutation / fast-path updates too"
            );
        }
        prev_ft = st.forrest_tomlin_updates;
        let _ = prev_ft;

        // Fresh BasicLu::factorize + LAPACK dgesv reference.
        let (cp_ref, ri_ref, va_ref) = dense_to_csc(n, &a, true);
        let mut lu_fresh = BasicLu::default();
        lu_fresh.factorize(&cp_ref, &ri_ref, &va_ref).unwrap();

        let b_dense: Vec<f64> = (0..n).map(|_| rng.next_f64_signed()).collect();
        let mut x_upd = b_dense.clone();
        let mut x_fresh = b_dense.clone();
        let mut x_lapack = b_dense.clone();
        let mut a_lapack = a.clone();

        lu.solve(&mut x_upd).unwrap();
        lu_fresh.solve(&mut x_fresh).unwrap();
        dgesv(&mut a_lapack, &mut x_lapack).unwrap();

        assert!(rel_diff(&x_upd, &x_lapack) < 1e-11, "step {step} dense FTRAN diverged");
        assert!(rel_diff(&x_upd, &x_fresh) < 1e-11, "step {step} fresh FTRAN diverged");

        // Dense BTRAN.
        let mut xt_upd = b_dense.clone();
        let mut xt_lapack = b_dense.clone();
        let mut at_lapack = transpose_dense(n, &a);
        lu.solve_transpose(&mut xt_upd).unwrap();
        dgesv(&mut at_lapack, &mut xt_lapack).unwrap();
        assert!(rel_diff(&xt_upd, &xt_lapack) < 1e-11, "step {step} dense BTRAN diverged");

        // Hyper-sparse FTRAN and BTRAN with 2-entry sparse RHS.
        let s_idx = vec![step % n, (step * 13 + 5) % n];
        let s_val = vec![1.5, -0.75];
        let mut sb = vec![0.0_f64; n];
        for (&idx, &v) in s_idx.iter().zip(s_val.iter()) {
            sb[idx] += v;
        }
        let mut sx_ref = sb.clone();
        lu.solve(&mut sx_ref).unwrap();
        lu.solve_sparse(&s_idx, &s_val, &mut out_idx, &mut out_val).unwrap();
        let mut sx_sp = vec![0.0_f64; n];
        for (&idx, &v) in out_idx.iter().zip(out_val.iter()) {
            sx_sp[idx] += v;
        }
        assert!(rel_diff(&sx_sp, &sx_ref) < 1e-12, "step {step} sparse FTRAN diverged");

        let mut sxt_ref = sb;
        lu.solve_transpose(&mut sxt_ref).unwrap();
        lu.solve_sparse_transpose(&s_idx, &s_val, &mut out_idx, &mut out_val).unwrap();
        let mut sxt_sp = vec![0.0_f64; n];
        for (&idx, &v) in out_idx.iter().zip(out_val.iter()) {
            sxt_sp[idx] += v;
        }
        assert!(rel_diff(&sxt_sp, &sxt_ref) < 1e-12, "step {step} sparse BTRAN diverged");
    }
}

#[test]
fn update_pure_permutation_and_tail_entries_past_m_are_exact() {
    // Upper-triangular 4x4 with u_row[0] = {col 1, col 3}:
    // [ 2  3  0  5 ]
    // [ 0  4  1  0 ]
    // [ 0  0  6  2 ]
    // [ 0  0  0  7 ]
    let mut a = [
        2.0, 0.0, 0.0, 0.0, // col 0
        3.0, 4.0, 0.0, 0.0, // col 1
        0.0, 1.0, 6.0, 0.0, // col 2
        5.0, 0.0, 2.0, 7.0, // col 3
    ];
    let (cp, ri, va) = dense_to_csc(4, &a, true);
    let mut lu = BasicLu::default();
    lu.factorize(&cp, &ri, &va).unwrap();
    assert_eq!(lu.stats().unwrap().bump_size, 0);

    // Case A: Replace col 0 with spike [1, 2, 0, 0] -> m = 1 < 3, while row 0 has
    // entry at col 3 (pos 3 > m = 1) AND col 1 (pos 1 <= m).
    // `w` MUST eliminate col 3 too (`w` not truncated at `m`), keeping solve exact!
    let new_r = [0, 1];
    let new_v = [1.0, 2.0];
    a[0] = 1.0;
    a[1] = 2.0;
    lu.update(0, &new_r, &new_v).unwrap();
    assert_eq!(lu.stats().unwrap().forrest_tomlin_updates, 1);

    let b = [1.0, -2.0, 3.0, 4.0];
    let mut x = b;
    let mut x_ref = b;
    let mut a_ref = a;
    lu.solve(&mut x).unwrap();
    dgesv(&mut a_ref, &mut x_ref).unwrap();
    assert!(rel_diff(&x, &x_ref) < 1e-13);

    let mut xt = b;
    let mut xt_ref = b;
    let mut at_ref = transpose_dense(4, &a);
    lu.solve_transpose(&mut xt).unwrap();
    dgesv(&mut at_ref, &mut xt_ref).unwrap();
    assert!(rel_diff(&xt, &xt_ref) < 1e-13);

    // Case B: Pure-permutation shortcut where row `p_row` has NO entries in `(p+1)..=m`,
    // only at `> m`!
    let b_mat = [
        2.0, 0.0, 0.0, 0.0, // col 0
        0.0, 3.0, 0.0, 0.0, // col 1
        0.0, 1.0, 5.0, 0.0, // col 2
        4.0, 0.0, 0.0, 6.0, // col 3: row 0 has entry ONLY at col 3 (pos 3)
    ];
    let (cpb, rib, vab) = dense_to_csc(4, &b_mat, true);
    let mut lu2 = BasicLu::default();
    lu2.factorize(&cpb, &rib, &vab).unwrap();
    // Replace col 0 with [2.0, 5.0, 0.0, 0.0] -> m = 1, u_row[0] only has col 3 (pos 3 > 1)!
    lu2.update(0, &[0, 1], &[2.0, 5.0]).unwrap();
    assert_eq!(lu2.stats().unwrap().updates, 1);
    assert_eq!(lu2.stats().unwrap().forrest_tomlin_updates, 0);

    let mut b_updated = b_mat;
    b_updated[0] = 2.0;
    b_updated[1] = 5.0;
    let mut x2 = b;
    let mut x2_ref = b;
    let mut a2_ref = b_updated;
    lu2.solve(&mut x2).unwrap();
    dgesv(&mut a2_ref, &mut x2_ref).unwrap();
    assert!(rel_diff(&x2, &x2_ref) < 1e-13);
}

#[test]
fn update_errors_and_singular_rollback_preserve_state() {
    let a = [
        2.0, 0.0, 0.0, // col 0
        3.0, 4.0, 0.0, // col 1
        1.0, 2.0, 5.0, // col 2
    ];
    let (cp, ri, va) = dense_to_csc(3, &a, true);
    let mut lu = BasicLu::new(BasicLuParams {
        max_updates: 2,
        ..BasicLuParams::default()
    });
    lu.factorize(&cp, &ri, &va).unwrap();

    // Pre-update reference solution.
    let rhs = [1.0, -3.0, 2.0];
    let mut x_before = rhs;
    lu.solve(&mut x_before).unwrap();

    // Validation errors.
    assert_eq!(lu.update(3, &[0], &[1.0]), Err(Error::InvalidColumn { col: 3, n: 3 }));
    assert_eq!(lu.update(0, &[0, 1], &[1.0]), Err(Error::BadShape));
    assert_eq!(lu.update(0, &[3], &[1.0]), Err(Error::InvalidStructure));
    assert_eq!(lu.update(0, &[0], &[f64::NAN]), Err(Error::NotFinite));

    // SingularUpdate: zero column.
    assert_eq!(lu.update(1, &[], &[]), Err(Error::SingularUpdate));
    // SingularUpdate: spike with m < p (replacing col 2 with a linear combination of col 0 and col 1 -> m < 2).
    assert_eq!(lu.update(2, &[0], &[5.0]), Err(Error::SingularUpdate));
    // SingularUpdate: Forrest–Tomlin cancellation producing zero pivot alpha.
    // Replacing col 0 with [3.0, 4.0, 0.0] makes col 0 proportional to col 1!
    assert_eq!(lu.update(0, &[0, 1], &[3.0, 4.0]), Err(Error::SingularUpdate));
    // SingularUpdate in pure-permutation path (v[p_row] < abs_pivot_tol while m > p).
    let pure_perm_a = [
        2.0, 0.0, 0.0, // col 0
        0.0, 3.0, 0.0, // col 1 (row 0 has NO entry at col 1!)
        0.0, 0.0, 4.0, // col 2
    ];
    let (cpp, rip, vap) = dense_to_csc(3, &pure_perm_a, true);
    let mut lu_pp = BasicLu::default();
    lu_pp.factorize(&cpp, &rip, &vap).unwrap();
    assert_eq!(lu_pp.update(0, &[1], &[5.0]), Err(Error::SingularUpdate));

    // Hyper-sparse update (`sparse_threshold = 0.95` on n = 20) exercising reach_known=true,
    // sparse U^T eta chaining (`u_row[r_k]` non-empty), and hyper-sparse FTRAN/BTRAN through etas!
    let n_sp = 20;
    let mut a_sp = vec![0.0_f64; n_sp * n_sp];
    for j in 0..n_sp {
        a_sp[j + j * n_sp] = 4.0;
        if j + 1 < n_sp {
            a_sp[j + (j + 1) * n_sp] = 0.5;
        }
        if j + 2 < n_sp {
            a_sp[j + (j + 2) * n_sp] = -0.25;
        }
    }
    let (cps, ris, vas) = dense_to_csc(n_sp, &a_sp, true);
    let mut lu_sp = BasicLu::new(BasicLuParams {
        sparse_threshold: 0.95,
        ..BasicLuParams::default()
    });
    lu_sp.factorize(&cps, &ris, &vas).unwrap();
    lu_sp.update(0, &[0, 1, 2], &[3.0, 1.0, 0.5]).unwrap();
    lu_sp.update(1, &[1, 2, 3], &[3.5, 0.8, -0.4]).unwrap();
    for i in 0..n_sp {
        a_sp[i] = 0.0;
        a_sp[n_sp + i] = 0.0;
    }
    a_sp[0] = 3.0;
    a_sp[1] = 1.0;
    a_sp[2] = 0.5;
    a_sp[n_sp + 1] = 3.5;
    a_sp[n_sp + 2] = 0.8;
    a_sp[n_sp + 3] = -0.4;
    assert_eq!(lu_sp.stats().unwrap().forrest_tomlin_updates, 2);

    let b_sp: Vec<f64> = (0..n_sp).map(|i| (i + 1) as f64).collect();
    let mut x_sp_lu = b_sp.clone();
    let mut x_sp_ref = b_sp;
    let mut a_sp_ref = a_sp.clone();
    lu_sp.solve(&mut x_sp_lu).unwrap();
    dgesv(&mut a_sp_ref, &mut x_sp_ref).unwrap();
    assert!(rel_diff(&x_sp_lu, &x_sp_ref) < 1e-13);

    // Hyper-sparse FTRAN and BTRAN directly through accumulated Forrest–Tomlin etas:
    let mut out_i = Vec::new();
    let mut out_v = Vec::new();
    for &rhs_pos in &[1usize, 3, 18] {
        let mut rhs_d = vec![0.0; n_sp];
        rhs_d[rhs_pos] = 2.0;

        let mut f_d = rhs_d.clone();
        lu_sp.solve(&mut f_d).unwrap();
        lu_sp.solve_sparse(&[rhs_pos], &[2.0], &mut out_i, &mut out_v).unwrap();
        let mut f_s = vec![0.0; n_sp];
        for (&idx, &val) in out_i.iter().zip(out_v.iter()) {
            f_s[idx] += val;
        }
        assert!(rel_diff(&f_s, &f_d) < 1e-13);

        let mut b_d = rhs_d;
        lu_sp.solve_transpose(&mut b_d).unwrap();
        lu_sp.solve_sparse_transpose(&[rhs_pos], &[2.0], &mut out_i, &mut out_v).unwrap();
        let mut b_s = vec![0.0; n_sp];
        for (&idx, &val) in out_i.iter().zip(out_v.iter()) {
            b_s[idx] += val;
        }
        assert!(rel_diff(&b_s, &b_d) < 1e-13);
    }

    // Exercise BTRAN mid-solve switch in L^T (after U^T DFS succeeds):
    // Rows 1..nl-2 are row singletons coupling into row `nl-1`, while col `nl-1` has an
    // off-diagonal `(0, nl-1)` so `col nl-1` is not peeled as a column singleton. Thus `L`
    // accumulates `nl - 2 = 28` entries in row `nl - 1`, keeping U^T reach <= 2 (within
    // `max_reach = 6`) while L^T DFS overflows `max_reach` and switches cleanly to dense.
    let nl = 30;
    let mut al = vec![0.0_f64; nl * nl];
    for j in 0..nl {
        al[j + j * nl] = 4.0;
        if j < nl - 1 {
            al[(nl - 1) + j * nl] = 0.2;
        }
    }
    al[0 + (nl - 1) * nl] = 0.1;
    let (cpl, ril, val) = dense_to_csc(nl, &al, true);
    let mut lu_lt = BasicLu::new(BasicLuParams {
        sparse_threshold: 0.2, // max_reach = 6
        ..BasicLuParams::default()
    });
    lu_lt.factorize(&cpl, &ril, &val).unwrap();
    let mut lt_dense = vec![0.0; nl];
    lt_dense[nl - 1] = 1.0;
    lu_lt.solve_transpose(&mut lt_dense).unwrap();
    lu_lt.solve_sparse_transpose(&[nl - 1], &[1.0], &mut out_i, &mut out_v).unwrap();
    let mut lt_sp = vec![0.0; nl];
    for (&i, &v) in out_i.iter().zip(out_v.iter()) {
        lt_sp[i] = v;
    }
    assert!(rel_diff(&lt_sp, &lt_dense) < 1e-14);

    // Unsorted/duplicate entries in `update`: canceling duplicates -> SingularUpdate rollback;
    // unsorted + duplicate accumulating entries -> exact solution matching `dgesv`.
    assert_eq!(lu_sp.update(0, &[2, 0, 2], &[1.0, 0.0, -1.0]), Err(Error::SingularUpdate));
    lu_sp.update(0, &[2, 0, 2, 1], &[0.2, 3.0, 0.3, 1.0]).unwrap();
    a_sp[0] = 3.0;
    a_sp[1] = 1.0;
    a_sp[2] = 0.5;
    let mut rhs_chk: Vec<f64> = (0..n_sp).map(|i| (i * 2 + 1) as f64).collect();
    let mut ref_chk = rhs_chk.clone();
    let mut a_chk = a_sp.clone();
    lu_sp.solve(&mut rhs_chk).unwrap();
    dgesv(&mut a_chk, &mut ref_chk).unwrap();
    assert!(rel_diff(&rhs_chk, &ref_chk) < 1e-13);

    // Verify factors & workspace are completely intact after all failed updates!
    assert_eq!(lu.stats().unwrap().updates, 0);
    assert_eq!(lu.stats().unwrap().forrest_tomlin_updates, 0);
    let mut x_after = rhs;
    lu.solve(&mut x_after).unwrap();
    assert!(rel_diff(&x_after, &x_before) < 1e-14);

    // Exhaust max_updates = 2, then verify MaxUpdatesReached.
    lu.update(0, &[0], &[4.0]).unwrap();
    lu.update(1, &[1], &[8.0]).unwrap();
    assert_eq!(lu.update(2, &[2], &[7.0]), Err(Error::MaxUpdatesReached));
}

/// Performance benchmark matching `bench_sparse_lu` in `src/sparse/sparse_lu.rs`:
///
/// `cargo test --release --lib bench_basiclu -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_basiclu() {
    let m = 4_usize;
    let n = m * m * m * 3;
    let node = |i: usize, j: usize, k: usize| (k * m * m + j * m + i) * 3;
    let mut col_ptr = vec![0_usize];
    let mut row_idx = Vec::new();
    for k in 0..m {
        for j in 0..m {
            for i in 0..m {
                for _d in 0..3 {
                    let mut rows = Vec::new();
                    for dk in 0..3_i32 {
                        for dj in 0..3_i32 {
                            for di in 0..3_i32 {
                                let (ii, jj, kk) = (i as i32 + di - 1, j as i32 + dj - 1, k as i32 + dk - 1);
                                if ii < 0 || jj < 0 || kk < 0 || ii >= m as i32 || jj >= m as i32 || kk >= m as i32 {
                                    continue;
                                }
                                let base = node(ii as usize, jj as usize, kk as usize);
                                for d in 0..3 {
                                    rows.push(base + d);
                                }
                            }
                        }
                    }
                    rows.sort_unstable();
                    row_idx.extend_from_slice(&rows);
                    col_ptr.push(row_idx.len());
                }
            }
        }
    }
    let nnz = row_idx.len();
    let mut values = vec![0.0_f64; nnz];
    for col in 0..n {
        let mut sum = 0.0;
        for p in col_ptr[col]..col_ptr[col + 1] {
            if row_idx[p] != col {
                values[p] = -1.0;
                sum += 1.0;
            }
        }
        for p in col_ptr[col]..col_ptr[col + 1] {
            if row_idx[p] == col {
                values[p] = 10.0 + sum;
            }
        }
    }

    let mut lu = BasicLu::default();
    let reps = 80;
    let t = Instant::now();
    for _ in 0..reps {
        lu.factorize(&col_ptr, &row_idx, &values).unwrap();
    }
    let per_factorize = t.elapsed().as_secs_f64() / reps as f64;
    let stats = *lu.stats().unwrap();

    let mut b = vec![1.0; n];
    let t = Instant::now();
    for _ in 0..(reps * 10) {
        b.fill(1.0);
        lu.solve(&mut b).unwrap();
    }
    let per_solve_dense = t.elapsed().as_secs_f64() / (reps * 10) as f64;

    let mut out_idx = Vec::new();
    let mut out_val = Vec::new();
    let t = Instant::now();
    for s in 0..(reps * 10) {
        lu.solve_sparse(&[s % n], &[1.0], &mut out_idx, &mut out_val).unwrap();
    }
    let per_solve_sparse = t.elapsed().as_secs_f64() / (reps * 10) as f64;

    let t = Instant::now();
    for s in 0..50 {
        let col = (s * 7) % n;
        lu.update(col, &[col], &[15.0]).unwrap();
    }
    let per_update = t.elapsed().as_secs_f64() / 50.0;

    println!(
        "basiclu: n={n}, nnz_a={nnz}, nnz_l={}, nnz_u={}, bump={} -> factorize {:.3} ms, dense solve {:.1} us, sparse solve {:.1} us, update {:.1} us",
        stats.nnz_l,
        stats.nnz_u,
        stats.bump_size,
        per_factorize * 1e3,
        per_solve_dense * 1e6,
        per_solve_sparse * 1e6,
        per_update * 1e6
    );
}
