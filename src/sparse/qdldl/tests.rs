// Derived/transcribed from `osqp/qdldl` v0.1.9 (Apache License 2.0, Copyright OSQP Developers),
// adapted and modified for `algebra_linear` in pure safe Rust (Apache-2.0 §4(b) notice).

use super::csc::{UpperTriCscPattern, UpperTriCscView};
use super::etree::{NONE, compute_etree, prefix_sum_l_col_ptr};
use super::factor::{FactorWorkspace, NumericFactor, factor_in_place};
use super::{Error, Qdldl, QdldlStats};
use crate::lapack::dgesv;
use crate::lapack::families::{Dist, Dlatms, DlatmsPack, DlatmsSym, dlatms};

/// Deterministic SplitMix64 PRNG for zero-dependency test matrix generation.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Returns a uniform float in `[0, 1)`.
    fn next_f64_unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Returns a uniform float in `[-1, 1)`.
    fn next_f64_signed(&mut self) -> f64 {
        2.0 * self.next_f64_unit() - 1.0
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0);
        (self.next_u64() % (bound as u64)) as usize
    }
}

fn upper_csc_to_dense(n: usize, col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        for p in col_ptr[j]..col_ptr[j + 1] {
            let i = row_idx[p];
            let v = values[p];
            a[i + j * n] = v;
            a[j + i * n] = v;
        }
    }
    a
}

fn dense_to_upper_csc(n: usize, a_dense: &[f64], drop_exact_zeros: bool) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::new();
    let mut values = Vec::new();
    col_ptr.push(0);
    for j in 0..n {
        for i in 0..=j {
            let v = a_dense[i + j * n];
            if !drop_exact_zeros || v != 0.0 {
                row_idx.push(i);
                values.push(v);
            }
        }
        col_ptr.push(row_idx.len());
    }
    (col_ptr, row_idx, values)
}

fn reconstruct_ldl_t(n: usize, l_col_ptr: &[usize], l_row_idx: &[usize], l_values: &[f64], d: &[f64]) -> Vec<f64> {
    let mut l_full = vec![0.0; n * n];
    for j in 0..n {
        l_full[j + j * n] = 1.0;
        for p in l_col_ptr[j]..l_col_ptr[j + 1] {
            let i = l_row_idx[p];
            l_full[i + j * n] = l_values[p];
        }
    }
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..=i.min(j) {
                sum += l_full[i + k * n] * d[k] * l_full[j + k * n];
            }
            out[i + j * n] = sum;
        }
    }
    out
}

fn matvec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for j in 0..n {
        let xj = x[j];
        for i in 0..n {
            y[i] += a[i + j * n] * xj;
        }
    }
    y
}

fn norm_inf_vec(x: &[f64]) -> f64 {
    x.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()))
}

fn norm_inf_mat(a: &[f64], n: usize) -> f64 {
    let mut max_row = 0.0f64;
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            row_sum += a[i + j * n].abs();
        }
        max_row = max_row.max(row_sum);
    }
    max_row
}

fn factor_residual_inf(a_dense: &[f64], solver: &Qdldl, n: usize) -> f64 {
    let d = solver.d().expect("factorized solver must expose d");
    let recon = reconstruct_ldl_t(n, &solver.symbolic.l_col_ptr, &solver.numeric.l_row_idx, &solver.numeric.l_values, d);
    let diff: Vec<f64> = a_dense.iter().zip(&recon).map(|(x, y)| x - y).collect();
    let a_norm = norm_inf_mat(a_dense, n);
    let diff_norm = norm_inf_mat(&diff, n);
    if a_norm == 0.0 { diff_norm } else { diff_norm / a_norm }
}

fn scaled_solve_residual(a_dense: &[f64], x: &[f64], b: &[f64], n: usize) -> f64 {
    let ax = matvec(a_dense, x, n);
    let r: Vec<f64> = ax.iter().zip(b).map(|(ai, bi)| ai - bi).collect();
    let denom = norm_inf_mat(a_dense, n) * norm_inf_vec(x) + norm_inf_vec(b);
    if denom == 0.0 { norm_inf_vec(&r) } else { norm_inf_vec(&r) / denom }
}

fn verify_triangular_solves(solver: &Qdldl, rng: &mut SplitMix64) {
    let n = solver.symbolic.n;
    let l_col_ptr = &solver.symbolic.l_col_ptr;
    let l_row_idx = &solver.numeric.l_row_idx;
    let l_values = &solver.numeric.l_values;

    let z: Vec<f64> = (0..n).map(|_| rng.next_f64_signed() + 0.5).collect();

    // Compute rhs_l = (I + L) * z
    let mut rhs_l = z.clone();
    for j in 0..n {
        for p in l_col_ptr[j]..l_col_ptr[j + 1] {
            rhs_l[l_row_idx[p]] += l_values[p] * z[j];
        }
    }
    solver.solve_l(&mut rhs_l).unwrap();
    let err_l: Vec<f64> = rhs_l.iter().zip(&z).map(|(a, b)| a - b).collect();
    let rel_l = norm_inf_vec(&err_l) / (norm_inf_vec(&z) + 1.0);
    assert!(rel_l < 1e-12, "solve_l error {rel_l:e}");

    // Compute rhs_lt = (I + L)^T * z
    let mut rhs_lt = z.clone();
    for j in 0..n {
        for p in l_col_ptr[j]..l_col_ptr[j + 1] {
            rhs_lt[j] += l_values[p] * z[l_row_idx[p]];
        }
    }
    solver.solve_lt(&mut rhs_lt).unwrap();
    let err_lt: Vec<f64> = rhs_lt.iter().zip(&z).map(|(a, b)| a - b).collect();
    let rel_lt = norm_inf_vec(&err_lt) / (norm_inf_vec(&z) + 1.0);
    assert!(rel_lt < 1e-12, "solve_lt error {rel_lt:e}");
}

fn verify_system(
    a_dense: &[f64],
    col_ptr: &[usize],
    row_idx: &[usize],
    values: &[f64],
    expected_positive_inertia: Option<usize>,
    rng: &mut SplitMix64,
    context: &str,
) {
    let n = col_ptr.len() - 1;
    let mut solver = Qdldl::analyze(col_ptr, row_idx).unwrap();
    assert!(solver.d().is_none());
    assert!(solver.stats().is_none());
    let mut dummy = vec![0.0; n];
    assert_eq!(solver.solve(&mut dummy).unwrap_err(), Error::NotFactorized);
    assert_eq!(solver.solve_l(&mut dummy).unwrap_err(), Error::NotFactorized);
    assert_eq!(solver.solve_lt(&mut dummy).unwrap_err(), Error::NotFactorized);

    solver.factorize(col_ptr, row_idx, values).unwrap();
    let stats = solver.stats().expect("stats must be present after factorize");
    assert_eq!(stats.n, n, "{context}: stats.n");
    assert_eq!(stats.nnz_a, values.len(), "{context}: stats.nnz_a");
    assert_eq!(stats.nnz_l, solver.symbolic.nnz_l(), "{context}: stats.nnz_l");
    if let Some(expected_pos) = expected_positive_inertia {
        assert_eq!(stats.positive_inertia, expected_pos, "{context}: positive_inertia");
    }

    let fac_res = factor_residual_inf(a_dense, &solver, n);
    assert!(fac_res < 1e-12, "{context}: factorization residual {fac_res:e} >= 1e-12");

    let b: Vec<f64> = (0..n).map(|i| 1.0 + rng.next_f64_signed() + (i % 5) as f64 * 0.1).collect();
    let mut x = b.clone();
    solver.solve(&mut x).unwrap();
    let solve_res = scaled_solve_residual(a_dense, &x, &b, n);
    assert!(solve_res < 1e-12, "{context}: solve residual {solve_res:e} >= 1e-12");

    let mut a_lu = a_dense.to_vec();
    let mut x_ref = b.clone();
    dgesv(&mut a_lu, &mut x_ref).unwrap();
    let diff: Vec<f64> = x.iter().zip(&x_ref).map(|(u, v)| u - v).collect();
    let rel_dgesv = norm_inf_vec(&diff) / (norm_inf_vec(&x_ref) + 1.0);
    assert!(rel_dgesv < 1e-10, "{context}: dgesv difference {rel_dgesv:e} >= 1e-10");

    verify_triangular_solves(&solver, rng);
}

#[test]
fn error_display_messages_and_std_error_are_exact() {
    let cases = [
        (Error::BadShape, "inconsistent shape or length"),
        (Error::InvalidStructure, "invalid upper-triangular CSC structure"),
        (Error::UnsortedRows, "row indices must be strictly increasing within a column"),
        (Error::DuplicateEntry, "duplicate entry in a column"),
        (
            Error::LowerTriangularEntry { row: 2, col: 1 },
            "lower-triangular entry at (2, 1); expected row <= col",
        ),
        (Error::EmptyColumn { col: 3 }, "empty column 3 in upper-triangular matrix"),
        (Error::IndexOverflow, "index or nonzero count overflow"),
        (Error::StructureMismatch, "the matrix pattern differs from the symbolic analysis"),
        (Error::NotFinite, "the matrix or pivot has a NaN or infinite value"),
        (Error::ZeroDiagonal { index: 4 }, "zero diagonal pivot at index 4"),
        (Error::NotFactorized, "the matrix has not been factorized"),
    ];
    for (err, expected) in cases {
        assert_eq!(err.to_string(), expected);
        let dyn_err: &dyn std::error::Error = &err;
        assert_eq!(dyn_err.to_string(), expected);
    }
}

#[test]
fn csc_validation_rejects_all_malformed_inputs() {
    // Empty col_ptr
    assert_eq!(UpperTriCscPattern::new(&[], &[]).unwrap_err(), Error::BadShape);
    assert_eq!(Qdldl::analyze(&[], &[]).unwrap_err(), Error::BadShape);

    // Nonzero col_ptr[0]
    assert_eq!(UpperTriCscPattern::new(&[1, 1], &[]).unwrap_err(), Error::InvalidStructure);

    // Length mismatch between col_ptr[n] and row_idx.len()
    assert_eq!(UpperTriCscPattern::new(&[0, 1, 3], &[0, 1]).unwrap_err(), Error::BadShape);

    // Decreasing col_ptr
    assert_eq!(UpperTriCscPattern::new(&[0, 2, 1], &[0]).unwrap_err(), Error::InvalidStructure);

    // Row index >= n
    assert_eq!(UpperTriCscPattern::new(&[0, 1, 2], &[0, 2]).unwrap_err(), Error::InvalidStructure);

    // Lower-triangular entry (row > col)
    assert_eq!(
        UpperTriCscPattern::new(&[0, 2, 2], &[0, 1]).unwrap_err(),
        Error::LowerTriangularEntry { row: 1, col: 0 }
    );

    // Duplicate entry in a column
    assert_eq!(UpperTriCscPattern::new(&[0, 1, 3], &[0, 0, 0]).unwrap_err(), Error::DuplicateEntry);

    // Unsorted rows in a column
    assert_eq!(UpperTriCscPattern::new(&[0, 1, 3], &[0, 1, 0]).unwrap_err(), Error::UnsortedRows);

    // Values length mismatch in UpperTriCscView
    assert_eq!(UpperTriCscView::new(&[0, 1, 2], &[0, 1], &[1.0]).unwrap_err(), Error::BadShape);

    // Valid pattern accessors and deterministic fingerprint
    let pat = UpperTriCscPattern::new(&[0, 1, 3], &[0, 0, 1]).unwrap();
    assert_eq!(pat.n(), 2);
    assert_eq!(pat.nnz(), 3);
    assert_eq!(pat.col_ptr(), &[0, 1, 3]);
    assert_eq!(pat.row_idx(), &[0, 0, 1]);
    assert_eq!(pat.row_idx_of_col(1), &[0, 1]);
    assert_eq!(pat.fingerprint(), UpperTriCscPattern::new(&[0, 1, 3], &[0, 0, 1]).unwrap().fingerprint());
    let view = UpperTriCscView::new(&[0, 1, 3], &[0, 0, 1], &[4.0, 1.0, 3.0]).unwrap();
    assert_eq!(view.values(), &[4.0, 1.0, 3.0]);
    assert_eq!(view.val_of_col(1), &[1.0, 3.0]);

    // Missing diagonal in a non-empty column is ALLOWED by UpperTriCscPattern & compute_etree
    let sqd_pat = UpperTriCscPattern::new(&[0, 1, 2], &[0, 0]).unwrap();
    assert!(compute_etree(sqd_pat).is_ok());

    // Empty column rejected by compute_etree / Qdldl::analyze
    assert_eq!(Qdldl::analyze(&[0, 0], &[]).unwrap_err(), Error::EmptyColumn { col: 0 });
    assert_eq!(Qdldl::analyze(&[0, 1, 1, 2], &[0, 2]).unwrap_err(), Error::EmptyColumn { col: 1 });
}

#[test]
fn prefix_sum_detects_index_overflow() {
    assert_eq!(prefix_sum_l_col_ptr(&[usize::MAX - 1, 2]).unwrap_err(), Error::IndexOverflow);
    let boundary = prefix_sum_l_col_ptr(&[usize::MAX, 0]).unwrap();
    assert_eq!(boundary, vec![0, usize::MAX, usize::MAX]);
}

#[test]
fn factorize_rejects_mismatch_non_finite_and_zero_diagonals() {
    let col_ptr = [0, 1, 3];
    let row_idx = [0, 0, 1];
    let mut solver = Qdldl::analyze(&col_ptr, &row_idx).unwrap();

    // StructureMismatch: different n
    assert_eq!(
        solver.factorize(&[0, 1, 3, 4], &[0, 0, 1, 2], &[4.0, 1.0, 3.0, 2.0]).unwrap_err(),
        Error::StructureMismatch
    );
    // StructureMismatch: same n, different nnz
    assert_eq!(solver.factorize(&[0, 1, 2], &[0, 1], &[4.0, 3.0]).unwrap_err(), Error::StructureMismatch);
    // StructureMismatch: same n and nnz, different row indices (SQD missing diag vs standard)
    assert_eq!(solver.factorize(&[0, 1, 2], &[0, 0], &[4.0, 1.0]).unwrap_err(), Error::StructureMismatch);

    // Non-finite inputs
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(solver.factorize(&col_ptr, &row_idx, &[4.0, bad, 3.0]).unwrap_err(), Error::NotFinite);
    }

    // Numeric overflow to non-finite pivot during elimination
    // A = [[1e-308, 1e160], [1e160, 1.0]] -> dval[1] = 1.0 - 1e320 / 1e-308 = -inf -> NotFinite
    assert_eq!(solver.factorize(&col_ptr, &row_idx, &[1e-308, 1e160, 1.0]).unwrap_err(), Error::NotFinite);

    // Zero diagonal at index 0
    assert_eq!(
        solver.factorize(&col_ptr, &row_idx, &[0.0, 1.0, 3.0]).unwrap_err(),
        Error::ZeroDiagonal { index: 0 }
    );

    // Zero diagonal at index 1 via Schur-complement cancellation: [[2, 4], [4, 8]]
    assert_eq!(
        solver.factorize(&col_ptr, &row_idx, &[2.0, 4.0, 8.0]).unwrap_err(),
        Error::ZeroDiagonal { index: 1 }
    );

    // Subnormal pivot reciprocal overflow (both positive and negative)
    assert_eq!(
        solver.factorize(&col_ptr, &row_idx, &[1e-315, 0.0, 1.0]).unwrap_err(),
        Error::ZeroDiagonal { index: 0 }
    );
    assert_eq!(
        solver.factorize(&col_ptr, &row_idx, &[-1e-315, 0.0, 1.0]).unwrap_err(),
        Error::ZeroDiagonal { index: 0 }
    );

    // Direct factor_in_place StructureMismatch check
    let pat = UpperTriCscPattern::new(&col_ptr, &row_idx).unwrap();
    let sym = compute_etree(pat).unwrap();
    let mut num = NumericFactor::from_symbolic(&sym);
    let mut work = FactorWorkspace::new(sym.n);
    let other = UpperTriCscView::new(&[0, 1, 2], &[0, 1], &[1.0, 2.0]).unwrap();
    assert_eq!(factor_in_place(other, &sym, &mut num, &mut work).unwrap_err(), Error::StructureMismatch);
}

#[test]
fn solve_checks_not_factorized_and_bad_shape() {
    let col_ptr = [0, 1, 3];
    let row_idx = [0, 0, 1];
    let values = [4.0, 1.0, 3.0];
    let mut solver = Qdldl::analyze(&col_ptr, &row_idx).unwrap();

    let mut b = [1.0, 2.0];
    assert_eq!(solver.solve(&mut b).unwrap_err(), Error::NotFactorized);
    assert_eq!(solver.solve_l(&mut b).unwrap_err(), Error::NotFactorized);
    assert_eq!(solver.solve_lt(&mut b).unwrap_err(), Error::NotFactorized);

    solver.factorize(&col_ptr, &row_idx, &values).unwrap();
    for bad_len in [0usize, 1, 3] {
        let mut wrong = vec![1.0; bad_len];
        assert_eq!(solver.solve(&mut wrong).unwrap_err(), Error::BadShape);
        assert_eq!(solver.solve_l(&mut wrong).unwrap_err(), Error::BadShape);
        assert_eq!(solver.solve_lt(&mut wrong).unwrap_err(), Error::BadShape);
    }
}

#[test]
fn empty_and_one_by_one_matrices() {
    // n == 0
    let mut solver0 = Qdldl::analyze(&[0], &[]).unwrap();
    solver0.factorize(&[0], &[], &[]).unwrap();
    assert_eq!(solver0.d(), Some(&[][..]));
    assert_eq!(
        solver0.stats(),
        Some(&QdldlStats {
            n: 0,
            nnz_a: 0,
            nnz_l: 0,
            positive_inertia: 0,
        })
    );
    let mut empty: [f64; 0] = [];
    solver0.solve(&mut empty).unwrap();
    solver0.solve_l(&mut empty).unwrap();
    solver0.solve_lt(&mut empty).unwrap();

    // n == 1 positive
    let mut solver1 = Qdldl::analyze(&[0, 1], &[0]).unwrap();
    solver1.factorize(&[0, 1], &[0], &[4.0]).unwrap();
    assert_eq!(solver1.d(), Some(&[4.0][..]));
    assert_eq!(
        solver1.stats(),
        Some(&QdldlStats {
            n: 1,
            nnz_a: 1,
            nnz_l: 0,
            positive_inertia: 1,
        })
    );
    let mut b1 = [10.0];
    solver1.solve(&mut b1).unwrap();
    assert!((b1[0] - 2.5).abs() < 1e-15);

    // n == 1 negative
    solver1.factorize(&[0, 1], &[0], &[-4.0]).unwrap();
    assert_eq!(solver1.d(), Some(&[-4.0][..]));
    assert_eq!(solver1.stats().unwrap().positive_inertia, 0);
    let mut b1 = [10.0];
    solver1.solve(&mut b1).unwrap();
    assert!((b1[0] + 2.5).abs() < 1e-15);

    // n == 1 zero and subnormal
    assert_eq!(solver1.factorize(&[0, 1], &[0], &[0.0]).unwrap_err(), Error::ZeroDiagonal { index: 0 });
    assert_eq!(solver1.factorize(&[0, 1], &[0], &[1e-315]).unwrap_err(), Error::ZeroDiagonal { index: 0 });
}

#[test]
fn multi_branch_elimination_tree_regression() {
    // Construct a 5x5 matrix where branches 0 -> 2 and 1 -> 2 merge at column 2
    // before column 3, and column 4 connects to column 2:
    //   col 0: [0]
    //   col 1: [1]
    //   col 2: [0, 1, 2]
    //   col 3: [0, 1, 3]  (A[2, 3] == 0: fill-in at L[3, 2] depends on BOTH 0 and 1!)
    //   col 4: [2, 4]     (catches dirty y[2] if topological order failed!)
    let col_ptr = [0, 1, 2, 5, 8, 10];
    let row_idx = [0, 1, 0, 1, 2, 0, 1, 3, 2, 4];
    let values = [
        4.0, // (0, 0)
        5.0, // (1, 1)
        1.0, -2.0, 6.0, // (0, 2), (1, 2), (2, 2)
        1.5, 2.5, 8.0, // (0, 3), (1, 3), (3, 3)
        -1.0, 7.0, // (2, 4), (4, 4)
    ];

    let mut solver = Qdldl::analyze(&col_ptr, &row_idx).unwrap();
    assert_eq!(solver.symbolic.etree, vec![2, 2, 3, 4, NONE]);
    assert_eq!(solver.symbolic.lnz, vec![2, 2, 2, 1, 0]);
    assert_eq!(solver.symbolic.l_col_ptr, vec![0, 2, 4, 6, 7, 7]);

    solver.factorize(&col_ptr, &row_idx, &values).unwrap();
    let d = solver.d().unwrap();
    // Hand-verify L[3, 2]:
    //   L[2, 0] = 1.0 / 4.0 = 0.25, L[3, 0] = 1.5 / 4.0 = 0.375
    //   L[2, 1] = -2.0 / 5.0 = -0.4, L[3, 1] = 2.5 / 5.0 = 0.5
    //   D[2] = 6.0 - 1.0*0.25 - (-2.0)*(-0.4) = 6.0 - 0.25 - 0.8 = 4.95
    //   y_3[2] = 0 - L[2,0]*y_3[0] - L[2,1]*y_3[1] = -0.25*1.5 - (-0.4)*2.5 = -0.375 + 1.0 = 0.625
    //   L[3, 2] = 0.625 / 4.95
    let expected_d2 = 4.95;
    let expected_l32 = 0.625 / 4.95;
    assert!((d[2] - expected_d2).abs() < 1e-14, "D[2] = {} vs {}", d[2], expected_d2);
    // Column 2 of L starts at l_col_ptr[2] = 4, row 3 is first entry in col 2 (index 4), row 4 is index 5.
    assert_eq!(solver.numeric.l_row_idx[4], 3);
    assert!(
        (solver.numeric.l_values[4] - expected_l32).abs() < 1e-14,
        "L[3, 2] = {} vs {expected_l32}",
        solver.numeric.l_values[4]
    );

    let a_dense = upper_csc_to_dense(5, &col_ptr, &row_idx, &values);
    let mut rng = SplitMix64::new(0x1234_5678);
    verify_system(&a_dense, &col_ptr, &row_idx, &values, Some(5), &mut rng, "multi_branch");
}

#[test]
fn arrow_tridiagonal_laplacian_and_disconnected_matrices() {
    let mut rng = SplitMix64::new(0xabcd_ef01);

    // 1a. Arrowhead with dense last column (zero fill: nnz_l = n - 1)
    let n = 12;
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::new();
    let mut values = Vec::new();
    col_ptr.push(0);
    for j in 0..n - 1 {
        row_idx.push(j);
        values.push(5.0 + j as f64 * 0.25);
        col_ptr.push(row_idx.len());
    }
    for i in 0..n - 1 {
        row_idx.push(i);
        values.push(0.2 * ((i as f64) + 1.0) * if i % 2 == 0 { 1.0 } else { -1.0 });
    }
    row_idx.push(n - 1);
    values.push(30.0);
    col_ptr.push(row_idx.len());
    let a_arrow_tail = upper_csc_to_dense(n, &col_ptr, &row_idx, &values);
    let solver_tail = Qdldl::analyze(&col_ptr, &row_idx).unwrap();
    assert_eq!(solver_tail.symbolic.nnz_l(), n - 1);
    verify_system(&a_arrow_tail, &col_ptr, &row_idx, &values, Some(n), &mut rng, "arrow_last");

    // 1b. Arrowhead with dense first row/column (full fill: nnz_l = n * (n - 1) / 2)
    let mut col_ptr_h = vec![0];
    let mut row_idx_h = Vec::new();
    let mut values_h = Vec::new();
    row_idx_h.push(0);
    values_h.push(40.0);
    col_ptr_h.push(row_idx_h.len());
    for j in 1..n {
        row_idx_h.push(0);
        values_h.push(0.5 / (j as f64));
        row_idx_h.push(j);
        values_h.push(4.0 + j as f64 * 0.1);
        col_ptr_h.push(row_idx_h.len());
    }
    let a_arrow_head = upper_csc_to_dense(n, &col_ptr_h, &row_idx_h, &values_h);
    let solver_head = Qdldl::analyze(&col_ptr_h, &row_idx_h).unwrap();
    assert_eq!(solver_head.symbolic.nnz_l(), n * (n - 1) / 2);
    verify_system(&a_arrow_head, &col_ptr_h, &row_idx_h, &values_h, Some(n), &mut rng, "arrow_first");

    // 2. Tridiagonal 1D SPD matrix (n = 30)
    let n_tri = 30;
    let mut col_ptr_t = vec![0];
    let mut row_idx_t = Vec::new();
    let mut values_t = Vec::new();
    for j in 0..n_tri {
        if j > 0 {
            row_idx_t.push(j - 1);
            values_t.push(-1.0);
        }
        row_idx_t.push(j);
        values_t.push(4.0);
        col_ptr_t.push(row_idx_t.len());
    }
    let a_tri = upper_csc_to_dense(n_tri, &col_ptr_t, &row_idx_t, &values_t);
    verify_system(&a_tri, &col_ptr_t, &row_idx_t, &values_t, Some(n_tri), &mut rng, "tridiag");

    // 3. 2D 5-point Laplacian grid (6x6 = 36 dofs)
    let nx = 6usize;
    let ny = 6usize;
    let n_grid = nx * ny;
    let mut col_ptr_g = vec![0];
    let mut row_idx_g = Vec::new();
    let mut values_g = Vec::new();
    for y in 0..ny {
        for x in 0..nx {
            let j = y * nx + x;
            let mut rows = Vec::new();
            if y > 0 {
                rows.push((y - 1) * nx + x);
            }
            if x > 0 {
                rows.push(y * nx + (x - 1));
            }
            rows.sort_unstable();
            for r in rows {
                row_idx_g.push(r);
                values_g.push(-1.0);
            }
            row_idx_g.push(j);
            values_g.push(5.0);
            col_ptr_g.push(row_idx_g.len());
        }
    }
    let a_grid = upper_csc_to_dense(n_grid, &col_ptr_g, &row_idx_g, &values_g);
    verify_system(&a_grid, &col_ptr_g, &row_idx_g, &values_g, Some(n_grid), &mut rng, "laplacian_2d");

    // 4. Disconnected 3-block diagonal matrix (multiple NONE roots in etree)
    let mut col_ptr_d = vec![0];
    let mut row_idx_d = Vec::new();
    let mut values_d = Vec::new();
    for blk in 0..3 {
        let base = blk * 3;
        for local in 0..3 {
            for r in 0..=local {
                row_idx_d.push(base + r);
                values_d.push(if r == local { 5.0 + blk as f64 } else { -0.5 });
            }
            col_ptr_d.push(row_idx_d.len());
        }
    }
    let sym_disc = compute_etree(UpperTriCscPattern::new(&col_ptr_d, &row_idx_d).unwrap()).unwrap();
    let roots = sym_disc.etree.iter().filter(|&&p| p == NONE).count();
    assert_eq!(roots, 3);
    let a_disc = upper_csc_to_dense(9, &col_ptr_d, &row_idx_d, &values_d);
    verify_system(&a_disc, &col_ptr_d, &row_idx_d, &values_d, Some(9), &mut rng, "disconnected");

    // 5. Negative-definite matrix (positive_inertia == 0)
    let neg_values: Vec<f64> = values_t.iter().map(|v| -v).collect();
    let neg_dense: Vec<f64> = a_tri.iter().map(|v| -v).collect();
    verify_system(&neg_dense, &col_ptr_t, &row_idx_t, &neg_values, Some(0), &mut rng, "neg_def");
}

#[test]
fn sqd_kkt_with_missing_dual_diagonals() {
    // K = [H  A^T; A  0], H = 3x3 SPD, A = 2x3 full row rank, C = 0 omitted from CSC.
    // Upper triangle columns:
    //   col 0: (0, 0) = 4
    //   col 1: (0, 1) = 1, (1, 1) = 5
    //   col 2: (1, 2) = -1, (2, 2) = 6
    //   col 3: (0, 3) = 1, (1, 3) = 2        <-- missing diagonal (3, 3)
    //   col 4: (1, 4) = -1, (2, 4) = 3       <-- missing diagonal (4, 4)
    let col_ptr = [0, 1, 3, 5, 7, 9];
    let row_idx = [0, 0, 1, 1, 2, 0, 1, 1, 2];
    let values = [4.0, 1.0, 5.0, -1.0, 6.0, 1.0, 2.0, -1.0, 3.0];

    let a_dense = upper_csc_to_dense(5, &col_ptr, &row_idx, &values);
    let mut rng = SplitMix64::new(0xfeed_face);
    verify_system(&a_dense, &col_ptr, &row_idx, &values, Some(3), &mut rng, "sqd_kkt_zero_c");

    // Check dual pivots are strictly negative
    let mut solver = Qdldl::analyze(&col_ptr, &row_idx).unwrap();
    solver.factorize(&col_ptr, &row_idx, &values).unwrap();
    let d = solver.d().unwrap();
    assert!(d[0] > 0.0 && d[1] > 0.0 && d[2] > 0.0);
    assert!(d[3] < 0.0 && d[4] < 0.0);

    // Rank-deficient A (col 4 = 2 * col 3) with exact dyadic arithmetic must fail
    // with ZeroDiagonal at index 4.
    let bad_values = [4.0, 4.0, 4.0, 1.0, 2.0, 2.0, 4.0];
    let bad_col_ptr = [0, 1, 2, 3, 5, 7];
    let bad_row_idx = [0, 1, 2, 0, 1, 0, 1];
    let mut bad_solver = Qdldl::analyze(&bad_col_ptr, &bad_row_idx).unwrap();
    assert_eq!(
        bad_solver.factorize(&bad_col_ptr, &bad_row_idx, &bad_values).unwrap_err(),
        Error::ZeroDiagonal { index: 4 }
    );
}

#[test]
fn refactorization_preserves_prior_on_validation_error_and_recovers_after_zero_diagonal() {
    let col_ptr = [0, 1, 3, 5];
    let row_idx = [0, 0, 1, 1, 2];
    let vals1 = [4.0, 1.0, 5.0, -2.0, 6.0];
    let vals2 = [8.0, -1.0, 7.0, 3.0, 9.0];

    let mut solver = Qdldl::analyze(&col_ptr, &row_idx).unwrap();
    solver.factorize(&col_ptr, &row_idx, &vals1).unwrap();
    let d1 = solver.d().unwrap().to_vec();
    let stats1 = *solver.stats().unwrap();

    let a1_dense = upper_csc_to_dense(3, &col_ptr, &row_idx, &vals1);
    let b = [1.0, -2.0, 3.0];
    let mut x1 = b;
    solver.solve(&mut x1).unwrap();
    assert!(scaled_solve_residual(&a1_dense, &x1, &b, 3) < 1e-13);

    // Validation errors MUST preserve prior valid factors and stats
    assert_eq!(solver.factorize(&col_ptr, &row_idx, &[1.0, 2.0]).unwrap_err(), Error::BadShape);
    assert_eq!(solver.factorize(&[1, 1, 3, 5], &row_idx, &vals1).unwrap_err(), Error::InvalidStructure);
    assert_eq!(
        solver.factorize(&[0, 1, 2, 5], &[0, 1, 0, 1, 2], &vals1).unwrap_err(),
        Error::StructureMismatch
    );
    assert_eq!(
        solver.factorize(&col_ptr, &row_idx, &[4.0, f64::NAN, 5.0, -2.0, 6.0]).unwrap_err(),
        Error::NotFinite
    );
    assert_eq!(solver.d().unwrap(), d1.as_slice());
    assert_eq!(solver.stats(), Some(&stats1));
    let mut x1_again = b;
    solver.solve(&mut x1_again).unwrap();
    assert_eq!(x1_again, x1);

    // Numeric failure (ZeroDiagonal at index 1) MUST invalidate prior factors
    let zero_pivot_vals = [2.0, 4.0, 8.0, -2.0, 6.0];
    assert_eq!(
        solver.factorize(&col_ptr, &row_idx, &zero_pivot_vals).unwrap_err(),
        Error::ZeroDiagonal { index: 1 }
    );
    assert!(solver.d().is_none());
    assert!(solver.stats().is_none());
    let mut x_fail = b;
    assert_eq!(solver.solve(&mut x_fail).unwrap_err(), Error::NotFactorized);
    assert_eq!(solver.solve_l(&mut x_fail).unwrap_err(), Error::NotFactorized);
    assert_eq!(solver.solve_lt(&mut x_fail).unwrap_err(), Error::NotFactorized);

    // Subsequent valid refactorization MUST recover cleanly
    solver.factorize(&col_ptr, &row_idx, &vals2).unwrap();
    let a2_dense = upper_csc_to_dense(3, &col_ptr, &row_idx, &vals2);
    let mut x2 = b;
    solver.solve(&mut x2).unwrap();
    assert!(scaled_solve_residual(&a2_dense, &x2, &b, 3) < 1e-13);
    assert!(factor_residual_inf(&a2_dense, &solver, 3) < 1e-13);
    assert_eq!(solver.stats().unwrap().positive_inertia, 3);
}

#[test]
fn randomized_spd_and_sqd_sweep_matches_dgesv() {
    let mut rng = SplitMix64::new(0x5eed_cafe_1234_5678);

    // 300 random SPD matrices across n in 2..=64
    for case in 0..300 {
        let n = 2 + (case % 63);
        let mut a_dense = vec![0.0; n * n];
        let density = if case % 3 == 0 {
            1.0
        } else if case % 3 == 1 {
            0.35
        } else {
            0.15
        };

        for j in 0..n {
            for i in 0..j {
                if rng.next_f64_unit() < density {
                    let v = rng.next_f64_signed();
                    a_dense[i + j * n] = v;
                    a_dense[j + i * n] = v;
                }
            }
        }
        for i in 0..n {
            let row_abs: f64 = (0..n).filter(|&j| j != i).map(|j| a_dense[i + j * n].abs()).sum();
            a_dense[i + i * n] = row_abs + 1.0 + rng.next_f64_unit() * 2.0;
        }

        let (col_ptr, row_idx, values) = dense_to_upper_csc(n, &a_dense, true);
        let context = format!("random_spd_{case}_n{n}");
        verify_system(&a_dense, &col_ptr, &row_idx, &values, Some(n), &mut rng, &context);
    }

    // 250 random SQD KKT matrices [H A^T; A -C] across n in 2..=64
    // (half with C = 0 and missing dual diagonals, half with C > 0)
    for case in 0..250 {
        let n = 2 + (case % 63);
        let n_d = (1 + rng.next_usize((n / 2).max(1))).min(n - 1);
        let n_p = n - n_d;
        let zero_c = case % 2 == 0;

        let mut a_dense = vec![0.0; n * n];

        // Primal SPD block H (n_p x n_p)
        for j in 0..n_p {
            for i in 0..j {
                if rng.next_f64_unit() < 0.35 {
                    let v = rng.next_f64_signed() * 0.5;
                    a_dense[i + j * n] = v;
                    a_dense[j + i * n] = v;
                }
            }
        }
        for i in 0..n_p {
            let row_abs: f64 = (0..n_p).filter(|&j| j != i).map(|j| a_dense[i + j * n].abs()).sum();
            a_dense[i + i * n] = row_abs + 1.5 + rng.next_f64_unit();
        }

        // Coupling block A (n_d x n_p), guaranteed full row rank by placing a
        // strong entry on (r, r % n_p) or diagonally dominant first n_d columns when n_d <= n_p.
        for r in 0..n_d {
            // Fill sparse random couplings
            for c in 0..n_p {
                if rng.next_f64_unit() < 0.3 {
                    let v = rng.next_f64_signed() * 0.2;
                    a_dense[c + (n_p + r) * n] = v;
                    a_dense[(n_p + r) + c * n] = v;
                }
            }
            // Ensure full row rank (n_d <= n_p holds since n_d <= n/2)
            let row_abs: f64 = (0..n_p).filter(|&c| c != r).map(|c| a_dense[c + (n_p + r) * n].abs()).sum();
            let anchor = (row_abs + 1.5 + rng.next_f64_unit()) * if rng.next_f64_signed() >= 0.0 { 1.0 } else { -1.0 };
            a_dense[r + (n_p + r) * n] = anchor;
            a_dense[(n_p + r) + r * n] = anchor;
        }

        // Dual block -C (either identically zero -> omitted diagonals, or SPD negative block)
        if !zero_c {
            for j in 0..n_d {
                for i in 0..j {
                    if rng.next_f64_unit() < 0.25 {
                        let v = -rng.next_f64_signed() * 0.25;
                        a_dense[(n_p + i) + (n_p + j) * n] = v;
                        a_dense[(n_p + j) + (n_p + i) * n] = v;
                    }
                }
            }
            for i in 0..n_d {
                let row_abs: f64 = (0..n_d).filter(|&j| j != i).map(|j| a_dense[(n_p + i) + (n_p + j) * n].abs()).sum();
                a_dense[(n_p + i) + (n_p + i) * n] = -(row_abs + 0.5 + rng.next_f64_unit());
            }
        }

        let (col_ptr, row_idx, values) = dense_to_upper_csc(n, &a_dense, true);
        let context = format!("random_sqd_{case}_np{n_p}_nd{n_d}_zeroc_{zero_c}");
        verify_system(&a_dense, &col_ptr, &row_idx, &values, Some(n_p), &mut rng, &context);
    }
}

#[test]
fn dlatms_symmetric_matrices_match_dgesv() {
    let mut rng = SplitMix64::new(0xdead_beef_9988);
    let sizes = [2usize, 5, 9, 16, 24, 32];
    let conds = [10.0, 1e2, 1e3, 1e4];

    for (si, &n) in sizes.iter().enumerate() {
        for (ci, &cond) in conds.iter().enumerate() {
            let params = Dlatms {
                dist: Dist::Uniform,
                sym: DlatmsSym::PositiveSymmetric,
                mode: 3,
                cond,
                dmax: 1.0,
                pack: DlatmsPack::Full,
            };
            let mut a = vec![0.0; n * n];
            let mut d = vec![0.0; n];
            let mut seed = [1001 + si as u32 * 13, 2002 + ci as u32 * 7, 3003 + si as u32, 4004 | 1];
            dlatms(&params, n, n, &mut a, &mut d, &mut seed).unwrap();

            let (col_ptr, row_idx, values) = dense_to_upper_csc(n, &a, false);
            verify_system(
                &a,
                &col_ptr,
                &row_idx,
                &values,
                Some(n),
                &mut rng,
                &format!("dlatms_spd_n{n}_cond{cond:e}"),
            );

            // Also test negative definite -A on representative cases
            if ci == 1 {
                let a_neg: Vec<f64> = a.iter().map(|v| -v).collect();
                let vals_neg: Vec<f64> = values.iter().map(|v| -v).collect();
                verify_system(&a_neg, &col_ptr, &row_idx, &vals_neg, Some(0), &mut rng, &format!("dlatms_negdef_n{n}"));
            }
        }
    }

    // SQD KKT matrices with dlatms(PositiveSymmetric) primal block H and C = 0
    for (idx, &(n_p, n_d)) in [(4usize, 2usize), (8, 4), (16, 6)].iter().enumerate() {
        let params = Dlatms {
            dist: Dist::Uniform,
            sym: DlatmsSym::PositiveSymmetric,
            mode: 3,
            cond: 1e2,
            dmax: 2.0,
            pack: DlatmsPack::Full,
        };
        let mut h = vec![0.0; n_p * n_p];
        let mut d_eig = vec![0.0; n_p];
        let mut seed = [701 + idx as u32 * 19, 803, 907, 1009];
        dlatms(&params, n_p, n_p, &mut h, &mut d_eig, &mut seed).unwrap();

        let n = n_p + n_d;
        let mut kkt = vec![0.0; n * n];
        for j in 0..n_p {
            for i in 0..n_p {
                kkt[i + j * n] = h[i + j * n_p];
            }
        }
        for r in 0..n_d {
            for c in 0..n_p {
                let v = if c == r {
                    1.5 + rng.next_f64_unit()
                } else {
                    0.1 * rng.next_f64_signed()
                };
                kkt[c + (n_p + r) * n] = v;
                kkt[(n_p + r) + c * n] = v;
            }
        }
        let (col_ptr, row_idx, values) = dense_to_upper_csc(n, &kkt, true);
        verify_system(
            &kkt,
            &col_ptr,
            &row_idx,
            &values,
            Some(n_p),
            &mut rng,
            &format!("dlatms_sqd_np{n_p}_nd{n_d}"),
        );
    }
}

/// What one sparse QDLDL factorization of a small 3D box costs.
///
/// `cargo test --release --lib bench_qdldl -- --ignored --nocapture`
///
/// Matches the 4x4x4 node lattice from `bench_sparse_lu` (192 dofs, 3 dofs per
/// node coupled to 6 neighbors), stored as upper-triangular CSC.
#[test]
#[ignore]
fn bench_qdldl() {
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    use std::time::Instant;
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    use web_time::Instant;
    let m = 4_usize;
    let n = m * m * m * 3;
    let node = |i: usize, j: usize, k: usize| (k * m * m + j * m + i) * 3;

    let mut full_rows: Vec<Vec<usize>> = Vec::with_capacity(n);
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
                    full_rows.push(rows);
                }
            }
        }
    }

    let mut col_ptr = vec![0_usize];
    let mut row_idx = Vec::new();
    let mut values = Vec::new();
    for (col, rows) in full_rows.iter().enumerate() {
        let mut off_count = 0usize;
        for &r in rows {
            if r != col {
                off_count += 1;
            }
        }
        for &r in rows {
            if r <= col {
                row_idx.push(r);
                values.push(if r == col { 10.0 + off_count as f64 } else { -1.0 });
            }
        }
        col_ptr.push(row_idx.len());
    }

    let nnz_a = row_idx.len();
    let mut solver = Qdldl::analyze(&col_ptr, &row_idx).unwrap();
    solver.factorize(&col_ptr, &row_idx, &values).unwrap();
    let nnz_l = solver.stats().unwrap().nnz_l;
    let fill = nnz_l as f64 / nnz_a as f64;

    let reps = 160;
    let t = Instant::now();
    for _ in 0..reps {
        solver.factorize(&col_ptr, &row_idx, &values).unwrap();
    }
    let per_factorize = t.elapsed().as_secs_f64() / reps as f64;

    let mut b = vec![1.0; n];
    let t = Instant::now();
    for _ in 0..(reps * 10) {
        b.fill(1.0);
        solver.solve(&mut b).unwrap();
    }
    let per_solve = t.elapsed().as_secs_f64() / (reps * 10) as f64;
    println!(
        "qdldl: n = {n}, nnz(A_tri) = {nnz_a}, nnz(L) = {nnz_l}, fill = {fill:.2} -> factorize {:.3} ms, solve {:.1} us",
        per_factorize * 1e3,
        per_solve * 1e6
    );
}
