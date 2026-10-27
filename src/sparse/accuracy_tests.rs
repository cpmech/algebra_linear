//! Accuracy of the sparse solvers on LAPACK's own random matrices.
//!
//! The dense generators in `lapack::families` build square matrices with a
//! prescribed condition number (`DLATMS`), which is exactly what a solver needs:
//! a known spectrum and a known worst case. The matrix is converted to CSC and
//! solved for a known vector; the test checks the forward error and the scaled
//! residual for the simplicial, supernodal and automatic paths.

use crate::lapack::families::{Dist, Dlatms, DlatmsPack, DlatmsSym, dlatms};
use crate::sparse::csc::{CscPattern, CscView};
use crate::sparse::solver::{LuMethod, LuSymbolic};
use crate::sparse::supernodal::SymbolicSupernodal;
use crate::sparse::symbolic::SymbolicLu;

/// Builds an `n x n` random matrix with singular values from `cond` down to 1.
fn random_square(n: usize, cond: f64, seed: [u32; 4]) -> Vec<f64> {
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
    let mut seed = seed;
    dlatms(&params, n, n, &mut a, &mut d, &mut seed).expect("dlatms");
    a
}

/// Converts a dense column-major matrix to CSC, keeping every entry.
fn to_csc(a: &[f64], n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::with_capacity(n * n);
    col_ptr.push(0);
    for _ in 0..n {
        for i in 0..n {
            row_idx.push(i);
        }
        col_ptr.push(row_idx.len());
    }
    (col_ptr, row_idx, a.to_vec())
}

fn matvec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut b = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            b[i] += a[i + j * n] * x[j];
        }
    }
    b
}

fn norm_inf(x: &[f64]) -> f64 {
    x.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// Scaled residual `||A x - b||∞ / (||A||∞ ||x||∞ + ||b||∞)`.
fn scaled_residual(a: &[f64], x: &[f64], b: &[f64], n: usize) -> f64 {
    let ax = matvec(a, x, n);
    let r: Vec<f64> = (0..n).map(|i| ax[i] - b[i]).collect();
    let a_norm = (0..n).map(|i| (0..n).map(|j| a[i + j * n].abs()).sum::<f64>()).fold(0.0f64, f64::max);
    let denom = a_norm * norm_inf(x) + norm_inf(b);
    if denom == 0.0 { norm_inf(&r) } else { norm_inf(&r) / denom }
}

fn forward_error(x: &[f64], exact: &[f64]) -> f64 {
    let e: Vec<f64> = x.iter().zip(exact).map(|(a, b)| a - b).collect();
    norm_inf(&e) / norm_inf(exact).max(1.0)
}

#[test]
fn dense_random_systems_are_solved_accurately() {
    let n_values = [4usize, 8, 16, 32];
    let conditions = [1e2, 1e4];
    for (case, &n) in n_values.iter().enumerate() {
        for (ci, &cond) in conditions.iter().enumerate() {
            let seed = [1234 + case as u32 * 17, 2345 + ci as u32, 3456 + case as u32, 4567];
            let a = random_square(n, cond, seed);
            let x_exact: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64 * 0.25).collect();
            let b = matvec(&a, &x_exact, n);
            let (col_ptr, row_idx, values) = to_csc(&a, n);
            let pattern = CscPattern::new(n, n, &col_ptr, &row_idx).unwrap();
            let view = CscView::new(n, n, &col_ptr, &row_idx, &values).unwrap();

            let simplicial = SymbolicLu::analyze(pattern).unwrap();
            let mut xs = b.clone();
            simplicial.factorize(view).unwrap().solve(&mut xs).unwrap();

            let supernodal = SymbolicSupernodal::analyze(pattern).unwrap();
            let mut xn = b.clone();
            supernodal.factorize(view).unwrap().solve(&mut xn).unwrap();

            // The default threshold (40.0) is tuned for large matrices; use a
            // low threshold so the dense case exercises the supernodal path.
            let auto = LuSymbolic::analyze(pattern, LuMethod::Auto { flop_ratio_threshold: 1.0 }).unwrap();
            let mut xa = b.clone();
            auto.factorize(view).unwrap().solve(&mut xa).unwrap();

            let tol_err = 1e-9 * cond;
            let tol_res = 1e-13 * cond;
            for (name, x) in [("simplicial", &xs), ("supernodal", &xn), ("auto", &xa)] {
                let err = forward_error(x, &x_exact);
                let res = scaled_residual(&a, x, &b, n);
                assert!(err <= tol_err, "{name}: n={n} cond={cond:e} forward error {err:e} > {tol_err:e}");
                assert!(res <= tol_res, "{name}: n={n} cond={cond:e} residual {res:e} > {tol_res:e}");
            }
            // Dense matrices must select the supernodal path from n = 16 on.
            if n >= 16 {
                assert!(auto.is_supernodal(), "n={n}: dense matrix should be supernodal");
            }
        }
    }
}

#[test]
fn a_complex_random_system_is_solved_accurately() {
    use crate::Complex64;
    let n = 8;
    let re = random_square(n, 1e2, [11, 22, 33, 44]);
    let im = random_square(n, 1e2, [55, 66, 77, 88]);
    let mut a = vec![Complex64::new(0.0, 0.0); n * n];
    for k in 0..n * n {
        // A = Re + i Im, with the real part dominant so the system stays well
        // conditioned enough for a 1e-10 forward-error check.
        a[k] = Complex64::new(re[k], 0.25 * im[k]);
    }
    let x_exact: Vec<Complex64> = (0..n).map(|i| Complex64::new(1.0 + (i % 5) as f64, -0.5)).collect();
    let mut b = vec![Complex64::new(0.0, 0.0); n];
    for i in 0..n {
        for j in 0..n {
            b[i] += a[i + j * n] * x_exact[j];
        }
    }
    let col_ptr: Vec<usize> = (0..=n).map(|j| j * n).collect();
    let row_idx: Vec<usize> = (0..n).flat_map(|_| 0..n).collect();
    let view = CscView::new(n, n, &col_ptr, &row_idx, &a).unwrap();
    let pattern = CscPattern::new(n, n, &col_ptr, &row_idx).unwrap();
    let symbolic = LuSymbolic::analyze(pattern, LuMethod::auto()).unwrap();
    let factors = symbolic.factorize(view).unwrap();
    let mut x = b.clone();
    factors.solve(&mut x).unwrap();
    let e: Vec<f64> = x.iter().zip(&x_exact).map(|(a, b)| (*a - *b).norm()).collect();
    let err = norm_inf(&e) / norm_inf(&x_exact.iter().map(|z| z.norm()).collect::<Vec<_>>());
    assert!(err < 1e-9, "complex forward error {err:e}");
}
