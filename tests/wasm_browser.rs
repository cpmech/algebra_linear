//! Browser-target smoke test for the sparse path.
//!
//! It exercises the `web-time` stopwatch used by the symbolic analysis and the
//! numeric factorization; without the shim, `Instant::now()` panics on
//! `wasm32-unknown-unknown`.
#![cfg(all(target_arch = "wasm32", target_os = "unknown"))]

use algebra_linear::sparse::amgcl::{Amg, AmgParams, Cg, CsrMatrix, Parallelism, SolverParams};
use algebra_linear::sparse::{CscPattern, CscView, SymbolicLu};
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn sparse_lu_runs_in_the_browser_target() {
    // A = [[4, 1], [1, 3]]
    let col_ptr = [0, 2, 4];
    let row_idx = [0, 1, 0, 1];
    let values = [4.0, 1.0, 1.0, 3.0];
    let pattern = CscPattern::new(2, 2, &col_ptr, &row_idx).unwrap();
    let symbolic = SymbolicLu::analyze(pattern).unwrap();
    let a = CscView::new(2, 2, &col_ptr, &row_idx, &values).unwrap();
    let lu = symbolic.factorize(a).unwrap();
    let mut x = [1.0, 2.0];
    lu.solve(&mut x).unwrap();
    assert!((x[0] - 1.0 / 11.0).abs() < 1e-14);
    assert!((x[1] - 7.0 / 11.0).abs() < 1e-14);
}

#[wasm_bindgen_test]
fn amgcl_uses_the_serial_browser_path() {
    let a = CsrMatrix::poisson_2d(4, 4).unwrap();
    let b = vec![1.0; a.nrows()];
    let mut x = vec![0.0; a.nrows()];
    let parallelism = Parallelism {
        threads: 4,
        min_parallel_len: 1,
    };
    let mut amg = Amg::new(
        &a,
        AmgParams {
            parallelism,
            ..AmgParams::default()
        },
    )
    .unwrap();
    let cg = Cg::new(SolverParams {
        tolerance: 1e-10,
        max_iterations: 50,
        parallelism,
        ..SolverParams::default()
    });
    let report = cg.solve(&a, &mut amg, &b, &mut x).unwrap();
    assert!(report.converged);
    assert!(x.iter().all(|value| value.is_finite()));
}
