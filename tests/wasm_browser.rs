//! Browser-target smoke test for the sparse path.
//!
//! It exercises the `web-time` stopwatch used by the symbolic analysis and the
//! numeric factorization; without the shim, `Instant::now()` panics on
//! `wasm32-unknown-unknown`.
#![cfg(all(target_arch = "wasm32", target_os = "unknown"))]

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
