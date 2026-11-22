//! Browser-target smoke test for the sparse path.
//!
//! It exercises the `web-time` stopwatch used by the symbolic analysis and the
//! numeric factorization; without the shim, `Instant::now()` panics on
//! `wasm32-unknown-unknown`.
#![cfg(all(target_arch = "wasm32", target_os = "unknown"))]

use algebra_linear::sparse::faer::{EffectiveLuMethod, LuMethod, SparseLu};
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn sparse_lu_runs_in_the_browser_target() {
    // A = [[4, 1], [1, 3]]
    let col_ptr = [0, 2, 4];
    let row_idx = [0, 1, 0, 1];
    let values = [4.0, 1.0, 1.0, 3.0];
    let mut lu = SparseLu::analyze(&col_ptr, &row_idx, LuMethod::auto()).unwrap();
    assert_eq!(lu.effective_method(), EffectiveLuMethod::Simplicial);
    assert!(lu.flop_ratio().is_some());
    lu.factorize(&col_ptr, &row_idx, &values).unwrap();
    assert_eq!(lu.stats().unwrap().method, EffectiveLuMethod::Simplicial);
    let mut x = [1.0, 2.0];
    lu.solve(&mut x).unwrap();
    assert!((x[0] - 1.0 / 11.0).abs() < 1e-14);
    assert!((x[1] - 7.0 / 11.0).abs() < 1e-14);
}
