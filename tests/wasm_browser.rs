//! Browser-target smoke test for the sparse path.
//!
//! It exercises the `web-time` stopwatch used by the symbolic analysis and the
//! numeric factorization; without the shim, `Instant::now()` panics on
//! `wasm32-unknown-unknown`.
#![cfg(all(target_arch = "wasm32", target_os = "unknown"))]

use algebra_linear::sparse::amgcl::{AmgParams, KrylovMethod, Parallelism, SolverParams, amg_setup, krylov_solve};
use algebra_linear::sparse::{LuMethod, SparseLu};
use wasm_bindgen_test::*;

/// Five-point Laplacian on an `nx` by `ny` grid as raw CSR arrays.
fn poisson_2d(nx: usize, ny: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let n = nx * ny;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for y in 0..ny {
        for x in 0..nx {
            let row = y * nx + x;
            if x > 0 {
                col_idx.push(row - 1);
                values.push(-1.0);
            }
            col_idx.push(row);
            values.push(4.0);
            if x + 1 < nx {
                col_idx.push(row + 1);
                values.push(-1.0);
            }
            if y > 0 {
                col_idx.push(row - nx);
                values.push(-1.0);
            }
            if y + 1 < ny {
                col_idx.push(row + nx);
                values.push(-1.0);
            }
            row_ptr.push(col_idx.len());
        }
    }
    (row_ptr, col_idx, values)
}

#[wasm_bindgen_test]
fn sparse_lu_runs_in_the_browser_target() {
    // A = [[4, 1], [1, 3]]
    let col_ptr = [0, 2, 4];
    let row_idx = [0, 1, 0, 1];
    let values = [4.0, 1.0, 1.0, 3.0];
    let mut lu = SparseLu::analyze(&col_ptr, &row_idx, LuMethod::auto()).unwrap();
    lu.factorize(&col_ptr, &row_idx, &values).unwrap();
    let mut x = [1.0, 2.0];
    lu.solve(&mut x).unwrap();
    assert!((x[0] - 1.0 / 11.0).abs() < 1e-14);
    assert!((x[1] - 7.0 / 11.0).abs() < 1e-14);
}

#[wasm_bindgen_test]
fn amgcl_uses_the_serial_browser_path() {
    let (row_ptr, col_idx, values) = poisson_2d(4, 4);
    let n = row_ptr.len() - 1;
    let b = vec![1.0; n];
    let mut x = vec![0.0; n];
    let parallelism = Parallelism {
        threads: 4,
        min_parallel_len: 1,
    };
    let mut amg = amg_setup(
        &row_ptr,
        &col_idx,
        &values,
        Some(AmgParams {
            parallelism,
            ..AmgParams::default()
        }),
    )
    .unwrap();
    let report = krylov_solve(
        KrylovMethod::Cg,
        &mut amg,
        &b,
        &mut x,
        SolverParams {
            tolerance: 1e-10,
            max_iterations: 50,
            parallelism,
            ..SolverParams::default()
        },
    )
    .unwrap();
    assert!(report.converged);
    assert!(x.iter().all(|value| value.is_finite()));
}
