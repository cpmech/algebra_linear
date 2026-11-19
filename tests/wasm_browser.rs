//! Browser-target smoke test for the sparse path.
//!
//! It exercises the `web-time` stopwatch used by the symbolic analysis and the
//! numeric factorization; without the shim, `Instant::now()` panics on
//! `wasm32-unknown-unknown`.
#![cfg(all(target_arch = "wasm32", target_os = "unknown"))]

use algebra_linear::sparse::amgcl::{AmgParams, KrylovMethod, Parallelism, SolverParams, amg_setup, amg_setup_owned, krylov_solve};
use algebra_linear::sparse::basiclu::{BasicLu, BasicLuParams};
use algebra_linear::sparse::faer::{EffectiveLuMethod, LuMethod, SparseLu};
use algebra_linear::sparse::qdldl::Qdldl;
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
    assert_eq!(lu.effective_method(), EffectiveLuMethod::Simplicial);
    assert!(lu.flop_ratio().is_some());
    lu.factorize(&col_ptr, &row_idx, &values).unwrap();
    assert_eq!(lu.stats().unwrap().method, EffectiveLuMethod::Simplicial);
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

    let mut owned = amg_setup_owned(
        row_ptr,
        col_idx,
        values,
        Some(AmgParams {
            parallelism,
            ..AmgParams::default()
        }),
    )
    .unwrap();
    assert!(owned.memory_bytes() > 0);
    let mut x_owned = vec![0.0; n];
    let owned_report = krylov_solve(KrylovMethod::Gmres, &mut owned, &b, &mut x_owned, SolverParams::default()).unwrap();
    assert!(owned_report.converged);
}

#[wasm_bindgen_test]
fn qdldl_runs_in_the_browser_target() {
    // Symmetric quasi-definite KKT matrix: [[4, 0, 1], [0, 3, -1], [1, -1, 0]] (upper triangle).
    let col_ptr = [0, 1, 2, 4];
    let row_idx = [0, 1, 0, 1];
    let values = [4.0, 3.0, 1.0, -1.0];

    let mut qdldl = Qdldl::analyze(&col_ptr, &row_idx).unwrap();
    qdldl.factorize(&col_ptr, &row_idx, &values).unwrap();
    assert_eq!(qdldl.stats().unwrap().positive_inertia, 2);

    let mut x = [5.0, 2.0, 0.0];
    qdldl.solve(&mut x).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-13);
    assert!((x[1] - 1.0).abs() < 1e-13);
    assert!((x[2] - 1.0).abs() < 1e-13);
}

#[wasm_bindgen_test]
fn basiclu_runs_in_the_browser_target() {
    let col_ptr = [0, 2, 4];
    let row_idx = [0, 1, 0, 1];
    let values = [4.0, 1.0, 1.0, 3.0];

    let mut lu = BasicLu::new(BasicLuParams::default());
    lu.factorize(&col_ptr, &row_idx, &values).unwrap();

    let mut x = [1.0, 2.0];
    lu.solve(&mut x).unwrap();
    assert!((x[0] - 1.0 / 11.0).abs() < 1e-13);
    assert!((x[1] - 7.0 / 11.0).abs() < 1e-13);

    // Replace column 0 with [2.0, 0.0]^T -> A_new = [[2, 1], [0, 3]], x = [1, 1] for b = [3, 3].
    lu.update(0, &[0], &[2.0]).unwrap();
    let mut rhs = [3.0, 3.0];
    lu.solve(&mut rhs).unwrap();
    assert!((rhs[0] - 1.0).abs() < 1e-13);
    assert!((rhs[1] - 1.0).abs() < 1e-13);
}
