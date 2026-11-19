use crate::sparse::amgcl::matrix::CsrMatrix;
use crate::sparse::amgcl::memory::{BYTES_PER_MIB, MemoryBudget, bytes_to_mib_ceil};
use crate::sparse::amgcl::{
    AggregationParams, AmgParams, ChebyshevParams, Coarsening, Error, Ilu0Params, KrylovMethod, Parallelism, SmoothedAggregationParams, Smoother,
    SolverParams, amg_setup, amg_setup_owned, krylov_solve,
};

fn tridiagonal_spd_raw(n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::with_capacity(3 * n);
    let mut values = Vec::with_capacity(3 * n);
    row_ptr.push(0);
    for row in 0..n {
        if row > 0 {
            col_idx.push(row - 1);
            values.push(-1.0);
        }
        col_idx.push(row);
        values.push(4.0);
        if row + 1 < n {
            col_idx.push(row + 1);
            values.push(-1.0);
        }
        row_ptr.push(col_idx.len());
    }
    (row_ptr, col_idx, values)
}

fn diagonal_raw(n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let row_ptr: Vec<usize> = (0..=n).collect();
    let col_idx: Vec<usize> = (0..n).collect();
    let values = vec![4.0; n];
    (row_ptr, col_idx, values)
}

fn nonsymmetric_banded(n: usize) -> CsrMatrix {
    let mut triplets = Vec::new();
    for i in 0..n {
        triplets.push((i, i, 4.0));
        if i > 0 {
            triplets.push((i, i - 1, -1.5));
        }
        if i + 1 < n {
            triplets.push((i, i + 1, -0.5));
        }
        if i + 3 < n {
            triplets.push((i, i + 3, -0.25));
        }
    }
    CsrMatrix::from_triplets(n, n, triplets).unwrap()
}

fn true_relative_residual(a: &CsrMatrix, b: &[f64], x: &[f64]) -> f64 {
    let mut r = vec![0.0; b.len()];
    a.residual(b, x, &mut r).unwrap();
    let num = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    let den = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    num / den
}

#[test]
fn matmul_budget_gates_cancelling_transient_peak() {
    let a = CsrMatrix::from_parts(1, 2, vec![0, 2], vec![0, 1], vec![1.0, 1.0]).unwrap();
    let ncols = 40_000;
    let row_offsets = vec![0, ncols, 2 * ncols];
    let mut col_indices = Vec::with_capacity(2 * ncols);
    let mut values = Vec::with_capacity(2 * ncols);
    for column in 0..ncols {
        col_indices.push(column);
        values.push(1.0);
    }
    for column in 0..ncols {
        col_indices.push(column);
        values.push(-1.0);
    }
    let b = CsrMatrix::from_parts(2, ncols, row_offsets, col_indices, values).unwrap();

    let err = a
        .matmul_with_budget(&b, Parallelism::default(), &MemoryBudget::new(1), Some(0))
        .unwrap_err();
    assert!(
        matches!(
            err,
            Error::OutOfMemory {
                budget_mb: 1,
                level: Some(0),
                ..
            }
        ),
        "{err:?}"
    );

    let product = a.matmul_with_budget(&b, Parallelism::default(), &MemoryBudget::new(0), Some(0)).unwrap();
    assert_eq!(product.row_offsets(), &[0, 0]);
    assert_eq!(product.nnz(), 0);
}

#[test]
fn matmul_budget_preserves_product() {
    let a = CsrMatrix::from_triplets(2, 2, [(0, 0, 1.0), (0, 1, 2.0), (1, 1, 3.0)]).unwrap();
    let b = CsrMatrix::from_triplets(2, 2, [(0, 0, 4.0), (1, 0, 5.0), (1, 1, 6.0)]).unwrap();
    let product = a.matmul_with_budget(&b, Parallelism::default(), &MemoryBudget::new(0), None).unwrap();
    assert_eq!(product.to_dense(), vec![14.0, 12.0, 15.0, 18.0]);
}

#[test]
fn setup_rejects_spai1_transient_peak() {
    let n = 600;
    let mut triplets = Vec::with_capacity(3 * n - 2);
    for column in 0..n {
        triplets.push((0, column, if column == 0 { 4.0 } else { -1.0 }));
    }
    for row in 1..n {
        triplets.push((row, 0, -1.0));
        triplets.push((row, row, 4.0));
    }
    let a = CsrMatrix::from_triplets(n, n, triplets).unwrap();
    let params = |max_memory_mb| {
        Some(AmgParams {
            smoother: Smoother::Spai1,
            max_levels: 1,
            direct_coarse: false,
            max_memory_mb,
            ..AmgParams::default()
        })
    };

    let err = amg_setup(a.row_offsets(), a.col_indices(), a.values(), params(1)).unwrap_err();
    assert!(
        matches!(
            err,
            Error::OutOfMemory {
                budget_mb: 1,
                level: Some(0),
                ..
            }
        ),
        "{err:?}"
    );

    let amg = amg_setup(a.row_offsets(), a.col_indices(), a.values(), params(16)).unwrap();
    assert!(amg.memory_bytes() <= 16 * BYTES_PER_MIB);
}

#[test]
fn amg_setup_and_amg_setup_owned_match_across_strategies_and_retained_bytes() {
    let a = CsrMatrix::poisson_2d(16, 16).unwrap();
    let b: Vec<f64> = (0..a.nrows()).map(|i| 1.0 + (i % 7) as f64 * 0.25).collect();

    let configs = [
        None,
        Some(AmgParams {
            coarse_enough: 16,
            ..AmgParams::default()
        }),
        Some(AmgParams {
            coarsening: Coarsening::Aggregation(AggregationParams::default()),
            smoother: Smoother::Ilu0(Ilu0Params::default()),
            coarse_enough: 16,
            ..AmgParams::default()
        }),
        Some(AmgParams {
            coarsening: Coarsening::SmoothedAggregation(SmoothedAggregationParams {
                estimate_spectral_radius: true,
                power_iterations: 4,
                ..SmoothedAggregationParams::default()
            }),
            smoother: Smoother::Chebyshev(ChebyshevParams {
                scale: true,
                power_iterations: 4,
                ..ChebyshevParams::default()
            }),
            coarse_enough: 16,
            ..AmgParams::default()
        }),
        Some(AmgParams {
            smoother: Smoother::Spai0,
            coarse_enough: 16,
            ..AmgParams::default()
        }),
        Some(AmgParams {
            smoother: Smoother::Spai1,
            coarse_enough: 16,
            ..AmgParams::default()
        }),
    ];

    for params in configs {
        let mut borrowed = amg_setup(a.row_offsets(), a.col_indices(), a.values(), params).unwrap();
        let mut owned = amg_setup_owned(a.row_offsets().to_vec(), a.col_indices().to_vec(), a.values().to_vec(), params).unwrap();

        let borrowed_levels: Vec<_> = borrowed.levels().collect();
        let owned_levels: Vec<_> = owned.levels().collect();
        assert_eq!(borrowed_levels, owned_levels);
        assert_eq!(borrowed.grid_complexity(), owned.grid_complexity());
        assert_eq!(borrowed.operator_complexity(), owned.operator_complexity());
        assert_eq!(borrowed.memory_bytes(), owned.memory_bytes());
        assert_eq!(borrowed.memory_bytes(), borrowed.computed_memory_bytes());
        assert_eq!(owned.memory_bytes(), owned.computed_memory_bytes());

        let mut x_borrowed = vec![0.0; a.nrows()];
        let mut x_owned = vec![0.0; a.nrows()];
        let method = match params.map(|p| p.smoother) {
            None | Some(Smoother::GaussSeidel) | Some(Smoother::DampedJacobi(_)) => KrylovMethod::Cg,
            _ => KrylovMethod::Gmres,
        };
        let solver_params = SolverParams {
            tolerance: 1e-10,
            max_iterations: 300,
            ..SolverParams::default()
        };
        let report_borrowed = krylov_solve(method, &mut borrowed, &b, &mut x_borrowed, solver_params).unwrap();
        let report_owned = krylov_solve(method, &mut owned, &b, &mut x_owned, solver_params).unwrap();
        assert!(report_borrowed.converged, "{params:?}: {report_borrowed:?}");
        assert_eq!(report_borrowed, report_owned);
        assert_eq!(x_borrowed, x_owned);
    }
}

#[test]
fn setup_rejects_level_zero_input_over_budget_and_succeeds_when_lifted() {
    let (row_ptr, col_idx, values) = tridiagonal_spd_raw(30_000);
    let tight = Some(AmgParams {
        max_memory_mb: 1,
        ..AmgParams::default()
    });

    let err_borrowed = amg_setup(&row_ptr, &col_idx, &values, tight).unwrap_err();
    assert!(
        matches!(
            err_borrowed,
            Error::OutOfMemory {
                budget_mb: 1,
                level: None,
                ..
            }
        ),
        "{err_borrowed:?}"
    );

    let err_owned = amg_setup_owned(row_ptr.clone(), col_idx.clone(), values.clone(), tight).unwrap_err();
    assert!(
        matches!(
            err_owned,
            Error::OutOfMemory {
                budget_mb: 1,
                level: None,
                ..
            }
        ),
        "{err_owned:?}"
    );

    let lifted = Some(AmgParams {
        max_memory_mb: 16,
        ..AmgParams::default()
    });
    let amg_borrowed = amg_setup(&row_ptr, &col_idx, &values, lifted).unwrap();
    let amg_owned = amg_setup_owned(row_ptr, col_idx, values, lifted).unwrap();
    assert_eq!(amg_borrowed.memory_bytes(), amg_owned.memory_bytes());
    assert!(amg_borrowed.memory_bytes() <= 16 * BYTES_PER_MIB);
}

#[test]
fn setup_enforces_budget_across_coarsening_strategies_and_direct_coarse() {
    let a = CsrMatrix::poisson_2d(100, 100).unwrap();
    for coarsening in [
        Coarsening::default(),
        Coarsening::Aggregation(AggregationParams::default()),
        Coarsening::SmoothedAggregation(SmoothedAggregationParams {
            estimate_spectral_radius: true,
            power_iterations: 4,
            ..SmoothedAggregationParams::default()
        }),
    ] {
        let tight = Some(AmgParams {
            coarsening,
            max_memory_mb: 1,
            ..AmgParams::default()
        });
        let err = amg_setup(a.row_offsets(), a.col_indices(), a.values(), tight).unwrap_err();
        assert!(
            matches!(
                err,
                Error::OutOfMemory {
                    budget_mb: 1,
                    level: Some(_),
                    ..
                }
            ),
            "{coarsening:?}: {err:?}"
        );

        let lifted = Some(AmgParams {
            coarsening,
            max_memory_mb: 16,
            ..AmgParams::default()
        });
        let amg = amg_setup(a.row_offsets(), a.col_indices(), a.values(), lifted).unwrap();
        assert_eq!(amg.memory_bytes(), amg.computed_memory_bytes());
        assert!(amg.memory_bytes() <= 16 * BYTES_PER_MIB);
    }

    // Coarse dense LU must propagate OutOfMemory instead of swallowing it.
    let small = CsrMatrix::poisson_2d(20, 20).unwrap();
    let err = amg_setup(
        small.row_offsets(),
        small.col_indices(),
        small.values(),
        Some(AmgParams {
            coarse_enough: 400,
            direct_coarse: true,
            max_memory_mb: 1,
            ..AmgParams::default()
        }),
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            Error::OutOfMemory {
                budget_mb: 1,
                level: Some(0),
                ..
            }
        ),
        "{err:?}"
    );
    let ok = amg_setup(
        small.row_offsets(),
        small.col_indices(),
        small.values(),
        Some(AmgParams {
            coarse_enough: 400,
            direct_coarse: true,
            max_memory_mb: 2,
            ..AmgParams::default()
        }),
    )
    .unwrap();
    assert!(ok.levels().next().unwrap().direct_solver);
}

#[test]
fn setup_enforces_budget_across_smoothers() {
    let n = if core::mem::size_of::<usize>() >= 8 { 32_000 } else { 40_000 };
    let (row_ptr, col_idx, values) = diagonal_raw(n);
    let smoothers = [
        Smoother::Ilu0(Ilu0Params::default()),
        Smoother::Chebyshev(ChebyshevParams {
            scale: true,
            power_iterations: 4,
            ..ChebyshevParams::default()
        }),
        Smoother::Spai0,
        Smoother::Spai1,
    ];
    for smoother in smoothers {
        let err = amg_setup(
            &row_ptr,
            &col_idx,
            &values,
            Some(AmgParams {
                smoother,
                max_levels: 1,
                direct_coarse: false,
                max_memory_mb: 1,
                ..AmgParams::default()
            }),
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                Error::OutOfMemory {
                    budget_mb: 1,
                    level: Some(0),
                    ..
                }
            ),
            "{smoother:?}: {err:?}"
        );

        let amg = amg_setup(
            &row_ptr,
            &col_idx,
            &values,
            Some(AmgParams {
                smoother,
                max_levels: 1,
                direct_coarse: false,
                max_memory_mb: 8,
                ..AmgParams::default()
            }),
        )
        .unwrap();
        assert_eq!(amg.memory_bytes(), amg.computed_memory_bytes());
        assert!(amg.memory_bytes() <= 8 * BYTES_PER_MIB);
    }
}

#[test]
fn krylov_solve_enforces_memory_budget_and_converges_when_lifted() {
    let n = 15_000;
    let (row_ptr, col_idx, values) = tridiagonal_spd_raw(n);
    let b = vec![1.0; n];

    for method in [KrylovMethod::Cg, KrylovMethod::BiCgStab, KrylovMethod::Gmres] {
        let mut identity = amg_setup(&row_ptr, &col_idx, &values, None).unwrap();
        let mut x = vec![0.0; n];
        let err = krylov_solve(
            method,
            &mut identity,
            &b,
            &mut x,
            SolverParams {
                gmres_restart: 10,
                max_memory_mb: 1,
                ..SolverParams::default()
            },
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                Error::OutOfMemory {
                    budget_mb: 1,
                    level: None,
                    ..
                }
            ),
            "{method:?}: {err:?}"
        );

        let report = krylov_solve(
            method,
            &mut identity,
            &b,
            &mut x,
            SolverParams {
                gmres_restart: 10,
                tolerance: 1e-10,
                max_iterations: 100,
                max_memory_mb: 8,
                ..SolverParams::default()
            },
        )
        .unwrap();
        assert!(report.converged, "{method:?}: {report:?}");
    }

    // AMG + GMRES: verify solver enforces the tighter nonzero budget alongside retained AMG memory.
    let small_n = 4_096;
    let (p_ptr, p_idx, p_val) = tridiagonal_spd_raw(small_n);
    let small_b = vec![1.0; small_n];
    let mut amg = amg_setup(&p_ptr, &p_idx, &p_val, Some(AmgParams::default())).unwrap();
    let amg_mb_floor = bytes_to_mib_ceil(amg.memory_bytes());
    let mut x = vec![0.0; small_n];
    let err = krylov_solve(
        KrylovMethod::Gmres,
        &mut amg,
        &small_b,
        &mut x,
        SolverParams {
            gmres_restart: 40,
            max_memory_mb: amg_mb_floor,
            ..SolverParams::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, Error::OutOfMemory { level: None, .. }), "{err:?}");

    let report = krylov_solve(
        KrylovMethod::Gmres,
        &mut amg,
        &small_b,
        &mut x,
        SolverParams {
            gmres_restart: 40,
            tolerance: 1e-10,
            max_iterations: 100,
            max_memory_mb: amg_mb_floor + 4,
            ..SolverParams::default()
        },
    )
    .unwrap();
    assert!(report.converged, "{report:?}");

    // Tighter AMG budget takes precedence over an unlimited or looser SolverParams budget.
    let mut tight_amg = amg_setup(
        &p_ptr,
        &p_idx,
        &p_val,
        Some(AmgParams {
            max_memory_mb: amg_mb_floor,
            ..AmgParams::default()
        }),
    )
    .unwrap();
    let err = krylov_solve(
        KrylovMethod::Gmres,
        &mut tight_amg,
        &small_b,
        &mut x,
        SolverParams {
            gmres_restart: 40,
            max_memory_mb: amg_mb_floor + 10,
            ..SolverParams::default()
        },
    )
    .unwrap_err();
    assert_eq!(
        err,
        Error::OutOfMemory {
            needed_mb: bytes_to_mib_ceil(
                tight_amg.memory_bytes() + 43 * small_n * 8 + (41 * 40 + 4 * 40 + 1) * 8 + 40 * core::mem::size_of::<Vec<f64>>()
            ),
            budget_mb: amg_mb_floor,
            level: None,
        }
    );
}

#[test]
fn single_basis_gmres_converges_across_multiple_restarts() {
    let n = 24;
    let a = nonsymmetric_banded(n);
    let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 5) as f64 * 0.5).collect();

    let preconditioners = [
        None,
        Some(AmgParams {
            smoother: Smoother::GaussSeidel,
            coarse_enough: 8,
            ..AmgParams::default()
        }),
        Some(AmgParams {
            smoother: Smoother::DampedJacobi(0.72),
            coarse_enough: 8,
            ..AmgParams::default()
        }),
        Some(AmgParams {
            smoother: Smoother::Ilu0(Ilu0Params::default()),
            coarse_enough: 8,
            ..AmgParams::default()
        }),
    ];

    for params in preconditioners {
        let mut amg = amg_setup(a.row_offsets(), a.col_indices(), a.values(), params).unwrap();
        let mut x = vec![0.0; n];
        let report = krylov_solve(
            KrylovMethod::Gmres,
            &mut amg,
            &b,
            &mut x,
            SolverParams {
                tolerance: 1e-11,
                max_iterations: 200,
                gmres_restart: 3,
                ..SolverParams::default()
            },
        )
        .unwrap();
        assert!(report.converged, "{params:?}: {report:?}");
        assert!(report.iterations > 3, "expected multiple restarts for {params:?}, got {report:?}");
        let rel = true_relative_residual(&a, &b, &x);
        assert!(rel < 1e-10, "{params:?}: relative residual {rel:e}");
    }
}

#[test]
fn ensure_nonzero_diagonal_canonicalizes_unsorted_duplicates_and_fills_missing_diagonal() {
    // Row 0: unsorted entries, duplicate (0, 0) summing to 0.0 -> filled with 1.0; duplicate (0, 2) canceling to 0.0 -> dropped.
    // Row 1: duplicate (1, 1) summing to 3.0; no fill needed.
    // Row 2: missing (2, 2) -> filled with 1.0.
    let row_ptr = vec![0, 5, 8, 10];
    let col_idx = vec![2, 0, 1, 0, 2, 1, 0, 1, 1, 0];
    let values = vec![0.5, 2.0, -1.0, -2.0, -0.5, 2.0, -0.5, 1.0, -0.5, 0.25];
    let raw = CsrMatrix::from_parts(3, 3, row_ptr.clone(), col_idx.clone(), values.clone()).unwrap();
    let canonical = raw.canonicalized().unwrap();
    assert_eq!(canonical.row(0).columns, &[1]);
    assert_eq!(canonical.row(0).values, &[-1.0]);

    let mut amg = amg_setup_owned(
        row_ptr,
        col_idx,
        values,
        Some(AmgParams {
            smoother: Smoother::Ilu0(Ilu0Params::default()),
            max_levels: 1,
            direct_coarse: false,
            ..AmgParams::default()
        }),
    )
    .unwrap();
    assert_eq!(amg.memory_bytes(), amg.computed_memory_bytes());
    let mut x = vec![0.0; 3];
    amg.apply(&[1.0, 2.0, 3.0], &mut x).unwrap();
    assert!(x.iter().all(|v| v.is_finite()), "{x:?}");
    // Row 0 had filled diagonal 1.0 and -1.0 at col 1, row 1 had [ -0.5, 3.0 ], row 2 had [ 0.25, -0.5, 1.0 ].
    let expected_filled = CsrMatrix::from_triplets(
        3,
        3,
        [
            (0, 0, 1.0),
            (0, 1, -1.0),
            (1, 0, -0.5),
            (1, 1, 3.0),
            (2, 0, 0.25),
            (2, 1, -0.5),
            (2, 2, 1.0),
        ],
    )
    .unwrap();
    assert!(true_relative_residual(&expected_filled, &[1.0, 2.0, 3.0], &x) < 1e-13, "{x:?}");
}

#[test]
fn parallel_budgeted_setup_and_multi_restart_gmres_succeed() {
    let a = CsrMatrix::poisson_2d(36, 36).unwrap();
    let n = a.nrows();
    let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 5) as f64 * 0.1).collect();
    let parallelism = crate::sparse::amgcl::Parallelism {
        threads: 4,
        min_parallel_len: 32,
    };

    let mut amg = amg_setup(
        a.row_offsets(),
        a.col_indices(),
        a.values(),
        Some(AmgParams {
            smoother: Smoother::Spai1,
            coarse_enough: 16,
            parallelism,
            max_memory_mb: 16,
            ..AmgParams::default()
        }),
    )
    .unwrap();
    assert_eq!(amg.memory_bytes(), amg.computed_memory_bytes());

    let mut x = vec![0.0; n];
    let report = krylov_solve(
        KrylovMethod::Gmres,
        &mut amg,
        &b,
        &mut x,
        SolverParams {
            tolerance: 1e-10,
            max_iterations: 200,
            gmres_restart: 3,
            parallelism,
            max_memory_mb: 16,
            ..SolverParams::default()
        },
    )
    .unwrap();
    assert!(report.converged, "{report:?}");
    assert!(true_relative_residual(&a, &b, &x) < 1e-9);
}

#[test]
fn validation_errors_take_priority_over_memory_budget() {
    let (row_ptr, col_idx, values) = tridiagonal_spd_raw(30_000);
    let err = amg_setup(
        &row_ptr,
        &col_idx,
        &values,
        Some(AmgParams {
            coarse_enough: 0,
            max_memory_mb: 1,
            ..AmgParams::default()
        }),
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidParameter(_)), "{err:?}");

    let (small_ptr, small_idx, small_val) = tridiagonal_spd_raw(15_000);
    let mut amg = amg_setup(&small_ptr, &small_idx, &small_val, None).unwrap();
    let mut x = vec![0.0; 15_000];
    let err = krylov_solve(
        KrylovMethod::Gmres,
        &mut amg,
        &[1.0; 10],
        &mut x,
        SolverParams {
            gmres_restart: 5,
            max_memory_mb: 1,
            ..SolverParams::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, Error::DimensionMismatch { .. }), "{err:?}");
}
