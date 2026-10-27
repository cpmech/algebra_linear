use crate::sparse::amgcl::matrix::CsrMatrix;
use crate::sparse::amgcl::{Amg, AmgParams, Error, KrylovMethod, SolverParams, amg_setup, krylov_solve};

/// Raw CSR arrays for the 1-D Laplacian on `n` unknowns.
fn poisson_1d(n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for row in 0..n {
        if row > 0 {
            col_idx.push(row - 1);
            values.push(-1.0);
        }
        col_idx.push(row);
        values.push(2.0);
        if row + 1 < n {
            col_idx.push(row + 1);
            values.push(-1.0);
        }
        row_ptr.push(col_idx.len());
    }
    (row_ptr, col_idx, values)
}

fn residual(a: &CsrMatrix, b: &[f64], x: &[f64]) -> f64 {
    let mut r = vec![0.0; b.len()];
    a.residual(b, x, &mut r).unwrap();
    let numerator = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    let denominator = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    numerator / denominator
}

#[test]
fn matrix_only_state_is_the_identity_preconditioner() {
    let n = 200;
    let (row_ptr, col_idx, values) = poisson_1d(n);
    let a = CsrMatrix::from_parts(n, n, row_ptr.clone(), col_idx.clone(), values.clone()).unwrap();
    let b = vec![1.0; n];
    let params = SolverParams {
        tolerance: 1e-9,
        max_iterations: 500,
        ..SolverParams::default()
    };

    let mut identity = amg_setup(&row_ptr, &col_idx, &values, None).unwrap();
    assert_eq!(identity.levels().len(), 1);
    let mut x_identity = vec![0.0; n];
    let identity_report = krylov_solve(KrylovMethod::Cg, &mut identity, &b, &mut x_identity, params).unwrap();
    assert!(identity_report.converged, "{identity_report:?}");
    assert!(residual(&a, &b, &x_identity) < 1e-8, "{identity_report:?}");

    let mut amg = amg_setup(&row_ptr, &col_idx, &values, Some(AmgParams::default())).unwrap();
    assert!(amg.levels().len() > 1);
    let mut x_amg = vec![0.0; n];
    let amg_report = krylov_solve(KrylovMethod::Cg, &mut amg, &b, &mut x_amg, params).unwrap();
    assert!(amg_report.converged, "{amg_report:?}");
    assert!(residual(&a, &b, &x_amg) < 1e-8, "{amg_report:?}");
    assert!(
        amg_report.iterations < identity_report.iterations,
        "AMG: {amg_report:?}, identity: {identity_report:?}"
    );
}

#[test]
fn matrix_only_apply_copies_the_right_hand_side() {
    let (row_ptr, col_idx, values) = poisson_1d(6);
    let mut amg = amg_setup(&row_ptr, &col_idx, &values, None).unwrap();
    let rhs = [1.0, -2.0, 3.5, 0.0, 4.0, -1.5];
    let mut x = [9.0; 6];
    amg.apply(&rhs, &mut x).unwrap();
    assert_eq!(x, rhs);

    let mut untouched = [7.0; 6];
    amg.cycle(&rhs, &mut untouched).unwrap();
    assert_eq!(untouched, [7.0; 6]);
}

#[test]
fn matrix_only_state_solves_nonsymmetric_systems_with_bicgstab_and_gmres() {
    let a = CsrMatrix::from_triplets(
        4,
        4,
        [
            (0, 0, 4.0),
            (0, 1, 1.0),
            (1, 0, 2.0),
            (1, 1, 5.0),
            (1, 2, -1.0),
            (2, 1, 1.0),
            (2, 2, 6.0),
            (2, 3, 2.0),
            (3, 2, -1.0),
            (3, 3, 3.0),
        ],
    )
    .unwrap();
    let b = [1.0, 2.0, 3.0, 4.0];
    for method in [KrylovMethod::BiCgStab, KrylovMethod::Gmres] {
        let mut amg = amg_setup(a.row_offsets(), a.col_indices(), a.values(), None).unwrap();
        let mut x = [0.0; 4];
        let report = krylov_solve(
            method,
            &mut amg,
            &b,
            &mut x,
            SolverParams {
                tolerance: 1e-12,
                max_iterations: 50,
                gmres_restart: 3,
                ..SolverParams::default()
            },
        )
        .unwrap();
        assert!(report.converged, "{method:?}: {report:?}");
        assert!(residual(&a, &b, &x) < 1e-10, "{method:?}: {report:?}");
    }
}

#[test]
fn poisson_hierarchy_has_multiple_levels() {
    let a = CsrMatrix::poisson_2d(24, 24).unwrap();
    let mut amg = amg_setup(
        a.row_offsets(),
        a.col_indices(),
        a.values(),
        Some(AmgParams {
            coarse_enough: 16,
            ..AmgParams::default()
        }),
    )
    .unwrap();
    let levels: Vec<_> = amg.levels().collect();
    assert!(levels.len() > 1, "{levels:?}");
    for pair in levels.windows(2) {
        assert!(pair[1].unknowns < pair[0].unknowns, "{levels:?}");
    }
    assert!(levels.last().unwrap().direct_solver, "{levels:?}");
    assert!(amg.grid_complexity() > 1.0);
    assert!(amg.operator_complexity() > 1.0);

    let b = vec![1.0; a.nrows()];
    let mut x = vec![0.0; a.nrows()];
    let report = krylov_solve(
        KrylovMethod::Cg,
        &mut amg,
        &b,
        &mut x,
        SolverParams {
            tolerance: 1e-10,
            max_iterations: 150,
            ..SolverParams::default()
        },
    )
    .unwrap();
    assert!(report.converged, "{report:?}");
}

#[test]
fn gmres_restart_is_honored() {
    let n = 24;
    let (row_ptr, col_idx, values) = poisson_1d(n);
    let b = vec![1.0; n];
    let params = |gmres_restart| SolverParams {
        tolerance: 1e-10,
        max_iterations: 5000,
        gmres_restart,
        ..SolverParams::default()
    };

    let mut short_cycle = amg_setup(&row_ptr, &col_idx, &values, None).unwrap();
    let mut short = vec![0.0; n];
    let short_report = krylov_solve(KrylovMethod::Gmres, &mut short_cycle, &b, &mut short, params(1)).unwrap();
    assert!(short_report.converged, "{short_report:?}");

    let mut long_cycle = amg_setup(&row_ptr, &col_idx, &values, None).unwrap();
    let mut long = vec![0.0; n];
    let long_report = krylov_solve(KrylovMethod::Gmres, &mut long_cycle, &b, &mut long, params(30)).unwrap();
    assert!(long_report.converged, "{long_report:?}");
    assert!(long_report.iterations <= n, "{long_report:?}");

    assert!(
        short_report.iterations > long_report.iterations,
        "GMRES(1): {short_report:?}, GMRES(30): {long_report:?}"
    );

    let mut invalid = amg_setup(&row_ptr, &col_idx, &values, None).unwrap();
    let mut x = vec![0.0; n];
    let error = krylov_solve(KrylovMethod::Gmres, &mut invalid, &b, &mut x, params(0)).unwrap_err();
    assert_eq!(error, Error::InvalidParameter("GMRES restart must be positive".into()));
}

#[test]
fn iteration_limit_returns_a_non_convergence_report() {
    let (row_ptr, col_idx, values) = poisson_1d(20);
    let b = vec![1.0; 20];
    let mut x = vec![0.0; 20];
    let mut amg = amg_setup(&row_ptr, &col_idx, &values, None).unwrap();
    let report = krylov_solve(
        KrylovMethod::Cg,
        &mut amg,
        &b,
        &mut x,
        SolverParams {
            max_iterations: 1,
            tolerance: 1e-14,
            ..SolverParams::default()
        },
    )
    .unwrap();
    assert_eq!(report.iterations, 1);
    assert!(!report.converged);
    assert!(report.relative_residual > 1e-14);
}

#[test]
fn malformed_csr_arrays_are_rejected() {
    assert!(matches!(amg_setup(&[], &[], &[], None), Err(Error::InvalidMatrix(_))));
    assert!(matches!(amg_setup(&[0], &[], &[], None), Err(Error::InvalidMatrix(_))));
    assert!(matches!(amg_setup(&[0, 1], &[1], &[1.0], None), Err(Error::InvalidMatrix(_))));
    assert!(matches!(amg_setup(&[0, 1], &[0], &[f64::NAN], None), Err(Error::InvalidMatrix(_))));
    assert!(matches!(amg_setup(&[0, 2], &[0], &[1.0], None), Err(Error::InvalidMatrix(_))));
}

#[test]
fn amg_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Amg>();
}
