use crate::sparse::amgcl::{
    AggregationParams, Amg, AmgParams, BiCgStab, Cg, ChebyshevParams, Coarsening, CsrMatrix, Error, Gmres, Identity, Ilu0Params, Parallelism,
    SmoothedAggregationParams, Smoother, SolverParams,
};

fn parallel() -> Parallelism {
    Parallelism {
        threads: 4,
        min_parallel_len: 32,
    }
}

fn residual(a: &CsrMatrix, b: &[f64], x: &[f64]) -> f64 {
    let mut r = vec![0.0; b.len()];
    a.residual(b, x, &mut r).unwrap();
    r.iter().map(|v| v * v).sum::<f64>().sqrt() / b.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn solver_params() -> SolverParams {
    SolverParams {
        tolerance: 1e-9,
        max_iterations: 150,
        parallelism: parallel(),
        ..SolverParams::default()
    }
}

#[test]
fn parallel_matrix_kernels_match_serial_results() {
    let a = CsrMatrix::poisson_2d(18, 16).unwrap();
    let x: Vec<f64> = (0..a.ncols()).map(|i| (i % 13) as f64 * 0.25 - 1.0).collect();
    let b = vec![0.75; a.nrows()];
    let mut serial = vec![0.0; a.nrows()];
    let mut threaded = vec![0.0; a.nrows()];
    a.spmv(&x, &mut serial).unwrap();
    a.spmv_with(&x, &mut threaded, parallel()).unwrap();
    assert_eq!(serial, threaded);

    a.spmv_add(&x, &mut serial).unwrap();
    a.spmv_add_with(&x, &mut threaded, parallel()).unwrap();
    assert_eq!(serial, threaded);

    a.residual(&b, &x, &mut serial).unwrap();
    a.residual_with(&b, &x, &mut threaded, parallel()).unwrap();
    assert_eq!(serial, threaded);

    let transpose = a.transpose();
    let serial_product = transpose.matmul(&a).unwrap();
    let parallel_product = transpose.matmul_with(&a, parallel()).unwrap();
    assert_eq!(serial_product, parallel_product);
}

#[test]
fn every_smoother_works_in_parallel_amg_gmres() {
    let smoothers = [
        Smoother::GaussSeidel,
        Smoother::DampedJacobi(0.72),
        Smoother::Ilu0(Ilu0Params::default()),
        Smoother::Chebyshev(ChebyshevParams {
            degree: 4,
            power_iterations: 6,
            scale: true,
            ..ChebyshevParams::default()
        }),
        Smoother::Spai0,
        Smoother::Spai1,
    ];
    let a = CsrMatrix::poisson_2d(20, 20).unwrap();
    let b: Vec<f64> = (0..a.nrows()).map(|i| 1.0 + (i % 5) as f64).collect();
    for smoother in smoothers {
        let mut amg = Amg::new(
            &a,
            AmgParams {
                smoother,
                parallelism: parallel(),
                coarse_enough: 16,
                ..AmgParams::default()
            },
        )
        .unwrap();
        let mut x = vec![0.0; a.nrows()];
        let report = Gmres::new(solver_params(), 20).unwrap().solve(&a, &mut amg, &b, &mut x).unwrap();
        assert!(report.converged, "{smoother:?}: {report:?}");
        assert!(residual(&a, &b, &x) < 1e-8, "{smoother:?}");
    }
}

#[test]
fn parallel_cg_is_deterministic_and_converges() {
    let a = CsrMatrix::poisson_2d(28, 28).unwrap();
    let b = vec![1.0; a.nrows()];
    let mut first = vec![0.0; a.nrows()];
    let mut second = vec![0.0; a.nrows()];
    let params = AmgParams {
        smoother: Smoother::DampedJacobi(0.72),
        parallelism: parallel(),
        ..AmgParams::default()
    };
    let mut first_amg = Amg::new(&a, params).unwrap();
    let mut second_amg = Amg::new(&a, params).unwrap();
    let first_report = Cg::new(solver_params()).solve(&a, &mut first_amg, &b, &mut first).unwrap();
    let second_report = Cg::new(solver_params()).solve(&a, &mut second_amg, &b, &mut second).unwrap();
    assert!(first_report.converged, "{first_report:?}");
    assert_eq!(first_report, second_report);
    assert_eq!(first, second);
    assert!(residual(&a, &b, &first) < 1e-8);
}

#[test]
fn parallel_aggregation_hierarchies_converge() {
    let a = CsrMatrix::poisson_2d(20, 20).unwrap();
    let b = vec![1.0; a.nrows()];
    let strategies = [
        Coarsening::Aggregation(AggregationParams::default()),
        Coarsening::SmoothedAggregation(SmoothedAggregationParams {
            estimate_spectral_radius: true,
            power_iterations: 6,
            ..SmoothedAggregationParams::default()
        }),
    ];
    for coarsening in strategies {
        let mut amg = Amg::new(
            &a,
            AmgParams {
                coarsening,
                parallelism: parallel(),
                coarse_enough: 16,
                ..AmgParams::default()
            },
        )
        .unwrap();
        let mut x = vec![0.0; a.nrows()];
        let report = Cg::new(solver_params()).solve(&a, &mut amg, &b, &mut x).unwrap();
        assert!(report.converged, "{coarsening:?}: {report:?}");
        assert!(residual(&a, &b, &x) < 1e-8);
    }
}

#[test]
fn parallel_bicgstab_and_gmres_handle_nonsymmetric_systems() {
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
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let params = SolverParams {
        parallelism: Parallelism {
            threads: 4,
            min_parallel_len: 1,
        },
        ..solver_params()
    };
    let mut x = vec![0.0; 4];
    let report = BiCgStab::new(params).solve(&a, &mut Identity, &b, &mut x).unwrap();
    assert!(report.converged, "{report:?}");
    assert!(residual(&a, &b, &x) < 1e-8);

    let mut x = vec![0.0; 4];
    let report = Gmres::new(params, 3).unwrap().solve(&a, &mut Identity, &b, &mut x).unwrap();
    assert!(report.converged, "{report:?}");
    assert!(residual(&a, &b, &x) < 1e-8);
}

#[test]
fn zero_threads_are_rejected_by_public_entry_points() {
    let a = CsrMatrix::from_triplets(1, 1, [(0, 0, 1.0)]).unwrap();
    let invalid = Parallelism::new(0);
    let error = Amg::new(
        &a,
        AmgParams {
            parallelism: invalid,
            ..AmgParams::default()
        },
    )
    .unwrap_err();
    assert!(matches!(error, Error::InvalidParameter(_)));

    let error = Cg::new(SolverParams {
        parallelism: invalid,
        ..SolverParams::default()
    })
    .solve(&a, &mut Identity, &[1.0], &mut [0.0])
    .unwrap_err();
    assert!(matches!(error, Error::InvalidParameter(_)));

    let error = a.spmv_with(&[1.0], &mut [0.0], invalid).unwrap_err();
    assert!(matches!(error, Error::InvalidParameter(_)));
}

#[test]
fn default_execution_remains_serial() {
    assert_eq!(Parallelism::default().threads, 1);
    assert_eq!(AmgParams::default().parallelism.threads, 1);
    assert_eq!(SolverParams::default().parallelism.threads, 1);
    assert!(Parallelism::auto().threads >= 1);
}
