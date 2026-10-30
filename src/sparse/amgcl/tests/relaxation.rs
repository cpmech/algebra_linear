use super::setup;
use crate::sparse::amgcl::matrix::CsrMatrix;
use crate::sparse::amgcl::{AmgParams, ChebyshevParams, Error, Ilu0Params, KrylovMethod, Smoother, SolverParams, amg_setup, krylov_solve};

fn relative_residual(a: &CsrMatrix, b: &[f64], x: &[f64]) -> f64 {
    let mut r = vec![0.0; b.len()];
    a.residual(b, x, &mut r).unwrap();
    r.iter().map(|v| v * v).sum::<f64>().sqrt() / b.iter().map(|v| v * v).sum::<f64>().sqrt()
}

#[test]
fn all_serial_smoothers_converge_in_amg_gmres() {
    let smoothers = [
        Smoother::Ilu0(Ilu0Params { damping: 0.9 }),
        Smoother::Chebyshev(ChebyshevParams::default()),
        Smoother::Chebyshev(ChebyshevParams {
            degree: 4,
            scale: true,
            power_iterations: 12,
            ..ChebyshevParams::default()
        }),
        Smoother::Spai0,
        Smoother::Spai1,
    ];
    let a = CsrMatrix::poisson_2d(16, 16).unwrap();
    let b: Vec<f64> = (0..a.nrows()).map(|i| 1.0 + (i % 5) as f64 / 10.0).collect();

    for smoother in smoothers {
        let mut amg = setup(
            &a,
            Some(AmgParams {
                smoother,
                ..AmgParams::default()
            }),
        );
        let mut x = vec![0.0; a.nrows()];
        let report = krylov_solve(
            KrylovMethod::Gmres,
            &mut amg,
            &b,
            &mut x,
            SolverParams {
                tolerance: 1e-9,
                max_iterations: 100,
                gmres_restart: 30,
                ..SolverParams::default()
            },
        )
        .unwrap();
        assert!(report.converged, "{smoother:?}: {report:?}");
        assert!(relative_residual(&a, &b, &x) < 1.1e-9, "{smoother:?}: {report:?}");
    }
}

#[test]
fn ilu0_setup_canonicalizes_unsorted_duplicate_entries() {
    let a = CsrMatrix::from_parts(
        2,
        2,
        vec![0, 4, 8],
        vec![1, 0, 0, 1, 1, 0, 1, 0],
        vec![-0.5, 1.0, 1.0, -0.5, 1.0, -0.5, 1.0, -0.5],
    )
    .unwrap();
    let b = vec![1.0, 2.0];
    let mut x = vec![0.0; 2];
    let mut amg = setup(
        &a,
        Some(AmgParams {
            smoother: Smoother::Ilu0(Ilu0Params::default()),
            max_levels: 1,
            direct_coarse: false,
            ..AmgParams::default()
        }),
    );
    amg.apply(&b, &mut x).unwrap();
    assert!(relative_residual(&a, &b, &x) < 1e-13, "{x:?}");
}

#[test]
fn relaxation_parameters_and_singular_patterns_are_checked() {
    let a = CsrMatrix::poisson_2d(4, 4).unwrap();
    for smoother in [
        Smoother::Ilu0(Ilu0Params { damping: 0.0 }),
        Smoother::Chebyshev(ChebyshevParams {
            degree: 0,
            ..ChebyshevParams::default()
        }),
        Smoother::Chebyshev(ChebyshevParams {
            lower: 0.0,
            ..ChebyshevParams::default()
        }),
    ] {
        assert!(matches!(
            amg_setup(
                a.row_offsets(),
                a.col_indices(),
                a.values(),
                Some(AmgParams {
                    smoother,
                    ..AmgParams::default()
                })
            ),
            Err(Error::InvalidParameter(_))
        ));
    }

    // A null diagonal no longer stops the setup: the hierarchy gives that row a unit diagonal for
    // its own operators, because a smoother cannot divide by zero and a structural matrix may
    // legitimately have a degree-zero node. The Krylov method still sees the caller's matrix.
    // row 0 has no diagonal entry; filling it with one must not make the matrix singular, so the
    // off-diagonal values are chosen to keep the 2x2 system invertible
    let missing_diagonal = CsrMatrix::from_triplets(2, 2, [(0, 1, 2.0), (1, 0, 1.0), (1, 1, 3.0)]).unwrap();
    let mut amg = amg_setup(
        missing_diagonal.row_offsets(),
        missing_diagonal.col_indices(),
        missing_diagonal.values(),
        Some(AmgParams {
            smoother: Smoother::Ilu0(Ilu0Params::default()),
            max_levels: 1,
            direct_coarse: false,
            ..AmgParams::default()
        }),
    )
    .unwrap();
    let mut x = vec![0.0; 2];
    amg.apply(&[1.0, 1.0], &mut x).unwrap();
    assert!(x.iter().all(|v| v.is_finite()), "{x:?}");
}
