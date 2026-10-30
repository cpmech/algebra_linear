use super::setup;
use crate::sparse::amgcl::matrix::CsrMatrix;
use crate::sparse::amgcl::preconditioner::Identity;
use crate::sparse::amgcl::solver::{BiCgStab, Cg, Gmres};
use crate::sparse::amgcl::{AmgParams, KrylovMethod, Smoother, SolverParams, krylov_solve};

fn residual(a: &CsrMatrix, b: &[f64], x: &[f64]) -> f64 {
    let mut r = vec![0.0; b.len()];
    a.residual(b, x, &mut r).unwrap();
    r.iter().map(|v| v * v).sum::<f64>().sqrt() / b.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn poisson_1d(n: usize) -> CsrMatrix {
    let mut triplets = Vec::new();
    for i in 0..n {
        triplets.push((i, i, 2.0));
        if i > 0 {
            triplets.push((i, i - 1, -1.0));
        }
        if i + 1 < n {
            triplets.push((i, i + 1, -1.0));
        }
    }
    CsrMatrix::from_triplets(n, n, triplets).unwrap()
}

#[test]
fn amg_cg_solves_1d_poisson() {
    let a = poisson_1d(200);
    let b = vec![1.0; a.nrows()];
    let mut x = vec![0.0; a.nrows()];
    let mut amg = setup(&a, Some(AmgParams::default()));
    assert!(amg.levels().len() >= 2);
    let report = krylov_solve(KrylovMethod::Cg, &mut amg, &b, &mut x, SolverParams::default()).unwrap();
    assert!(report.converged, "{report:?}");
    assert!(report.iterations < 30, "{report:?}");
    assert!(residual(&a, &b, &x) < 1e-8);
}

#[test]
fn amg_cg_solves_2d_poisson_from_nonzero_guess() {
    let a = CsrMatrix::poisson_2d(24, 24).unwrap();
    let b: Vec<f64> = (0..a.nrows()).map(|i| 1.0 + (i % 7) as f64 / 10.0).collect();
    let mut x = vec![0.01; a.nrows()];
    let params = AmgParams {
        smoother: Smoother::DampedJacobi(0.72),
        ..AmgParams::default()
    };
    let mut amg = setup(&a, Some(params));
    let report = krylov_solve(
        KrylovMethod::Cg,
        &mut amg,
        &b,
        &mut x,
        SolverParams {
            tolerance: 1e-10,
            max_iterations: 100,
            ..SolverParams::default()
        },
    )
    .unwrap();
    assert!(report.converged, "{report:?}");
    assert!(report.iterations < 50, "{report:?}");
    assert!(residual(&a, &b, &x) < 1e-10);
}

#[test]
fn bicgstab_and_gmres_handle_nonsymmetric_system() {
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
        tolerance: 1e-12,
        max_iterations: 50,
        ..SolverParams::default()
    };
    let mut x = vec![0.0; 4];
    let report = BiCgStab::new(params).solve(&a, &mut Identity, &b, &mut x).unwrap();
    assert!(report.converged, "{report:?}");
    assert!(residual(&a, &b, &x) < 1e-11);

    let mut y = vec![0.0; 4];
    let report = Gmres::new(params, 3).unwrap().solve(&a, &mut Identity, &b, &mut y).unwrap();
    assert!(report.converged, "{report:?}");
    assert!(residual(&a, &b, &y) < 1e-11);
}

#[test]
fn zero_rhs_returns_zero_without_iterations() {
    let a = CsrMatrix::poisson_2d(4, 4).unwrap();
    let b = vec![0.0; a.nrows()];
    let mut x = vec![3.0; a.nrows()];
    let report = Cg::default().solve(&a, &mut Identity, &b, &mut x).unwrap();
    assert_eq!(report.iterations, 0);
    assert!(report.converged);
    assert!(x.iter().all(|&v| v == 0.0));
}

#[test]
fn coarse_lu_uses_partial_pivoting() {
    // The direct coarse solve takes the operator as it is: a zero diagonal is fine, pivoting
    // reorders the rows. (The hierarchy above it fills a null diagonal for its smoothers, which is a
    // different concern -- there the matrix must be smoothable, here it only has to be factorable.)
    use crate::sparse::amgcl::coarse::DenseLu;
    let a = CsrMatrix::from_triplets(2, 2, [(0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)]).unwrap();
    let lu = DenseLu::new(&a).unwrap();
    let mut x = vec![0.0; 2];
    lu.solve(&[2.0, 7.0], &mut x);
    assert!((x[0] - 1.0).abs() < 1e-14, "{x:?}");
    assert!((x[1] - 1.0).abs() < 1e-14, "{x:?}");
}

#[test]
fn iteration_limit_is_a_report_not_an_error() {
    let a = poisson_1d(20);
    let b = vec![1.0; 20];
    let mut x = vec![0.0; 20];
    let report = Cg::new(SolverParams {
        max_iterations: 1,
        tolerance: 1e-14,
        ..SolverParams::default()
    })
    .solve(&a, &mut Identity, &b, &mut x)
    .unwrap();
    assert_eq!(report.iterations, 1);
    assert!(!report.converged);
}

#[test]
fn direct_small_amg_level_is_exact() {
    let a = poisson_1d(8);
    let b: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let mut x = vec![0.0; 8];
    let mut amg = setup(&a, Some(AmgParams::default()));
    amg.apply(&b, &mut x).unwrap();
    assert!(residual(&a, &b, &x) < 1e-13);
}
