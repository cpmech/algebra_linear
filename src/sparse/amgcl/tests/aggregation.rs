use super::setup;
use crate::sparse::amgcl::matrix::CsrMatrix;
use crate::sparse::amgcl::{
    AggregationParams, AmgParams, Coarsening, Error, KrylovMethod, PlainAggregationParams, SmoothedAggregationParams, SolverParams, amg_setup,
    krylov_solve,
};

fn residual(a: &CsrMatrix, b: &[f64], x: &[f64]) -> f64 {
    let mut r = vec![0.0; b.len()];
    a.residual(b, x, &mut r).unwrap();
    r.iter().map(|v| v * v).sum::<f64>().sqrt() / b.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn solve(coarsening: Coarsening) -> (crate::sparse::amgcl::SolveReport, f64, usize) {
    let a = CsrMatrix::poisson_2d(24, 24).unwrap();
    let b: Vec<f64> = (0..a.nrows()).map(|i| 0.5 + (i % 11) as f64 / 10.0).collect();
    let mut x = vec![0.02; a.nrows()];
    let params = AmgParams {
        coarsening,
        coarse_enough: 16,
        ..AmgParams::default()
    };
    let mut amg = setup(&a, Some(params));
    let levels = amg.levels().len();
    assert!(levels >= 3, "expected at least three levels, got {levels}");
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
    let true_residual = residual(&a, &b, &x);
    (report, true_residual, levels)
}

#[test]
fn aggregation_amg_cg_solves_poisson() {
    let (report, residual, _) = solve(Coarsening::Aggregation(AggregationParams::default()));
    assert!(report.converged, "{report:?}");
    assert!(residual < 1e-9, "true residual: {residual:e}");
}

#[test]
fn smoothed_aggregation_amg_cg_solves_poisson() {
    let (report, residual, _) = solve(Coarsening::SmoothedAggregation(SmoothedAggregationParams::default()));
    assert!(report.converged, "{report:?}");
    assert!(report.iterations < 60, "{report:?}");
    assert!(residual < 1e-9, "true residual: {residual:e}");
}

#[test]
fn smoothed_aggregation_supports_both_spectral_estimators() {
    for power_iterations in [0, 10] {
        let (report, residual, _) = solve(Coarsening::SmoothedAggregation(SmoothedAggregationParams {
            estimate_spectral_radius: true,
            power_iterations,
            ..SmoothedAggregationParams::default()
        }));
        assert!(report.converged, "power_iterations={power_iterations}: {report:?}");
        assert!(residual < 1e-9, "true residual: {residual:e}");
    }
}

#[test]
fn aggregation_handles_unsorted_duplicate_diagonals() {
    let a = CsrMatrix::from_parts(
        3,
        3,
        vec![0, 3, 7, 9],
        vec![1, 0, 0, 2, 0, 1, 1, 2, 1],
        vec![-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 2.0, -1.0],
    )
    .unwrap();
    let b = vec![1.0; 3];
    let mut x = vec![0.0; 3];
    let mut amg = setup(
        &a,
        Some(AmgParams {
            coarsening: Coarsening::Aggregation(AggregationParams::default()),
            coarse_enough: 1,
            ..AmgParams::default()
        }),
    );
    let report = krylov_solve(KrylovMethod::Cg, &mut amg, &b, &mut x, SolverParams::default()).unwrap();
    assert!(report.converged, "{report:?}");
    assert!(residual(&a, &b, &x) < 1e-8);
}

#[test]
fn invalid_aggregation_parameters_are_rejected_before_setup() {
    let a = CsrMatrix::from_triplets(1, 1, [(0, 0, 1.0)]).unwrap();
    let cases = [
        Coarsening::Aggregation(AggregationParams {
            over_interp: f64::INFINITY,
            ..AggregationParams::default()
        }),
        Coarsening::Aggregation(AggregationParams {
            aggregation: PlainAggregationParams { eps_strong: 1.0 },
            ..AggregationParams::default()
        }),
        Coarsening::SmoothedAggregation(SmoothedAggregationParams {
            relax: -1.0,
            ..SmoothedAggregationParams::default()
        }),
        Coarsening::SmoothedAggregation(SmoothedAggregationParams {
            aggregation: PlainAggregationParams { eps_strong: 0.0 },
            ..SmoothedAggregationParams::default()
        }),
    ];
    for coarsening in cases {
        let error = amg_setup(
            a.row_offsets(),
            a.col_indices(),
            a.values(),
            Some(AmgParams {
                coarsening,
                ..AmgParams::default()
            }),
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidParameter(_)), "{error:?}");
    }
}
