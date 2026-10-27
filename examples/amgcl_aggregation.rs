use algebra_linear::sparse::amgcl::{AggregationParams, Amg, AmgParams, Cg, Coarsening, CsrMatrix, SmoothedAggregationParams, SolverParams};

fn solve(name: &str, a: &CsrMatrix, coarsening: Coarsening) -> Result<(), algebra_linear::sparse::amgcl::Error> {
    let b = vec![1.0; a.nrows()];
    let mut x = vec![0.0; a.nrows()];
    let mut amg = Amg::new(
        a,
        AmgParams {
            coarsening,
            coarse_enough: 16,
            ..AmgParams::default()
        },
    )?;
    let report = Cg::new(SolverParams {
        tolerance: 1e-10,
        max_iterations: 150,
        ..SolverParams::default()
    })
    .solve(a, &mut amg, &b, &mut x)?;
    println!(
        "{name:22} {:2} levels, {:3} iterations, residual {:.3e}",
        amg.levels().len(),
        report.iterations,
        report.relative_residual
    );
    Ok(())
}

fn main() -> Result<(), algebra_linear::sparse::amgcl::Error> {
    let a = CsrMatrix::poisson_2d(32, 32)?;
    solve("Ruge-Stuben", &a, Coarsening::default())?;
    solve("Aggregation", &a, Coarsening::Aggregation(AggregationParams::default()))?;
    solve(
        "Smoothed aggregation",
        &a,
        Coarsening::SmoothedAggregation(SmoothedAggregationParams::default()),
    )?;
    solve(
        "SA + spectral estimate",
        &a,
        Coarsening::SmoothedAggregation(SmoothedAggregationParams {
            estimate_spectral_radius: true,
            power_iterations: 10,
            ..SmoothedAggregationParams::default()
        }),
    )
}
