use algebra_linear::sparse::amgcl::{
    AggregationParams, AmgParams, Coarsening, KrylovMethod, SmoothedAggregationParams, SolverParams, amg_setup, krylov_solve,
};

pub mod common;

fn solve(name: &str, a: &common::Csr, coarsening: Coarsening) -> Result<(), algebra_linear::sparse::amgcl::Error> {
    let unknowns = a.rows();
    let b = vec![1.0; unknowns];
    let mut x = vec![0.0; unknowns];
    let mut amg = amg_setup(
        &a.row_ptr,
        &a.col_idx,
        &a.values,
        Some(AmgParams {
            coarsening,
            coarse_enough: 16,
            ..AmgParams::default()
        }),
    )?;
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
    )?;
    println!(
        "{name:22} {:2} levels, {:3} iterations, residual {:.3e}",
        amg.levels().len(),
        report.iterations,
        report.relative_residual
    );
    Ok(())
}

fn main() -> Result<(), algebra_linear::sparse::amgcl::Error> {
    let a = common::poisson_2d(32, 32);
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
