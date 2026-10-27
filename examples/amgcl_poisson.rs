use algebra_linear::sparse::amgcl::{AmgParams, KrylovMethod, SolverParams, amg_setup, krylov_solve};

pub mod common;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = std::env::args().nth(1).map(|value| value.parse::<usize>()).transpose()?.unwrap_or(32);
    let a = common::poisson_2d(n, n);
    let unknowns = a.rows();
    let b = vec![1.0; unknowns];
    let mut x = vec![0.0; unknowns];

    let mut amg = amg_setup(&a.row_ptr, &a.col_idx, &a.values, Some(AmgParams::default()))?;
    println!("AMG hierarchy ({unknowns} unknowns):");
    for (depth, level) in amg.levels().enumerate() {
        println!(
            "  {depth:2}: {:8} rows, {:10} nonzeros{}",
            level.unknowns,
            level.nonzeros,
            if level.direct_solver { ", direct LU" } else { "" }
        );
    }
    println!(
        "grid complexity: {:.3}, operator complexity: {:.3}",
        amg.grid_complexity(),
        amg.operator_complexity()
    );

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
    )?;
    println!(
        "{} after {} iterations; relative residual = {:.3e}",
        if report.converged { "converged" } else { "not converged" },
        report.iterations,
        report.relative_residual
    );
    Ok(())
}
