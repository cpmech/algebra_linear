use algebra_linear::sparse::amgcl::{Amg, AmgParams, Cg, CsrMatrix, SolverParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = std::env::args().nth(1).map(|value| value.parse::<usize>()).transpose()?.unwrap_or(32);
    let a = CsrMatrix::poisson_2d(n, n)?;
    let b = vec![1.0; a.nrows()];
    let mut x = vec![0.0; a.nrows()];

    let mut amg = Amg::new(&a, AmgParams::default())?;
    println!("AMG hierarchy ({} unknowns):", a.nrows());
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

    let solver = Cg::new(SolverParams {
        tolerance: 1e-10,
        max_iterations: 100,
        ..SolverParams::default()
    });
    let report = solver.solve(&a, &mut amg, &b, &mut x)?;
    println!(
        "{} after {} iterations; relative residual = {:.3e}",
        if report.converged { "converged" } else { "not converged" },
        report.iterations,
        report.relative_residual
    );
    Ok(())
}
