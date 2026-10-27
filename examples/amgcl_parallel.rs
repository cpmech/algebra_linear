use algebra_linear::sparse::amgcl::{Amg, AmgParams, Cg, CsrMatrix, Parallelism, Smoother, SolveReport, SolverParams};

fn solve(a: &CsrMatrix, parallelism: Parallelism) -> Result<SolveReport, algebra_linear::sparse::amgcl::Error> {
    let mut amg = Amg::new(
        a,
        AmgParams {
            smoother: Smoother::DampedJacobi(0.72),
            parallelism,
            ..AmgParams::default()
        },
    )?;
    let b = vec![1.0; a.nrows()];
    let mut x = vec![0.0; a.nrows()];
    Cg::new(SolverParams {
        tolerance: 1e-10,
        max_iterations: 100,
        parallelism,
        ..SolverParams::default()
    })
    .solve(a, &mut amg, &b, &mut x)
}

fn main() -> Result<(), algebra_linear::sparse::amgcl::Error> {
    let a = CsrMatrix::poisson_2d(64, 64)?;
    let threads = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .map(Parallelism::new)
        .unwrap_or_else(Parallelism::auto);
    let serial = solve(&a, Parallelism::default())?;
    let threaded = solve(&a, threads)?;
    println!("serial:   {serial:?}");
    println!("threaded: {threaded:?} ({:?})", threads);
    Ok(())
}
