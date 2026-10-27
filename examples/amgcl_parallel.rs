use algebra_linear::sparse::amgcl::{AmgParams, KrylovMethod, Parallelism, Smoother, SolveReport, SolverParams, amg_setup, krylov_solve};

pub mod common;

fn solve(a: &common::Csr, parallelism: Parallelism) -> Result<SolveReport, algebra_linear::sparse::amgcl::Error> {
    let unknowns = a.rows();
    let mut amg = amg_setup(
        &a.row_ptr,
        &a.col_idx,
        &a.values,
        Some(AmgParams {
            smoother: Smoother::DampedJacobi(0.72),
            parallelism,
            ..AmgParams::default()
        }),
    )?;
    let b = vec![1.0; unknowns];
    let mut x = vec![0.0; unknowns];
    krylov_solve(
        KrylovMethod::Cg,
        &mut amg,
        &b,
        &mut x,
        SolverParams {
            tolerance: 1e-10,
            max_iterations: 100,
            parallelism,
            ..SolverParams::default()
        },
    )
}

fn main() -> Result<(), algebra_linear::sparse::amgcl::Error> {
    let a = common::poisson_2d(64, 64);
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
