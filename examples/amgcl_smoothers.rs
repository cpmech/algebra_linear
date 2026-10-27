use algebra_linear::sparse::amgcl::{Amg, AmgParams, ChebyshevParams, CsrMatrix, Gmres, Ilu0Params, Smoother, SolverParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = CsrMatrix::poisson_2d(24, 24)?;
    let b = vec![1.0; a.nrows()];
    let smoothers = [
        ("Gauss-Seidel", Smoother::GaussSeidel),
        ("Damped Jacobi", Smoother::DampedJacobi(0.72)),
        ("ILU(0)", Smoother::Ilu0(Ilu0Params::default())),
        ("Chebyshev", Smoother::Chebyshev(ChebyshevParams::default())),
        (
            "Scaled Chebyshev",
            Smoother::Chebyshev(ChebyshevParams {
                scale: true,
                power_iterations: 12,
                ..ChebyshevParams::default()
            }),
        ),
        ("SPAI(0)", Smoother::Spai0),
        ("SPAI(1)", Smoother::Spai1),
    ];

    for (name, smoother) in smoothers {
        let mut amg = Amg::new(
            &a,
            AmgParams {
                smoother,
                ..AmgParams::default()
            },
        )?;
        let mut x = vec![0.0; a.nrows()];
        let report = Gmres::new(
            SolverParams {
                tolerance: 1e-9,
                max_iterations: 100,
                ..SolverParams::default()
            },
            30,
        )?
        .solve(&a, &mut amg, &b, &mut x)?;
        println!(
            "{name:20} {:2} iterations, residual {:.3e}, converged: {}",
            report.iterations, report.relative_residual, report.converged
        );
    }
    Ok(())
}
