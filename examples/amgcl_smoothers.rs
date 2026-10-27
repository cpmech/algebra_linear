use algebra_linear::sparse::amgcl::{AmgParams, ChebyshevParams, Ilu0Params, KrylovMethod, Smoother, SolverParams, amg_setup, krylov_solve};

pub mod common;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = common::poisson_2d(24, 24);
    let unknowns = a.rows();
    let b = vec![1.0; unknowns];
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
        let mut amg = amg_setup(
            &a.row_ptr,
            &a.col_idx,
            &a.values,
            Some(AmgParams {
                smoother,
                ..AmgParams::default()
            }),
        )?;
        let mut x = vec![0.0; unknowns];
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
        )?;
        println!(
            "{name:20} {:2} iterations, residual {:.3e}, converged: {}",
            report.iterations, report.relative_residual, report.converged
        );
    }
    Ok(())
}
