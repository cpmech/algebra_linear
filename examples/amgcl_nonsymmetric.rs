use algebra_linear::sparse::amgcl::{KrylovMethod, SolverParams, amg_setup, krylov_solve};

pub mod common;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A small nonsymmetric diagonally dominant system.
    let a = common::from_triplets(
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
    );
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let params = SolverParams {
        tolerance: 1e-12,
        max_iterations: 50,
        ..SolverParams::default()
    };

    // No hierarchy parameters: the state applies as the identity preconditioner.
    let mut amg = amg_setup(&a.row_ptr, &a.col_idx, &a.values, None)?;
    let mut x = vec![0.0; 4];
    let report = krylov_solve(KrylovMethod::BiCgStab, &mut amg, &b, &mut x, params)?;
    println!("BiCGStab: {report:?}; x = {x:?}");

    let mut y = vec![0.0; 4];
    let report = krylov_solve(KrylovMethod::Gmres, &mut amg, &b, &mut y, SolverParams { gmres_restart: 3, ..params })?;
    println!("GMRES(3): {report:?}; x = {y:?}");
    Ok(())
}
