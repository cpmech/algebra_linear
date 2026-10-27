use algebra_linear::sparse::amgcl::{BiCgStab, CsrMatrix, Gmres, Identity, SolverParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A small nonsymmetric diagonally dominant system.
    let a = CsrMatrix::from_triplets(
        4,
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
    )?;
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let params = SolverParams {
        tolerance: 1e-12,
        max_iterations: 50,
        ..SolverParams::default()
    };

    let mut x = vec![0.0; 4];
    let report = BiCgStab::new(params).solve(&a, &mut Identity, &b, &mut x)?;
    println!("BiCGStab: {report:?}; x = {x:?}");

    let mut y = vec![0.0; 4];
    let report = Gmres::new(params, 3)?.solve(&a, &mut Identity, &b, &mut y)?;
    println!("GMRES(3): {report:?}; x = {y:?}");
    Ok(())
}
