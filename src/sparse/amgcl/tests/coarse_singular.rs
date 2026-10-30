//! A coarsest operator that is singular must not stop the hierarchy
//!
//! Both cases below are what a SuiteSparse matrix produced in practice: dependent rows when the
//! strength graph splits a component, and an isolated row whose column was eliminated.

use super::setup;
use crate::sparse::amgcl::AmgParams;
use crate::sparse::amgcl::matrix::CsrMatrix;

#[test]
fn dependent_rows_in_the_coarsest_operator_are_shifted() {
    // rank one: [[1, 1], [1, 1]] cannot be factorized without a shift
    let a = CsrMatrix::from_triplets(2, 2, vec![(0, 0, 1.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.0)]).unwrap();
    let mut amg = setup(&a, Some(AmgParams::default()));
    let levels: Vec<_> = amg.levels().collect();
    assert_eq!(levels.len(), 1, "two unknowns are coarsest already");
    assert!(levels[0].direct_solver, "the shifted factorization should still be used");

    let rhs = vec![1.0, 1.0];
    let mut x = vec![0.0; 2];
    amg.apply(&rhs, &mut x).unwrap();
    assert!(x.iter().all(|v| v.is_finite()), "{x:?}");
}

#[test]
fn an_isolated_row_in_the_coarsest_operator_is_shifted() {
    // a 1D Poisson operator with one row and column eliminated, which leaves an exact zero row
    let n = 8;
    let mut triplets = Vec::new();
    for i in 0..n {
        if i == 4 {
            continue;
        }
        triplets.push((i, i, 2.0));
        if i > 0 && i - 1 != 4 {
            triplets.push((i, i - 1, -1.0));
        }
        if i + 1 < n && i + 1 != 4 {
            triplets.push((i, i + 1, -1.0));
        }
    }
    let a = CsrMatrix::from_triplets(n, n, triplets).unwrap();
    let mut amg = setup(&a, Some(AmgParams::default()));
    let rhs = vec![1.0; n];
    let mut x = vec![0.0; n];
    amg.apply(&rhs, &mut x).unwrap();
    assert!(x.iter().all(|v| v.is_finite()), "{x:?}");
}
