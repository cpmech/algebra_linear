mod aggregation;
mod api;
mod irregular;
mod matrix;
#[cfg(not(target_arch = "wasm32"))]
mod parallel;
mod relaxation;
mod solvers;

use crate::sparse::amgcl::matrix::CsrMatrix;
use crate::sparse::amgcl::{Amg, AmgParams, amg_setup};

/// Builds AMG state from a matrix through the public entry point.
fn setup(a: &CsrMatrix, params: Option<AmgParams>) -> Amg {
    amg_setup(a.row_offsets(), a.col_indices(), a.values(), params).unwrap()
}
