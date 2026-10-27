mod aggregation;
mod irregular;
mod matrix;
#[cfg(not(target_arch = "wasm32"))]
mod parallel;
mod relaxation;
mod solvers;
