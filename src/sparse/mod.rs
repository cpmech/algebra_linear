//! Sparse direct solvers.
//!
//! [`faer`] is the general sparse LU direct solver: [`faer::SparseLu`] analyzes a
//! compressed sparse column (CSC) pattern once and then factorizes any values on
//! that pattern, the COLAMD ordering reduces fill, [`faer::LuMethod`] chooses a
//! simplicial or supernodal factorization, and [`faer::SparseLuStats`] reports
//! the structure, cost and timing.
//!
//! # Provenance
//!
//! [`faer`] is translated from faer commit `ddbd2ae`
//! (<https://codeberg.org/sarah-quinones/faer>): `faer/src/sparse/linalg/lu.rs`
//! is the simplicial LU (MIT), `faer/src/sparse/linalg/colamd.rs` is the COLAMD
//! ordering (BSD 3-Clause, faer's transcription of SuiteSparse COLAMD), and the
//! column elimination tree and postorder helpers come from
//! `faer/src/sparse/linalg/qr.rs` and `.../cholesky.rs` (MIT). The module's own
//! documentation carries the file-by-file map and the per-file licenses.
//!
//! See `LICENSE` for the notices.
//!
//! # Conventions
//!
//! The crate's conventions hold here as well: column-major dense storage,
//! 0-based `usize` indices, no workspace arguments, and errors instead of
//! `INFO`. Sparse LU callers provide CSC.
//!
//! # Example
//!
//! ```
//! use algebra_linear::sparse::faer::{SparseLu, LuMethod};
//!
//! let col_ptr = [0, 2, 4];
//! let row_idx = [0, 1, 0, 1];
//! let values = [4.0, 1.0, 1.0, 3.0];
//! let mut lu = SparseLu::analyze(&col_ptr, &row_idx, LuMethod::auto()).unwrap();
//! lu.factorize(&col_ptr, &row_idx, &values).unwrap();
//! let mut b = [1.0, 2.0];
//! lu.solve(&mut b).unwrap();
//! assert!((b[0] - 1.0 / 11.0).abs() < 1e-14);
//! assert!((b[1] - 7.0 / 11.0).abs() < 1e-14);
//! ```

pub mod faer;
