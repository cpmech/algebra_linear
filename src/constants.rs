//! Constants shared by the kernels and the reference-parity harness.
//!
//! These mirror the numeric codes the crate's tests and the reference harness
//! pass to LAPACK, plus the boolean spelling of the C interface. Nothing here is
//! part of the public API: a kernel takes `bool` or a character, never a code.
#![allow(dead_code)] // each item is used by a subset of the configurations

/// The integer type of a boolean flag in the C interface.
pub(crate) type CcBool = i32;

/// True, in the C interface's spelling.
pub(crate) const C_TRUE: i32 = 1;
/// False, in the C interface's spelling.
pub(crate) const C_FALSE: i32 = 0;

/// Euclidean norm code.
pub(crate) const NORM_EUC: isize = 0;
/// Frobenius norm code.
pub(crate) const NORM_FRO: isize = 1;
/// Infinity norm code.
pub(crate) const NORM_INF: isize = 2;
/// Maximum absolute entry code.
pub(crate) const NORM_MAX: isize = 3;
/// One norm code.
pub(crate) const NORM_ONE: isize = 4;
