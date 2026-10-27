use crate::sparse::amgcl::Result;

/// Linear preconditioner interface used by all Krylov solvers.
///
/// Implementations may keep reusable scratch space, hence the mutable receiver.
pub trait Preconditioner {
    /// Computes `x = M⁻¹ rhs` (or another fixed linear approximation).
    fn apply(&mut self, rhs: &[f64], x: &mut [f64]) -> Result<()>;
}

/// Identity preconditioner for unpreconditioned solves.
///
/// Test-only: the public entry point represents the identity preconditioner by
/// the matrix-only [`Amg`](crate::sparse::amgcl::Amg) state, so the library
/// itself never constructs this type.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Identity;

#[cfg(test)]
impl Preconditioner for Identity {
    fn apply(&mut self, rhs: &[f64], x: &mut [f64]) -> Result<()> {
        x.copy_from_slice(rhs);
        Ok(())
    }
}
