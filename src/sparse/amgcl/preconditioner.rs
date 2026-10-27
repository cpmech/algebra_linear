use crate::sparse::amgcl::Result;

/// Linear preconditioner interface used by all Krylov solvers.
///
/// Implementations may keep reusable scratch space, hence the mutable receiver.
pub trait Preconditioner {
    /// Computes `x = M⁻¹ rhs` (or another fixed linear approximation).
    fn apply(&mut self, rhs: &[f64], x: &mut [f64]) -> Result<()>;
}

/// Identity preconditioner for unpreconditioned solves.
#[derive(Clone, Copy, Debug, Default)]
pub struct Identity;

impl Preconditioner for Identity {
    fn apply(&mut self, rhs: &[f64], x: &mut [f64]) -> Result<()> {
        x.copy_from_slice(rhs);
        Ok(())
    }
}
