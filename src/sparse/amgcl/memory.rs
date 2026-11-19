//! Memory accounting, soft memory budgets, and fallible bulk allocations.

use core::cell::Cell;

use crate::sparse::amgcl::{Error, Result};

/// Number of bytes per MiB.
pub(crate) const BYTES_PER_MIB: usize = 1024 * 1024;

/// Converts `bytes` to MiB, rounding up (returning `0` when `bytes == 0`).
pub(crate) fn bytes_to_mib_ceil(bytes: usize) -> usize {
    if bytes == 0 { 0 } else { bytes.div_ceil(BYTES_PER_MIB).max(1) }
}

/// Byte footprint of a `Vec<T>` buffer with `len` elements, using saturating arithmetic.
pub(crate) fn vec_bytes<T>(len: usize) -> usize {
    len.saturating_mul(core::mem::size_of::<T>())
}

/// Byte footprint of a CSR matrix with `nrows` rows and `nnz` stored entries.
pub(crate) fn csr_bytes(nrows: usize, nnz: usize) -> usize {
    let row_ptr_bytes = nrows.saturating_add(1).saturating_mul(core::mem::size_of::<usize>());
    let entry_bytes = core::mem::size_of::<usize>() + core::mem::size_of::<f64>();
    row_ptr_bytes.saturating_add(nnz.saturating_mul(entry_bytes))
}

/// Tracks retained memory and enforces an optional MiB budget.
#[derive(Debug)]
pub(crate) struct MemoryBudget {
    budget_mb: usize,
    budget_bytes: usize,
    retained_bytes: Cell<usize>,
}

impl MemoryBudget {
    /// Creates a budget with limit `budget_mb` MiB (`0` disables the soft limit).
    pub(crate) fn new(budget_mb: usize) -> Self {
        Self {
            budget_mb,
            budget_bytes: budget_mb.saturating_mul(BYTES_PER_MIB),
            retained_bytes: Cell::new(0),
        }
    }

    /// Configured budget in MiB (`0` when unlimited).
    pub(crate) fn budget_mb(&self) -> usize {
        self.budget_mb
    }

    /// Currently retained bytes recorded in this budget.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.retained_bytes.get()
    }

    /// Checks whether adding `additional_bytes` fits within the configured budget.
    pub(crate) fn check_additional(&self, additional_bytes: usize, level: Option<usize>) -> Result<()> {
        if self.budget_mb == 0 {
            return Ok(());
        }
        let total = self.retained_bytes.get().saturating_add(additional_bytes);
        if total > self.budget_bytes {
            return Err(Error::OutOfMemory {
                needed_mb: bytes_to_mib_ceil(total),
                budget_mb: self.budget_mb,
                level,
            });
        }
        Ok(())
    }

    /// Adds `bytes` to the retained footprint.
    pub(crate) fn record_retained(&self, bytes: usize) {
        self.retained_bytes.set(self.retained_bytes.get().saturating_add(bytes));
    }

    /// Subtracts `bytes` from the retained footprint.
    pub(crate) fn release_retained(&self, bytes: usize) {
        self.retained_bytes.set(self.retained_bytes.get().saturating_sub(bytes));
    }

    /// Records `bytes` as retained and returns an RAII guard that releases them on drop.
    pub(crate) fn guard(&self, bytes: usize) -> BudgetGuard<'_> {
        self.record_retained(bytes);
        BudgetGuard { budget: self, bytes }
    }

    /// Builds an [`Error::OutOfMemory`] for a failed allocation of `requested_bytes`.
    pub(crate) fn alloc_error(&self, requested_bytes: usize, level: Option<usize>) -> Error {
        out_of_memory(requested_bytes, self.budget_mb, level)
    }
}

/// RAII guard that tracks transient retained bytes and releases them when dropped.
#[derive(Debug)]
pub(crate) struct BudgetGuard<'a> {
    budget: &'a MemoryBudget,
    bytes: usize,
}

impl BudgetGuard<'_> {
    /// Transfers `other`'s tracked bytes into `self` without changing net retained bytes.
    pub(crate) fn merge(&mut self, mut other: Self) {
        self.bytes = self.bytes.saturating_add(other.bytes);
        other.bytes = 0;
    }
}

impl Drop for BudgetGuard<'_> {
    fn drop(&mut self) {
        self.budget.release_retained(self.bytes);
    }
}

/// Builds an [`Error::OutOfMemory`] from raw byte count and MiB budget.
pub(crate) fn out_of_memory(requested_bytes: usize, budget_mb: usize, level: Option<usize>) -> Error {
    Error::OutOfMemory {
        needed_mb: bytes_to_mib_ceil(requested_bytes),
        budget_mb,
        level,
    }
}

/// Reserves an exact `Vec<T>` capacity without checking `MemoryBudget` (for pre-checked worker threads).
pub(crate) fn try_reserve_vec_with_capacity<T>(cap: usize, budget_mb: usize, level: Option<usize>) -> Result<Vec<T>> {
    let mut vec = Vec::new();
    vec.try_reserve_exact(cap)
        .map_err(|_| out_of_memory(vec_bytes::<T>(cap), budget_mb, level))?;
    Ok(vec)
}

/// Allocates a `Vec<T>` filled with `len` copies of `value` without checking `MemoryBudget`.
pub(crate) fn try_reserve_vec_filled<T: Clone>(value: T, len: usize, budget_mb: usize, level: Option<usize>) -> Result<Vec<T>> {
    let mut vec = try_reserve_vec_with_capacity::<T>(len, budget_mb, level)?;
    vec.resize(len, value);
    Ok(vec)
}

/// Checks `budget` and allocates a `Vec<T>` with exact capacity `cap`.
pub(crate) fn try_vec_with_capacity<T>(cap: usize, budget: &MemoryBudget, level: Option<usize>) -> Result<Vec<T>> {
    budget.check_additional(vec_bytes::<T>(cap), level)?;
    try_reserve_vec_with_capacity(cap, budget.budget_mb(), level)
}

/// Checks `budget` and allocates a `Vec<T>` filled with `len` copies of `value`.
pub(crate) fn try_vec_filled<T: Clone>(value: T, len: usize, budget: &MemoryBudget, level: Option<usize>) -> Result<Vec<T>> {
    let mut vec = try_vec_with_capacity::<T>(len, budget, level)?;
    vec.resize(len, value);
    Ok(vec)
}

/// Checks `budget` and clones `slice` into a newly reserved `Vec<T>`.
pub(crate) fn try_clone_slice<T: Clone>(slice: &[T], budget: &MemoryBudget, level: Option<usize>) -> Result<Vec<T>> {
    let mut vec = try_vec_with_capacity::<T>(slice.len(), budget, level)?;
    vec.extend_from_slice(slice);
    Ok(vec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_to_mib_ceil_rounds_up_and_preserves_zero() {
        assert_eq!(bytes_to_mib_ceil(0), 0);
        assert_eq!(bytes_to_mib_ceil(1), 1);
        assert_eq!(bytes_to_mib_ceil(BYTES_PER_MIB), 1);
        assert_eq!(bytes_to_mib_ceil(BYTES_PER_MIB + 1), 2);
        assert_eq!(bytes_to_mib_ceil(usize::MAX), usize::MAX.div_ceil(BYTES_PER_MIB));
    }

    #[test]
    fn csr_bytes_matches_pointer_and_entry_sizes() {
        let usize_bytes = core::mem::size_of::<usize>();
        let f64_bytes = core::mem::size_of::<f64>();
        assert_eq!(csr_bytes(10, 25), 11 * usize_bytes + 25 * (usize_bytes + f64_bytes));
        assert_eq!(csr_bytes(usize::MAX, usize::MAX), usize::MAX);
    }

    #[test]
    fn memory_budget_tracks_retained_bytes_and_enforces_limit() {
        let unlimited = MemoryBudget::new(0);
        assert_eq!(unlimited.budget_mb(), 0);
        assert!(unlimited.check_additional(100 * BYTES_PER_MIB, Some(0)).is_ok());
        unlimited.record_retained(512);
        assert_eq!(unlimited.retained_bytes(), 512);
        unlimited.release_retained(1024);
        assert_eq!(unlimited.retained_bytes(), 0);

        let budget = MemoryBudget::new(2);
        assert_eq!(budget.budget_mb(), 2);
        budget.record_retained(BYTES_PER_MIB);
        assert!(budget.check_additional(BYTES_PER_MIB, None).is_ok());

        let err = budget.check_additional(BYTES_PER_MIB + 1, Some(1)).unwrap_err();
        assert_eq!(
            err,
            Error::OutOfMemory {
                needed_mb: 3,
                budget_mb: 2,
                level: Some(1),
            }
        );

        {
            let mut guard_a = budget.guard(512);
            let guard_b = budget.guard(256);
            guard_a.merge(guard_b);
            assert_eq!(budget.retained_bytes(), BYTES_PER_MIB + 768);
        }
        assert_eq!(budget.retained_bytes(), BYTES_PER_MIB);
        assert_eq!(
            budget.alloc_error(2 * BYTES_PER_MIB, None),
            Error::OutOfMemory {
                needed_mb: 2,
                budget_mb: 2,
                level: None,
            }
        );
    }

    #[test]
    fn fallible_vec_helpers_check_soft_budget_and_capacity_overflow() {
        let tight = MemoryBudget::new(1);
        let filled = try_vec_filled(3.5_f64, 4, &tight, None).unwrap();
        assert_eq!(filled, vec![3.5; 4]);
        let cloned = try_clone_slice(&filled, &tight, None).unwrap();
        assert_eq!(cloned, filled);

        let soft_err = try_vec_filled(0_u8, BYTES_PER_MIB + 1, &tight, Some(0)).unwrap_err();
        assert_eq!(
            soft_err,
            Error::OutOfMemory {
                needed_mb: 2,
                budget_mb: 1,
                level: Some(0),
            }
        );

        let overflow_cap = usize::MAX / 8;
        let hard_err = try_vec_with_capacity::<f64>(overflow_cap, &MemoryBudget::new(0), Some(2)).unwrap_err();
        assert_eq!(
            hard_err,
            Error::OutOfMemory {
                needed_mb: bytes_to_mib_ceil(vec_bytes::<f64>(overflow_cap)),
                budget_mb: 0,
                level: Some(2),
            }
        );
    }
}
