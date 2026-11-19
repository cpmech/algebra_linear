//! Owned CSR storage, checked constructors, serial and parallel SpMV, transpose, and SpGEMM —
//! AMGCL's `backend/builtin.hpp`, with the row-accumulator product from `detail/spgemm.hpp`.
//!
//! Parallel kernels preserve each row's arithmetic order and match their serial counterparts, and
//! products and transposes emit sorted rows, so their output is deterministic.

use crate::sparse::amgcl::memory::{
    BudgetGuard, MemoryBudget, csr_bytes, try_clone_slice, try_reserve_vec_filled, try_reserve_vec_with_capacity, try_vec_filled,
    try_vec_with_capacity, vec_bytes,
};
use crate::sparse::amgcl::parallel::{Parallelism, for_each_mut, map_chunks};
use crate::sparse::amgcl::{Error, Result};

/// Compressed sparse row matrix with `f64` values.
///
/// Rows may be unsorted and may contain duplicate columns. Constructors that
/// build matrices from triplets canonicalize both. Most AMGCL-style workflows
/// should use [`CsrMatrix::from_triplets`].
#[derive(Clone, Debug, PartialEq)]
pub struct CsrMatrix {
    nrows: usize,
    ncols: usize,
    row_offsets: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<f64>,
}

/// Borrowed view of one CSR row.
#[derive(Clone, Copy, Debug)]
pub struct CsrRow<'a> {
    /// Column indices in this row.
    pub columns: &'a [usize],
    /// Values aligned with `columns`.
    pub values: &'a [f64],
}

impl CsrMatrix {
    /// Builds a matrix from raw CSR arrays after validating their structure.
    pub fn from_parts(nrows: usize, ncols: usize, row_offsets: Vec<usize>, col_indices: Vec<usize>, values: Vec<f64>) -> Result<Self> {
        if row_offsets.len() != nrows + 1 {
            return Err(Error::InvalidMatrix(format!(
                "row_offsets has length {}, expected {}",
                row_offsets.len(),
                nrows + 1
            )));
        }
        if col_indices.len() != values.len() {
            return Err(Error::InvalidMatrix(format!(
                "column/value lengths differ ({} versus {})",
                col_indices.len(),
                values.len()
            )));
        }
        if row_offsets[0] != 0 || row_offsets[nrows] != values.len() {
            return Err(Error::InvalidMatrix("row offsets must start at zero and end at nnz".into()));
        }
        if row_offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err(Error::InvalidMatrix("row offsets are not monotone".into()));
        }
        if let Some(&column) = col_indices.iter().find(|&&column| column >= ncols) {
            return Err(Error::InvalidMatrix(format!("column index {column} is outside 0..{ncols}")));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidMatrix("matrix contains NaN or infinity".into()));
        }
        Ok(Self {
            nrows,
            ncols,
            row_offsets,
            col_indices,
            values,
        })
    }

    /// Builds a canonical CSR matrix from `(row, column, value)` triplets.
    /// Duplicate entries are summed and exact zeros are removed.
    // Test-only: setup builds canonical CSR matrices directly row-by-row.
    #[cfg(test)]
    pub fn from_triplets<I>(nrows: usize, ncols: usize, triplets: I) -> Result<Self>
    where
        I: IntoIterator<Item = (usize, usize, f64)>,
    {
        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nrows];
        for (row, column, value) in triplets {
            if row >= nrows || column >= ncols {
                return Err(Error::InvalidMatrix(format!(
                    "triplet ({row}, {column}) is outside {nrows} by {ncols} matrix"
                )));
            }
            if !value.is_finite() {
                return Err(Error::InvalidMatrix("triplet contains NaN or infinity".into()));
            }
            rows[row].push((column, value));
        }

        let mut row_offsets = Vec::with_capacity(nrows + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        row_offsets.push(0);
        for row in &mut rows {
            push_canonical_row(row, &mut col_indices, &mut values);
            row_offsets.push(col_indices.len());
        }
        Self::from_parts(nrows, ncols, row_offsets, col_indices, values)
    }

    /// Builds a canonical CSR matrix row-by-row using a single reused scratch buffer.
    ///
    /// Callers must pass upper bounds `nnz_cap` and `row_cap` so all allocations are
    /// checked against `budget` and reserved up front without reallocation.
    pub(crate) fn from_row_producer_with_budget<F>(
        nrows: usize,
        ncols: usize,
        nnz_cap: usize,
        row_cap: usize,
        budget: &MemoryBudget,
        level: Option<usize>,
        mut fill_row: F,
    ) -> Result<Self>
    where
        F: FnMut(usize, &mut Vec<(usize, f64)>) -> Result<()>,
    {
        budget.check_additional(csr_bytes(nrows, nnz_cap).saturating_add(vec_bytes::<(usize, f64)>(row_cap)), level)?;
        let mut row_offsets = try_vec_with_capacity::<usize>(nrows + 1, budget, level)?;
        let mut col_indices = try_vec_with_capacity::<usize>(nnz_cap, budget, level)?;
        let mut values = try_vec_with_capacity::<f64>(nnz_cap, budget, level)?;
        let mut row_buf = try_vec_with_capacity::<(usize, f64)>(row_cap, budget, level)?;
        row_offsets.push(0);

        for row in 0..nrows {
            row_buf.clear();
            fill_row(row, &mut row_buf)?;
            debug_assert!(row_buf.len() <= row_cap);
            push_canonical_row(&mut row_buf, &mut col_indices, &mut values);
            debug_assert!(col_indices.len() <= nnz_cap);
            row_offsets.push(col_indices.len());
        }
        Self::from_parts(nrows, ncols, row_offsets, col_indices, values)
    }

    /// Returns sorted rows with duplicate columns summed and exact zeros removed.
    #[cfg(test)]
    pub(crate) fn canonicalized(&self) -> Result<Self> {
        self.canonicalized_with_budget(&MemoryBudget::new(0), None)
    }

    /// Returns sorted rows with duplicate columns summed and exact zeros removed.
    pub(crate) fn canonicalized_with_budget(&self, budget: &MemoryBudget, level: Option<usize>) -> Result<Self> {
        Self::from_row_producer_with_budget(self.nrows, self.ncols, self.nnz(), self.max_row_nnz(), budget, level, |row, buf| {
            let entries = self.row(row);
            for (&column, &value) in entries.columns.iter().zip(entries.values) {
                buf.push((column, value));
            }
            Ok(())
        })
    }

    /// Estimated heap footprint in bytes for this CSR matrix.
    #[inline]
    pub(crate) fn memory_bytes(&self) -> usize {
        csr_bytes(self.nrows, self.nnz())
    }

    /// Maximum stored entry count across all rows.
    pub(crate) fn max_row_nnz(&self) -> usize {
        (0..self.nrows)
            .map(|row| self.row_offsets[row + 1] - self.row_offsets[row])
            .max()
            .unwrap_or(0)
    }

    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Number of stored values, including any duplicates in raw CSR input.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Whether row and column counts are equal.
    #[inline]
    pub fn is_square(&self) -> bool {
        self.nrows == self.ncols
    }

    /// CSR row-start array of length `nrows + 1`.
    #[inline]
    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    /// Column index array in CSR storage order.
    #[inline]
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Values in CSR storage order.
    #[inline]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Borrows one row.
    ///
    /// # Panics
    /// Panics if `row >= nrows`.
    #[inline]
    pub fn row(&self, row: usize) -> CsrRow<'_> {
        let range = self.row_offsets[row]..self.row_offsets[row + 1];
        CsrRow {
            columns: &self.col_indices[range.clone()],
            values: &self.values[range],
        }
    }

    /// Returns the diagonal, or an error if the matrix is not square.
    #[cfg(test)]
    pub fn diagonal(&self) -> Result<Vec<f64>> {
        self.diagonal_with_budget(&MemoryBudget::new(0), None)
    }

    /// Returns the diagonal while checking `budget`, or an error if the matrix is not square.
    pub(crate) fn diagonal_with_budget(&self, budget: &MemoryBudget, level: Option<usize>) -> Result<Vec<f64>> {
        if !self.is_square() {
            return Err(Error::InvalidMatrix("diagonal of a rectangular matrix was requested".into()));
        }
        let mut diagonal = try_vec_filled(0.0, self.nrows, budget, level)?;
        for (row, diagonal_entry) in diagonal.iter_mut().enumerate() {
            for index in self.row_offsets[row]..self.row_offsets[row + 1] {
                if self.col_indices[index] == row {
                    *diagonal_entry += self.values[index];
                }
            }
        }
        Ok(diagonal)
    }

    pub(crate) fn scale(&mut self, factor: f64) {
        for value in &mut self.values {
            *value *= factor;
        }
    }

    /// Computes `y = A * x` serially.
    // Test-only: the solvers call the unchecked kernels directly.
    #[cfg(test)]
    pub fn spmv(&self, x: &[f64], y: &mut [f64]) -> Result<()> {
        self.spmv_with(x, y, Parallelism::default())
    }

    /// Computes `y = A * x` with the selected shared-memory configuration.
    // Test-only: the solvers call the unchecked kernels directly.
    #[cfg(test)]
    pub fn spmv_with(&self, x: &[f64], y: &mut [f64], params: Parallelism) -> Result<()> {
        params.validate()?;
        self.check_vector(x.len(), self.ncols, "SpMV input")?;
        self.check_vector(y.len(), self.nrows, "SpMV output")?;
        self.spmv_unchecked_with(x, y, params);
        Ok(())
    }

    /// Computes `y += A * x` serially.
    // Test-only: the solvers call the unchecked kernels directly.
    #[cfg(test)]
    pub fn spmv_add(&self, x: &[f64], y: &mut [f64]) -> Result<()> {
        self.spmv_add_with(x, y, Parallelism::default())
    }

    /// Computes `y += A * x` with the selected shared-memory configuration.
    // Test-only: the solvers call the unchecked kernels directly.
    #[cfg(test)]
    pub fn spmv_add_with(&self, x: &[f64], y: &mut [f64], params: Parallelism) -> Result<()> {
        params.validate()?;
        self.check_vector(x.len(), self.ncols, "SpMV input")?;
        self.check_vector(y.len(), self.nrows, "SpMV output")?;
        self.spmv_add_unchecked_with(x, y, params);
        Ok(())
    }

    /// Computes `r = rhs - A * x` serially.
    // Test-only: the solvers call the unchecked kernel directly.
    #[cfg(test)]
    pub fn residual(&self, rhs: &[f64], x: &[f64], r: &mut [f64]) -> Result<()> {
        self.residual_with(rhs, x, r, Parallelism::default())
    }

    /// Computes `r = rhs - A * x` with the selected shared-memory configuration.
    // Test-only: the solvers call the unchecked kernel directly.
    #[cfg(test)]
    pub fn residual_with(&self, rhs: &[f64], x: &[f64], r: &mut [f64], params: Parallelism) -> Result<()> {
        params.validate()?;
        self.check_vector(x.len(), self.ncols, "residual solution")?;
        self.check_vector(rhs.len(), self.nrows, "residual right-hand side")?;
        self.check_vector(r.len(), self.nrows, "residual output")?;
        self.residual_unchecked_with(rhs, x, r, params);
        Ok(())
    }

    pub(crate) fn spmv_unchecked_with(&self, x: &[f64], y: &mut [f64], params: Parallelism) {
        for_each_mut(y, params, |offset, output| {
            for (local, output) in output.iter_mut().enumerate() {
                let row = offset + local;
                let mut sum = 0.0;
                for index in self.row_offsets[row]..self.row_offsets[row + 1] {
                    sum += self.values[index] * x[self.col_indices[index]];
                }
                *output = sum;
            }
        });
    }

    pub(crate) fn spmv_add_unchecked_with(&self, x: &[f64], y: &mut [f64], params: Parallelism) {
        for_each_mut(y, params, |offset, output| {
            for (local, output) in output.iter_mut().enumerate() {
                let row = offset + local;
                let mut sum = 0.0;
                for index in self.row_offsets[row]..self.row_offsets[row + 1] {
                    sum += self.values[index] * x[self.col_indices[index]];
                }
                *output += sum;
            }
        });
    }

    pub(crate) fn residual_unchecked_with(&self, rhs: &[f64], x: &[f64], r: &mut [f64], params: Parallelism) {
        for_each_mut(r, params, |offset, output| {
            for (local, output) in output.iter_mut().enumerate() {
                let row = offset + local;
                let mut sum = 0.0;
                for index in self.row_offsets[row]..self.row_offsets[row + 1] {
                    sum += self.values[index] * x[self.col_indices[index]];
                }
                *output = rhs[row] - sum;
            }
        });
    }

    /// Sparse transpose. Duplicate entries in each output row are combined.
    #[cfg(test)]
    pub fn transpose(&self) -> Self {
        self.transpose_with_budget(&MemoryBudget::new(0), None)
            .expect("unbounded transpose allocation succeeded")
    }

    /// Sparse transpose while checking `budget` and using fallible allocations.
    pub(crate) fn transpose_with_budget(&self, budget: &MemoryBudget, level: Option<usize>) -> Result<Self> {
        let nnz = self.nnz();
        let needed = csr_bytes(self.ncols, nnz).saturating_add(vec_bytes::<usize>(self.ncols));
        budget.check_additional(needed, level)?;
        let mut row_offsets = try_vec_filled(0, self.ncols + 1, budget, level)?;
        for &column in &self.col_indices {
            row_offsets[column + 1] += 1;
        }
        for row in 0..self.ncols {
            row_offsets[row + 1] += row_offsets[row];
        }
        let mut next = try_clone_slice(&row_offsets[..self.ncols], budget, level)?;
        let mut col_indices = try_vec_filled(0, nnz, budget, level)?;
        let mut values = try_vec_filled(0.0, nnz, budget, level)?;
        for row in 0..self.nrows {
            for index in self.row_offsets[row]..self.row_offsets[row + 1] {
                let output_row = self.col_indices[index];
                let target = next[output_row];
                next[output_row] += 1;
                col_indices[target] = row;
                values[target] = self.values[index];
            }
        }
        Ok(Self {
            nrows: self.ncols,
            ncols: self.nrows,
            row_offsets,
            col_indices,
            values,
        })
    }

    /// Sparse matrix product `self * rhs` using serial per-row accumulators.
    // Test-only: the setup uses the explicit-parallelism variant.
    #[cfg(test)]
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        self.matmul_with(rhs, Parallelism::default())
    }

    /// Sparse matrix product with independent row blocks and deterministic output.
    #[cfg(test)]
    pub fn matmul_with(&self, rhs: &Self, params: Parallelism) -> Result<Self> {
        self.matmul_with_budget(rhs, params, &MemoryBudget::new(0), None)
    }

    /// Sparse matrix product with independent row blocks, deterministic output, and memory budgeting.
    pub(crate) fn matmul_with_budget(&self, rhs: &Self, params: Parallelism, budget: &MemoryBudget, level: Option<usize>) -> Result<Self> {
        params.validate()?;
        if self.ncols != rhs.nrows {
            return Err(Error::DimensionMismatch {
                expected: self.ncols,
                actual: rhs.nrows,
                context: "sparse matrix product",
            });
        }
        let workers = params.workers(self.nrows);
        let per_worker_bytes = rhs.ncols.saturating_mul(core::mem::size_of::<f64>() + core::mem::size_of::<usize>());
        budget.check_additional(workers.saturating_mul(per_worker_bytes), level)?;

        let budget_mb = budget.budget_mb();
        let raw_chunks = map_chunks(
            self.nrows,
            params,
            |start, end| -> Result<(Vec<(Vec<usize>, Vec<f64>)>, BudgetGuard<'_>)> {
                let mut chunk_guard = budget.try_guard(vec_bytes::<(Vec<usize>, Vec<f64>)>(end - start), level)?;
                let mut rows = try_reserve_vec_with_capacity(end - start, budget_mb, level)?;
                let _workspace_guard = budget.try_guard(rhs.ncols.saturating_mul(3 * core::mem::size_of::<f64>()), level)?;
                let mut accumulator = try_reserve_vec_filled(0.0, rhs.ncols, budget_mb, level)?;
                let mut marker = try_reserve_vec_filled(usize::MAX, rhs.ncols, budget_mb, level)?;
                let mut touched = try_reserve_vec_with_capacity::<usize>(rhs.ncols, budget_mb, level)?;
                for row in start..end {
                    touched.clear();
                    for left_index in self.row_offsets[row]..self.row_offsets[row + 1] {
                        let inner = self.col_indices[left_index];
                        let left_value = self.values[left_index];
                        for right_index in rhs.row_offsets[inner]..rhs.row_offsets[inner + 1] {
                            let column = rhs.col_indices[right_index];
                            if marker[column] != row {
                                marker[column] = row;
                                accumulator[column] = 0.0;
                                touched.push(column);
                            }
                            accumulator[column] += left_value * rhs.values[right_index];
                        }
                    }
                    touched.sort_unstable();
                    let row_guard = budget.try_guard(
                        touched.len().saturating_mul(core::mem::size_of::<usize>() + core::mem::size_of::<f64>()),
                        level,
                    )?;
                    let mut columns = try_reserve_vec_with_capacity(touched.len(), budget_mb, level)?;
                    let mut values = try_reserve_vec_with_capacity(touched.len(), budget_mb, level)?;
                    for &column in &touched {
                        let value = accumulator[column];
                        if value != 0.0 {
                            columns.push(column);
                            values.push(value);
                        }
                    }
                    rows.push((columns, values));
                    chunk_guard.merge(row_guard);
                }
                Ok((rows, chunk_guard))
            },
        );

        let mut products = try_vec_with_capacity(raw_chunks.len(), budget, level)?;
        let mut guards: Vec<BudgetGuard<'_>> = try_vec_with_capacity(raw_chunks.len(), budget, level)?;
        let mut total_nnz = 0_usize;
        for chunk in raw_chunks {
            let (rows, guard) = chunk?;
            for (columns, _) in &rows {
                total_nnz = total_nnz.saturating_add(columns.len());
            }
            products.push(rows);
            guards.push(guard);
        }

        let mut row_offsets = try_vec_with_capacity::<usize>(self.nrows + 1, budget, level)?;
        let mut col_indices = try_vec_with_capacity::<usize>(total_nnz, budget, level)?;
        let mut values = try_vec_with_capacity::<f64>(total_nnz, budget, level)?;
        row_offsets.push(0);
        for chunk in products {
            for (columns, chunk_values) in chunk {
                debug_assert_eq!(columns.len(), chunk_values.len());
                col_indices.extend(columns);
                values.extend(chunk_values);
                row_offsets.push(values.len());
            }
        }
        drop(guards);
        Self::from_parts(self.nrows, rhs.ncols, row_offsets, col_indices, values)
    }

    /// Row-major dense representation, primarily for tests and small systems.
    // Test-only: a diagnostic view used to check the sparse kernels against dense expectations.
    #[cfg(test)]
    pub fn to_dense(&self) -> Vec<f64> {
        let mut dense = vec![0.0; self.nrows * self.ncols];
        for row in 0..self.nrows {
            for index in self.row_offsets[row]..self.row_offsets[row + 1] {
                dense[row * self.ncols + self.col_indices[index]] += self.values[index];
            }
        }
        dense
    }

    /// Five-point finite-difference Laplacian on an `nx` by `ny` grid.
    /// Boundary conditions are Dirichlet and the grid spacing is omitted.
    // Test-only: callers now assemble their own CSR arrays.
    #[cfg(test)]
    pub fn poisson_2d(nx: usize, ny: usize) -> Result<Self> {
        if nx == 0 || ny == 0 {
            return Err(Error::InvalidParameter("Poisson grid dimensions must be positive".into()));
        }
        let n = nx
            .checked_mul(ny)
            .ok_or_else(|| Error::InvalidParameter("Poisson grid dimensions overflow usize".into()))?;
        let mut triplets = Vec::with_capacity(n * 5);
        for y in 0..ny {
            for x in 0..nx {
                let row = y * nx + x;
                triplets.push((row, row, 4.0));
                if x > 0 {
                    triplets.push((row, row - 1, -1.0));
                }
                if x + 1 < nx {
                    triplets.push((row, row + 1, -1.0));
                }
                if y > 0 {
                    triplets.push((row, row - nx, -1.0));
                }
                if y + 1 < ny {
                    triplets.push((row, row + nx, -1.0));
                }
            }
        }
        Self::from_triplets(n, n, triplets)
    }

    // Test-only: the checked wrappers that call it are test-only.
    #[cfg(test)]
    fn check_vector(&self, actual: usize, expected: usize, context: &'static str) -> Result<()> {
        if actual == expected {
            Ok(())
        } else {
            Err(Error::DimensionMismatch { expected, actual, context })
        }
    }
}

/// Sorts one row's `(column, value)` entries, sums duplicates, and emits nonzero values.
fn push_canonical_row(row: &mut [(usize, f64)], col_indices: &mut Vec<usize>, values: &mut Vec<f64>) {
    row.sort_unstable_by_key(|entry| entry.0);
    let mut i = 0;
    while i < row.len() {
        let column = row[i].0;
        let mut value = 0.0;
        while i < row.len() && row[i].0 == column {
            value += row[i].1;
            i += 1;
        }
        if value != 0.0 {
            col_indices.push(column);
            values.push(value);
        }
    }
}
