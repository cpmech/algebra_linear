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
            row_offsets.push(col_indices.len());
        }
        Self::from_parts(nrows, ncols, row_offsets, col_indices, values)
    }

    /// Returns sorted rows with duplicate columns summed and exact zeros removed.
    pub(crate) fn canonicalized(&self) -> Result<Self> {
        Self::from_triplets(
            self.nrows,
            self.ncols,
            (0..self.nrows).flat_map(|row| {
                (self.row_offsets[row]..self.row_offsets[row + 1]).map(move |index| (row, self.col_indices[index], self.values[index]))
            }),
        )
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
    pub fn diagonal(&self) -> Result<Vec<f64>> {
        if !self.is_square() {
            return Err(Error::InvalidMatrix("diagonal of a rectangular matrix was requested".into()));
        }
        let mut diagonal = vec![0.0; self.nrows];
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
    pub fn transpose(&self) -> Self {
        let mut row_offsets = vec![0; self.ncols + 1];
        for &column in &self.col_indices {
            row_offsets[column + 1] += 1;
        }
        for row in 0..self.ncols {
            row_offsets[row + 1] += row_offsets[row];
        }
        let mut next = row_offsets[..self.ncols].to_vec();
        let mut col_indices = vec![0; self.nnz()];
        let mut values = vec![0.0; self.nnz()];
        for row in 0..self.nrows {
            for index in self.row_offsets[row]..self.row_offsets[row + 1] {
                let output_row = self.col_indices[index];
                let target = next[output_row];
                next[output_row] += 1;
                col_indices[target] = row;
                values[target] = self.values[index];
            }
        }
        Self {
            nrows: self.ncols,
            ncols: self.nrows,
            row_offsets,
            col_indices,
            values,
        }
    }

    /// Sparse matrix product `self * rhs` using serial per-row accumulators.
    // Test-only: the setup uses the explicit-parallelism variant.
    #[cfg(test)]
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        self.matmul_with(rhs, Parallelism::default())
    }

    /// Sparse matrix product with independent row blocks and deterministic output.
    pub fn matmul_with(&self, rhs: &Self, params: Parallelism) -> Result<Self> {
        params.validate()?;
        if self.ncols != rhs.nrows {
            return Err(Error::DimensionMismatch {
                expected: self.ncols,
                actual: rhs.nrows,
                context: "sparse matrix product",
            });
        }
        let products = map_chunks(self.nrows, params, |start, end| {
            let mut rows = Vec::with_capacity(end - start);
            let mut accumulator = vec![0.0; rhs.ncols];
            let mut touched = Vec::new();
            let mut marker = vec![usize::MAX; rhs.ncols];
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
                let mut columns = Vec::with_capacity(touched.len());
                let mut values = Vec::with_capacity(touched.len());
                for &column in &touched {
                    let value = accumulator[column];
                    if value != 0.0 {
                        columns.push(column);
                        values.push(value);
                    }
                }
                rows.push((columns, values));
            }
            rows
        });
        let mut row_offsets = Vec::with_capacity(self.nrows + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        row_offsets.push(0);
        for chunk in products {
            for (columns, chunk_values) in chunk {
                debug_assert_eq!(columns.len(), chunk_values.len());
                col_indices.extend(columns);
                values.extend(chunk_values);
                row_offsets.push(values.len());
            }
        }
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
