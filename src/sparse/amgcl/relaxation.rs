//! Serial relaxation schemes and their per-level setup data.

use crate::sparse::amgcl::parallel::{Parallelism, for_each_mut, map_chunks};
use crate::sparse::amgcl::vector::{axpy_with, norm, norm_with};
use crate::sparse::amgcl::{CsrMatrix, Error, Result};

/// Incomplete LU with zero fill-in parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ilu0Params {
    /// Damping applied to each incomplete-factorization correction.
    pub damping: f64,
}

impl Default for Ilu0Params {
    fn default() -> Self {
        Self { damping: 1.0 }
    }
}

/// Chebyshev polynomial smoother parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChebyshevParams {
    /// Polynomial degree (number of residual evaluations per sweep).
    pub degree: usize,
    /// Safety multiplier for the upper spectral bound.
    pub higher: f64,
    /// Lowest-to-highest target eigenvalue ratio.
    pub lower: f64,
    /// Power iterations for spectral estimation; zero uses Gershgorin discs.
    pub power_iterations: usize,
    /// Apply inverse-diagonal scaling before polynomial relaxation.
    pub scale: bool,
}

impl Default for ChebyshevParams {
    fn default() -> Self {
        Self {
            degree: 5,
            higher: 1.0,
            lower: 1.0 / 30.0,
            power_iterations: 0,
            scale: false,
        }
    }
}

/// Relaxation used on each non-coarse level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Smoother {
    /// Forward pre-sweeps and backward post-sweeps of serial Gauss–Seidel.
    GaussSeidel,
    /// Damped Jacobi with the supplied damping factor (AMGCL default: `0.72`).
    DampedJacobi(f64),
    /// Incomplete LU with zero fill-in.
    Ilu0(Ilu0Params),
    /// Chebyshev polynomial relaxation.
    Chebyshev(ChebyshevParams),
    /// Diagonal sparse approximate inverse.
    Spai0,
    /// Sparse approximate inverse with the sparsity pattern of the system matrix.
    Spai1,
}

impl Smoother {
    pub(crate) fn validate(self) -> Result<()> {
        match self {
            Self::DampedJacobi(damping) => {
                if !damping.is_finite() || damping <= 0.0 {
                    return Err(Error::InvalidParameter("Jacobi damping must be finite and positive".into()));
                }
            }
            Self::Ilu0(params) => {
                if !params.damping.is_finite() || params.damping <= 0.0 {
                    return Err(Error::InvalidParameter("ILU(0) damping must be finite and positive".into()));
                }
            }
            Self::Chebyshev(params) => {
                if params.degree == 0 {
                    return Err(Error::InvalidParameter("Chebyshev degree must be positive".into()));
                }
                if !params.higher.is_finite() || !params.lower.is_finite() || params.lower <= 0.0 || params.higher < params.lower {
                    return Err(Error::InvalidParameter("Chebyshev bounds require finite higher >= lower > 0".into()));
                }
            }
            Self::GaussSeidel | Self::Spai0 | Self::Spai1 => {}
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) enum Relaxation {
    GaussSeidel { inverse_diagonal: Vec<f64> },
    DampedJacobi { damping: f64, inverse_diagonal: Vec<f64> },
    Ilu0(IncompleteLu),
    Chebyshev(Chebyshev),
    Spai0(Vec<f64>),
    Spai1(CsrMatrix),
}

impl Relaxation {
    #[cfg(test)]
    pub(crate) fn new(a: &CsrMatrix, smoother: Smoother) -> Result<Self> {
        Self::new_with(a, smoother, Parallelism::default())
    }

    pub(crate) fn new_with(a: &CsrMatrix, smoother: Smoother, parallel: Parallelism) -> Result<Self> {
        match smoother {
            Smoother::GaussSeidel => Ok(Self::GaussSeidel {
                inverse_diagonal: inverse_diagonal(a)?,
            }),
            Smoother::DampedJacobi(damping) => Ok(Self::DampedJacobi {
                damping,
                inverse_diagonal: inverse_diagonal(a)?,
            }),
            Smoother::Ilu0(params) => Ok(Self::Ilu0(IncompleteLu::new(a, params)?)),
            Smoother::Chebyshev(params) => Ok(Self::Chebyshev(Chebyshev::new_with(a, params, parallel)?)),
            Smoother::Spai0 => Ok(Self::Spai0(spai0_with(a, parallel)?)),
            Smoother::Spai1 => Ok(Self::Spai1(spai1_with(a, parallel)?)),
        }
    }

    #[cfg(test)]
    pub(crate) fn apply(&mut self, a: &CsrMatrix, rhs: &[f64], x: &mut [f64], residual: &mut [f64], forward: bool) {
        self.apply_with(a, rhs, x, residual, forward, Parallelism::default());
    }

    pub(crate) fn apply_with(&mut self, a: &CsrMatrix, rhs: &[f64], x: &mut [f64], residual: &mut [f64], forward: bool, parallel: Parallelism) {
        match self {
            Self::GaussSeidel { inverse_diagonal } => {
                if forward {
                    for i in 0..x.len() {
                        gauss_seidel_row(a, inverse_diagonal, i, rhs, x);
                    }
                } else {
                    for i in (0..x.len()).rev() {
                        gauss_seidel_row(a, inverse_diagonal, i, rhs, x);
                    }
                }
            }
            Self::DampedJacobi { damping, inverse_diagonal } => {
                a.residual_unchecked_with(rhs, x, residual, parallel);
                for_each_mut(x, parallel, |offset, chunk| {
                    for (local, value) in chunk.iter_mut().enumerate() {
                        let i = offset + local;
                        *value += *damping * inverse_diagonal[i] * residual[i];
                    }
                });
            }
            Self::Ilu0(ilu) => {
                a.residual_unchecked_with(rhs, x, residual, parallel);
                ilu.solve(residual);
                for_each_mut(x, parallel, |offset, chunk| {
                    for (local, value) in chunk.iter_mut().enumerate() {
                        *value += ilu.damping * residual[offset + local];
                    }
                });
            }
            Self::Chebyshev(chebyshev) => chebyshev.apply_with(a, rhs, x, residual, parallel),
            Self::Spai0(inverse) => {
                a.residual_unchecked_with(rhs, x, residual, parallel);
                for_each_mut(x, parallel, |offset, chunk| {
                    for (local, value) in chunk.iter_mut().enumerate() {
                        let i = offset + local;
                        *value += inverse[i] * residual[i];
                    }
                });
            }
            Self::Spai1(inverse) => {
                a.residual_unchecked_with(rhs, x, residual, parallel);
                inverse.spmv_add_unchecked_with(residual, x, parallel);
            }
        }
    }
}

pub(crate) fn inverse_diagonal(a: &CsrMatrix) -> Result<Vec<f64>> {
    let diagonal = a.diagonal()?;
    let mut inverse = Vec::with_capacity(diagonal.len());
    for (row, value) in diagonal.into_iter().enumerate() {
        if value == 0.0 || !value.is_finite() {
            return Err(Error::SingularMatrix { row });
        }
        inverse.push(1.0 / value);
    }
    Ok(inverse)
}

fn gauss_seidel_row(a: &CsrMatrix, inverse_diagonal: &[f64], i: usize, rhs: &[f64], x: &mut [f64]) {
    let row = a.row(i);
    let mut value = rhs[i];
    for (&column, &entry) in row.columns.iter().zip(row.values) {
        if column != i {
            value -= entry * x[column];
        }
    }
    x[i] = value * inverse_diagonal[i];
}

#[derive(Debug)]
pub(crate) struct IncompleteLu {
    lower: CsrMatrix,
    upper: CsrMatrix,
    inverse_diagonal: Vec<f64>,
    damping: f64,
}

impl IncompleteLu {
    fn new(a: &CsrMatrix, params: Ilu0Params) -> Result<Self> {
        let a = a.canonicalized()?;
        let n = a.nrows();
        let mut lower_offsets = vec![0; n + 1];
        let mut upper_offsets = vec![0; n + 1];
        let mut lower_columns = Vec::with_capacity(a.nnz());
        let mut upper_columns = Vec::with_capacity(a.nnz());
        let mut lower_values = Vec::with_capacity(a.nnz());
        let mut upper_values = Vec::with_capacity(a.nnz());
        let mut inverse_diagonal = vec![0.0; n];
        let mut work = vec![0.0; n];
        let mut position = vec![usize::MAX; n];

        for i in 0..n {
            let row_start = a.row_offsets()[i];
            let row_end = a.row_offsets()[i + 1];
            for index in row_start..row_end {
                let column = a.col_indices()[index];
                let value = a.values()[index];
                work[column] = value;
                if column < i {
                    position[column] = lower_values.len();
                    lower_columns.push(column);
                    lower_values.push(value);
                } else if column == i {
                    position[column] = usize::MAX - 1;
                } else if column > i {
                    position[column] = upper_values.len();
                    upper_columns.push(column);
                    upper_values.push(value);
                }
            }
            lower_offsets[i + 1] = lower_values.len();
            upper_offsets[i + 1] = upper_values.len();

            let mut has_diagonal = false;
            for index in row_start..row_end {
                let column = a.col_indices()[index];
                if column == i {
                    has_diagonal = true;
                    break;
                }
                let multiplier = work[column] * inverse_diagonal[column];
                work[column] = multiplier;
                for upper_index in upper_offsets[column]..upper_offsets[column + 1] {
                    let target = upper_columns[upper_index];
                    if position[target] != usize::MAX {
                        work[target] -= multiplier * upper_values[upper_index];
                    }
                }
            }
            if !has_diagonal {
                return Err(Error::InvalidMatrix(format!("ILU(0) requires a diagonal entry in row {i}")));
            }
            let pivot = work[i];
            if pivot == 0.0 || !pivot.is_finite() {
                return Err(Error::SingularMatrix { row: i });
            }
            inverse_diagonal[i] = 1.0 / pivot;

            for index in row_start..row_end {
                let column = a.col_indices()[index];
                if column < i {
                    lower_values[position[column]] = work[column];
                } else if column > i {
                    upper_values[position[column]] = work[column];
                }
                position[column] = usize::MAX;
                work[column] = 0.0;
            }
        }

        Ok(Self {
            lower: CsrMatrix::from_parts(n, n, lower_offsets, lower_columns, lower_values)?,
            upper: CsrMatrix::from_parts(n, n, upper_offsets, upper_columns, upper_values)?,
            inverse_diagonal,
            damping: params.damping,
        })
    }

    fn solve(&self, vector: &mut [f64]) {
        for i in 0..vector.len() {
            let mut value = vector[i];
            let row = self.lower.row(i);
            for (&column, &entry) in row.columns.iter().zip(row.values) {
                value -= entry * vector[column];
            }
            vector[i] = value;
        }
        for i in (0..vector.len()).rev() {
            let mut value = vector[i];
            let row = self.upper.row(i);
            for (&column, &entry) in row.columns.iter().zip(row.values) {
                value -= entry * vector[column];
            }
            vector[i] = value * self.inverse_diagonal[i];
        }
    }
}

#[derive(Debug)]
pub(crate) struct Chebyshev {
    degree: usize,
    scale: bool,
    inverse_diagonal: Vec<f64>,
    center: f64,
    semi_axis: f64,
    polynomial: Vec<f64>,
}

impl Chebyshev {
    fn new_with(a: &CsrMatrix, params: ChebyshevParams, parallel: Parallelism) -> Result<Self> {
        let inverse_diagonal = if params.scale { inverse_diagonal(a)? } else { Vec::new() };
        let radius = spectral_radius_with(a, params.power_iterations, &inverse_diagonal, parallel)?;
        if radius <= 0.0 || !radius.is_finite() {
            return Err(Error::InvalidMatrix(
                "Chebyshev smoother requires a positive finite spectral radius".into(),
            ));
        }
        let upper = radius * params.higher;
        let lower = radius * params.lower;
        let center = 0.5 * (upper + lower);
        let semi_axis = 0.5 * (upper - lower);
        if center == 0.0 || !center.is_finite() || !semi_axis.is_finite() {
            return Err(Error::InvalidParameter("Chebyshev spectral interval is degenerate".into()));
        }
        Ok(Self {
            degree: params.degree,
            scale: params.scale,
            inverse_diagonal,
            center,
            semi_axis,
            polynomial: vec![0.0; a.nrows()],
        })
    }

    fn apply_with(&mut self, a: &CsrMatrix, rhs: &[f64], x: &mut [f64], residual: &mut [f64], parallel: Parallelism) {
        let mut alpha = 0.0;
        for k in 0..self.degree {
            a.residual_unchecked_with(rhs, x, residual, parallel);
            if self.scale {
                for_each_mut(residual, parallel, |offset, chunk| {
                    for (local, value) in chunk.iter_mut().enumerate() {
                        *value *= self.inverse_diagonal[offset + local];
                    }
                });
            }
            if k == 0 {
                alpha = 1.0 / self.center;
            } else if k == 1 {
                alpha = 2.0 * self.center / (2.0 * self.center * self.center - self.semi_axis * self.semi_axis);
            } else {
                alpha = 1.0 / (self.center - 0.25 * alpha * self.semi_axis * self.semi_axis);
            }
            let beta = if k == 0 { 0.0 } else { alpha * self.center - 1.0 };
            for_each_mut(&mut self.polynomial, parallel, |offset, chunk| {
                for (local, value) in chunk.iter_mut().enumerate() {
                    *value = alpha * residual[offset + local] + beta * *value;
                }
            });
            axpy_with(1.0, &self.polynomial, x, parallel);
        }
    }
}

pub(crate) fn spectral_radius(a: &CsrMatrix, power_iterations: usize, inverse_diagonal: &[f64]) -> Result<f64> {
    spectral_radius_with(a, power_iterations, inverse_diagonal, Parallelism::default())
}

pub(crate) fn spectral_radius_with(a: &CsrMatrix, power_iterations: usize, inverse_diagonal: &[f64], parallel: Parallelism) -> Result<f64> {
    if power_iterations == 0 {
        return Ok(gershgorin_radius(a, inverse_diagonal));
    }
    let n = a.nrows();
    let mut state = 0x4d59_5df4_d0f3_3173_u64;
    let mut current = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let uniform = ((state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64));
        current.push(2.0 * uniform - 1.0);
    }
    let initial_norm = norm_with(&current, parallel);
    if initial_norm == 0.0 {
        return Ok(gershgorin_radius(a, inverse_diagonal));
    }
    for_each_mut(&mut current, parallel, |_, chunk| {
        for value in chunk {
            *value /= initial_norm;
        }
    });
    let mut next = vec![0.0; n];
    let mut radius = 0.0;
    for iteration in 0..power_iterations {
        a.spmv_unchecked_with(&current, &mut next, parallel);
        if !inverse_diagonal.is_empty() {
            for_each_mut(&mut next, parallel, |offset, chunk| {
                for (local, value) in chunk.iter_mut().enumerate() {
                    *value *= inverse_diagonal[offset + local];
                }
            });
        }
        radius = map_chunks(n, parallel, |start, end| {
            next[start..end]
                .iter()
                .zip(&current[start..end])
                .map(|(next, current)| (next * current).abs())
                .sum::<f64>()
        })
        .into_iter()
        .sum();
        if iteration + 1 < power_iterations {
            let next_norm = norm_with(&next, parallel);
            if next_norm == 0.0 || !next_norm.is_finite() {
                return Ok(gershgorin_radius(a, inverse_diagonal));
            }
            for_each_mut(&mut current, parallel, |offset, chunk| {
                for (local, value) in chunk.iter_mut().enumerate() {
                    *value = next[offset + local] / next_norm;
                }
            });
        }
    }
    if radius > 0.0 && radius.is_finite() {
        Ok(radius)
    } else {
        Ok(gershgorin_radius(a, inverse_diagonal))
    }
}

fn gershgorin_radius(a: &CsrMatrix, inverse_diagonal: &[f64]) -> f64 {
    let mut radius = 0.0_f64;
    for i in 0..a.nrows() {
        let mut sum = 0.0;
        for index in a.row_offsets()[i]..a.row_offsets()[i + 1] {
            sum += a.values()[index].abs();
        }
        if !inverse_diagonal.is_empty() {
            sum *= inverse_diagonal[i].abs();
        }
        radius = radius.max(sum);
    }
    radius
}

fn spai0_with(a: &CsrMatrix, parallel: Parallelism) -> Result<Vec<f64>> {
    let chunks = map_chunks(a.nrows(), parallel, |start, end| {
        let mut inverse = Vec::with_capacity(end - start);
        for i in start..end {
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for index in a.row_offsets()[i]..a.row_offsets()[i + 1] {
                let value = a.values()[index];
                denominator += value * value;
                if a.col_indices()[index] == i {
                    numerator += value;
                }
            }
            if denominator == 0.0 || !denominator.is_finite() {
                return Err(Error::SingularMatrix { row: i });
            }
            inverse.push(numerator / denominator);
        }
        Ok(inverse)
    });
    let mut inverse = Vec::with_capacity(a.nrows());
    for chunk in chunks {
        inverse.extend(chunk?);
    }
    Ok(inverse)
}

fn spai1_with(a: &CsrMatrix, parallel: Parallelism) -> Result<CsrMatrix> {
    let a = a.canonicalized()?;
    let n = a.nrows();
    let chunks = map_chunks(n, parallel, |start, end| -> Result<Vec<f64>> {
        let mut inverse_values = Vec::with_capacity(a.row_offsets()[end] - a.row_offsets()[start]);
        let mut marker = vec![usize::MAX; n];
        let mut neighborhood = Vec::new();

        for i in start..end {
            let row_start = a.row_offsets()[i];
            let row_end = a.row_offsets()[i + 1];
            let columns = &a.col_indices()[row_start..row_end];
            if columns.is_empty() {
                return Err(Error::SingularMatrix { row: i });
            }
            neighborhood.clear();
            for &column in columns {
                for index in a.row_offsets()[column]..a.row_offsets()[column + 1] {
                    let reachable = a.col_indices()[index];
                    if marker[reachable] == usize::MAX {
                        marker[reachable] = 0;
                        neighborhood.push(reachable);
                    }
                }
            }
            neighborhood.sort_unstable();
            let rows = neighborhood.len();
            let cols = columns.len();
            for (row, &original) in neighborhood.iter().enumerate() {
                marker[original] = row;
            }
            let mut local = vec![0.0; rows * cols];
            let mut rhs = vec![0.0; rows];
            for (row, &original) in neighborhood.iter().enumerate() {
                if original == i {
                    rhs[row] = 1.0;
                }
            }
            for (local_column, &source_row) in columns.iter().enumerate() {
                for index in a.row_offsets()[source_row]..a.row_offsets()[source_row + 1] {
                    let local_row = marker[a.col_indices()[index]];
                    local[local_row * cols + local_column] = a.values()[index];
                }
            }
            let solution = least_squares(&local, rows, cols, &rhs)
                .map_err(|_| Error::InvalidMatrix(format!("SPAI(1) neighborhood for row {i} is rank deficient")))?;
            inverse_values.extend(solution);
            for &original in &neighborhood {
                marker[original] = usize::MAX;
            }
        }
        Ok(inverse_values)
    });
    let mut inverse_values = Vec::with_capacity(a.nnz());
    for chunk in chunks {
        inverse_values.extend(chunk?);
    }

    CsrMatrix::from_parts(n, n, a.row_offsets().to_vec(), a.col_indices().to_vec(), inverse_values)
}

/// Full-rank least squares. Tall systems use B=QR; wide systems use Bᵀ=QR
/// and return the minimum-norm solution.
fn least_squares(matrix: &[f64], rows: usize, cols: usize, rhs: &[f64]) -> Result<Vec<f64>> {
    if rows >= cols {
        let mut factor = matrix.to_vec();
        let reflectors = householder_qr(&mut factor, rows, cols)?;
        let mut transformed = rhs.to_vec();
        for (k, reflector) in reflectors.iter().enumerate() {
            apply_reflector(k, reflector, &mut transformed);
        }
        let mut solution = vec![0.0; cols];
        for i in (0..cols).rev() {
            let mut value = transformed[i];
            for j in i + 1..cols {
                value -= factor[i * cols + j] * solution[j];
            }
            let diagonal = factor[i * cols + i];
            if diagonal == 0.0 || !diagonal.is_finite() {
                return Err(Error::SolverBreakdown("rank-deficient QR system"));
            }
            solution[i] = value / diagonal;
        }
        Ok(solution)
    } else {
        // Transpose into an cols-by-rows tall matrix.
        let mut factor = vec![0.0; matrix.len()];
        for row in 0..rows {
            for col in 0..cols {
                factor[col * rows + row] = matrix[row * cols + col];
            }
        }
        let reflectors = householder_qr(&mut factor, cols, rows)?;
        let mut transformed = vec![0.0; cols];
        transformed[..rows].copy_from_slice(rhs);
        for i in 0..rows {
            let diagonal = factor[i * rows + i];
            if diagonal == 0.0 || !diagonal.is_finite() {
                return Err(Error::SolverBreakdown("rank-deficient QR system"));
            }
            transformed[i] /= diagonal;
            for j in i + 1..rows {
                transformed[j] -= factor[i * rows + j] * transformed[i];
            }
        }
        for (k, reflector) in reflectors.iter().enumerate().rev() {
            apply_reflector(k, reflector, &mut transformed);
        }
        Ok(transformed)
    }
}

/// Returns reflectors with implicit leading one; R remains in `matrix`.
fn householder_qr(matrix: &mut [f64], rows: usize, cols: usize) -> Result<Vec<Vec<f64>>> {
    let mut reflectors = Vec::with_capacity(rows.min(cols));
    for k in 0..rows.min(cols) {
        let length = rows - k;
        let mut reflector = vec![0.0; length];
        for (offset, value) in reflector.iter_mut().enumerate() {
            *value = matrix[(k + offset) * cols + k];
        }
        let column_norm = norm(&reflector);
        if column_norm == 0.0 || !column_norm.is_finite() {
            return Err(Error::SolverBreakdown("rank-deficient QR column"));
        }
        let alpha = if reflector[0] >= 0.0 { -column_norm } else { column_norm };
        reflector[0] -= alpha;
        let reflector_norm = reflector.iter().map(|value| value * value).sum::<f64>();
        if reflector_norm == 0.0 || !reflector_norm.is_finite() {
            return Err(Error::SolverBreakdown("degenerate Householder reflector"));
        }
        matrix[k * cols + k] = alpha;
        for row in k + 1..rows {
            matrix[row * cols + k] = 0.0;
        }
        for column in k + 1..cols {
            let mut projection = 0.0;
            for (offset, &v) in reflector.iter().enumerate() {
                projection += v * matrix[(k + offset) * cols + column];
            }
            projection *= 2.0 / reflector_norm;
            for (offset, &v) in reflector.iter().enumerate() {
                matrix[(k + offset) * cols + column] -= projection * v;
            }
        }
        reflectors.push(reflector);
    }
    Ok(reflectors)
}

fn apply_reflector(k: usize, reflector: &[f64], vector: &mut [f64]) {
    let reflector_norm = reflector.iter().map(|value| value * value).sum::<f64>();
    let mut projection = 0.0;
    for (offset, &v) in reflector.iter().enumerate() {
        projection += v * vector[k + offset];
    }
    projection *= 2.0 / reflector_norm;
    for (offset, &v) in reflector.iter().enumerate() {
        vector[k + offset] -= projection * v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tridiagonal(n: usize) -> CsrMatrix {
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push((i, i, 2.0));
            if i > 0 {
                triplets.push((i, i - 1, -1.0));
            }
            if i + 1 < n {
                triplets.push((i, i + 1, -1.0));
            }
        }
        CsrMatrix::from_triplets(n, n, triplets).unwrap()
    }

    #[test]
    fn ilu0_is_exact_for_tridiagonal_matrix() {
        let a = tridiagonal(6);
        let rhs: Vec<f64> = (0..6).map(|i| i as f64 + 1.0).collect();
        let mut x = vec![0.0; 6];
        let mut residual = vec![0.0; 6];
        Relaxation::new(&a, Smoother::Ilu0(Ilu0Params::default()))
            .unwrap()
            .apply(&a, &rhs, &mut x, &mut residual, true);
        a.residual(&rhs, &x, &mut residual).unwrap();
        assert!(norm(&residual) < 1e-13, "{residual:?}");
    }

    #[test]
    fn spai0_uses_least_squares_diagonal() {
        let a = CsrMatrix::from_triplets(1, 1, [(0, 0, 2.0)]).unwrap();
        let Relaxation::Spai0(inverse) = Relaxation::new(&a, Smoother::Spai0).unwrap() else {
            panic!("wrong relaxation variant");
        };
        assert_eq!(inverse, vec![0.5]);
    }

    #[test]
    fn spai1_is_exact_for_small_dense_matrix() {
        let a = CsrMatrix::from_triplets(
            3,
            3,
            [
                (0, 0, 4.0),
                (0, 1, 1.0),
                (0, 2, 0.5),
                (1, 0, 1.0),
                (1, 1, 3.0),
                (1, 2, 1.0),
                (2, 0, 0.5),
                (2, 1, 1.0),
                (2, 2, 2.0),
            ],
        )
        .unwrap();
        let Relaxation::Spai1(inverse) = Relaxation::new(&a, Smoother::Spai1).unwrap() else {
            panic!("wrong relaxation variant");
        };
        let product = a.matmul(&inverse).unwrap();
        let dense = product.to_dense();
        for row in 0..3 {
            for col in 0..3 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!((dense[row * 3 + col] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn householder_least_squares_handles_tall_and_wide_systems() {
        let tall = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let x = least_squares(&tall, 3, 2, &[1.0, 2.0, 3.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-13);
        assert!((x[1] - 2.0).abs() < 1e-13);

        let wide = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let x = least_squares(&wide, 2, 3, &[1.0, 2.0]).unwrap();
        let expected = [0.0, 1.0, 1.0];
        for i in 0..3 {
            assert!((x[i] - expected[i]).abs() < 1e-13, "{x:?}");
        }
    }

    #[test]
    fn chebyshev_gershgorin_radius_matches_diagonal_matrix() {
        let a = CsrMatrix::from_triplets(2, 2, [(0, 0, 2.0), (1, 1, 5.0)]).unwrap();
        assert_eq!(spectral_radius(&a, 0, &[]).unwrap(), 5.0);
        Relaxation::new(&a, Smoother::Chebyshev(ChebyshevParams::default())).unwrap();
        Relaxation::new(
            &a,
            Smoother::Chebyshev(ChebyshevParams {
                scale: true,
                power_iterations: 8,
                ..ChebyshevParams::default()
            }),
        )
        .unwrap();
    }
}
