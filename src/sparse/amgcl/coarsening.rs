//! Classical and aggregation-based AMG coarsening strategies.

mod aggregation;

pub use aggregation::{AggregationParams, PlainAggregationParams, SmoothedAggregationParams};

use crate::sparse::amgcl::matrix::CsrMatrix;
use crate::sparse::amgcl::memory::{BudgetGuard, MemoryBudget, try_clone_slice, try_vec_filled, vec_bytes};
use crate::sparse::amgcl::{Error, Result};
use aggregation::Transfer;

const U: u8 = 0;
const F: u8 = 1;
const C: u8 = 2;
const EPS: f64 = 2.0 * f64::EPSILON;

/// Parameters for strength of connection, C/F splitting, and interpolation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoarseningParams {
    /// Strong negative coupling threshold, normally `0.25`.
    pub eps_strong: f64,
    /// Truncate small interpolation weights and rescale the rest.
    pub truncate: bool,
    /// Interpolation truncation threshold, normally `0.2`.
    pub eps_trunc: f64,
}

impl Default for CoarseningParams {
    fn default() -> Self {
        Self {
            eps_strong: 0.25,
            truncate: true,
            eps_trunc: 0.2,
        }
    }
}

impl CoarseningParams {
    pub(crate) fn validate(&self) -> Result<()> {
        if !(0.0 < self.eps_strong && self.eps_strong < 1.0) {
            return Err(Error::InvalidParameter("eps_strong must lie strictly between zero and one".into()));
        }
        if !(0.0 <= self.eps_trunc && self.eps_trunc < 1.0) {
            return Err(Error::InvalidParameter("eps_trunc must be in [0, 1)".into()));
        }
        Ok(())
    }
}

/// Coarsening strategy used at every nonterminal AMG level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Coarsening {
    /// Classical Ruge–Stüben C/F splitting with direct interpolation.
    RugeStuben(CoarseningParams),
    /// Piecewise-constant tentative prolongation with scaled Galerkin products.
    Aggregation(AggregationParams),
    /// Strongly filtered, damped-Jacobi-smoothed tentative prolongation.
    SmoothedAggregation(SmoothedAggregationParams),
}

impl Default for Coarsening {
    fn default() -> Self {
        Self::RugeStuben(CoarseningParams::default())
    }
}

impl Coarsening {
    pub(crate) fn validate(&self) -> Result<()> {
        match self {
            Self::RugeStuben(params) => params.validate(),
            Self::Aggregation(params) => params.validate(),
            Self::SmoothedAggregation(params) => params.validate(),
        }
    }
}

pub(crate) fn transfer_with_budget(a: &CsrMatrix, params: Coarsening, budget: &MemoryBudget, level: usize) -> Result<Option<Transfer>> {
    match params {
        Coarsening::RugeStuben(params) => prolongation(a, params, budget, level).map(|p| {
            p.map(|prolongation| Transfer {
                prolongation,
                coarse_scale: 1.0,
            })
        }),
        Coarsening::Aggregation(params) => aggregation::aggregation(a, params, budget, level),
        Coarsening::SmoothedAggregation(params) => aggregation::smoothed_aggregation(a, params, budget, level),
    }
}

struct Strength<'a> {
    /// Strong-connection flags indexed exactly like `A.values`.
    aligned: Vec<bool>,
    /// Transposed strong graph in CSR form (without values).
    transpose_offsets: Vec<usize>,
    transpose_columns: Vec<usize>,
    _guard: BudgetGuard<'a>,
}

/// Constructs prolongation `P`; restriction is `P^T`.
pub(crate) fn prolongation(a: &CsrMatrix, params: CoarseningParams, budget: &MemoryBudget, level: usize) -> Result<Option<CsrMatrix>> {
    let n = a.nrows();
    let mut cf = try_vec_filled(U, n, budget, Some(level))?;
    let _cf_guard = budget.guard(vec_bytes::<u8>(n));
    let strength = connect(a, params.eps_strong, &mut cf, budget, level)?;
    cfsplit(a, &strength, &mut cf, budget, level)?;

    let mut coarse_index = try_vec_filled(usize::MAX, n, budget, Some(level))?;
    let _coarse_idx_guard = budget.guard(vec_bytes::<usize>(n));
    let mut coarse_count = 0;
    for (i, &marker) in cf.iter().enumerate() {
        if marker == C {
            coarse_index[i] = coarse_count;
            coarse_count += 1;
        }
    }
    if coarse_count == 0 || coarse_count == n {
        return Ok(None);
    }

    let mut p_nnz_cap = coarse_count;
    let mut row_cap = 1_usize;
    for i in 0..n {
        if cf[i] != C {
            let mut count = 0_usize;
            for index in a.row_offsets()[i]..a.row_offsets()[i + 1] {
                if strength.aligned[index] && cf[a.col_indices()[index]] == C {
                    count += 1;
                }
            }
            p_nnz_cap += count;
            row_cap = row_cap.max(count);
        }
    }

    CsrMatrix::from_row_producer_with_budget(n, coarse_count, p_nnz_cap, row_cap, budget, Some(level), |i, row_buf| {
        if cf[i] == C {
            row_buf.push((coarse_index[i], 1.0));
            return Ok(());
        }

        let row_start = a.row_offsets()[i];
        let row_end = a.row_offsets()[i + 1];
        let mut diagonal = 0.0;
        let mut negative_numerator = 0.0;
        let mut negative_denominator = 0.0;
        let mut positive_numerator = 0.0;
        let mut positive_denominator = 0.0;
        let mut discarded_negative = 0.0;
        let mut discarded_positive = 0.0;
        let mut minimum = 0.0_f64;
        let mut maximum = 0.0_f64;

        if params.truncate {
            for index in row_start..row_end {
                let column = a.col_indices()[index];
                if strength.aligned[index] && cf[column] == C {
                    minimum = minimum.min(a.values()[index]);
                    maximum = maximum.max(a.values()[index]);
                }
            }
            minimum *= params.eps_trunc;
            maximum *= params.eps_trunc;
        }

        for index in row_start..row_end {
            let column = a.col_indices()[index];
            let value = a.values()[index];
            if column == i {
                diagonal += value;
                continue;
            }
            let interpolatory = strength.aligned[index] && cf[column] == C;
            if value < 0.0 {
                negative_numerator += value;
                if interpolatory {
                    negative_denominator += value;
                    if params.truncate && minimum < value {
                        discarded_negative += value;
                    }
                }
            } else {
                positive_numerator += value;
                if interpolatory {
                    positive_denominator += value;
                    if params.truncate && value < maximum {
                        discarded_positive += value;
                    }
                }
            }
        }

        let mut negative_scale = 1.0;
        let mut positive_scale = 1.0;
        if params.truncate {
            let remaining_negative = negative_denominator - discarded_negative;
            let remaining_positive = positive_denominator - discarded_positive;
            if remaining_negative.abs() > EPS {
                negative_scale = negative_denominator.abs() / remaining_negative.abs();
            }
            if remaining_positive.abs() > EPS {
                positive_scale = positive_denominator.abs() / remaining_positive.abs();
            }
        }
        if positive_numerator > 0.0 && positive_denominator.abs() < EPS {
            diagonal += positive_numerator;
        }
        if diagonal.abs() <= EPS && (negative_denominator.abs() > EPS || positive_denominator.abs() > EPS) {
            return Err(Error::SingularMatrix { row: i });
        }
        let alpha = if negative_denominator.abs() > EPS {
            -negative_scale * negative_numerator.abs() / (diagonal.abs() * negative_denominator.abs())
        } else {
            0.0
        };
        let beta = if positive_denominator.abs() > EPS {
            -positive_scale * positive_numerator.abs() / (diagonal.abs() * positive_denominator.abs())
        } else {
            0.0
        };

        for index in row_start..row_end {
            let column = a.col_indices()[index];
            if !strength.aligned[index] || cf[column] != C {
                continue;
            }
            let value = a.values()[index];
            if params.truncate && minimum <= value && value <= maximum {
                continue;
            }
            let weight = if value < 0.0 { alpha } else { beta } * value;
            if weight != 0.0 {
                row_buf.push((coarse_index[column], weight));
            }
        }
        Ok(())
    })
    .map(Some)
}

fn connect<'a>(a: &CsrMatrix, eps_strong: f64, cf: &mut [u8], budget: &'a MemoryBudget, level: usize) -> Result<Strength<'a>> {
    let n = a.nrows();
    let mut aligned = try_vec_filled(false, a.nnz(), budget, Some(level))?;
    let mut guard = budget.guard(vec_bytes::<bool>(a.nnz()));
    for (i, marker) in cf.iter_mut().enumerate() {
        let start = a.row_offsets()[i];
        let end = a.row_offsets()[i + 1];
        let mut minimum = 0.0_f64;
        for index in start..end {
            if a.col_indices()[index] != i {
                minimum = minimum.min(a.values()[index]);
            }
        }
        if minimum.abs() < EPS {
            *marker = F;
            continue;
        }
        let threshold = minimum * eps_strong;
        for (index, strong) in aligned.iter_mut().enumerate().take(end).skip(start) {
            *strong = a.col_indices()[index] != i && a.values()[index] < threshold;
        }
    }

    let mut transpose_offsets = try_vec_filled(0, n + 1, budget, Some(level))?;
    guard.merge(budget.guard(vec_bytes::<usize>(n + 1)));
    for (index, &strong) in aligned.iter().enumerate() {
        if strong {
            transpose_offsets[a.col_indices()[index] + 1] += 1;
        }
    }
    for i in 0..n {
        transpose_offsets[i + 1] += transpose_offsets[i];
    }
    let next_bytes = vec_bytes::<usize>(n);
    let tcol_bytes = vec_bytes::<usize>(transpose_offsets[n]);
    budget.check_additional(next_bytes.saturating_add(tcol_bytes), Some(level))?;
    let mut next = try_clone_slice(&transpose_offsets[..n], budget, Some(level))?;
    let _next_guard = budget.guard(next_bytes);
    let mut transpose_columns = try_vec_filled(0, transpose_offsets[n], budget, Some(level))?;
    guard.merge(budget.guard(tcol_bytes));
    for i in 0..n {
        for index in a.row_offsets()[i]..a.row_offsets()[i + 1] {
            if aligned[index] {
                let target = next[a.col_indices()[index]];
                next[a.col_indices()[index]] += 1;
                transpose_columns[target] = i;
            }
        }
    }
    Ok(Strength {
        aligned,
        transpose_offsets,
        transpose_columns,
        _guard: guard,
    })
}

/// First-pass Ruge–Stüben C/F splitting. This follows AMGCL's bucketed
/// implementation so that strong-neighbor measures are updated incrementally.
fn cfsplit(a: &CsrMatrix, strength: &Strength<'_>, cf: &mut [u8], budget: &MemoryBudget, level: usize) -> Result<()> {
    let n = a.nrows();
    if n == 0 {
        return Ok(());
    }
    let work_bytes = vec_bytes::<usize>(n.saturating_mul(5).saturating_add(1));
    budget.check_additional(work_bytes, Some(level))?;
    let mut lambda = try_vec_filled(0_usize, n, budget, Some(level))?;
    for (i, measure) in lambda.iter_mut().enumerate() {
        for index in strength.transpose_offsets[i]..strength.transpose_offsets[i + 1] {
            *measure += if cf[strength.transpose_columns[index]] == U { 1 } else { 2 };
        }
    }

    let mut group_start = try_vec_filled(0_usize, n + 1, budget, Some(level))?;
    let mut group_count = try_vec_filled(0_usize, n, budget, Some(level))?;
    let mut position_to_node = try_vec_filled(0_usize, n, budget, Some(level))?;
    let mut node_to_position = try_vec_filled(0_usize, n, budget, Some(level))?;
    for &measure in &lambda {
        group_start[measure + 1] += 1;
    }
    for i in 0..n {
        group_start[i + 1] += group_start[i];
    }
    for i in 0..n {
        let measure = lambda[i];
        let position = group_start[measure] + group_count[measure];
        group_count[measure] += 1;
        position_to_node[position] = i;
        node_to_position[i] = position;
    }

    for top in (0..n).rev() {
        let i = position_to_node[top];
        let measure = lambda[i];
        if measure == 0 {
            for marker in cf.iter_mut() {
                if *marker == U {
                    *marker = C;
                }
            }
            break;
        }
        group_count[measure] -= 1;
        if cf[i] == F {
            continue;
        }
        cf[i] = C;

        // Strongly influenced undecided neighbors become F points.
        for index in strength.transpose_offsets[i]..strength.transpose_offsets[i + 1] {
            let fine = strength.transpose_columns[index];
            if cf[fine] != U {
                continue;
            }
            cf[fine] = F;
            for a_index in a.row_offsets()[fine]..a.row_offsets()[fine + 1] {
                if !strength.aligned[a_index] {
                    continue;
                }
                let neighbor = a.col_indices()[a_index];
                let old_measure = lambda[neighbor];
                if cf[neighbor] != U || old_measure + 1 >= n {
                    continue;
                }
                let old_position = node_to_position[neighbor];
                let new_position = group_start[old_measure] + group_count[old_measure] - 1;
                swap_positions(old_position, new_position, &mut position_to_node, &mut node_to_position);
                group_count[old_measure] -= 1;
                group_count[old_measure + 1] += 1;
                group_start[old_measure + 1] = group_start[old_measure] + group_count[old_measure];
                lambda[neighbor] = old_measure + 1;
            }
        }

        // Outgoing strong neighbors of a new C point lose one vote.
        for index in a.row_offsets()[i]..a.row_offsets()[i + 1] {
            if !strength.aligned[index] {
                continue;
            }
            let neighbor = a.col_indices()[index];
            let old_measure = lambda[neighbor];
            if cf[neighbor] != U || old_measure == 0 {
                continue;
            }
            let old_position = node_to_position[neighbor];
            let new_position = group_start[old_measure];
            swap_positions(old_position, new_position, &mut position_to_node, &mut node_to_position);
            group_count[old_measure] -= 1;
            group_count[old_measure - 1] += 1;
            group_start[old_measure] += 1;
            lambda[neighbor] = old_measure - 1;
        }
    }
    Ok(())
}

fn swap_positions(first: usize, second: usize, position_to_node: &mut [usize], node_to_position: &mut [usize]) {
    node_to_position[position_to_node[first]] = second;
    node_to_position[position_to_node[second]] = first;
    position_to_node.swap(first, second);
}
