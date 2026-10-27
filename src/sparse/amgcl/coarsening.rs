//! Classical and aggregation-based AMG coarsening strategies.

mod aggregation;

pub use aggregation::{AggregationParams, PlainAggregationParams, SmoothedAggregationParams};

use crate::sparse::amgcl::matrix::CsrMatrix;
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

pub(crate) fn transfer(a: &CsrMatrix, params: Coarsening) -> Result<Option<Transfer>> {
    match params {
        Coarsening::RugeStuben(params) => prolongation(a, params).map(|p| {
            p.map(|prolongation| Transfer {
                prolongation,
                coarse_scale: 1.0,
            })
        }),
        Coarsening::Aggregation(params) => aggregation::aggregation(a, params),
        Coarsening::SmoothedAggregation(params) => aggregation::smoothed_aggregation(a, params),
    }
}

struct Strength {
    /// Strong-connection flags indexed exactly like `A.values`.
    aligned: Vec<bool>,
    /// Transposed strong graph in CSR form (without values).
    transpose_offsets: Vec<usize>,
    transpose_columns: Vec<usize>,
}

/// Constructs prolongation `P`; restriction is `P^T`.
pub(crate) fn prolongation(a: &CsrMatrix, params: CoarseningParams) -> Result<Option<CsrMatrix>> {
    let n = a.nrows();
    let mut cf = vec![U; n];
    let strength = connect(a, params.eps_strong, &mut cf);
    cfsplit(a, &strength, &mut cf);

    let mut coarse_index = vec![usize::MAX; n];
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

    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for i in 0..n {
        if cf[i] == C {
            rows[i].push((coarse_index[i], 1.0));
            continue;
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
                rows[i].push((coarse_index[column], weight));
            }
        }
    }

    let triplets = rows
        .into_iter()
        .enumerate()
        .flat_map(|(row, entries)| entries.into_iter().map(move |(column, value)| (row, column, value)));
    CsrMatrix::from_triplets(n, coarse_count, triplets).map(Some)
}

fn connect(a: &CsrMatrix, eps_strong: f64, cf: &mut [u8]) -> Strength {
    let n = a.nrows();
    let mut aligned = vec![false; a.nnz()];
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

    let mut transpose_offsets = vec![0; n + 1];
    for (index, &strong) in aligned.iter().enumerate() {
        if strong {
            transpose_offsets[a.col_indices()[index] + 1] += 1;
        }
    }
    for i in 0..n {
        transpose_offsets[i + 1] += transpose_offsets[i];
    }
    let mut next = transpose_offsets[..n].to_vec();
    let mut transpose_columns = vec![0; transpose_offsets[n]];
    for i in 0..n {
        for index in a.row_offsets()[i]..a.row_offsets()[i + 1] {
            if aligned[index] {
                let target = next[a.col_indices()[index]];
                next[a.col_indices()[index]] += 1;
                transpose_columns[target] = i;
            }
        }
    }
    Strength {
        aligned,
        transpose_offsets,
        transpose_columns,
    }
}

/// First-pass Ruge–Stüben C/F splitting. This follows AMGCL's bucketed
/// implementation so that strong-neighbor measures are updated incrementally.
fn cfsplit(a: &CsrMatrix, strength: &Strength, cf: &mut [u8]) {
    let n = a.nrows();
    if n == 0 {
        return;
    }
    let mut lambda = vec![0_usize; n];
    for (i, measure) in lambda.iter_mut().enumerate() {
        for index in strength.transpose_offsets[i]..strength.transpose_offsets[i + 1] {
            *measure += if cf[strength.transpose_columns[index]] == U { 1 } else { 2 };
        }
    }

    let mut group_start = vec![0_usize; n + 1];
    let mut group_count = vec![0_usize; n];
    let mut position_to_node = vec![0_usize; n];
    let mut node_to_position = vec![0_usize; n];
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
}

fn swap_positions(first: usize, second: usize, position_to_node: &mut [usize], node_to_position: &mut [usize]) {
    node_to_position[position_to_node[first]] = second;
    node_to_position[position_to_node[second]] = first;
    position_to_node.swap(first, second);
}
