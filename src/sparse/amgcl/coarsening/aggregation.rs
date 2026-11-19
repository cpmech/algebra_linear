//! Greedy aggregation and aggregation-based transfer operators.
//!
//! The aggregates come from AMGCL's `plain_aggregates.hpp` (symmetric strength graph, greedy
//! construction), and the piecewise-constant tentative prolongation from
//! `tentative_prolongation.hpp`. Non-smoothed transfers divide the coarse Galerkin operator by
//! `over_interp`; smoothed ones filter weak off-diagonal entries, adjust the filtered diagonal, and
//! interpolate with damped Jacobi.

use crate::sparse::amgcl::matrix::CsrMatrix;
use crate::sparse::amgcl::memory::{BudgetGuard, MemoryBudget, try_vec_filled, try_vec_with_capacity, vec_bytes};
use crate::sparse::amgcl::parallel::Parallelism;
use crate::sparse::amgcl::relaxation::{inverse_diagonal_with_budget, spectral_radius_with_budget};
use crate::sparse::amgcl::{Error, Result};

const UNDEFINED: isize = -1;
const REMOVED: isize = -2;

/// Parameters for greedy plain aggregation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlainAggregationParams {
    /// Symmetric strong-connection threshold, normally `0.08`.
    ///
    /// Off-diagonal entries satisfy `a_ij² > eps_strong² a_ii a_jj`.
    pub eps_strong: f64,
}

impl Default for PlainAggregationParams {
    fn default() -> Self {
        Self { eps_strong: 0.08 }
    }
}

impl PlainAggregationParams {
    fn validate(&self) -> Result<()> {
        if !(0.0 < self.eps_strong && self.eps_strong < 1.0) {
            return Err(Error::InvalidParameter(
                "aggregation eps_strong must lie strictly between zero and one".into(),
            ));
        }
        Ok(())
    }
}

/// Parameters for non-smoothed aggregation coarsening.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AggregationParams {
    /// Aggregate construction parameters.
    pub aggregation: PlainAggregationParams,
    /// Over-interpolation factor; the Galerkin operator is divided by this value.
    pub over_interp: f64,
}

impl AggregationParams {
    pub(crate) fn validate(&self) -> Result<()> {
        self.aggregation.validate()?;
        if !self.over_interp.is_finite() || self.over_interp <= 0.0 {
            return Err(Error::InvalidParameter("over_interp must be finite and positive".into()));
        }
        Ok(())
    }
}

impl Default for AggregationParams {
    fn default() -> Self {
        Self {
            aggregation: PlainAggregationParams::default(),
            over_interp: 1.5,
        }
    }
}

/// Parameters for smoothed aggregation coarsening.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmoothedAggregationParams {
    /// Aggregate construction and strong-connection parameters.
    pub aggregation: PlainAggregationParams,
    /// Scaling of the Jacobi prolongation damping factor.
    pub relax: f64,
    /// Estimate the spectral radius of `D⁻¹A` instead of using the fixed `2/3` damping.
    pub estimate_spectral_radius: bool,
    /// Power iterations for spectral estimation; zero uses Gershgorin discs.
    pub power_iterations: usize,
}

impl SmoothedAggregationParams {
    pub(crate) fn validate(&self) -> Result<()> {
        self.aggregation.validate()?;
        if !self.relax.is_finite() || self.relax <= 0.0 {
            return Err(Error::InvalidParameter("smoothed aggregation relax must be finite and positive".into()));
        }
        Ok(())
    }
}

impl Default for SmoothedAggregationParams {
    fn default() -> Self {
        Self {
            aggregation: PlainAggregationParams::default(),
            relax: 1.0,
            estimate_spectral_radius: false,
            power_iterations: 0,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Transfer {
    pub(crate) prolongation: CsrMatrix,
    pub(crate) coarse_scale: f64,
}

struct Aggregates<'a> {
    count: usize,
    ids: Vec<isize>,
    strong: Vec<bool>,
    _guard: BudgetGuard<'a>,
}

pub(crate) fn aggregation(a: &CsrMatrix, params: AggregationParams, budget: &MemoryBudget, level: usize) -> Result<Option<Transfer>> {
    params.aggregation.validate()?;
    if !params.over_interp.is_finite() || params.over_interp <= 0.0 {
        return Err(Error::InvalidParameter("over_interp must be finite and positive".into()));
    }
    let aggregates = Aggregates::new(a, params.aggregation, budget, level)?;
    let Some(prolongation) = tentative(a, &aggregates, budget, level)? else {
        return Ok(None);
    };
    Ok(Some(Transfer {
        prolongation,
        coarse_scale: 1.0 / params.over_interp,
    }))
}

pub(crate) fn smoothed_aggregation(
    a: &CsrMatrix,
    params: SmoothedAggregationParams,
    budget: &MemoryBudget,
    level: usize,
) -> Result<Option<Transfer>> {
    params.aggregation.validate()?;
    if !params.relax.is_finite() || params.relax <= 0.0 {
        return Err(Error::InvalidParameter("smoothed aggregation relax must be finite and positive".into()));
    }
    let aggregates = Aggregates::new(a, params.aggregation, budget, level)?;
    let Some(tentative) = tentative(a, &aggregates, budget, level)? else {
        return Ok(None);
    };
    let _tentative_guard = budget.guard(tentative.memory_bytes());

    let omega = if params.estimate_spectral_radius {
        let inv_diag = inverse_diagonal_with_budget(a, budget, Some(level))?;
        let _inv_guard = budget.guard(vec_bytes::<f64>(inv_diag.len()));
        let radius = spectral_radius_with_budget(a, params.power_iterations, &inv_diag, Parallelism::default(), budget, Some(level))?;
        if radius <= 0.0 || !radius.is_finite() {
            return Err(Error::InvalidMatrix(
                "smoothed aggregation requires a positive finite spectral radius".into(),
            ));
        }
        params.relax * (4.0 / 3.0) / radius
    } else {
        params.relax * (2.0 / 3.0)
    };
    if !omega.is_finite() {
        return Err(Error::NonFinite);
    }

    let n = a.nrows();
    let mut nnz_cap = 0_usize;
    let mut row_cap = 0_usize;
    for i in 0..n {
        let mut count = 0_usize;
        for index in a.row_offsets()[i]..a.row_offsets()[i + 1] {
            let column = a.col_indices()[index];
            if column == i || aggregates.strong[index] {
                count += tentative.row_offsets()[column + 1] - tentative.row_offsets()[column];
            }
        }
        nnz_cap += count;
        row_cap = row_cap.max(count);
    }

    CsrMatrix::from_row_producer_with_budget(n, tentative.ncols(), nnz_cap, row_cap, budget, Some(level), |i, row_buf| {
        let mut filtered_diagonal = 0.0;
        for index in a.row_offsets()[i]..a.row_offsets()[i + 1] {
            if a.col_indices()[index] == i || !aggregates.strong[index] {
                filtered_diagonal += a.values()[index];
            }
        }
        let off_diagonal_scale = if filtered_diagonal == 0.0 { 0.0 } else { -omega / filtered_diagonal };

        for index in a.row_offsets()[i]..a.row_offsets()[i + 1] {
            let column = a.col_indices()[index];
            if column != i && !aggregates.strong[index] {
                continue;
            }
            let value = if column == i {
                1.0 - omega
            } else {
                off_diagonal_scale * a.values()[index]
            };
            for p_index in tentative.row_offsets()[column]..tentative.row_offsets()[column + 1] {
                row_buf.push((tentative.col_indices()[p_index], value * tentative.values()[p_index]));
            }
        }
        Ok(())
    })
    .map(|p| {
        Some(Transfer {
            prolongation: p,
            coarse_scale: 1.0,
        })
    })
}

impl<'a> Aggregates<'a> {
    fn new(a: &CsrMatrix, params: PlainAggregationParams, budget: &'a MemoryBudget, level: usize) -> Result<Self> {
        let n = a.nrows();
        let diagonal = a.diagonal_with_budget(budget, Some(level))?;
        let diag_guard = budget.guard(vec_bytes::<f64>(n));
        let eps_squared = params.eps_strong * params.eps_strong;
        let mut strong = try_vec_filled(false, a.nnz(), budget, Some(level))?;
        let mut guard = budget.guard(vec_bytes::<bool>(a.nnz()));
        for i in 0..n {
            for (index, is_strong) in strong.iter_mut().enumerate().take(a.row_offsets()[i + 1]).skip(a.row_offsets()[i]) {
                let column = a.col_indices()[index];
                let value = a.values()[index];
                *is_strong = column != i && eps_squared * diagonal[i] * diagonal[column] < value * value;
            }
        }
        drop(diagonal);
        drop(diag_guard);

        let mut ids = try_vec_filled(REMOVED, n, budget, Some(level))?;
        guard.merge(budget.guard(vec_bytes::<isize>(n)));
        for (i, id) in ids.iter_mut().enumerate() {
            if (a.row_offsets()[i]..a.row_offsets()[i + 1]).any(|index| strong[index]) {
                *id = UNDEFINED;
            }
        }

        let mut count = 0;
        let max_row_len = a.max_row_nnz();
        let mut neighbors = try_vec_with_capacity::<usize>(max_row_len, budget, Some(level))?;
        let neighbors_guard = budget.guard(vec_bytes::<usize>(max_row_len));
        for i in 0..n {
            if ids[i] != UNDEFINED {
                continue;
            }
            let current = count as isize;
            count += 1;
            ids[i] = current;

            neighbors.clear();
            for (index, &is_strong) in strong.iter().enumerate().take(a.row_offsets()[i + 1]).skip(a.row_offsets()[i]) {
                let column = a.col_indices()[index];
                if is_strong && ids[column] != REMOVED {
                    ids[column] = current;
                    neighbors.push(column);
                }
            }
            for &neighbor in &neighbors {
                for (index, &is_strong) in strong
                    .iter()
                    .enumerate()
                    .take(a.row_offsets()[neighbor + 1])
                    .skip(a.row_offsets()[neighbor])
                {
                    let column = a.col_indices()[index];
                    if is_strong && ids[column] == UNDEFINED {
                        ids[column] = current;
                    }
                }
            }
        }
        drop(neighbors);
        drop(neighbors_guard);
        if count == 0 {
            return Ok(Self {
                count: 0,
                ids,
                strong,
                _guard: guard,
            });
        }

        // Expansion can empty an aggregate. Remove gaps in coarse indices.
        let mut occupied = try_vec_filled(false, count, budget, Some(level))?;
        let _occ_guard = budget.guard(vec_bytes::<bool>(count));
        for &id in &ids {
            if id >= 0 {
                occupied[id as usize] = true;
            }
        }
        let mut renumber = try_vec_filled(0, count, budget, Some(level))?;
        let _renum_guard = budget.guard(vec_bytes::<usize>(count));
        let mut next = 0;
        for (old, &present) in occupied.iter().enumerate() {
            renumber[old] = next;
            next += usize::from(present);
        }
        for id in &mut ids {
            if *id >= 0 {
                *id = renumber[*id as usize] as isize;
            }
        }
        Ok(Self {
            count: next,
            ids,
            strong,
            _guard: guard,
        })
    }
}

fn tentative(a: &CsrMatrix, aggregates: &Aggregates<'_>, budget: &MemoryBudget, level: usize) -> Result<Option<CsrMatrix>> {
    let n = a.nrows();
    if aggregates.count == 0 || aggregates.count == n {
        return Ok(None);
    }
    let nnz = aggregates.ids.iter().filter(|&&id| id >= 0).count();
    CsrMatrix::from_row_producer_with_budget(n, aggregates.count, nnz, 1, budget, Some(level), |row, buf| {
        let id = aggregates.ids[row];
        if id >= 0 {
            buf.push((id as usize, 1.0));
        }
        Ok(())
    })
    .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_matrix(n: usize) -> CsrMatrix {
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
    fn plain_aggregates_are_disjoint_and_cover_connected_points() {
        let a = path_matrix(8);
        let budget = MemoryBudget::new(0);
        let aggregates = Aggregates::new(&a, PlainAggregationParams::default(), &budget, 0).unwrap();
        assert!(aggregates.count > 0 && aggregates.count < a.nrows());
        assert!(aggregates.ids.iter().all(|&id| { id >= 0 && (id as usize) < aggregates.count }));
        let p = tentative(&a, &aggregates, &budget, 0).unwrap().unwrap();
        assert_eq!(p.nnz(), a.nrows());
        assert!(p.row_offsets().windows(2).all(|w| w[1] - w[0] == 1));
    }

    #[test]
    fn isolated_rows_have_empty_tentative_rows() {
        let a = CsrMatrix::from_triplets(3, 3, [(0, 0, 1.0), (1, 1, 2.0), (1, 2, -1.0), (2, 1, -1.0), (2, 2, 2.0)]).unwrap();
        let budget = MemoryBudget::new(0);
        let aggregates = Aggregates::new(&a, PlainAggregationParams::default(), &budget, 0).unwrap();
        assert_eq!(aggregates.ids[0], REMOVED);
        let p = tentative(&a, &aggregates, &budget, 0).unwrap().unwrap();
        assert_eq!(p.row_offsets()[0], p.row_offsets()[1]);
    }

    #[test]
    fn smoothed_path_prolongation_has_expected_entries() {
        let a = path_matrix(4);
        let params = SmoothedAggregationParams::default();
        let p = smoothed_aggregation(&a, params, &MemoryBudget::new(0), 0).unwrap().unwrap().prolongation;
        assert_eq!(p.ncols(), 2);
        let dense = p.to_dense();
        let expected = [2.0 / 3.0, 0.0, 2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 0.0, 2.0 / 3.0];
        for (actual, expected) in dense.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-14, "{dense:?}");
        }
    }

    #[test]
    fn aggregation_rejects_invalid_parameters() {
        let a = path_matrix(8);
        let budget = MemoryBudget::new(0);
        let error = aggregation(
            &a,
            AggregationParams {
                over_interp: 0.0,
                ..AggregationParams::default()
            },
            &budget,
            0,
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidParameter(_)));
        let error = smoothed_aggregation(
            &a,
            SmoothedAggregationParams {
                relax: f64::NAN,
                ..SmoothedAggregationParams::default()
            },
            &budget,
            0,
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidParameter(_)));
    }
}
