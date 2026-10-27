//! Greedy aggregation and aggregation-based transfer operators.

use crate::sparse::amgcl::matrix::CsrMatrix;
use crate::sparse::amgcl::relaxation::{inverse_diagonal, spectral_radius};
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

struct Aggregates {
    count: usize,
    ids: Vec<isize>,
    strong: Vec<bool>,
}

pub(crate) fn aggregation(a: &CsrMatrix, params: AggregationParams) -> Result<Option<Transfer>> {
    params.aggregation.validate()?;
    if !params.over_interp.is_finite() || params.over_interp <= 0.0 {
        return Err(Error::InvalidParameter("over_interp must be finite and positive".into()));
    }
    let aggregates = Aggregates::new(a, params.aggregation)?;
    let Some(prolongation) = tentative(a, &aggregates)? else {
        return Ok(None);
    };
    Ok(Some(Transfer {
        prolongation,
        coarse_scale: 1.0 / params.over_interp,
    }))
}

pub(crate) fn smoothed_aggregation(a: &CsrMatrix, params: SmoothedAggregationParams) -> Result<Option<Transfer>> {
    params.aggregation.validate()?;
    if !params.relax.is_finite() || params.relax <= 0.0 {
        return Err(Error::InvalidParameter("smoothed aggregation relax must be finite and positive".into()));
    }
    let aggregates = Aggregates::new(a, params.aggregation)?;
    let Some(tentative) = tentative(a, &aggregates)? else {
        return Ok(None);
    };

    let omega = if params.estimate_spectral_radius {
        let radius = spectral_radius(a, params.power_iterations, &inverse_diagonal(a)?)?;
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
    let mut rows = vec![Vec::new(); n];
    for (i, row) in rows.iter_mut().enumerate() {
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
                row.push((tentative.col_indices()[p_index], value * tentative.values()[p_index]));
            }
        }
    }
    let triplets = rows
        .into_iter()
        .enumerate()
        .flat_map(|(row, entries)| entries.into_iter().map(move |(column, value)| (row, column, value)));
    CsrMatrix::from_triplets(n, tentative.ncols(), triplets).map(|p| {
        Some(Transfer {
            prolongation: p,
            coarse_scale: 1.0,
        })
    })
}

impl Aggregates {
    fn new(a: &CsrMatrix, params: PlainAggregationParams) -> Result<Self> {
        let n = a.nrows();
        let diagonal = a.diagonal()?;
        let eps_squared = params.eps_strong * params.eps_strong;
        let mut strong = vec![false; a.nnz()];
        for i in 0..n {
            for (index, is_strong) in strong.iter_mut().enumerate().take(a.row_offsets()[i + 1]).skip(a.row_offsets()[i]) {
                let column = a.col_indices()[index];
                let value = a.values()[index];
                *is_strong = column != i && eps_squared * diagonal[i] * diagonal[column] < value * value;
            }
        }

        let mut ids = vec![REMOVED; n];
        for (i, id) in ids.iter_mut().enumerate() {
            if (a.row_offsets()[i]..a.row_offsets()[i + 1]).any(|index| strong[index]) {
                *id = UNDEFINED;
            }
        }

        let mut count = 0;
        let mut neighbors = Vec::new();
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
        if count == 0 {
            return Ok(Self { count: 0, ids, strong });
        }

        // Expansion can empty an aggregate. Remove gaps in coarse indices.
        let mut occupied = vec![false; count];
        for &id in &ids {
            if id >= 0 {
                occupied[id as usize] = true;
            }
        }
        let mut renumber = vec![0; count];
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
        Ok(Self { count: next, ids, strong })
    }
}

fn tentative(a: &CsrMatrix, aggregates: &Aggregates) -> Result<Option<CsrMatrix>> {
    let n = a.nrows();
    if aggregates.count == 0 || aggregates.count == n {
        return Ok(None);
    }
    let triplets = aggregates
        .ids
        .iter()
        .enumerate()
        .filter_map(|(row, &id)| (id >= 0).then_some((row, id as usize, 1.0)));
    CsrMatrix::from_triplets(n, aggregates.count, triplets).map(Some)
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
        let aggregates = Aggregates::new(&a, PlainAggregationParams::default()).unwrap();
        assert!(aggregates.count > 0 && aggregates.count < a.nrows());
        assert!(aggregates.ids.iter().all(|&id| { id >= 0 && (id as usize) < aggregates.count }));
        let p = tentative(&a, &aggregates).unwrap().unwrap();
        assert_eq!(p.nnz(), a.nrows());
        assert!(p.row_offsets().windows(2).all(|w| w[1] - w[0] == 1));
    }

    #[test]
    fn isolated_rows_have_empty_tentative_rows() {
        let a = CsrMatrix::from_triplets(3, 3, [(0, 0, 1.0), (1, 1, 2.0), (1, 2, -1.0), (2, 1, -1.0), (2, 2, 2.0)]).unwrap();
        let aggregates = Aggregates::new(&a, PlainAggregationParams::default()).unwrap();
        assert_eq!(aggregates.ids[0], REMOVED);
        let p = tentative(&a, &aggregates).unwrap().unwrap();
        assert_eq!(p.row_offsets()[0], p.row_offsets()[1]);
    }

    #[test]
    fn smoothed_path_prolongation_has_expected_entries() {
        let a = path_matrix(4);
        let params = SmoothedAggregationParams::default();
        let p = smoothed_aggregation(&a, params).unwrap().unwrap().prolongation;
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
        let error = aggregation(
            &a,
            AggregationParams {
                over_interp: 0.0,
                ..AggregationParams::default()
            },
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidParameter(_)));
        let error = smoothed_aggregation(
            &a,
            SmoothedAggregationParams {
                relax: f64::NAN,
                ..SmoothedAggregationParams::default()
            },
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidParameter(_)));
    }
}
