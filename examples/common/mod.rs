//! CSR assembly shared by the AMGCL examples.
//!
//! The public AMGCL entry points take raw compressed sparse row arrays, so each
//! example assembles its own matrix here rather than handing a matrix object to
//! the solver.

/// Compressed sparse row arrays for a square matrix.
pub struct Csr {
    /// Row-start offsets, one more than the number of rows.
    pub row_ptr: Vec<usize>,
    /// Column indices in row-major storage order.
    pub col_idx: Vec<usize>,
    /// Values aligned with `col_idx`.
    pub values: Vec<f64>,
}

impl Csr {
    /// Number of rows and columns.
    pub fn rows(&self) -> usize {
        self.row_ptr.len() - 1
    }
}

/// Builds canonical CSR arrays from `(row, column, value)` triplets.
///
/// Each row is sorted by column, duplicate columns are summed, and exact zeros
/// are dropped.
pub fn from_triplets<I>(n: usize, triplets: I) -> Csr
where
    I: IntoIterator<Item = (usize, usize, f64)>,
{
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (row, column, value) in triplets {
        assert!(row < n && column < n, "triplet ({row}, {column}) is outside the {n} by {n} matrix");
        rows[row].push((column, value));
    }

    let mut csr = Csr {
        row_ptr: vec![0],
        col_idx: Vec::new(),
        values: Vec::new(),
    };
    for row in &mut rows {
        row.sort_unstable_by_key(|entry| entry.0);
        let mut index = 0;
        while index < row.len() {
            let column = row[index].0;
            let mut value = 0.0;
            while index < row.len() && row[index].0 == column {
                value += row[index].1;
                index += 1;
            }
            if value != 0.0 {
                csr.col_idx.push(column);
                csr.values.push(value);
            }
        }
        csr.row_ptr.push(csr.col_idx.len());
    }
    csr
}

/// Five-point finite-difference Laplacian on an `nx` by `ny` grid.
///
/// Boundary conditions are Dirichlet and the grid spacing is omitted.
pub fn poisson_2d(nx: usize, ny: usize) -> Csr {
    assert!(nx > 0 && ny > 0, "Poisson grid dimensions must be positive");
    let mut triplets = Vec::with_capacity(nx * ny * 5);
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
    from_triplets(nx * ny, triplets)
}
