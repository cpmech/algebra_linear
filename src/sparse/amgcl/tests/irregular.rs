use crate::sparse::amgcl::{Amg, AmgParams, Cg, CsrMatrix, SolverParams};

/// Deterministic irregular symmetric M-matrices exercise C/F splitting and
/// interpolation without relying on a random-number dependency.
fn irregular_m_matrix(n: usize, seed: u64) -> CsrMatrix {
    let mut triplets = Vec::new();
    let mut row_sums = vec![0.0; n];
    for i in 0..n {
        for j in i + 1..n {
            let hash = seed.wrapping_add((i as u64 + 1) * 0x9e37_79b9).wrapping_add((j as u64 + 3) * 0x85eb_ca6b);
            if hash % 5 < 2 || j == i + 1 {
                let value = -0.2 - (hash % 100) as f64 / 150.0;
                triplets.push((i, j, value));
                triplets.push((j, i, value));
                row_sums[i] += value.abs();
                row_sums[j] += value.abs();
            }
        }
    }
    for (i, &row_sum) in row_sums.iter().enumerate() {
        triplets.push((i, i, row_sum + 0.5));
    }
    CsrMatrix::from_triplets(n, n, triplets).unwrap()
}

#[test]
fn amg_converges_on_irregular_spd_matrices() {
    for case in 0..12 {
        let n = 9 + case * 7;
        let a = irregular_m_matrix(n, case as u64 + 11);
        let b: Vec<f64> = (0..n).map(|i| ((i * 13 + case) % 17) as f64 - 4.0).collect();
        let mut x = vec![0.0; n];
        let mut amg = Amg::new(
            &a,
            AmgParams {
                coarse_enough: 8,
                ..AmgParams::default()
            },
        )
        .unwrap();
        let report = Cg::new(SolverParams {
            tolerance: 1e-9,
            max_iterations: 100,
            ..SolverParams::default()
        })
        .solve(&a, &mut amg, &b, &mut x)
        .unwrap();
        assert!(report.converged, "case {case}, n {n}: {report:?}");
        let mut r = vec![0.0; n];
        a.residual(&b, &x, &mut r).unwrap();
        let numerator = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        let denominator = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(numerator <= 1.1e-9 * denominator, "case {case}");
    }
}
