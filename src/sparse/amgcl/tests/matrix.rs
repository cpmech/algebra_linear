use crate::sparse::amgcl::{CsrMatrix, Error};

#[test]
fn triplets_are_sorted_combined_and_zero_filtered() {
    let a = CsrMatrix::from_triplets(2, 3, [(1, 2, 2.0), (0, 1, 1.0), (1, 2, 3.0), (0, 0, 1.0), (0, 0, -1.0)]).unwrap();
    assert_eq!(a.row_offsets(), &[0, 1, 2]);
    assert_eq!(a.col_indices(), &[1, 2]);
    assert_eq!(a.values(), &[1.0, 5.0]);
}

#[test]
fn spmv_transpose_and_product_match_dense_results() {
    let a = CsrMatrix::from_triplets(2, 3, [(0, 0, 1.0), (0, 2, 2.0), (1, 1, -1.0), (1, 2, 4.0)]).unwrap();
    let mut y = vec![0.0; 2];
    a.spmv(&[2.0, 3.0, 5.0], &mut y).unwrap();
    assert_eq!(y, vec![12.0, 17.0]);

    let transpose = a.transpose();
    assert_eq!(transpose.to_dense(), vec![1.0, 0.0, 0.0, -1.0, 2.0, 4.0]);
    let gram = transpose.matmul(&a).unwrap();
    assert_eq!(gram.to_dense(), vec![1.0, 0.0, 2.0, 0.0, 1.0, -4.0, 2.0, -4.0, 20.0]);
}

#[test]
fn malformed_csr_is_rejected() {
    let error = CsrMatrix::from_parts(1, 1, vec![0, 2], vec![0], vec![1.0]).unwrap_err();
    assert!(matches!(error, Error::InvalidMatrix(_)));
    let error = CsrMatrix::from_triplets(1, 1, [(0, 1, 2.0)]).unwrap_err();
    assert!(matches!(error, Error::InvalidMatrix(_)));
}

#[test]
fn poisson_matrix_has_expected_dimensions_and_nonzeros() {
    let a = CsrMatrix::poisson_2d(4, 3).unwrap();
    assert_eq!(a.nrows(), 12);
    assert_eq!(a.ncols(), 12);
    assert_eq!(a.nnz(), 46);
    assert_eq!(a.diagonal().unwrap(), vec![4.0; 12]);
}
