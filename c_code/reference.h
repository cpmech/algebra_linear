// Reference-LAPACK bindings for the reference-parity harness.
//
// Development only. `build.rs` compiles this file against a locally built LAPACK
// 3.12.1 when the `dev_lapack` feature is on; a normal build never sees it, and
// the crate ships no reference library.
//
// They call the reference's column-major C interface (`LAPACK_*`). Never use
// `LAPACKE_*`: it is the row-major API and would transpose every matrix, making
// the comparison meaningless.
//
// LAPACK is BSD-3-Clause; see LICENSE.

#pragma once

#include <inttypes.h>

#include "lapack.h"

// `COMPLEX64` is the reference's opaque double-complex, two doubles in sequence,
// which is what `Complex64` is on the Rust side.
#define COMPLEX64 lapack_complex_double

// Boolean spelling of the C interface, and the SVD job codes.
#define C_BOOL int32_t
#define C_TRUE 1
#define C_FALSE 0
#define SVD_CODE_A 0
#define SVD_CODE_S 1
#define SVD_CODE_O 2
#define SVD_CODE_N 3

// Computes the solution to a system of linear equations
// <https://www.netlib.org/lapack/explore-html/d8/d72/dgesv_8f.html>
void c_dgesv(const int32_t *n,
             const int32_t *nrhs,
             double *a,
             const int32_t *lda,
             int32_t *ipiv,
             double *b,
             const int32_t *ldb,
             int32_t *info);

// Computes the solution to a real system of linear equations
// <https://www.netlib.org/lapack/explore-html/d1/ddc/zgesv_8f.html>
void c_zgesv(const int32_t *n,
             const int32_t *nrhs,
             COMPLEX64 *a,
             const int32_t *lda,
             int32_t *ipiv,
             COMPLEX64 *b,
             const int32_t *ldb,
             int32_t *info);

// Computes the Cholesky factorization of a real symmetric positive definite matrix
// <https://www.netlib.org/lapack/explore-html/d0/d8a/dpotrf_8f.html>
void c_dpotrf(C_BOOL upper,
              const int32_t *n,
              double *a,
              const int32_t *lda,
              int32_t *info);

// Computes the Cholesky factorization of a complex Hermitian positive definite matrix A.
// <https://www.netlib.org/lapack/explore-html/d1/db9/zpotrf_8f.html>
void c_zpotrf(C_BOOL upper,
              const int32_t *n,
              COMPLEX64 *a,
              const int32_t *lda,
              int32_t *info);

// Computes the eigenvalues and eigenvectors of a symmetric matrix
// <https://netlib.org/lapack/explore-html/dd/d4c/dsyev_8f.html>
void c_dsyev(C_BOOL calc_v,
             C_BOOL upper,
             const int32_t *n,
             double *a,
             const int32_t *lda,
             double *w,
             double *work,
             const int32_t *lwork,
             int32_t *info);

// Computes the eigenvalues and eigenvectors of a general matrix
// <https://www.netlib.org/lapack/explore-html/d9/d28/dgeev_8f.html>
void c_dgeev(C_BOOL calc_vl,
             C_BOOL calc_vr,
             const int32_t *n,
             double *a,
             const int32_t *lda,
             double *wr,
             double *wi,
             double *vl,
             const int32_t *ldvl,
             double *vr,
             const int32_t *ldvr,
             double *work,
             const int32_t *lwork,
             int32_t *info);

// Computes the eigenvalues and, optionally, the left and/or right eigenvectors for HE matrices
// <https://www.netlib.org/lapack/explore-html/d6/dee/zheev_8f.html>
void c_zheev(
    C_BOOL calc_v,
    C_BOOL upper,
    int32_t const *n,
    COMPLEX64 *a,
    int32_t const *lda,
    double *w,
    COMPLEX64 *work,
    int32_t const *lwork,
    double *rwork,
    int32_t *info);

// Computes the singular value decomposition (SVD)
// <https://www.netlib.org/lapack/explore-html/d8/d2d/dgesvd_8f.html>
void c_dgesvd(int32_t jobu_code,
              int32_t jobvt_code,
              const int32_t *m,
              const int32_t *n,
              double *a,
              const int32_t *lda,
              double *s,
              double *u,
              const int32_t *ldu,
              double *vt,
              const int32_t *ldvt,
              double *work,
              const int32_t *lwork,
              int32_t *info);

// Computes the LU factorization of a general (m,n) matrix
// <https://www.netlib.org/lapack/explore-html/d3/d6a/dgetrf_8f.html>
void c_dgetrf(const int32_t *m,
              const int32_t *n,
              double *a,
              const int32_t *lda,
              int32_t *ipiv,
              int32_t *info);

// Computes the inverse of a matrix using the LU factorization computed by dgetrf
// <https://www.netlib.org/lapack/explore-html/df/da4/dgetri_8f.html>
void c_dgetri(const int32_t *n,
              double *a,
              const int32_t *lda,
              const int32_t *ipiv,
              double *work,
              const int32_t *lwork,
              int32_t *info);

// Computes the LU factorization of a general (m,n) matrix
// <https://www.netlib.org/lapack/explore-html/dd/dd1/zgetrf_8f.html>
void c_zgetrf(const int32_t *m,
              const int32_t *n,
              COMPLEX64 *a,
              const int32_t *lda,
              int32_t *ipiv,
              int32_t *info);

// Computes the inverse of a matrix using the LU factorization computed by zgetrf
// <https://www.netlib.org/lapack/explore-html/d0/db3/zgetri_8f.html>
void c_zgetri(const int32_t *n,
              COMPLEX64 *a,
              const int32_t *lda,
              const int32_t *ipiv,
              COMPLEX64 *work,
              const int32_t *lwork,
              int32_t *info);
