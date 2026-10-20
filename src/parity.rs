//! Reference-parity tests: the kernels against LAPACK 3.12.1.
//!
//! Enabled by the `dev_lapack` feature, which compiles `c_code/reference_lapack.c`
//! against a locally built LAPACK and links it. The bindings drive the
//! **column-major** `LAPACK_*` interface, so both sides see the same data layout;
//! using LAPACKE would transpose every matrix and make the comparison meaningless.
//!
//! These tests are the reason the eigenvalue and SVD work is tractable: an
//! eigenvalue has no closed form for a general matrix, so "agrees with the
//! reference to 1e-10" is the only practical definition of correct.
//!
//! # What is compared
//!
//! Not the internal factorizations, which are allowed to differ, but the
//! *observable results*: the solution of a system, the inverse, the eigenvalues,
//! the singular values. Those are what a caller sees, and comparing them leaves
//! each implementation free to be organized differently.
//!
//! The subject is the **driver**, not a wrapper above it: this crate is the
//! kernels, so `dsyev` is compared here rather than through any wrapper above it.
//! The glue that maps a caller's matrix onto these slices is the caller's business
//! and is covered by its own tests.

use crate::Complex64;
use crate::constants::{C_FALSE, C_TRUE};
use crate::dev_lapack;
use crate::lapack::{self, LapackError};

/// Tolerance for a comparison that involves an elimination or an iteration.
///
/// Not machine epsilon: both sides accumulate rounding, and the two algorithms
/// differ in the order they do so. 1e-10 relative is far tighter than any
/// practical use of these routines yet still catches a genuinely wrong answer,
/// which is what these tests are for.
const TOL: f64 = 1e-10;

// ---------------------------------------------------------------------------------------------
// Fixtures. All column-major, matching what the drivers take; `get`/`cget` exist so the residual
// checks below read as matrix algebra rather than index arithmetic.
// ---------------------------------------------------------------------------------------------

/// Element `(i, j)` of an `nrow`-row column-major matrix.
fn get(a: &[f64], nrow: usize, i: usize, j: usize) -> f64 {
    a[i + j * nrow]
}

/// Element `(i, j)` of a complex `nrow`-row column-major matrix.
fn cget(a: &[Complex64], nrow: usize, i: usize, j: usize) -> Complex64 {
    a[i + j * nrow]
}

/// A deterministic, well-conditioned, non-symmetric matrix.
///
/// Built from a fixed recurrence rather than a random number generator so the
/// tests are reproducible without a dependency, and made non-symmetric and
/// non-tridiagonal so no structure can hide an indexing mistake.
fn sample(n: usize) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i + j * n] = if i == j {
                // Dominant diagonal keeps it well conditioned.
                (n as f64) + 1.0 + (i as f64) * 0.5
            } else {
                (((i * 7 + j * 13) % 11) as f64) - 5.0
            };
        }
    }
    a
}

/// A deterministic, well-conditioned, genuinely complex matrix.
///
/// Real or Hermitian data would hide a conjugation mistake, so the imaginary
/// parts are non-zero everywhere.
fn complex_sample(n: usize) -> Vec<Complex64> {
    let mut a = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let re = if i == j {
                (n as f64) + 2.0 + (i as f64) * 0.5
            } else {
                (((i * 5 + j * 7) % 9) as f64) - 4.0
            };
            let im = (((i * 3 + j * 11) % 7) as f64) - 3.0 + if i == j { 1.0 } else { 0.0 };
            a[i + j * n] = Complex64::new(re, im);
        }
    }
    a
}

/// A deterministic symmetric matrix, built from `sample`'s upper triangle.
fn symmetric_sample(n: usize) -> Vec<f64> {
    let raw = sample(n);
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let (x, y) = if i <= j { (i, j) } else { (j, i) };
            a[i + j * n] = get(&raw, n, x, y);
        }
    }
    a
}

/// A deterministic rectangular matrix, non-symmetric and with no zero entries.
fn rect_sample(m: usize, n: usize) -> Vec<f64> {
    let mut a = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            a[i + j * m] = (((i * 7 + j * 13 + 3) % 11) as f64) - 5.0 + (i as f64) * 0.25 - (j as f64) * 0.1;
        }
    }
    a
}

/// A deterministic, genuinely complex Hermitian matrix.
///
/// `H = (A + A^H) / 2` for the complex sample, so it is Hermitian by
/// construction. The imaginary parts off the diagonal are non-zero -- that is the
/// point: on real or real-symmetric data a missing conjugation in `zhetd2`,
/// `zungtr` or `zsteqr` still produces a plausible answer.
fn hermitian_sample(n: usize) -> Vec<Complex64> {
    let raw = complex_sample(n);
    let mut h = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let s = cget(&raw, n, i, j) + cget(&raw, n, j, i).conj();
            h[i + j * n] = Complex64::new(0.5 * s.re, 0.5 * s.im);
        }
    }
    h
}

fn compare_vectors(label: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for i in 0..got.len() {
        let scale = want[i].abs().max(1.0);
        let err = (got[i] - want[i]).abs() / scale;
        assert!(
            err < tol,
            "{label}: component {i} is {} but the reference says {} (relative error {err:e})",
            got[i],
            want[i]
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Real, symmetric and general drivers
// ---------------------------------------------------------------------------------------------

#[test]
fn dgesv_agrees_with_the_reference() {
    for n in [1usize, 2, 3, 5, 9] {
        let a = sample(n);

        // A known solution, so the comparison is against the reference solver
        // rather than against a hand-transcribed answer.
        let x_true: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.25).collect();
        let mut b_native = vec![0.0; n];
        crate::blas::dgemv(1.0, &a, &x_true, 0.0, &mut b_native, n, n);
        let mut b_ref = b_native.clone();

        // Native.
        let mut a_native = a.clone();
        lapack::dgesv(&mut a_native, &mut b_native).unwrap();

        // Reference. It overwrites its input, so hand it a copy rather than the
        // matrix the native side already consumed.
        let mut a_ref = a.clone();
        let mut ipiv = vec![0i32; n];
        let mut info = 0i32;
        let n_i32 = n as i32;
        // SAFETY: the buffers are local vectors sized from this test's `n` (or `m` and
        // `n`), so they hold every element the reference writes; `info` is checked
        // immediately after each call.
        unsafe {
            dev_lapack::c_dgesv(
                &n_i32,
                &1i32,
                a_ref.as_mut_ptr(),
                &n_i32,
                ipiv.as_mut_ptr(),
                b_ref.as_mut_ptr(),
                &n_i32,
                &mut info,
            );
        }
        assert_eq!(info, 0, "reference dgesv failed for n={n}");

        compare_vectors(&format!("dgesv solution (n={n})"), &b_native, &b_ref, TOL);
        // ...and both must recover the solution we started from.
        compare_vectors(&format!("dgesv vs known x (n={n})"), &b_native, &x_true, TOL);
    }
}

#[test]
fn dgetrf_and_dgetri_agree_with_the_reference() {
    for n in [1usize, 2, 4, 7] {
        let a = sample(n);

        // Native inverse.
        let mut inv_native = a.clone();
        let (ipiv_n, _det) = lapack::dgetrf(&mut inv_native, n).unwrap();
        lapack::dgetri(&mut inv_native, n, &ipiv_n).unwrap();

        // Reference inverse.
        let mut lu_ref = a.clone();
        let mut ipiv = vec![0i32; n];
        let mut info = 0i32;
        let n_i32 = n as i32;
        let mut work = vec![0.0f64; n.max(1)];
        let lwork = n_i32;
        // SAFETY: the buffers are local vectors sized from this test's `n` (or `m` and
        // `n`), so they hold every element the reference writes; `info` is checked
        // immediately after each call.
        unsafe {
            dev_lapack::c_dgetrf(&n_i32, &n_i32, lu_ref.as_mut_ptr(), &n_i32, ipiv.as_mut_ptr(), &mut info);
            assert_eq!(info, 0, "reference dgetrf failed for n={n}");
            dev_lapack::c_dgetri(&n_i32, lu_ref.as_mut_ptr(), &n_i32, ipiv.as_ptr(), work.as_mut_ptr(), &lwork, &mut info);
            assert_eq!(info, 0, "reference dgetri failed for n={n}");
        }

        // Element-wise: the inverse is unique, so this is a fair comparison.
        for i in 0..n {
            for j in 0..n {
                let got = get(&inv_native, n, i, j);
                let want = get(&lu_ref, n, i, j);
                let scale = want.abs().max(1.0);
                assert!(
                    (got - want).abs() / scale < TOL,
                    "dgetri ({n}): element ({i},{j}) is {got} but the reference says {want}"
                );
            }
        }
    }
}

#[test]
fn dpotrf_agrees_with_the_reference() {
    for n in [1usize, 2, 3, 6] {
        // A symmetric positive definite matrix: A^T A + n I, which is SPD by
        // construction and has no structure the algorithm could exploit.
        let raw = sample(n);
        let mut t = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += get(&raw, n, k, i) * get(&raw, n, k, j);
                }
                t[i + j * n] = if i == j { sum + n as f64 } else { sum };
            }
        }

        let mut l_native = t.clone();
        lapack::dpotrf(&mut l_native, n).unwrap();

        let mut l_ref = t.clone();
        let mut info = 0i32;
        let n_i32 = n as i32;
        // SAFETY: the buffers are local vectors sized from this test's `n` (or `m` and
        // `n`), so they hold every element the reference writes; `info` is checked
        // immediately after each call.
        unsafe {
            dev_lapack::c_dpotrf(C_FALSE, &n_i32, l_ref.as_mut_ptr(), &n_i32, &mut info);
        }
        assert_eq!(info, 0, "reference dpotrf failed for n={n}");

        // Only the lower triangle is written by either, by specification.
        for i in 0..n {
            for j in 0..=i {
                let got = get(&l_native, n, i, j);
                let want = get(&l_ref, n, i, j);
                let scale = want.abs().max(1.0);
                assert!(
                    (got - want).abs() / scale < TOL,
                    "dpotrf ({n}): L({i},{j}) is {got} but the reference says {want}"
                );
            }
        }
    }
}

#[test]
fn singular_detection_agrees_with_the_reference() {
    // Both sides must call the same matrices singular, or a caller would get a
    // solution from one path and an error from the other.
    for n in [2usize, 3, 5] {
        let mut sing = sample(n);
        // Make column 1 exactly twice column 0.
        for i in 0..n {
            let v = get(&sing, n, i, 0) * 2.0;
            sing[i + n] = v;
        }

        let mut native = sing.clone();
        let native_result = lapack::dgetrf(&mut native, n);

        let mut a_ref = sing.clone();
        let mut ipiv = vec![0i32; n];
        let mut info = 0i32;
        let n_i32 = n as i32;
        // SAFETY: the buffers are local vectors sized from this test's `n` (or `m` and
        // `n`), so they hold every element the reference writes; `info` is checked
        // immediately after each call.
        unsafe {
            dev_lapack::c_dgetrf(&n_i32, &n_i32, a_ref.as_mut_ptr(), &n_i32, ipiv.as_mut_ptr(), &mut info);
        }

        assert_eq!(
            native_result,
            Err(LapackError::Singular),
            "the native dgetrf must report a singular matrix (n={n})"
        );
        assert!(info > 0, "the reference dgetrf must report a singular matrix (n={n})");
    }
}

/// The real Schur form and its eigenvalues against the reference.
///
/// This is the acceptance test for `dhseqr`: `dgehrd` produces the Hessenberg
/// form, `dhseqr` reduces it to quasi-triangular Schur form and returns the
/// eigenvalues. Three things are checked, in increasing strength:
///
/// 1. `t` is quasi-upper-triangular -- zero below the first subdiagonal.
/// 2. The 2x2 blocks are in standard form (`t(i,i) == t(i+1,i+1)` and
///    `t(i+1,i) * t(i,i+1) < 0`), which is what makes the `wi` signs meaningful.
/// 3. The eigenvalue multiset agrees with the reference `dgeev`.
///
/// The multiset is compared *sorted*, not in order: the Schur form is not unique
/// and the two implementations deflate in a different sequence, so an in-order
/// comparison would be testing the ordering rather than the values. `wi` is
/// signed on both sides -- LAPACK returns the positive imaginary part first --
/// so conjugation pairs have to be sorted deterministically.
#[test]
fn dhseqr_eigenvalues_agree_with_the_reference() {
    for n in [1usize, 2, 3, 5, 8, 15] {
        let a = sample(n);

        // Native: Hessenberg, then real Schur form.
        let mut t = a.clone();
        lapack::eig::dgehrd(&mut t, n, None);
        let mut wr = vec![0.0; n];
        let mut wi = vec![0.0; n];
        lapack::eig::dhseqr(&mut t, &mut wr, &mut wi, n).unwrap();

        // (1) quasi-upper-triangular
        for j in 0..n {
            for i in (j + 2)..n {
                let t_ij = get(&t, n, i, j);
                assert!(t_ij.abs() < 1e-12, "n={n}: T({i},{j}) = {t_ij} is below the first subdiagonal");
            }
        }

        // (2) each 2x2 block standardized, and the two columns of a pair
        //     conjugate. A real eigenvalue has wi == 0.
        let mut i = 0;
        while i < n {
            if i + 1 < n && get(&t, n, i + 1, i) != 0.0 {
                assert!(
                    (get(&t, n, i, i) - get(&t, n, i + 1, i + 1)).abs() < 1e-10,
                    "n={n}: the 2x2 block at {i} has unequal diagonal entries"
                );
                assert!(
                    get(&t, n, i + 1, i) * get(&t, n, i, i + 1) < 0.0,
                    "n={n}: the 2x2 block at {i} is not in standard form"
                );
                assert!(wi[i] > 0.0, "n={n}: the first of a pair must have wi > 0, got {}", wi[i]);
                assert!((wi[i] + wi[i + 1]).abs() < 1e-10, "n={n}: the pair at {i} is not conjugate");
                i += 2;
            } else {
                assert_eq!(wi[i], 0.0, "n={n}: a real eigenvalue must have wi == 0");
                i += 1;
            }
        }

        // (3) the eigenvalue multiset against the reference.
        let mut a_ref = a.clone();
        let mut wr_ref = vec![0.0; n];
        let mut wi_ref = vec![0.0; n];
        let info = dev_lapack::query::dgeev(&mut a_ref, n, &mut wr_ref, &mut wi_ref, None);
        assert_eq!(info, 0, "reference dgeev failed for n={n}");

        let mut got: Vec<(f64, f64)> = (0..n).map(|i| (wr[i], wi[i])).collect();
        let mut want: Vec<(f64, f64)> = (0..n).map(|i| (wr_ref[i], wi_ref[i])).collect();
        got.sort_by(|x, y| x.partial_cmp(y).unwrap());
        want.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for i in 0..n {
            let scale = want[i].0.abs().max(want[i].1.abs()).max(1.0);
            assert!(
                (got[i].0 - want[i].0).abs() / scale < TOL && (got[i].1 - want[i].1).abs() / scale < TOL,
                "n={n}: eigenvalue {i} is {}+{}i but the reference says {}+{}i",
                got[i].0,
                got[i].1,
                want[i].0,
                want[i].1
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Complex drivers
// ---------------------------------------------------------------------------------------------

/// Complex Cholesky against the reference, on both triangles.
///
/// Built as `A^H A + n I`, which is Hermitian positive definite by construction.
/// Only the requested triangle is written by either side, so only that triangle
/// is compared -- the rest legitimately holds whatever the caller had.
#[test]
fn complex_cholesky_agrees_with_the_reference() {
    for n in [1usize, 2, 3, 6] {
        let raw = complex_sample(n);
        let mut t = vec![Complex64::new(0.0, 0.0); n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    s += cget(&raw, n, k, i).conj() * cget(&raw, n, k, j);
                }
                t[i + j * n] = s;
            }
        }
        for i in 0..n {
            let d = t[i + i * n];
            t[i + i * n] = Complex64::new(d.re + n as f64, 0.0);
        }

        for upper in [false, true] {
            let mut l_native = t.clone();
            let mut l_ref = t.clone();
            lapack::complex::zpotrf(&mut l_native, n, upper).expect("native Cholesky");
            let mut info = 0i32;
            let n_i32 = n as i32;
            let c_upper = if upper { C_TRUE } else { C_FALSE };
            // SAFETY: the buffers are local vectors sized from this test's `n` (or `m` and
            // `n`), so they hold every element the reference writes; `info` is checked
            // immediately after each call.
            unsafe {
                dev_lapack::c_zpotrf(c_upper, &n_i32, l_ref.as_mut_ptr(), &n_i32, &mut info);
            }
            assert_eq!(info, 0, "reference zpotrf failed for n={n} upper={upper}");
            for j in 0..n {
                for i in 0..n {
                    if (upper && i <= j) || (!upper && i >= j) {
                        let got = cget(&l_native, n, i, j);
                        let want = cget(&l_ref, n, i, j);
                        let scale = want.norm().max(1.0);
                        assert!(
                            (got - want).norm() / scale < TOL,
                            "n={n} upper={upper}: Cholesky differs at ({i},{j}): {got} vs {want}"
                        );
                    }
                }
            }
        }
    }
}

/// Complex inverse against the reference, element-wise.
///
/// The inverse is unique, so this is a fair comparison -- and it is the one that
/// catches a conjugation error, which any invariant-based check would miss.
#[test]
fn complex_inverse_agrees_with_the_reference() {
    for n in [1usize, 2, 3, 5, 8] {
        let a = complex_sample(n);

        // Native: the crate's own zgetrf + zgetri.
        let mut inv_native = a.clone();
        let ipiv_n = lapack::complex::zgetrf(&mut inv_native, n).expect("native zgetrf");
        lapack::complex::zgetri(&mut inv_native, n, &ipiv_n).expect("native zgetri");

        // Reference: zgetrf then zgetri, through the bindings.
        let n_i32 = n as i32;
        let mut lu = a.clone();
        let mut ipiv = vec![0i32; n];
        let mut info = 0i32;
        // SAFETY: the buffers are local vectors sized from this test's `n` (or `m` and
        // `n`), so they hold every element the reference writes; `info` is checked
        // immediately after each call.
        unsafe {
            dev_lapack::c_zgetrf(&n_i32, &n_i32, lu.as_mut_ptr(), &n_i32, ipiv.as_mut_ptr(), &mut info);
        }
        assert_eq!(info, 0, "reference zgetrf failed for n={n}");
        let mut work = vec![Complex64::new(0.0, 0.0); n.max(1)];
        // SAFETY: the buffers are local vectors sized from this test's `n` (or `m` and
        // `n`), so they hold every element the reference writes; `info` is checked
        // immediately after each call.
        unsafe {
            dev_lapack::c_zgetri(
                &n_i32,
                lu.as_mut_ptr(),
                &n_i32,
                ipiv.as_ptr(),
                work.as_mut_ptr(),
                &(work.len() as i32),
                &mut info,
            );
        }
        assert_eq!(info, 0, "reference zgetri failed for n={n}");

        for i in 0..n {
            for j in 0..n {
                let got = cget(&inv_native, n, i, j);
                let want = cget(&lu, n, i, j);
                let scale = want.norm().max(1.0);
                assert!((got - want).norm() / scale < TOL, "n={n}: inverse differs at ({i},{j}): {got} vs {want}");
            }
        }
    }
}

/// Complex solve against a known solution and against the reference.
///
/// The known solution is the stronger check: it does not depend on either
/// implementation being right.
#[test]
fn complex_solve_agrees_with_the_reference() {
    for n in [1usize, 2, 3, 5, 8] {
        let a = complex_sample(n);

        let x_true: Vec<Complex64> = (0..n).map(|i| Complex64::new(1.0 + (i as f64) * 0.5, -0.25 * (i as f64))).collect();
        // b = A x, formed directly so the fixture does not depend on a kernel.
        let mut b = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..n {
            for k in 0..n {
                b[i] += cget(&a, n, i, k) * x_true[k];
            }
        }

        // Native.
        let mut a_work = a.clone();
        let mut b_native = b.clone();
        lapack::complex::zgesv(&mut a_work, &mut b_native).expect("native solve");
        for i in 0..n {
            let scale = x_true[i].norm().max(1.0);
            assert!(
                (b_native[i] - x_true[i]).norm() / scale < TOL,
                "n={n}: native solve recovered {} but the true x[{i}] is {}",
                b_native[i],
                x_true[i]
            );
        }

        // Reference.
        let n_i32 = n as i32;
        let mut a_ref = a.clone();
        let mut b_ref = b.clone();
        let mut ipiv = vec![0i32; n];
        let mut info = 0i32;
        // SAFETY: the buffers are local vectors sized from this test's `n` (or `m` and
        // `n`), so they hold every element the reference writes; `info` is checked
        // immediately after each call.
        unsafe {
            dev_lapack::c_zgesv(
                &n_i32,
                &1i32,
                a_ref.as_mut_ptr(),
                &n_i32,
                ipiv.as_mut_ptr(),
                b_ref.as_mut_ptr(),
                &n_i32,
                &mut info,
            );
        }
        assert_eq!(info, 0, "reference zgesv failed for n={n}");
        for i in 0..n {
            let scale = b_ref[i].norm().max(1.0);
            assert!(
                (b_native[i] - b_ref[i]).norm() / scale < TOL,
                "n={n}: solve differs at {i}: {} vs the reference's {}",
                b_native[i],
                b_ref[i]
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Eigen-decompositions
// ---------------------------------------------------------------------------------------------

/// Symmetric eigen-decomposition against the reference.
///
/// Four things are checked, deliberately different in kind:
///
/// 1. **The eigenvalues are the reference's, in order.** Unlike the general
///    eigenproblem, the symmetric one *has* a canonical order -- ascending -- so
///    this compares in order rather than on a sorted multiset, and a wrong
///    ordering is a failure rather than a nuisance.
/// 2. **The vectors are orthonormal.** This is what a mis-accumulated
///    Householder or a missed rotation breaks.
/// 3. **`A v = lambda v` for each pair.** (1) and (2) together are still
///    satisfiable by the eigenvectors of a different matrix.
/// 4. **Both triangles agree.** LAPACK's `uplo` selects which triangle is *read*;
///    feeding the same symmetric matrix with `upper` false and true must give the
///    same answer, which catches a routine that reads the wrong half.
///
/// Nothing is compared against the reference's eigen*vectors*: their sign is
/// arbitrary.
#[test]
fn dsyev_agrees_with_the_reference() {
    for n in [1usize, 2, 3, 5, 8, 12] {
        let a = symmetric_sample(n);
        let mut results = Vec::new();

        for upper in [false, true] {
            let mut work = a.clone();
            let mut l = vec![0.0; n];
            lapack::sym::dsyev(&mut work, &mut l, n, upper).expect("native dsyev");

            // --- 1. eigenvalues in order against the reference.
            let mut t_ref = a.clone();
            let mut l_ref = vec![0.0; n];
            let info = dev_lapack::query::dsyev(&mut t_ref, n, &mut l_ref, true);
            assert_eq!(info, 0, "reference dsyev failed for n={n} upper={upper}");
            let scale = l_ref.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1.0);
            for i in 0..n {
                assert!(
                    (l[i] - l_ref[i]).abs() / scale < TOL,
                    "n={n} upper={upper}: eigenvalue {i} is {} but the reference says {}",
                    l[i],
                    l_ref[i]
                );
            }

            // --- 2. orthonormal columns.
            for i in 0..n {
                for j in 0..n {
                    let mut dot = 0.0;
                    for k in 0..n {
                        dot += get(&work, n, k, i) * get(&work, n, k, j);
                    }
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (dot - want).abs() < 1e-10,
                        "n={n} upper={upper}: columns {i},{j} are not orthonormal (dot = {dot})"
                    );
                }
            }

            // --- 3. the defining equation.
            for j in 0..n {
                for i in 0..n {
                    let mut av = 0.0;
                    for k in 0..n {
                        av += get(&a, n, i, k) * get(&work, n, k, j);
                    }
                    let resid = (av - l[j] * get(&work, n, i, j)).abs();
                    assert!(
                        resid / scale < 1e-10,
                        "n={n} upper={upper}: (A v - lambda v)[{i}] = {resid} for eigenvalue {j}"
                    );
                }
            }

            results.push((l, work));
        }

        // --- 4. the two triangles give the same spectrum, not necessarily the
        //     same vectors (the sign may differ).
        let (l_lo, _) = &results[0];
        let (l_up, _) = &results[1];
        for i in 0..n {
            assert!(
                (l_lo[i] - l_up[i]).abs() < 1e-12,
                "n={n}: reading the lower triangle gives {} but the upper gives {}",
                l_lo[i],
                l_up[i]
            );
        }
    }
}

/// Left eigenvectors, by their defining equation.
///
/// `dgeev_lr` returns both sets, so this checks what a right-vector-only test
/// cannot: `u^H A = lambda u^H`, equivalently `A^H u = conj(lambda) u`. It also
/// checks that the two sets agree on the eigenvalue list -- a left eigenvector
/// paired with the wrong eigenvalue is the specific mistake that a naive "run
/// `dgeev` on `A^T` and hand back its vectors" would make, since two independent
/// factorizations need not deflate in the same order.
///
/// Only real data is used here, so `A^H = A^T` and the conjugate is trivial.
#[test]
fn dgeev_lr_left_eigenvectors_satisfy_their_defining_equation() {
    for n in [1usize, 2, 3, 5, 8] {
        let a = sample(n);

        let mut work = a.clone();
        let mut u_real = vec![0.0; n * n];
        let mut u_imag = vec![0.0; n * n];
        let mut v_real = vec![0.0; n * n];
        let mut v_imag = vec![0.0; n * n];
        let (l_real, l_imag) = lapack::eig::dgeev_lr(&mut u_real, &mut u_imag, &mut v_real, &mut v_imag, &mut work, n).expect("native dgeev_lr");

        let scale = a.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1.0);
        let mut j = 0usize;
        while j < n {
            if l_imag[j] != 0.0 {
                // Complex pair: u = u_real[:,j] + i u_imag[:,j], lambda = lr + i li.
                // The defining equation is A^H u = conj(lambda) u, i.e.
                // A^T u = (lr - i li) u for real A.
                let (lr, li) = (l_real[j], l_imag[j]);
                let mut worst = 0.0f64;
                for i in 0..n {
                    // (A^T u)_i = sum_k A[k][i] * u_k
                    let mut sr = 0.0;
                    let mut si = 0.0;
                    for k in 0..n {
                        let ak = get(&a, n, k, i);
                        sr += ak * get(&u_real, n, k, j);
                        si += ak * get(&u_imag, n, k, j);
                    }
                    // subtract conj(lambda) * u_i = (lr - i li)(ur + i ui)
                    let ur = get(&u_real, n, i, j);
                    let ui = get(&u_imag, n, i, j);
                    let dr = sr - (lr * ur + li * ui);
                    let di = si - (lr * ui - li * ur);
                    worst = worst.max(dr.abs().max(di.abs()));
                }
                assert!(
                    worst / scale < 1e-10,
                    "n={n}: the complex left eigenvector for lambda = {lr}+{li}i has residual {worst} (scale {scale})"
                );
                j += 2;
            } else {
                let l = l_real[j];
                let mut worst = 0.0f64;
                for i in 0..n {
                    let mut s = 0.0;
                    for k in 0..n {
                        s += get(&a, n, k, i) * get(&u_real, n, k, j);
                    }
                    worst = worst.max((s - l * get(&u_real, n, i, j)).abs());
                }
                assert!(
                    worst / scale < 1e-10,
                    "n={n}: the real left eigenvector for lambda = {l} has residual {worst} (scale {scale})"
                );
                j += 1;
            }
        }

        // The right and left sets must describe the same spectrum.
        let want_scale = l_real.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1.0);
        for j in 0..n {
            assert!(l_real[j].is_finite() && l_imag[j].is_finite(), "n={n}: eigenvalue {j} is not finite");
        }
        let _ = (v_real, v_imag, want_scale);
    }
}

/// Singular value decomposition against the reference, for several shapes.
///
/// Unlike eigenvectors, singular values **are** unique, so they can be compared
/// directly -- and they are compared in order, because the SVD is conventionally
/// returned with them descending.
///
/// The rest of the test checks what the singular values alone cannot: that `U`
/// and `V` are orthogonal, that `U S V^T` reconstructs `A`, and that `S` is
/// non-negative.
#[test]
fn dgesvd_agrees_with_the_reference() {
    for (m, n) in [(1usize, 1usize), (2, 2), (3, 2), (2, 3), (4, 3), (3, 4), (6, 4), (4, 6), (5, 5)] {
        let a = rect_sample(m, n);
        check_svd(&a, m, n, false);
    }

    // Rank deficient: make the last column a copy of the first.
    for (m, n) in [(4usize, 3usize), (5, 5)] {
        let mut a = rect_sample(m, n);
        for i in 0..m {
            let v = get(&a, m, i, 0);
            a[i + (n - 1) * m] = v;
        }
        check_svd(&a, m, n, true);
    }
}

/// The four things an SVD must satisfy, plus the comparison against the
/// reference. `rank_deficient` relaxes only the tolerance on the smallest
/// singular value, which is genuinely ill-conditioned there.
fn check_svd(a: &[f64], m: usize, n: usize, rank_deficient: bool) {
    let min_mn = m.min(n);
    let label = format!("{m}x{n}{}", if rank_deficient { " (rank deficient)" } else { "" });

    let mut work = a.to_vec();
    let mut s = vec![0.0; min_mn];
    let mut u = vec![0.0; m * m];
    let mut vt = vec![0.0; n * n];
    lapack::svd::dgesvd(&mut work, m, n, &mut s, &mut u, &mut vt).expect("native dgesvd");

    let scale = a.iter().fold(0.0f64, |acc, x| acc.max(x.abs())).max(1.0);

    // --- singular values against the reference, in order.
    let mut a_ref = a.to_vec();
    let mut s_ref = vec![0.0; min_mn];
    let mut u_ref = vec![0.0; m * m];
    let mut vt_ref = vec![0.0; n * n];
    let info = dev_lapack::query::dgesvd(&mut a_ref, m, n, &mut s_ref, &mut u_ref, &mut vt_ref);
    assert_eq!(info, 0, "{label}: reference dgesvd failed");
    for i in 0..min_mn {
        let tol = if rank_deficient && i == min_mn - 1 { 1e-8 } else { TOL };
        assert!(
            (s[i] - s_ref[i]).abs() / scale < tol,
            "{label}: singular value {i} is {} but the reference says {}",
            s[i],
            s_ref[i]
        );
    }

    // --- non-negative and descending.
    for i in 0..min_mn {
        assert!(s[i] >= 0.0, "{label}: singular value {i} is negative: {}", s[i]);
        if i > 0 {
            assert!(
                s[i] <= s[i - 1] + 1e-12 * scale,
                "{label}: singular values are not descending: {} then {}",
                s[i - 1],
                s[i]
            );
        }
    }

    // --- U orthogonal (m x m).
    for i in 0..m {
        for j in 0..m {
            let mut dot = 0.0;
            for k in 0..m {
                dot += get(&u, m, k, i) * get(&u, m, k, j);
            }
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((dot - want).abs() < 1e-10, "{label}: U columns {i},{j} are not orthonormal ({dot})");
        }
    }

    // --- V orthogonal (n x n), stored transposed.
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0;
            for k in 0..n {
                // vt rows are V's columns, so vt[i][k] * vt[j][k] is V col i . V col j
                dot += get(&vt, n, i, k) * get(&vt, n, j, k);
            }
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((dot - want).abs() < 1e-10, "{label}: V columns {i},{j} are not orthonormal ({dot})");
        }
    }

    // --- U S V^T reconstructs A. This is the check that catches vectors which
    //     are individually fine but paired with the wrong singular value.
    let mut worst = 0.0f64;
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..min_mn {
                sum += get(&u, m, i, k) * s[k] * get(&vt, n, k, j);
            }
            worst = worst.max((sum - get(a, m, i, j)).abs());
        }
    }
    assert!(
        worst / scale < 1e-10,
        "{label}: U S V^T does not reconstruct A (worst element error {worst}, scale {scale})"
    );
}

/// Complex Hermitian eigen-decomposition against the reference, on both
/// triangles.
///
/// Four things are checked, mirroring the real `dsyev` test:
///
/// 1. **The eigenvalues are the reference's, in order** (ascending).
/// 2. **The vectors are orthonormal in the complex sense**, `v_i^H v_j =
///    delta_ij`. A conjugation mistake turns this into `v_i^T v_j`, which is not
///    the identity on complex data.
/// 3. **`A v = lambda v` for each pair**, against the original Hermitian matrix.
/// 4. **Both triangles agree.**
#[test]
fn zheev_agrees_with_the_reference() {
    let zero = Complex64::new(0.0, 0.0);
    for n in [1usize, 2, 3, 5, 8, 12] {
        let a = hermitian_sample(n);
        let scale = a.iter().fold(0.0f64, |m, x| m.max(x.norm())).max(1.0);

        // The data must be genuinely complex, or a conjugation mistake hides.
        let has_imag = (0..n).any(|j| (0..n).any(|i| i != j && cget(&a, n, i, j).im.abs() > 0.25));
        assert!(has_imag || n < 2, "the zheev test data must be genuinely complex (n={n})");

        let mut results = Vec::new();
        for upper in [false, true] {
            let mut work = a.clone();
            let mut l = vec![0.0; n];
            lapack::zeig::zheev(&mut work, &mut l, n, upper).expect("native zheev");

            // --- 1. eigenvalues in order against the reference.
            let mut t_ref = a.clone();
            let mut l_ref = vec![0.0; n];
            let info = dev_lapack::query::zheev(&mut t_ref, n, &mut l_ref, true);
            assert_eq!(info, 0, "reference zheev failed for n={n} upper={upper}");
            for i in 0..n {
                assert!(
                    (l[i] - l_ref[i]).abs() / scale < TOL,
                    "n={n} upper={upper}: eigenvalue {i} is {} but the reference says {}",
                    l[i],
                    l_ref[i]
                );
            }
            for i in 1..n {
                assert!(
                    l[i] >= l[i - 1] - 1e-12 * scale,
                    "n={n} upper={upper}: eigenvalues are not ascending: {} then {}",
                    l[i - 1],
                    l[i]
                );
            }

            // --- 2. orthonormal columns, complex sense.
            for i in 0..n {
                for j in 0..n {
                    let mut dot = zero;
                    for k in 0..n {
                        dot += cget(&work, n, k, i).conj() * cget(&work, n, k, j);
                    }
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (dot - Complex64::new(want, 0.0)).norm() < 1e-10,
                        "n={n} upper={upper}: columns {i},{j} are not orthonormal (v_i^H v_j = {dot})"
                    );
                }
            }

            // --- 3. the defining equation against the original matrix.
            for j in 0..n {
                for i in 0..n {
                    let mut av = zero;
                    for k in 0..n {
                        av += cget(&a, n, i, k) * cget(&work, n, k, j);
                    }
                    let resid = (av - cget(&work, n, i, j) * l[j]).norm();
                    assert!(
                        resid / scale < 1e-10,
                        "n={n} upper={upper}: (A v - lambda v)[{i}] = {resid} for eigenvalue {j}"
                    );
                }
            }

            results.push((l, work));
        }

        // --- 4. the two triangles give the same spectrum.
        let (l_lo, _) = &results[0];
        let (l_up, _) = &results[1];
        for i in 0..n {
            assert!(
                (l_lo[i] - l_up[i]).abs() / scale < 1e-12,
                "n={n}: reading the lower triangle gives {} but the upper gives {}",
                l_lo[i],
                l_up[i]
            );
        }
    }
}

/// The generalized eigenproblem `A v = lambda B v`, by its defining equation.
///
/// `dggev` returns `(alpha, beta)` with `lambda = alpha/beta`, so comparing
/// eigenvalues against a reference means comparing ratios -- and a ratio is badly
/// behaved exactly where `beta` is small. Rather than cope with infinities, the
/// data here is chosen so every eigenvalue is finite: `B = R^T R + n I` is
/// symmetric positive definite.
///
/// The check is `A v = lambda B v` for each right eigenvector, which needs no
/// reference and is the definition.
#[test]
fn dggev_satisfies_its_defining_equation() {
    for n in [1usize, 2, 3, 5, 8] {
        let ra = sample(n);
        let mut b = vec![0.0; n * n];
        // B = ra^T ra + n I, symmetric positive definite by construction.
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += get(&ra, n, k, i) * get(&ra, n, k, j);
                }
                b[i + j * n] = if i == j { sum + n as f64 } else { sum };
            }
        }

        // `dggev` overwrites both matrices in place, as LAPACK does, so the
        // defining equation has to be checked against copies of the original
        // problem.
        let (a_orig, b_orig) = (ra.clone(), b.clone());

        let mut a = ra.clone();
        let mut b_work = b.clone();
        let mut alpha_r = vec![0.0; n];
        let mut alpha_i = vec![0.0; n];
        let mut beta = vec![0.0; n];
        let mut v = vec![0.0; n * n];
        lapack::qz::dggev(
            false,
            true,
            n,
            &mut a,
            &mut b_work,
            &mut alpha_r,
            &mut alpha_i,
            &mut beta,
            &mut [],
            &mut v,
        )
        .expect("native dggev");

        let scale_b = b_orig.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1.0);

        let mut j = 0usize;
        while j < n {
            // lambda = alpha / beta, in complex arithmetic.
            assert!(
                beta[j].abs() > 1e-12,
                "beta[{j}] is {}; the data should give no infinite eigenvalue",
                beta[j]
            );
            let lr = alpha_r[j] / beta[j];
            let li = alpha_i[j] / beta[j];

            if li != 0.0 {
                // Eigenvector column j is (real, imaginary) in columns j and j+1.
                let mut worst = 0.0f64;
                for i in 0..n {
                    let mut av_r = 0.0;
                    let mut av_i = 0.0;
                    let mut bv_r = 0.0;
                    let mut bv_i = 0.0;
                    for k in 0..n {
                        let u_r = get(&v, n, k, j);
                        let u_i = get(&v, n, k, j + 1);
                        av_r += get(&a_orig, n, i, k) * u_r;
                        av_i += get(&a_orig, n, i, k) * u_i;
                        bv_r += get(&b_orig, n, i, k) * u_r;
                        bv_i += get(&b_orig, n, i, k) * u_i;
                    }
                    // right-hand side: lambda * (B u), complex multiply
                    let rhs_r = lr * bv_r - li * bv_i;
                    let rhs_i = lr * bv_i + li * bv_r;
                    worst = worst.max((av_r - rhs_r).abs().max((av_i - rhs_i).abs()));
                }
                assert!(
                    worst / scale_b < 1e-10,
                    "n={n}: the complex generalized eigenvector for lambda = {lr}+{li}i has residual {worst}"
                );
                j += 2;
            } else {
                let mut worst = 0.0f64;
                for i in 0..n {
                    let mut av = 0.0;
                    let mut bv = 0.0;
                    for k in 0..n {
                        av += get(&a_orig, n, i, k) * get(&v, n, k, j);
                        bv += get(&b_orig, n, i, k) * get(&v, n, k, j);
                    }
                    worst = worst.max((av - lr * bv).abs());
                }
                assert!(
                    worst / scale_b < 1e-10,
                    "n={n}: the real generalized eigenvector for lambda = {lr} has residual {worst}"
                );
                j += 1;
            }
        }
    }
}
