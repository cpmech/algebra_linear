//! Permutation helpers.
//!
//! A permutation is a `[usize]` in which every index appears exactly once. The
//! routines here allocate their results and return [`SparseError`] rather than
//! panicking on malformed input.

use super::SparseError;

/// Returns `true` if every index in `0..perm.len()` appears exactly once.
///
/// The check allocates a `Vec<bool>`; it is not on a hot path.
pub fn is_permutation(perm: &[usize]) -> bool {
    let n = perm.len();
    let mut seen = vec![false; n];
    for &p in perm {
        if p >= n || seen[p] {
            return false;
        }
        seen[p] = true;
    }
    true
}

/// Returns the inverse of `perm`.
///
/// If `perm` maps a new position to an old index (`perm[new] = old`), the
/// inverse maps an old index to its new position (`inverse[old] = new`).
///
/// # Errors
///
/// [`SparseError::BadShape`] if `perm` is not a permutation.
pub fn invert(perm: &[usize]) -> Result<Vec<usize>, SparseError> {
    if !is_permutation(perm) {
        return Err(SparseError::BadShape);
    }
    let mut inverse = vec![0; perm.len()];
    for (new, &old) in perm.iter().enumerate() {
        inverse[old] = new;
    }
    Ok(inverse)
}

/// Composes two permutations: `compose(a, b)[i] = a[b[i]]`.
///
/// # Errors
///
/// [`SparseError::BadShape`] if the lengths differ or either argument is not a
/// permutation.
pub fn compose(a: &[usize], b: &[usize]) -> Result<Vec<usize>, SparseError> {
    if a.len() != b.len() || !is_permutation(a) || !is_permutation(b) {
        return Err(SparseError::BadShape);
    }
    Ok(b.iter().map(|&bi| a[bi]).collect())
}

/// Permutes `x` in place: `x[i] = old[perm[i]]`.
///
/// This is the direction the sparse solve uses for the row permutation, so it
/// matches the element order of a permuted right-hand side.
///
/// # Errors
///
/// [`SparseError::BadShape`] if `x.len() != perm.len()` or `perm` is not a
/// permutation.
pub fn permute<T: Copy>(x: &mut [T], perm: &[usize]) -> Result<(), SparseError> {
    if x.len() != perm.len() || !is_permutation(perm) {
        return Err(SparseError::BadShape);
    }
    let copy = x.to_vec();
    for (i, &p) in perm.iter().enumerate() {
        x[i] = copy[p];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_permutations() {
        assert!(is_permutation(&[]));
        assert!(is_permutation(&[0]));
        assert!(is_permutation(&[1, 0]));
        assert!(is_permutation(&[2, 0, 1]));
        assert!(!is_permutation(&[1, 1]));
        assert!(!is_permutation(&[0, 3]));
    }

    #[test]
    fn inverts_a_permutation() {
        let p = [2, 0, 1];
        let inv = invert(&p).unwrap();
        assert_eq!(inv, [1, 2, 0]);
        assert_eq!(invert(&[]).unwrap(), Vec::<usize>::new());
        assert_eq!(invert(&[0, 1, 1]).unwrap_err(), SparseError::BadShape);
    }

    #[test]
    fn composes_two_permutations() {
        let a = [2, 0, 1];
        let b = [1, 2, 0];
        assert_eq!(compose(&a, &b).unwrap(), [a[1], a[2], a[0]]);
        assert_eq!(compose(&a, &[0, 0, 1]).unwrap_err(), SparseError::BadShape);
        assert_eq!(compose(&a, &[0, 1]).unwrap_err(), SparseError::BadShape);
    }

    #[test]
    fn permutes_a_vector() {
        let mut x = [10, 20, 30];
        permute(&mut x, &[2, 0, 1]).unwrap();
        assert_eq!(x, [30, 10, 20]);
        assert_eq!(permute(&mut x, &[0, 1]).unwrap_err(), SparseError::BadShape);
    }
}
