//! Permutation helpers.
//!
//! A permutation is a `[usize]` in which every index appears exactly once. The
//! routines here allocate their results and return [`SparseError`] rather than
//! panicking on malformed input.

#[cfg(test)]
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
#[cfg(test)]
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
}
