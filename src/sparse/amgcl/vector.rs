use crate::sparse::amgcl::parallel::{Parallelism, for_each_mut, map_chunks};
use crate::sparse::amgcl::{Error, Result};

#[inline]
pub(crate) fn dot(x: &[f64], y: &[f64]) -> f64 {
    // Four accumulators reduce dependency chains without changing the API.
    let mut sums = [0.0; 4];
    let chunks = x.len() / 4;
    for i in 0..chunks {
        let base = 4 * i;
        for lane in 0..4 {
            sums[lane] += x[base + lane] * y[base + lane];
        }
    }
    for i in 4 * chunks..x.len() {
        sums[0] += x[i] * y[i];
    }
    (sums[0] + sums[1]) + (sums[2] + sums[3])
}

pub(crate) fn dot_with(x: &[f64], y: &[f64], params: Parallelism) -> f64 {
    map_chunks(x.len(), params, |start, end| dot(&x[start..end], &y[start..end]))
        .into_iter()
        .sum()
}

#[derive(Clone, Copy)]
struct NormPart {
    scale: f64,
    ssq: f64,
}

fn norm_part(x: &[f64]) -> NormPart {
    // Scaled sum of squares avoids unnecessary overflow/underflow.
    let mut scale = 0.0_f64;
    let mut ssq = 1.0_f64;
    for &value in x {
        let magnitude = value.abs();
        if magnitude == 0.0 {
            continue;
        }
        if scale < magnitude {
            let ratio = if scale == 0.0 { 0.0 } else { scale / magnitude };
            ssq = 1.0 + ssq * ratio * ratio;
            scale = magnitude;
        } else {
            let ratio = magnitude / scale;
            ssq += ratio * ratio;
        }
    }
    NormPart { scale, ssq }
}

fn finish_norm(part: NormPart) -> f64 {
    if part.scale == 0.0 { 0.0 } else { part.scale * part.ssq.sqrt() }
}

#[inline]
pub(crate) fn norm(x: &[f64]) -> f64 {
    finish_norm(norm_part(x))
}

pub(crate) fn norm_with(x: &[f64], params: Parallelism) -> f64 {
    let parts = map_chunks(x.len(), params, |start, end| norm_part(&x[start..end]));
    let mut combined = NormPart { scale: 0.0, ssq: 1.0 };
    for part in parts {
        if part.scale == 0.0 {
            continue;
        }
        if combined.scale < part.scale {
            let ratio = if combined.scale == 0.0 { 0.0 } else { combined.scale / part.scale };
            combined.ssq = part.ssq + combined.ssq * ratio * ratio;
            combined.scale = part.scale;
        } else {
            let ratio = part.scale / combined.scale;
            combined.ssq += part.ssq * ratio * ratio;
        }
    }
    finish_norm(combined)
}

pub(crate) fn axpy_with(alpha: f64, x: &[f64], y: &mut [f64], params: Parallelism) {
    for_each_mut(y, params, |offset, chunk| {
        let input = &x[offset..offset + chunk.len()];
        for (value, &input) in chunk.iter_mut().zip(input) {
            *value += alpha * input;
        }
    });
}

pub(crate) fn set_scaled_with(x: &[f64], alpha: f64, y: &mut [f64], params: Parallelism) {
    for_each_mut(y, params, |offset, chunk| {
        let input = &x[offset..offset + chunk.len()];
        for (value, &input) in chunk.iter_mut().zip(input) {
            *value = alpha * input;
        }
    });
}

pub(crate) fn check_finite_with(x: &[f64], params: Parallelism) -> Result<()> {
    let valid = map_chunks(x.len(), params, |start, end| x[start..end].iter().all(|value| value.is_finite()))
        .into_iter()
        .all(|valid| valid);
    if valid { Ok(()) } else { Err(Error::NonFinite) }
}
