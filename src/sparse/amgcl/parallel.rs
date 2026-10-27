//! Deterministic shared-memory execution using only standard-library threads.

use crate::sparse::amgcl::{Error, Result};

/// Shared-memory execution parameters.
///
/// The default is serial. Parallel kernels use scoped threads and disjoint
/// chunks, so borrowed data remains valid and no worker outlives the operation.
/// WebAssembly targets always execute serially, even when `threads` exceeds one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Parallelism {
    /// Maximum number of worker threads, including the calling thread.
    pub threads: usize,
    /// Minimum vector length or matrix row count before threads are created.
    pub min_parallel_len: usize,
}

impl Parallelism {
    /// Uses the specified number of threads with the default size threshold.
    pub const fn new(threads: usize) -> Self {
        Self {
            threads,
            min_parallel_len: 4096,
        }
    }

    /// Uses all hardware threads reported by the standard library.
    ///
    /// This returns one thread on WebAssembly, where AMGCL is always serial.
    pub fn auto() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        #[cfg(target_arch = "wasm32")]
        let threads = 1;
        Self::new(threads)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.threads == 0 {
            return Err(Error::InvalidParameter("parallel thread count must be positive".into()));
        }
        Ok(())
    }

    pub(crate) fn workers(&self, len: usize) -> usize {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = self;
            let _ = len;
            1
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if len == 0 || len < self.min_parallel_len {
                1
            } else {
                self.threads.min(len).max(1)
            }
        }
    }
}

impl Default for Parallelism {
    fn default() -> Self {
        Self::new(1)
    }
}

fn bounds(len: usize, workers: usize, part: usize) -> (usize, usize) {
    let base = len / workers;
    let remainder = len % workers;
    let start = part * base + part.min(remainder);
    let end = start + base + usize::from(part < remainder);
    (start, end)
}

/// Applies `f` to disjoint mutable chunks. The first chunk runs on the caller.
pub(crate) fn for_each_mut<T, F>(values: &mut [T], params: Parallelism, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let workers = params.workers(values.len());
    if workers == 1 {
        f(0, values);
        return;
    }
    let len = values.len();
    // Split at computed bounds rather than using chunks_mut, whose final short
    // chunk would change the deterministic partition layout.
    let (_, first_end) = bounds(len, workers, 0);
    let (first, mut remaining) = values.split_at_mut(first_end);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers - 1);
        for part in 1..workers {
            let (start, end) = bounds(len, workers, part);
            let chunk_len = end - start;
            let (chunk, tail) = remaining.split_at_mut(chunk_len);
            remaining = tail;
            let function = &f;
            handles.push(scope.spawn(move || function(start, chunk)));
        }
        f(0, first);
        for handle in handles {
            if let Err(payload) = handle.join() {
                std::panic::resume_unwind(payload);
            }
        }
    });
}

/// Maps each deterministic partition and returns results in partition order.
pub(crate) fn map_chunks<T, F>(len: usize, params: Parallelism, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize, usize) -> T + Sync,
{
    let workers = params.workers(len);
    if workers == 1 {
        return vec![f(0, len)];
    }
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers - 1);
        for part in 1..workers {
            let (start, end) = bounds(len, workers, part);
            let function = &f;
            handles.push(scope.spawn(move || function(start, end)));
        }
        let (start, end) = bounds(len, workers, 0);
        let mut results = vec![f(start, end)];
        for handle in handles {
            match handle.join() {
                Ok(result) => results.push(result),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        results
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(target_arch = "wasm32"))]
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn partitions_cover_every_index_without_overlap() {
        for len in 0..67 {
            for workers in 1..10 {
                let workers = workers.min(len.max(1));
                let mut covered = vec![0; len];
                for part in 0..workers {
                    let (start, end) = bounds(len, workers, part);
                    for count in covered.iter_mut().take(end).skip(start) {
                        *count += 1;
                    }
                }
                assert!(covered.iter().all(|&count| count == 1));
            }
        }
    }

    #[test]
    fn mutable_chunks_preserve_global_offsets() {
        let mut values = vec![0; 1000];
        let params = Parallelism {
            threads: 4,
            min_parallel_len: 1,
        };
        for_each_mut(&mut values, params, |offset, chunk| {
            for (local, value) in chunk.iter_mut().enumerate() {
                *value = offset + local;
            }
        });
        assert!(values.iter().copied().eq(0..1000));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn scoped_workers_execute_concurrently() {
        static ACTIVE: AtomicUsize = AtomicUsize::new(0);
        static MAXIMUM: AtomicUsize = AtomicUsize::new(0);
        let params = Parallelism {
            threads: 4,
            min_parallel_len: 1,
        };
        let results = map_chunks(4, params, |_, _| {
            let active = ACTIVE.fetch_add(1, Ordering::SeqCst) + 1;
            MAXIMUM.fetch_max(active, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(20));
            ACTIVE.fetch_sub(1, Ordering::SeqCst);
            active
        });
        assert_eq!(results.len(), 4);
        assert!(MAXIMUM.load(Ordering::SeqCst) > 1);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn wasm_always_uses_serial_execution() {
        let params = Parallelism {
            threads: 4,
            min_parallel_len: 1,
        };
        assert_eq!(Parallelism::auto().threads, 1);
        assert_eq!(params.workers(100), 1);
        assert_eq!(map_chunks(100, params, |start, end| (start, end)), vec![(0, 100)]);
    }

    #[test]
    fn zero_threads_are_rejected() {
        assert!(Parallelism::new(0).validate().is_err());
    }
}
