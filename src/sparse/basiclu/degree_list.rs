//! Doubly-linked degree buckets for Markowitz pivot selection.
//!
//! Ported from `ERGO-Code/HiGHS` (`highs/ipm/basiclu/lu_list.h`, MIT).
//! Buckets cover degrees `0..=n` explicitly so entries whose active degree drops
//! to zero due to numerical cancellation remain valid until pruned.

use super::file::NONE;

/// Doubly-linked degree buckets covering `0..=n`.
#[derive(Clone, Debug)]
pub(crate) struct DegreeLists {
    /// First node in bucket `d` (`0..=n`), or [`NONE`] when empty.
    pub head: Vec<usize>,
    /// Previous node in the same bucket, or [`NONE`] if first/inactive.
    pub prev: Vec<usize>,
    /// Next node in the same bucket, or [`NONE`] if last/inactive.
    pub next: Vec<usize>,
    /// Current degree of `node` in `0..=n`, or [`NONE`] once removed.
    pub deg: Vec<usize>,
}

impl DegreeLists {
    /// Creates an empty bucket structure for `n` nodes with degrees in `0..=n`.
    pub(crate) fn new(n: usize) -> Self {
        Self {
            head: vec![NONE; n + 1],
            prev: vec![NONE; n],
            next: vec![NONE; n],
            deg: vec![NONE; n],
        }
    }

    /// Inserts `node` into bucket `d` (`0..=n`).
    #[inline]
    pub(crate) fn insert(&mut self, node: usize, d: usize) {
        debug_assert!(node < self.deg.len());
        debug_assert!(d < self.head.len());
        debug_assert_eq!(self.deg[node], NONE);
        let first = self.head[d];
        self.prev[node] = NONE;
        self.next[node] = first;
        if first != NONE {
            self.prev[first] = node;
        }
        self.head[d] = node;
        self.deg[node] = d;
    }

    /// Removes `node` from its current bucket and sets `deg[node] = NONE`.
    #[inline]
    pub(crate) fn remove(&mut self, node: usize) {
        let d = self.deg[node];
        if d == NONE {
            return;
        }
        let p = self.prev[node];
        let nx = self.next[node];
        if p != NONE {
            self.next[p] = nx;
        } else {
            self.head[d] = nx;
        }
        if nx != NONE {
            self.prev[nx] = p;
        }
        self.prev[node] = NONE;
        self.next[node] = NONE;
        self.deg[node] = NONE;
    }

    /// Moves an active `node` to bucket `new_d`.
    #[inline]
    pub(crate) fn move_to(&mut self, node: usize, new_d: usize) {
        if self.deg[node] == NONE {
            return;
        }
        self.remove(node);
        self.insert(node, new_d);
    }

    /// Increments the degree of an active `node` by `1`.
    #[inline]
    pub(crate) fn increment(&mut self, node: usize) {
        let d = self.deg[node];
        if d != NONE {
            self.move_to(node, d + 1);
        }
    }

    /// Decrements the degree of an active `node` by `1` (moving to bucket `0` if `d == 1`).
    #[inline]
    pub(crate) fn decrement(&mut self, node: usize) {
        let d = self.deg[node];
        if d != NONE {
            debug_assert!(d > 0);
            self.move_to(node, d - 1);
        }
    }

    /// Pops and removes the first node from bucket `d`, if non-empty.
    #[inline]
    pub(crate) fn pop_bucket(&mut self, d: usize) -> Option<usize> {
        let node = self.head[d];
        if node == NONE {
            None
        } else {
            self.remove(node);
            Some(node)
        }
    }
}
