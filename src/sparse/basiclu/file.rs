//! Dynamic sparse line pools (`LuFileVal` and `LuFileIdx`).
//!
//! Ported from `ERGO-Code/HiGHS` (`highs/ipm/basiclu/lu_file.c` and
//! `lu_garbage_perm.c`, MIT). Each file stores `n` variable-length lines in
//! contiguous arrays, keeping non-empty lines in a circular doubly-linked
//! physical-order list around sentinel index `n`.

/// Sentinel value representing an absent line, node, or bucket link.
pub(crate) const NONE: usize = usize::MAX;

/// Shared metadata and payload operations for [`LuFileIdx`] and [`LuFileVal`].
trait LuFileOps {
    fn n_lines(&self) -> usize;
    fn cap(&self) -> usize;
    fn threshold(&self) -> f64;
    fn meta_parts(&mut self) -> (&mut [usize], &mut [usize], &mut [usize], &mut [usize], &mut usize, &mut usize);
    fn resize_storage(&mut self, new_cap: usize);
    fn copy_storage_within(&mut self, src_start: usize, src_end: usize, dst: usize);
    fn swap_storage(&mut self, a: usize, b: usize);
}

#[inline]
fn unlink_line(n: usize, flink: &mut [usize], blink: &mut [usize], line: usize) {
    let prev = blink[line];
    let next = flink[line];
    if prev != NONE && next != NONE {
        flink[prev] = next;
        blink[next] = prev;
    }
    let _ = n;
    flink[line] = NONE;
    blink[line] = NONE;
}

#[inline]
fn link_at_tail(n: usize, flink: &mut [usize], blink: &mut [usize], line: usize) {
    let prev = blink[n];
    flink[prev] = line;
    blink[line] = prev;
    flink[line] = n;
    blink[n] = line;
}

fn compress_core<F: LuFileOps>(store: &mut F) {
    let n = store.n_lines();
    let mut write_pos = 0usize;
    let mut cur = {
        let (_, _, flink, _, _, _) = store.meta_parts();
        flink[n]
    };

    while cur != n {
        let (next, b, e) = {
            let (begin, end, flink, _, _, _) = store.meta_parts();
            (flink[cur], begin[cur], end[cur])
        };
        let len = e - b;
        debug_assert!(len > 0);
        if b != write_pos {
            store.copy_storage_within(b, e, write_pos);
            let (begin, end, _, _, _, _) = store.meta_parts();
            begin[cur] = write_pos;
            end[cur] = write_pos + len;
        }
        write_pos += len;
        cur = next;
    }

    let (_, _, _, _, live_nnz, tail_used) = store.meta_parts();
    debug_assert_eq!(write_pos, *live_nnz);
    *tail_used = write_pos;
}

fn prepare_push<F: LuFileOps>(store: &mut F, line: usize, extra: usize) -> usize {
    let n = store.n_lines();
    let cap = store.cap();

    // Fast path: line is already linked and has physical slack before the next line.
    {
        let (begin, end, flink, _, live_nnz, tail_used) = store.meta_parts();
        let next = flink[line];
        if next != NONE {
            let next_begin = if next == n { cap } else { begin[next] };
            let end_k = end[line];
            if end_k < next_begin {
                end[line] = end_k + 1;
                *live_nnz += 1;
                if end_k + 1 > *tail_used {
                    *tail_used = end_k + 1;
                }
                return end_k;
            }
        }
    }

    // Slow path (`lu_file_reappend`): optionally compact dead space first.
    let threshold = store.threshold();
    {
        let (_, _, _, _, live_nnz, tail_used) = store.meta_parts();
        if *tail_used > 0 {
            let dead = *tail_used - *live_nnz;
            if (dead as f64) / (*tail_used as f64) > threshold {
                compress_core(store);
            }
        }
    }

    let (begin_k, end_k, flink_k, tail_used_val) = {
        let (begin, end, flink, _, _, tail_used) = store.meta_parts();
        (begin[line], end[line], flink[line], *tail_used)
    };
    let cap = store.cap();
    let len = end_k - begin_k;
    let elbow = 2 * (len + 1) + extra;

    // If `line` is already the physical last line, extend capacity in place without moving it.
    if flink_k == n {
        let needed_cap = begin_k + elbow;
        if cap < needed_cap {
            let new_cap = cap.saturating_mul(2).max(needed_cap);
            store.resize_storage(new_cap);
        }
        let (_, end, _, _, live_nnz, tail_used) = store.meta_parts();
        end[line] = end_k + 1;
        *live_nnz += 1;
        if end_k + 1 > *tail_used {
            *tail_used = end_k + 1;
        }
        return end_k;
    }

    // Otherwise unlink `line` and relocate it to `tail_used` with `elbow` slots.
    if flink_k != NONE {
        let (_, _, flink, blink, _, _) = store.meta_parts();
        unlink_line(n, flink, blink, line);
    }

    let new_begin = tail_used_val;
    let new_end = new_begin + len;
    let new_tail = new_begin + elbow;
    if cap < new_tail {
        let new_cap = cap.saturating_mul(2).max(new_tail);
        store.resize_storage(new_cap);
    }
    if len > 0 {
        store.copy_storage_within(begin_k, end_k, new_begin);
    }

    let (begin, end, flink, blink, live_nnz, tail_used) = store.meta_parts();
    begin[line] = new_begin;
    end[line] = new_end + 1;
    link_at_tail(n, flink, blink, line);
    *live_nnz += 1;
    *tail_used = new_tail;
    new_end
}

fn swap_remove_core<F: LuFileOps>(store: &mut F, line: usize, pos: usize) {
    let n = store.n_lines();
    let (idx_pos, last, became_empty) = {
        let (begin, end, _, _, _, _) = store.meta_parts();
        let b = begin[line];
        let e = end[line];
        debug_assert!(pos < e - b);
        let idx_pos = b + pos;
        let last = e - 1;
        (idx_pos, last, last == b)
    };

    if idx_pos != last {
        store.swap_storage(idx_pos, last);
    }

    let (begin, end, flink, blink, live_nnz, _) = store.meta_parts();
    end[line] = last;
    *live_nnz -= 1;
    if became_empty {
        unlink_line(n, flink, blink, line);
        begin[line] = 0;
        end[line] = 0;
    }
}

fn clear_line_core<F: LuFileOps>(store: &mut F, line: usize) {
    let n = store.n_lines();
    let (begin, end, flink, blink, live_nnz, _) = store.meta_parts();
    let len = end[line] - begin[line];
    *live_nnz -= len;
    if flink[line] != NONE {
        unlink_line(n, flink, blink, line);
    }
    begin[line] = 0;
    end[line] = 0;
}

/// Lays out `n` lines sequentially with `slack` trailing slots after each non-empty line,
/// returning a mutable cursor vector pointing to `begin[line]` for populating entries.
fn allocate_sequential_core<F: LuFileOps>(store: &mut F, counts: &[usize], slack: usize) -> Vec<usize> {
    let n = store.n_lines();
    debug_assert_eq!(counts.len(), n);
    let mut total = 0usize;
    let mut nnz = 0usize;
    for &c in counts {
        if c > 0 {
            total += c + slack;
            nnz += c;
        }
    }
    if store.cap() < total {
        store.resize_storage(total);
    }
    let (begin, end, flink, blink, live_nnz, tail_used) = store.meta_parts();
    begin.fill(0);
    end.fill(0);
    flink.fill(NONE);
    blink.fill(NONE);
    flink[n] = n;
    blink[n] = n;

    let mut cursors = vec![0usize; n];
    let mut pos = 0usize;
    for (line, &c) in counts.iter().enumerate() {
        if c > 0 {
            begin[line] = pos;
            end[line] = pos + c;
            cursors[line] = pos;
            link_at_tail(n, flink, blink, line);
            pos += c + slack;
        }
    }
    *live_nnz = nnz;
    *tail_used = total;
    cursors
}

/// Dynamic sparse line pool storing row/column indices and `f64` values (`lu_file.c`).
#[derive(Clone, Debug)]
pub(crate) struct LuFileVal {
    pub begin: Vec<usize>,
    pub end: Vec<usize>,
    pub flink: Vec<usize>,
    pub blink: Vec<usize>,
    pub idx: Vec<usize>,
    pub val: Vec<f64>,
    pub live_nnz: usize,
    pub tail_used: usize,
    compress_threshold: f64,
}

impl LuFileVal {
    pub(crate) fn new(n: usize, init_cap: usize, compress_threshold: f64) -> Self {
        let mut flink = vec![NONE; n + 1];
        let mut blink = vec![NONE; n + 1];
        flink[n] = n;
        blink[n] = n;
        Self {
            begin: vec![0; n + 1],
            end: vec![0; n + 1],
            flink,
            blink,
            idx: vec![0; init_cap],
            val: vec![0.0; init_cap],
            live_nnz: 0,
            tail_used: 0,
            compress_threshold,
        }
    }

    pub(crate) fn allocate_sequential(&mut self, counts: &[usize], slack: usize) -> Vec<usize> {
        allocate_sequential_core(self, counts, slack)
    }

    #[inline]
    pub(crate) fn len(&self, line: usize) -> usize {
        self.end[line] - self.begin[line]
    }

    #[inline]
    pub(crate) fn idx_slice(&self, line: usize) -> &[usize] {
        &self.idx[self.begin[line]..self.end[line]]
    }

    #[inline]
    pub(crate) fn val_slice(&self, line: usize) -> &[f64] {
        &self.val[self.begin[line]..self.end[line]]
    }

    #[inline]
    pub(crate) fn val_slice_mut(&mut self, line: usize) -> &mut [f64] {
        &mut self.val[self.begin[line]..self.end[line]]
    }

    pub(crate) fn push_with_slack(&mut self, line: usize, idx: usize, val: f64, extra: usize) {
        let pos = prepare_push(self, line, extra);
        self.idx[pos] = idx;
        self.val[pos] = val;
    }

    pub(crate) fn swap_remove_at(&mut self, line: usize, pos: usize) {
        swap_remove_core(self, line, pos);
    }

    pub(crate) fn remove_idx(&mut self, line: usize, target_idx: usize) -> Option<f64> {
        let pos = self.idx_slice(line).iter().position(|&x| x == target_idx)?;
        let val = self.val_slice(line)[pos];
        self.swap_remove_at(line, pos);
        Some(val)
    }

    pub(crate) fn clear_line(&mut self, line: usize) {
        clear_line_core(self, line);
    }

    #[cfg(test)]
    pub(crate) fn compress(&mut self) {
        compress_core(self);
    }

    pub(crate) fn find_val(&self, line: usize, target_idx: usize) -> Option<f64> {
        let pos = self.idx_slice(line).iter().position(|&x| x == target_idx)?;
        Some(self.val_slice(line)[pos])
    }
}

impl LuFileOps for LuFileVal {
    #[inline]
    fn n_lines(&self) -> usize {
        self.begin.len() - 1
    }

    #[inline]
    fn cap(&self) -> usize {
        self.idx.len()
    }

    #[inline]
    fn threshold(&self) -> f64 {
        self.compress_threshold
    }

    #[inline]
    fn meta_parts(&mut self) -> (&mut [usize], &mut [usize], &mut [usize], &mut [usize], &mut usize, &mut usize) {
        (
            &mut self.begin,
            &mut self.end,
            &mut self.flink,
            &mut self.blink,
            &mut self.live_nnz,
            &mut self.tail_used,
        )
    }

    fn resize_storage(&mut self, new_cap: usize) {
        self.idx.resize(new_cap, 0);
        self.val.resize(new_cap, 0.0);
    }

    fn copy_storage_within(&mut self, src_start: usize, src_end: usize, dst: usize) {
        self.idx.copy_within(src_start..src_end, dst);
        self.val.copy_within(src_start..src_end, dst);
    }

    fn swap_storage(&mut self, a: usize, b: usize) {
        self.idx.swap(a, b);
        self.val.swap(a, b);
    }
}

/// Dynamic sparse line pool storing structural indices only (`lu_file.c`).
#[derive(Clone, Debug)]
pub(crate) struct LuFileIdx {
    pub begin: Vec<usize>,
    pub end: Vec<usize>,
    pub flink: Vec<usize>,
    pub blink: Vec<usize>,
    pub idx: Vec<usize>,
    pub live_nnz: usize,
    pub tail_used: usize,
    compress_threshold: f64,
}

impl LuFileIdx {
    pub(crate) fn new(n: usize, init_cap: usize, compress_threshold: f64) -> Self {
        let mut flink = vec![NONE; n + 1];
        let mut blink = vec![NONE; n + 1];
        flink[n] = n;
        blink[n] = n;
        Self {
            begin: vec![0; n + 1],
            end: vec![0; n + 1],
            flink,
            blink,
            idx: vec![0; init_cap],
            live_nnz: 0,
            tail_used: 0,
            compress_threshold,
        }
    }

    pub(crate) fn allocate_sequential(&mut self, counts: &[usize], slack: usize) -> Vec<usize> {
        allocate_sequential_core(self, counts, slack)
    }

    #[inline]
    pub(crate) fn len(&self, line: usize) -> usize {
        self.end[line] - self.begin[line]
    }

    #[inline]
    pub(crate) fn idx_slice(&self, line: usize) -> &[usize] {
        &self.idx[self.begin[line]..self.end[line]]
    }

    pub(crate) fn push_with_slack(&mut self, line: usize, idx: usize, extra: usize) {
        let pos = prepare_push(self, line, extra);
        self.idx[pos] = idx;
    }

    pub(crate) fn swap_remove_at(&mut self, line: usize, pos: usize) {
        swap_remove_core(self, line, pos);
    }

    pub(crate) fn remove_idx(&mut self, line: usize, target_idx: usize) -> bool {
        if let Some(pos) = self.idx_slice(line).iter().position(|&x| x == target_idx) {
            self.swap_remove_at(line, pos);
            true
        } else {
            false
        }
    }

    pub(crate) fn clear_line(&mut self, line: usize) {
        clear_line_core(self, line);
    }

    #[cfg(test)]
    pub(crate) fn compress(&mut self) {
        compress_core(self);
    }
}

impl LuFileOps for LuFileIdx {
    #[inline]
    fn n_lines(&self) -> usize {
        self.begin.len() - 1
    }

    #[inline]
    fn cap(&self) -> usize {
        self.idx.len()
    }

    #[inline]
    fn threshold(&self) -> f64 {
        self.compress_threshold
    }

    #[inline]
    fn meta_parts(&mut self) -> (&mut [usize], &mut [usize], &mut [usize], &mut [usize], &mut usize, &mut usize) {
        (
            &mut self.begin,
            &mut self.end,
            &mut self.flink,
            &mut self.blink,
            &mut self.live_nnz,
            &mut self.tail_used,
        )
    }

    fn resize_storage(&mut self, new_cap: usize) {
        self.idx.resize(new_cap, 0);
    }

    fn copy_storage_within(&mut self, src_start: usize, src_end: usize, dst: usize) {
        self.idx.copy_within(src_start..src_end, dst);
    }

    fn swap_storage(&mut self, a: usize, b: usize) {
        self.idx.swap(a, b);
    }
}
