# Implementation Plan: Porting BasicLU to Pure Safe Rust in `algebra_linear`

## 1. Project Metadata, Role in `algebra_linear`, & Licensing

- **Target Integration:** Internal submodule `algebra_linear::sparse::basiclu` (`src/sparse/basiclu/`, re-exported via `pub mod basiclu;` in `src/sparse/mod.rs`), paralleling `src/sparse/amgcl/` and [`AMGCL_PORTING.md`](../AMGCL_PORTING.md). This is **not** a standalone crate.
- **Role Alongside Existing Solvers:**
  - `SparseLu` (`src/sparse/`): static COLAMD ordering + partial pivoting + simplicial/supernodal BLAS-3 LU (real `f64` and complex `Complex64`), optimized for fixed PDE/FEM sparsity patterns.
  - `sparse::amgcl`: real (`f64`) CSR algebraic multigrid and Krylov iterative solvers.
  - `sparse::qdldl`: real (`f64`) upper-triangular CSC unpivoted $L D L^T$ for SPD and SQD KKT systems.
  - `sparse::basiclu`: real (`f64`) general unsymmetric CSC sparse LU with triangular singleton peeling, dynamic threshold Markowitz pivoting, rank-deficiency detection/completion, hyper-sparse Gilbert–Peierls FTRAN/BTRAN solves, and Forrest–Tomlin rank-1 basis column updates — designed for LP Simplex bases and interior-point crossover (IPX):
    $$P_r^{(0)} A P_c^{(0)} = L R_1^{-1} \cdots R_t^{-1} U$$
- **Upstream Reference:** `ERGO-Code/HiGHS` under **`highs/ipm/basiclu/`** (moved from `src/ipm/basiclu/` in HiGHS v1.7+; originally standalone `ERGO-Code/basiclu` by **Lukas Schork and Jacek Gondzio**). Pin the exact HiGHS release tag and commit SHA before transcription begins, and cite the exact upstream `.c`/`.h` file and routine name in every function comment (`AGENTS.md`).
- **Upstream & Target Licensing:**
  - Upstream BasicLU / HiGHS is distributed under the **MIT License** (verify the exact verbatim copyright notice from the pinned HiGHS commit, e.g., `Copyright (c) 2018-2021 ERGO-Code` / HiGHS MIT license text).
  - **Fully compatible with `algebra_linear`'s `BSD-3-Clause` license:** Follow the exact precedent established in `LICENSE` for MIT-licensed `faer` and `AMGCL` — keep `Cargo.toml` at `license = "BSD-3-Clause"` and append BasicLU's verbatim MIT notice, upstream URL, and pinned commit SHA to `LICENSE`.
- **Explicit Scope Boundaries:**
  - **In v1 scope:** Input CSC validation, triangular singleton peeling, dynamic `LuFile` storage with garbage collection (`compress`), column-oriented bump setup and Zlatev/Markowitz threshold pivoting, Schur complement elimination with cancellation and fill-in management, dual-format `L`/`LT` and mutable dual-format `U`/`UT` factor construction, opt-in rank-deficient unit-column completion (`allow_singular`), dense and iterative Gilbert–Peierls hyper-sparse FTRAN/BTRAN solves with automatic `sparse_threshold` switching, and `solve_for_update` + Forrest–Tomlin (plus Schork–Gondzio pure-permutation) rank-1 column updates with stability monitoring and rollback.
  - **Deferred (optional Phase 7 diagnostics):** `lu_matrix_norm.c`, `lu_condest.c`, `lu_residual_test.c`, and `basiclu_obj_maxvolume.c` are deferred unless downstream LP/IPX callers require them.
- **Engineering Constraints (`AGENTS.md`, `Cargo.toml`, `.github/workflows/ci.yml`):**
  - `#![forbid(unsafe_code)]` in `src/sparse/basiclu/mod.rs` (matching `src/sparse/amgcl/mod.rs`).
  - `#![warn(missing_docs)]` (inherited from `src/lib.rs`) — document all public structs, fields, methods, variants, and errors for `cargo doc --no-deps`.
  - **Zero new dependencies** (`Cargo.toml` allows only `num-complex`, `web-time` on `wasm32-unknown-unknown`, and `wasm-bindgen-test` as a `wasm32` dev-dependency; CI rejects `path =` dependencies).
  - **0-based `usize` indices and `f64` scalars** (`AGENTS.md`).
  - **No workspace arguments and no manual reallocation retry loops in the public API:** Upstream C's manual `BASICLU_REALLOCATE` status/retry loop over flat `istore`/`xstore` buffers is completely eliminated; `BasicLu` owns dynamic `Vec`-backed `LuFile` pools and solve scratch internally, while `LuFile::compress()` keeps internal memory bounded.
  - **Full WebAssembly compatibility (`wasm32-unknown-unknown` and `wasm32-wasip1`):** All graph traversals (`lu_dfs`) MUST use an **explicit iterative stack** (never recursion, which risks stack overflow on large triangular systems under WASI/browser stacks). Any timing statistics use the `web_time::Instant` cfg shim from `src/sparse/sparse_lu.rs`.
  - **American English everywhere** (`AGENTS.md`), Rust edition 2024, `rustfmt.toml` (`max_width = 150`), `cargo clippy --all-targets` warning-free, and zero developer filesystem paths in source.

---

## 2. Complete Upstream C to `algebra_linear` Module Mapping

Upstream `highs/ipm/basiclu/` separates the low-level `lu_*` core from the `basiclu_*` / `basiclu_obj_*` API wrappers. Note that `lu_list.h` is header-only (there is no separate list `.c` file) and the object-oriented wrappers live in `basiclu_obj_*.c`.

| Upstream C File (`highs/ipm/basiclu/`) | Target Module (`src/sparse/basiclu/`) | Core Responsibilities |
| :--- | :--- | :--- |
| `basiclu.h`, `lu_def.h`, `lu_internal.h`, `lu_internal.c`, `lu_initialize.c`, `basiclu_initialize.c`, `basiclu_obj_initialize.c`, `basiclu_obj_free.c` | `mod.rs`, `params.rs`, `error.rs` | Public `BasicLu` handle, `BasicLuParams`, `BasicLuStats`, `Error`, `Result<T>`, internal buffer management. |
| `lu_file.h`, `lu_file.c`, `lu_garbage_perm.c` | `file.rs` | `LuFileVal` & `LuFileIdx` dynamic sparse line pools (`flink`/`blink` physical ordering, `reappend`, `compress`). |
| `lu_list.h` | `degree_list.rs` | $O(1)$ doubly-linked degree buckets covering `0..=n` (including degree-0 tracking). |
| `lu_singletons.c` | `singletons.rs` | Column and row singleton triangular peeling directly on caller CSC prior to bump setup. |
| `lu_setup_bump.c`, `lu_markowitz.c` | `markowitz.rs` | Column-oriented `ActiveBump` setup, dead-line pruning, Zlatev/Markowitz search with columnwise threshold pivoting. |
| `lu_pivot.c`, `lu_factorize_bump.c`, `basiclu_factorize.c`, `basiclu_obj_factorize.c` | `eliminate.rs` | Column-oriented rank-1 Schur complement update, cancellation removal in both files, fill-in, `col_max` updates. |
| `lu_build_factors.c`, `basiclu_get_factors.c`, `basiclu_obj_get_factors.c` | `factors.rs` | Immutable dual `L`/`LT` (CSC+CSR), mutable dual `U`/`UT` (`LuFileVal` + `pivot` + orderings), rank-deficient completion. |
| `lu_dfs.c`, `lu_solve_symbolic.c`, `lu_solve_triangular.c`, `lu_solve_dense.c`, `lu_solve_sparse.c`, `basiclu_solve_dense.c`, `basiclu_solve_sparse.c` | `solve.rs` | Dense and iterative Gilbert–Peierls hyper-sparse FTRAN/BTRAN with `sparse_threshold` switching and `EtaStorage`. |
| `lu_solve_for_update.c`, `lu_update.c`, `basiclu_solve_for_update.c`, `basiclu_update.c` | `update.rs` | Spike computation through $L$ and $R_1 \dots R_t$, Schork–Gondzio pure-permutation shortcut, Forrest–Tomlin row eta updates, rollback on breakdown. |
| *(Deferred)* `lu_matrix_norm.c`, `lu_condest.c`, `lu_residual_test.c`, `basiclu_obj_maxvolume.c` | *(Optional later phase)* | Matrix norm, Hager–Higham condition estimator, residual check, max-volume basis utilities. |

---

## 3. Repository Integration & Module Layout

```text
src/
├── lib.rs                          # Update crate docs to mention `sparse::basiclu`
└── sparse/
    ├── mod.rs                      # Add `pub mod basiclu;`
    └── basiclu/
        ├── mod.rs                  # `#![forbid(unsafe_code)]`, `BasicLu`, re-exports `BasicLuParams`, `BasicLuStats`, `Error`, `Result`
        ├── params.rs               # `BasicLuParams` & `BasicLuStats`
        ├── error.rs                # `Error` & `Result<T>` (`Display`, `std::error::Error`)
        ├── file.rs                 # `LuFileVal` & `LuFileIdx` dynamic line pools + garbage collection
        ├── degree_list.rs          # `DegreeLists` doubly-linked buckets for degrees `0..=n`
        ├── singletons.rs           # Column & row singleton peeling pre-pass
        ├── markowitz.rs            # `ActiveBump`, `col_max`, and Markowitz threshold pivot search
        ├── eliminate.rs            # Column-oriented Schur complement elimination & fill-in
        ├── factors.rs              # `LuFactors`: dual `L`/`LT`, mutable dual `U`/`UT`, `pivot`, `EtaStorage`
        ├── solve.rs                # Dense + iterative hyper-sparse FTRAN & BTRAN
        ├── update.rs               # Forrest–Tomlin & pure-permutation rank-1 column replacement
        └── tests.rs                # `#[cfg(test)]` unit, LP-basis, hyper-sparse, 50-update, & `bench_basiclu`
tests/
└── wasm_browser.rs                 # Add `#[wasm_bindgen_test] fn basiclu_runs_in_the_browser_target()`
```

Additional repository files touched upon implementation:
- `Cargo.toml`: add `port-basiclu/` to `exclude` (or remove the folder once implementation lands).
- `LICENSE`: append BasicLU's verbatim MIT copyright notice, repository URL, and pinned HiGHS commit SHA.
- `README.md`: document `sparse::basiclu` alongside `SparseLu`, `sparse::amgcl`, and `sparse::qdldl`.
- `AGENTS.md`: record any genuinely unreachable coverage lines in the coverage table in the same PR.

---

## 4. Data Structures, Storage Orientation, & Public API Specifications

### 4.1 Module-Local Error Type (`src/sparse/basiclu/error.rs`)
Because `SparseError` in `src/sparse/mod.rs` is not marked `#[non_exhaustive]`, adding BasicLU-specific variants (`SingularUpdate`, `MaxUpdatesReached`, `InvalidColumn`) to `SparseError` would be semver-breaking. Following `sparse::amgcl::Error`, define a module-local error type:

```rust
/// Errors reported by `sparse::basiclu`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Dimension mismatch in matrix arrays or right-hand-side / update slices.
    BadShape,
    /// `col_ptr` does not start at 0, is not non-decreasing, or a row index is `>= n`.
    InvalidStructure,
    /// A column's row indices are not strictly increasing.
    UnsortedRows,
    /// A column contains duplicate row indices.
    DuplicateEntry,
    /// An input matrix entry or update entry is NaN or infinite.
    NotFinite,
    /// Factorization stopped with rank `rank < n` while `allow_singular` is `false`.
    Singular { rank: usize, n: usize },
    /// Rank-1 column update produced a numerically singular diagonal pivot (`|alpha| < abs_pivot_tol`)
    /// or structural rank deficiency; prior factors remain intact.
    SingularUpdate,
    /// Number of updates since the last factorization reached `BasicLuParams::max_updates`.
    MaxUpdatesReached,
    /// Replacement column index `col` is outside `0..n`.
    InvalidColumn { col: usize, n: usize },
    /// Solve or update was called before a successful `factorize`.
    NotFactorized,
}

pub type Result<T> = core::result::Result<T, Error>;
```

### 4.2 Distinct Tolerances, Parameters, & Statistics (`src/sparse/basiclu/params.rs`)
Upstream BasicLU strictly separates three tolerances that must **not** be conflated:
- `abs_pivot_tol` (`abstol`, default `1e-14`): minimum acceptable pivot magnitude (enforced in singleton peeling, Markowitz search, and Forrest–Tomlin updates).
- `rel_pivot_tol` (`reltol`, default `0.1`): column-wise threshold partial pivoting factor (`|a_ij| >= rel_pivot_tol * col_max[j]`).
- `drop_tol` (`droptol`, default `1e-20`): numerical drop tolerance below which near-zero entries are pruned as cancelled.

```rust
/// Tuning parameters for [`BasicLu`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BasicLuParams {
    /// Absolute minimum pivot magnitude (`BASICLU_ABS_PIVOT_TOLERANCE`). Default: `1e-14`.
    pub abs_pivot_tol: f64,
    /// Column-wise threshold partial pivoting factor in `(0.0, 1.0]` (`BASICLU_REL_PIVOT_TOLERANCE`). Default: `0.1`.
    pub rel_pivot_tol: f64,
    /// Drop tolerance for zero / cancelled entries (`BASICLU_DROP_TOLERANCE`). Default: `1e-20`.
    pub drop_tol: f64,
    /// Switch hyper-sparse solves to dense mode once reached fraction exceeds this (`0.0..=1.0`). Default: `0.05`.
    pub sparse_threshold: f64,
    /// Maximum number of rows inspected in Markowitz bucket search once a valid pivot is found. Default: `4`.
    pub search_rows: usize,
    /// Maximum number of column-replacement updates allowed before refactorization is required. Default: `100`.
    pub max_updates: usize,
    /// Dead-space fraction triggering `LuFile::compress()` compaction. Default: `0.5`.
    pub compress_threshold: f64,
    /// If `true`, rank-deficient matrices (`rank < n`) are completed with unit columns so LP basis
    /// repair can inspect dependent columns; if `false`, `factorize` returns `Err(Error::Singular)`.
    pub allow_singular: bool,
}

/// Structural, rank, and update statistics of the active factorization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BasicLuStats {
    pub n: usize,
    pub rank: usize,
    pub nnz_a: usize,
    pub nnz_l: usize,
    pub nnz_u: usize,
    pub nnz_r: usize,
    pub bump_size: usize,
    pub updates: usize,
    pub forrest_tomlin_updates: usize,
    /// Estimated relative pivot discrepancy recorded during updates (FTRAN vs BTRAN pivot check).
    pub pivot_error: f64,
}
```

### 4.3 Dynamic Sparse Line Pool (`LuFile` in `src/sparse/basiclu/file.rs`)
Replaces upstream's offset macros into untyped `istore`/`xstore` arrays with two strongly typed structs: `LuFileVal` (storing `idx: Vec<usize>` + `val: Vec<f64>`) and `LuFileIdx` (storing `idx: Vec<usize>` only).

```rust
pub(crate) const NONE: usize = usize::MAX;

/// Dynamic pool of variable-length sparse lines (rows or columns) supporting
/// O(1) append within slack capacity, tail relocation with elbow room, and
/// linear-time physical-order garbage collection (`lu_file.c`, `lu_garbage_perm.c`).
#[derive(Clone, Debug)]
pub(crate) struct LuFileVal {
    pub begin: Vec<usize>,  // start offset of line k
    pub end: Vec<usize>,    // exclusive end offset of line k (len = end[k] - begin[k])
    pub flink: Vec<usize>,  // next line in physical memory order (or head sentinel)
    pub blink: Vec<usize>,  // previous line in physical memory order
    pub idx: Vec<usize>,
    pub val: Vec<f64>,
    pub live_nnz: usize,    // sum of active entry counts across all lines
    pub tail_used: usize,   // high-water mark in `idx`/`val`
}
```
(`LuFileIdx` has the identical metadata without `val: Vec<f64>`.)

**Invariants:**
- Each line sits in a doubly-linked physical-order list (`flink`/`blink`), so the slack after line `k` is `next_begin - end[k]` (where `next_begin = begin[flink[k]]`, or `idx.len()` if `k` is last).
- Appending an entry into line `k` takes $O(1)$ while `end[k] < next_begin`.
- Removing an entry from line `k` swaps it with `end[k] - 1` and decrements `end[k]` in $O(1)$!
- When `end[k] == next_begin` (`reappend`), line `k` is unlinked from its physical neighbors, relocated to `tail_used` with `2 * (len + 1) + extra` elbow room (growing `idx`/`val` if needed, or triggering `compress()` first when dead space `(tail_used - live_nnz) / tail_used > compress_threshold`), and relinked at the tail.
- `compress()` walks lines in `flink` physical order (always moving left, never overlapping!) in $O(\text{live\_nnz})$ time with zero auxiliary heap allocations.
- **Safety invariant:** Any caller mutating lines during elimination or updates MUST copy the read-only pivot/spike line into scratch vectors *before* modifying active lines so a `reappend()`/`compress()` cannot invalidate borrowed offsets.

### 4.4 Column-Oriented `ActiveBump` & `DegreeLists` (`markowitz.rs`, `degree_list.rs`)
Upstream BasicLU stores **numerical values in columns** (`cols: LuFileVal`) and **structural indices only in rows** (`rows: LuFileIdx`) inside the active bump:

```rust
#[derive(Clone, Debug)]
pub(crate) struct ActiveBump {
    pub n: usize,
    pub cols: LuFileVal,   // active columns with indices and numerical values
    pub rows: LuFileIdx,   // active rows with column indices only (no duplicate floats)
    pub col_max: Vec<f64>, // current max |a_ij| per active column (maintained after each update)
}
```
**Why values live in columns (NOT rows):**
- Threshold partial pivoting in Markowitz search tests `|a_ij| >= rel_pivot_tol * col_max[j]`, guaranteeing every multiplier $L_{r, k} = a_{r, p\_col} / a_{p\_row, p\_col}$ satisfies $|L_{r, k}| \le 1 / \text{rel\_pivot\_tol}$.
- Column-oriented Gaussian elimination (scattering column `c` and subtracting multiples of the scaled pivot column `L[:, k]`) touches only column values and updates `col_max[c]` in the exact same pass.
- (Storing values only in rows while performing column-threshold checks or column division in singletons/elimination would force expensive $O(\text{deg})$ row scans or fail altogether.)

**Markowitz `DegreeLists` (`src/sparse/basiclu/degree_list.rs`):**
```rust
#[derive(Clone, Debug)]
pub(crate) struct DegreeLists {
    pub head: Vec<usize>, // len == n + 1; head[d] = first active node with degree d in 0..=n
    pub prev: Vec<usize>, // len == n
    pub next: Vec<usize>, // len == n
    pub deg: Vec<usize>,  // len == n; current degree in 0..=n, or NONE once eliminated/removed
}
```
Bucket `0` is explicitly present so numerical cancellation dropping a line's degree to `0` moves it cleanly into bucket `0` instead of panicking or corrupting `head`.

### 4.5 Dual-Format Finalized Factors & Update Storage (`src/sparse/basiclu/factors.rs`)
To support **both** hyper-sparse FTRAN and hyper-sparse BTRAN **and** $O(\text{nnz}(v) + \text{nnz}(\text{row } p))$ Forrest–Tomlin updates, `LuFactors` stores dual orientations (matching `lu_build_factors.c`):

```rust
#[derive(Clone, Debug)]
pub(crate) struct EtaStorage {
    pub eta_pivot_row: Vec<usize>, // row ID modified by update s in 0..t
    pub eta_ptr: Vec<usize>,       // len == t + 1
    pub eta_idx: Vec<usize>,       // original/stable column-position IDs
    pub eta_val: Vec<f64>,         // row elimination weights w_k
}

#[derive(Clone, Debug)]
pub(crate) struct LuFactors {
    pub n: usize,
    pub rank: usize,

    // Initial factorization row permutation (fixed across Forrest-Tomlin updates):
    // Permutes original matrix rows into L's elimination step order.
    pub p_row_l: Vec<usize>,     // L step -> original row
    pub p_row_l_inv: Vec<usize>, // original row -> L step

    // Immutable unit lower-triangular L in dual form (unit diagonal implicit):
    pub l_col_ptr: Vec<usize>,   // CSC for FTRAN & DFS reachability
    pub l_row_idx: Vec<usize>,
    pub l_val: Vec<f64>,
    pub lt_row_ptr: Vec<usize>,  // CSR (columns of L^T) for hyper-sparse BTRAN & DFS
    pub lt_col_idx: Vec<usize>,
    pub lt_val: Vec<f64>,

    // Mutable upper-triangular U in dual LuFile storage (strictly off-diagonal entries,
    // indexed by stable IDs, ordered dynamically by `u_row_order` / `u_col_order`):
    pub u_col: LuFileVal,        // for FTRAN & DFS
    pub u_row: LuFileVal,        // for BTRAN, DFS on U^T, & Forrest-Tomlin row deletion
    pub pivot: Vec<f64>,         // diagonal pivot value indexed by stable ID (no separate `u_diag` mystery)

    // Current triangular ordering of U (mutated in O(m - p) on Forrest-Tomlin updates):
    pub u_row_order: Vec<usize>, // triangular step -> stable row ID
    pub u_row_pos: Vec<usize>,   // stable row ID -> triangular step
    pub u_col_order: Vec<usize>, // triangular step -> original col ID
    pub u_col_pos: Vec<usize>,   // original col ID -> triangular step

    // Forrest-Tomlin row eta factors R_1, ..., R_t:
    pub etas: EtaStorage,
}
```
**Key architectural insight:** `L`, `LT`, `U` (`u_col`/`u_row`), and earlier etas `R_s` store **stable IDs** while `u_row_order`/`u_row_pos` and `u_col_order`/`u_col_pos` maintain the current upper-triangular ordering. Consequently, a Forrest–Tomlin cyclic shift `p -> m` updates only those ordering arrays and **never renumbers indices inside `L`, `LT`, `U`, or prior `R_s`**!

### 4.6 Public Solver Handle (`src/sparse/basiclu/mod.rs`)

```rust
pub struct BasicLu {
    params: BasicLuParams,
    factors: Option<LuFactors>,
    stats: Option<BasicLuStats>,
    work: SolveWorkspace, // dense f64 accumulators, u32 generation markers, iterative DFS stack
}
```

**Public Methods (`&mut self` on solves reuses owned `SolveWorkspace` with zero heap allocations, mirroring `Amg` in `AMGCL_PORTING.md`):**
- `pub fn new(params: BasicLuParams) -> Self` and `impl Default for BasicLu` (`BasicLuParams::default()`).
- `pub fn factorize(&mut self, col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> Result<()>`
  - Validates square CSC (`CscView` contract: strict ascending rows, `col_ptr[0] == 0`, `col_ptr[n] == row_idx.len()`, finiteness of `values`).
- `pub fn solve(&mut self, b: &mut [f64]) -> Result<()>` — dense FTRAN $A x = b$.
- `pub fn solve_transpose(&mut self, b: &mut [f64]) -> Result<()>` — dense BTRAN $A^T x = b$.
- `pub fn solve_sparse(&mut self, rhs_idx: &[usize], rhs_val: &[f64], out_idx: &mut Vec<usize>, out_val: &mut Vec<f64>) -> Result<()>` — hyper-sparse FTRAN with automatic dense fallback above `params.sparse_threshold`.
- `pub fn solve_sparse_transpose(&mut self, rhs_idx: &[usize], rhs_val: &[f64], out_idx: &mut Vec<usize>, out_val: &mut Vec<f64>) -> Result<()>` — hyper-sparse BTRAN with automatic dense fallback above `params.sparse_threshold`.
- `pub fn update(&mut self, col: usize, new_row_idx: &[usize], new_val: &[f64]) -> Result<()>` — rank-1 basis column replacement (validates `new_row_idx`/`new_val` finiteness and bounds before mutating factors).
- *(Optional advanced simplex pair)* `pub fn solve_for_update(&mut self, ...)` + `pub fn apply_update(&mut self, ...)` so revised-simplex callers that already solved the FTRAN spike column and BTRAN pivot row avoid repeating work.
- `pub fn stats(&self) -> Option<&BasicLuStats>` and `pub fn params(&self) -> &BasicLuParams`.

---

## 5. Exact Step-by-Step Algorithm Specification

### Phase 1: Input Validation & Triangular Singleton Peeling (`singletons.rs`, `lu_singletons.c`)

1. Handle `n == 0` cleanly (`rank = 0`, empty factors, `solve(&mut [])` succeeds).
2. Validate caller CSC (`col_ptr.len() >= 1`, `n = col_ptr.len() - 1`, strict increasing rows per column, finite `values`).
3. Peel directly on the caller's CSC structure **before** allocating/populating `ActiveBump` (never copy triangular border entries into the bump workspace).
4. Compute initial row counts `row_deg[0..n]` and column counts `col_deg[0..n]` (ignoring entries with `|v| <= params.drop_tol`), maintaining two queues: `col_singletons` (`col_deg[j] == 1`) and `row_singletons` (`row_deg[i] == 1`):
   - **Step 1a — Peel all column singletons first (into positions `0..n_cs`):**
     - Pop column `j` with `col_deg[j] == 1`, locate its sole active row `i` and value `piv = a[i, j]`.
     - If `piv.abs() < params.abs_pivot_tol`, defer `(i, j)` out of singleton elimination (it cannot serve as a numerically valid pivot).
     - Record `(i, j)` at current front step; mark row `i` and column `j` eliminated.
     - Remaining active entries in row `i` belong directly to row `i` of $U$ (zero divisions, zero fill-in, because column `j` has no other active rows!). Save them in temporary original-index singleton $U$ buffers.
     - Decrement `col_deg[c]` for every other active column `c` appearing in row `i`; if `col_deg[c] == 1`, push `c` onto `col_singletons`. (Note: peeling a column singleton deletes one row and one column; any newly exposed `col_deg == 1` stays in `col_singletons`, while newly exposed `row_deg == 1` only occurs if a row's remaining entry is checked afterwards.)
   - **Step 1b — Peel row singletons (into positions `n_cs..n_cs + n_rs`):**
     - Scan remaining uneliminated rows to populate `row_singletons` (`row_deg[i] == 1`, where deleting column-singleton rows cannot have left any unpeeled column singletons with `col_deg == 1` unless deferred).
     - Pop row `i`, locate its sole active column `j` and value `piv = a[i, j]`.
     - Check stability: `piv.abs() >= params.abs_pivot_tol` (and verify column pivot quality if required; otherwise defer to bump).
     - Record `(i, j)` at current step; mark row `i` and column `j` eliminated.
     - Remaining active entries `(r, val)` in column `j` are divided by `piv` (`val / piv`) and stored into temporary original-index $L$ buffers (keeping entries `> drop_tol`).
     - Decrement `row_deg[r]` for each such `r`; if `row_deg[r] == 1`, push `r` onto `row_singletons`.
5. **Exact 3×3 decoupling proof:** Placing column singletons first (`1`), row singletons second (`2`), and the remaining square bump `B` last (`3`) yields:
   $$P_r A P_c = \begin{pmatrix} U_{11} & A_{12} & A_{13} \\ 0 & L_{22} D_2 & 0 \\ 0 & A_{32} & B \end{pmatrix} = \begin{pmatrix} I & 0 & 0 \\ 0 & L_{22} & 0 \\ 0 & A_{32} D_2^{-1} & L_{33} \end{pmatrix} \begin{pmatrix} U_{11} & A_{12} & A_{13} \\ 0 & D_2 & 0 \\ 0 & 0 & U_{33} \end{pmatrix}$$
   Row singletons (`2`) have $A_{23} = 0$, so **no entries couple `2` to `B` in $U$** and $A_{32} D_2^{-1}$ enters $L_{32}$ directly with zero modification to $B$!

### Phase 2: Bump Setup & Threshold Markowitz Pivot Search (`markowitz.rs`, `lu_setup_bump.c`, `lu_markowitz.c`)

1. Copy only the remaining `(n - n_cs - n_rs) × (n - n_cs - n_rs)` active bump entries (with `|v| > params.drop_tol`) into column-oriented `ActiveBump` (`cols: LuFileVal`, `rows: LuFileIdx`).
2. Compute `col_max[j] = max_i |a_ij|` for every bump column `j`, and populate `row_deg_list` and `col_deg_list` across buckets `0..=bump_n`.
3. For each bump elimination step `step` until no active lines remain:
   - **Prune degree-0 lines immediately:** Pop any nodes in `row_deg_list.head[0]` or `col_deg_list.head[0]` and mark them structurally empty (never leave degree-0 nodes in the active search, and count them toward rank deficiency at factor assembly).
   - **Fast-path degree-1 bump singletons:**
     - If `col_deg_list.head[1] != NONE`, check its sole entry `(r, val)`: if `|val| >= abs_pivot_tol`, select immediately (`cost = 0`). (If `|val| < abs_pivot_tol`, remove the dead column from search so it cannot starve subsequent steps.)
     - If `row_deg_list.head[1] != NONE`, check its sole entry `(c, val)`: if `|val| >= abs_pivot_tol && |val| >= rel_pivot_tol * col_max[c]`, select immediately (`cost = 0`).
   - **Zlatev / Markowitz search (`d = 2..=bump_n`):**
     - Track `best_cost = usize::MAX`, `best_pivot = None`, `best_rel = 0.0`, `inspected_rows = 0`.
     - **Scan columns in `col_deg_list.head[d]`:**
       - If `col_max[c] < params.abs_pivot_tol`, remove column `c` from active search immediately (prevents dead-column starvation!).
       - Otherwise compute threshold `thresh = (params.rel_pivot_tol * col_max[c]).max(params.abs_pivot_tol)`.
       - Scan entries `(r, val)` of `cols[c]`: if `|val| >= thresh`, compute `cost = (row_deg[r] - 1) * (d - 1)`. Update `best_pivot` on strictly smaller `cost`, breaking ties by larger `|val| / col_max[c]`.
     - **Scan rows in `row_deg_list.head[d]`:**
       - For each column `c` in `rows[r]`, inspect `a[r, c]` in `cols[c]` (checking `|a[r, c]| >= rel_pivot_tol * col_max[c] && |a[r, c]| >= abs_pivot_tol`), updating `best_cost = (d - 1) * (col_deg[c] - 1)` and `best_pivot`.
       - Increment `inspected_rows += 1`. If `inspected_rows >= params.search_rows && best_pivot.is_some()`, break row scan.
     - **Provably optimal early-exit bounds:** After scanning rows of degree `d`, every uninspected entry has `row_deg >= d` and `col_deg >= d + 1` (if cols `<= d` were already scanned), so `best_cost <= d * (d - 1)` is optimal; after scanning **both** columns and rows of degree `d`, every remaining uninspected entry has `row_deg >= d + 1` and `col_deg >= d + 1`, so break as soon as `best_cost <= d * d`!
   - If no eligible pivot with `|piv| >= abs_pivot_tol` exists in the remaining bump, exit the bump loop (remaining lines form the rank-deficient kernel).

### Phase 3: Column-Oriented Gaussian Elimination & Schur Complement (`eliminate.rs`, `lu_pivot.c`, `lu_factorize_bump.c`)

Given selected pivot `(p_row, p_col)` with `piv = a[p_row, p_col]`:
1. **Copy pivot lines to scratch BEFORE mutating `ActiveBump`:**
   - Extract `piv_col_row[]` and scaled multipliers `piv_col_lval[k] = a[r, p_col] / piv` for every `r != p_row` in `cols[p_col]` (dropping any `|lval| <= drop_tol`; exclude `p_row` itself!).
   - Extract `piv_row_col[]` and pivot-row entries `piv_row_uval[]` for every active `c != p_col` appearing in `rows[p_row]` (by intersecting `rows[p_row]` with `cols[c]`, and simultaneously deleting `p_row` from `cols[c]` and `p_col` from `rows[r]`).
   - Why copying first is mandatory: Step 3 can trigger `cols.compress()` or `rows.compress()`, which shifts offsets inside `ActiveBump`.
2. Record `piv_col_lval` into the bump $L$ buffer, `piv_row_uval` into the bump $U$ buffer, `piv` as the step's diagonal pivot, and remove `p_row` and `p_col` from `row_deg_list` and `col_deg_list` (`deg = NONE`).
3. **Eliminate each affected column `c` in `piv_row_col` with `u_val = a[p_row, c]`:**
   - **Remove `p_row` from `cols[c]`** (swap-remove in $O(1)$ within `cols[c]`, decrement `col_deg[c]`).
   - **Scatter target column `c`** into dense `work[r]` using a fresh monotonic `u32` generation stamp `gen` (`mark[r] = gen`):
     - For every `(r, v)` currently in `cols[c]`: `work[r] = v; mark[r] = gen;`.
   - **Apply scaled pivot column `piv_col`:**
     - Maintain a small temporary `fill_rows` list for this column `c`.
     - For each `(r, l_val)` in `piv_col`:
       - `let delta = -l_val * u_val;`
       - If `mark[r] == gen`: `work[r] += delta;` (existing entry update — marker stays `gen` for `c`, and a fresh `gen += 1` isolates the next target column `c2` completely!).
       - Else if `delta.abs() > params.drop_tol`: `work[r] = delta; mark[r] = gen; fill_rows.push(r);`
   - **Write back updated column `c`, remove numerical cancellations from BOTH files, and recompute `col_max[c]`:**
     - `let mut new_max = 0.0_f64;`
     - Walk existing slots of `cols[c]` (iterating backward or using swap-remove cursor):
       - `let new_v = work[r]; work[r] = 0.0;`
       - If `new_v.abs() <= params.drop_tol`:
         - **Delete from BOTH `cols[c]` AND `rows[r]`**!
         - Decrement `col_deg[c]` and `row_deg[r]` in `DegreeLists` (if `row_deg[r]` reaches `0`, it moves cleanly into bucket `0`).
       - Else: store `new_v` back into `cols[c]`, update `new_max = new_max.max(new_v.abs())`.
     - For each `r` in `fill_rows`:
       - `let fill_v = work[r]; work[r] = 0.0;`
       - Insert `(r, fill_v)` into `cols[c]` and `c` into `rows[r]` (calling `reappend`/`compress` as needed).
       - Increment `col_deg[c]` and `row_deg[r]` in `DegreeLists`.
       - `new_max = new_max.max(fill_v.abs());`
     - Store `col_max[c] = new_max`.
4. **Remove `p_col` from active rows:** For every `r` in `piv_col_row`, swap-remove `p_col` from `rows[r]` and decrement `row_deg[r]` in `DegreeLists`.

### Phase 4: Factor Assembly & Rank-Deficiency Completion (`factors.rs`, `lu_build_factors.c`)

1. **Rank check (`rank < n`):**
   - If `!params.allow_singular`, return `Err(Error::Singular { rank, n })`.
   - If `params.allow_singular`, assign arbitrary remaining unpivoted rows and columns one-to-one into positions `rank..n` with `pivot = 1.0` and empty $L$/$U$ off-diagonal columns, recording `rank` in `BasicLuStats` so LP basis repair can still solve systems and inspect dependent positions.
2. **Assemble immutable dual `L`/`LT`:**
   - Map all singleton and bump $L$ entries into fixed step indices `p_row_l_inv[orig_row]`.
   - Pack strictly subdiagonal entries of $L$ into columnwise CSC (`l_col_ptr`, `l_row_idx`, `l_val`) — unit diagonal is implicit.
   - Build the exact structural transpose `LT` (`lt_row_ptr`, `lt_col_idx`, `lt_val`) in two linear counting-sort passes over `L`.
3. **Assemble mutable dual `U`/`UT`:**
   - Initialize `u_row_order[step] = step`, `u_row_pos[step] = step`, `u_col_order[step] = orig_col`, `u_col_pos[orig_col] = step`.
   - Pack strictly off-diagonal entries of $U$ (singleton rows + bump $U$ rows) simultaneously into `u_col: LuFileVal` (keyed by column ID) and `u_row: LuFileVal` (keyed by row ID), leaving modest elbow slack (e.g., `+2` entries per line) for subsequent Forrest–Tomlin updates, and store diagonal pivots in `pivot[row_id]`.
   - Initialize empty `EtaStorage`.

### Phase 5: Dense & Iterative Hyper-Sparse FTRAN / BTRAN (`solve.rs`)

Given $P_r^{(0)} A P_c^{(0)} = L R_1^{-1} \cdots R_t^{-1} U$:
- **FTRAN ($A x = b$):**
  1. Permute into working vector `w[step] = b[p_row_l[step]]`.
  2. Solve $L y = w$ (forward over `0..n` using `l_col_ptr`, or reverse post-order of iterative DFS reachability).
  3. Apply row etas $R_1, \dots, R_t$ **in forward order** `s = 0..t`:
     $$w[p_s] \mathrel{-}= \sum_{k \in \text{eta}_s} \text{val}_k \cdot w[k]$$
  4. Solve $U z = w$ (backward over `(0..n).rev()` along `u_row_order`/`u_col_order`: divide `w[r_id]` by `pivot[r_id]`, subtract `u_col` off-diagonal multipliers).
  5. Unpermute: `x[u_col_order[step]] = w[r_id]` where `r_id = u_row_order[step]`.
- **BTRAN ($A^T x = b$):**
  1. Permute into `w[r_id] = b[u_col_order[step]]` where `r_id = u_row_order[step]`.
  2. Solve $U^T y = w$ (forward over `0..n` along `u_row_order`: divide `w[r_id]` by `pivot[r_id]`, subtract off-diagonal entries from `u_row[r_id]`, i.e., columns of $U^T$).
  3. Apply transposes $R_t^T, \dots, R_1^T$ **in reverse order** `s = (0..t).rev()`:
     - Let `s_val = w[p_s]`. Skip in $O(1)$ if `s_val == 0.0`! Otherwise for each `(k, val_k)` in `eta_s`, do `w[k] -= val_k * s_val`.
  4. Solve $L^T z = w$ (backward over `(0..n).rev()` using `lt_row_ptr` — columns of $L^T$ — or reverse post-order of iterative DFS reachability on `LT`).
  5. Unpermute: `x[p_row_l[step]] = w[step]`.
- **Iterative Gilbert–Peierls Hyper-Sparse Mechanics (`lu_dfs.c`, `lu_solve_sparse.c`):**
  - Never use recursion: maintain explicit `(node, cursor)` frames on a pre-allocated `Vec<(usize, usize)>` inside `SolveWorkspace`, safe for `wasm32`'s limited stack.
  - Perform DFS over the column graph of `L` (FTRAN step 2) / `U` (FTRAN step 4) / `U^T` via `u_row` (BTRAN step 2) / `L^T` via `LT` (BTRAN step 4), pushing finished nodes in post-order onto `reach_stack`, then iterating `reach_stack` in **reverse** (topological order).
  - Track visited nodes via a monotonic `u32` `generation` counter (resetting `visited.fill(0)` only on `u32::MAX` wraparound).
  - **Dynamic `sparse_threshold` switching:** Before DFS and after applying etas, if `active_nnz as f64 > params.sparse_threshold * (n as f64)` (or if DFS discovers `reach_stack.len() as f64 > params.sparse_threshold * (n as f64)`), abort hyper-sparse tracking seamlessly and execute the dense loop for the remaining stages, clearing `w` cleanly before returning.

### Phase 6: Forrest–Tomlin & Pure-Permutation Rank-1 Updates (`update.rs`, `lu_solve_for_update.c`, `lu_update.c`)

To replace column `col` of $A$ with `(new_row_idx, new_val)`:
1. **Validate preconditions** before modifying any factor state:
   - Check `factors.is_some()` (`NotFactorized`), `col < n` (`InvalidColumn`), `stats.updates < params.max_updates` (`MaxUpdatesReached`), `new_row_idx.len() == new_val.len()`, `new_row_idx` indices `< n` and finite `new_val` (`NotFinite`).
2. **Compute spike vector $v$ through BOTH $L$ and ALL existing row etas $R_1, \dots, R_t$:**
   - Permute `a_new` by `p_row_l`, forward-solve $L$, then apply $R_1, \dots, R_t$ in order:
     $$v = R_t \cdots R_1 L^{-1} P_r^{(0)} a_{\text{new}}$$
   - *(Omitting $R_1 \dots R_t$ breaks every update after the first!)*
3. **Locate spike extent in current triangular ordering:**
   - Let `p = u_col_pos[col]`, `p_row = u_row_order[p]`.
   - Find `m = max { u_row_pos[r] | r in supp(v) }` (if `supp(v)` is empty or `m < p`, the updated matrix is structurally singular → return `Err(Error::SingularUpdate)` leaving factors untouched!).
4. **Fast path (`m == p`) & Schork–Gondzio pure-permutation check:**
   - If `m == p` (no subdiagonal spike) or if the spiked columns `p..=m` can be reordered into upper-triangular form purely by row/column permutations (upstream `lu_update.c` singleton/topological permutation pass), check `|new_pivot| >= params.abs_pivot_tol`. If valid, replace column `col` in `u_col` (and its entries in `u_row`), update `pivot` and `u_row_order`/`u_col_order`, increment `stats.updates` (and leave `forrest_tomlin_updates` unchanged — **zero** eta added!).
5. **General Forrest–Tomlin elimination (`m > p`):**
   - **Compute row eta $w$:** Starting from `u_row[p_row]` (positions `> p`, truncated at `<= m`), perform a sparse triangular forward solve with $U^T$ along `u_row_order` over `p+1..=m` using `u_row` (rows of $U$ = columns of $U^T$), computing multipliers $w_k$ for the row elimination $e_p^T - \sum_{k > p} w_k e_k^T$ that zeros out row `p_row` across columns `p+1..=m`.
   - **Compute and verify new pivot BEFORE mutating `U` or `EtaStorage`:**
     $$\alpha = v[p\_row] - \sum_{k} w_k \cdot v[k]$$
     - If `!alpha.is_finite() || alpha.abs() < params.abs_pivot_tol`, return `Err(Error::SingularUpdate)` **without modifying `U`, `pivot`, ordering arrays, or `EtaStorage`** (clean rollback guarantee).
     - Optionally track `pivot_error` when `solve_for_update` supplies an independent BTRAN pivot estimate (`|alpha - alpha_btran| / |alpha|`).
   - **Commit update to `U` and `EtaStorage`:**
     - Delete every off-diagonal entry `(c_id, _)` in `u_row[p_row]`, and swap-remove `p_row` from `u_col[c_id]` for each such column.
     - Clear old entries of `u_col[col]` (and swap-remove `col` from each old row in `u_row`), then insert the cleaned spike entries `(r_id, v[r_id])` (with `|v| > drop_tol` and `r_id != p_row`) into `u_col[col]` and `u_row[r_id]`.
     - Set `pivot[p_row] = alpha`.
     - Apply the cyclic shift `p -> m` in `u_row_order`/`u_row_pos` and `u_col_order`/`u_col_pos` only (zero renumbering inside `L`, `LT`, `U`, or prior `R_s`).
     - Append `(p_row, sparse w)` as `R_{t+1}` into `EtaStorage`; increment `stats.updates` and `stats.forrest_tomlin_updates`.

---

## 6. Phased Execution Schedule, Verification Suite, & CI Gates

### Implementation Phases
| Phase | Deliverable | Gate |
| :--- | :--- | :--- |
| **1** | `src/sparse/mod.rs` (`pub mod basiclu;`), `params.rs`, `error.rs`, `file.rs`, `degree_list.rs` | `LuFile` append, swap-remove, `reappend`, `compress()` invariants, and `DegreeLists` `0..=n` bucket operations pass under forced tight capacities. |
| **2** | `singletons.rs`, `markowitz.rs` | Triangular / permuted-triangular matrices peel with zero fill-in; `abstol` singleton deferral and Markowitz early-exit bounds verified. |
| **3** | `eliminate.rs`, `factors.rs` | Full bump factorization passes $\|P_r A P_c - L U\|_\infty / \|A\|_\infty < 10^{-12}$, cancellation dual removal, and both `allow_singular` modes verified. |
| **4** | `solve.rs` | Dense & hyper-sparse FTRAN/BTRAN agree within $10^{-14}$ across `sparse_threshold` transitions and iterative DFS paths. |
| **5** | `update.rs`, `BasicLu` handle | 50 consecutive Forrest–Tomlin & pure-permutation updates verified at every step `t = 1..50` against fresh factorization + `dgesv`. |
| **6** | `tests.rs`, `tests/wasm_browser.rs`, `bench_basiclu`, docs, `LICENSE`, `Cargo.toml`, `AGENTS.md` | All CI, clippy, fmt, doc, WASM, coverage ($\ge 92\%$ crate, $\ge 95\%$ new module), and `cargo publish --dry-run` checks pass. |

*(Optional local parity check, never in CI: follow the `dev_lapack` env-var pattern in `build.rs` with an opt-in `dev_basiclu` feature pointing `ALGEBRA_LINEAR_DEV_BASICLU_ROOT` at a local HiGHS build if differential testing against C `basiclu` is desired locally.)*

### Verification Suite (`src/sparse/basiclu/tests.rs` & `tests/wasm_browser.rs`)

All unit, regression, and randomized tests live inside `src/sparse/basiclu/tests.rs` (`#[cfg(test)]`) so they are measured by `cargo llvm-cov --lib` (`min-lines: 92` in `.github/workflows/ci.yml`) and executed on `wasm32-wasip1`. Only the browser smoke test goes in `tests/wasm_browser.rs`.

Required test coverage:
1. **Error, Params, & Storage Unit Tests:**
   - Every `Error` variant + exact `Display` string + `&dyn std::error::Error`.
   - `n == 0` and `n == 1` edge cases.
   - `LuFileVal` & `LuFileIdx`: tight initial capacity forcing multiple `reappend()` and `compress()` runs, verifying 100% entry preservation and monotonic physical ordering.
   - `DegreeLists`: insert/remove/degree changes down to bucket `0` and out.
2. **Factorization & Elimination Regression Suite:**
   - Purely triangular & permuted-triangular systems (column singletons, row singletons, mixed, `abs_pivot_tol` deferral).
   - Multi-column elimination sharing pivot-row columns (explicit regression test catching marker-loss across columns).
   - Numerical cancellation dropping entries cleanly from **both** `cols` and `rows` (catching ghost entries).
   - Dead-column pruning (`col_max < abs_pivot_tol`) and degree-0 eviction during bump factorization.
   - Rank-deficient systems tested under both `allow_singular = false` (`Err(Error::Singular { rank, n })`) and `allow_singular = true` (unit-column completion with `stats.rank < n`).
   - Random dense/sparse matrices generated via `crate::lapack::families::dlatms` and a deterministic in-tree PRNG (`SplitMix64`), verified against `crate::lapack::dgesv` ($\|A x - b\|_\infty / (\|A\|_\infty \|x\|_\infty + \|b\|_\infty) < 10^{-12}$).
3. **Hyper-Sparse & Dense Solve Equivalence:**
   - Verify `solve_sparse` matches `solve` (FTRAN) and `solve_sparse_transpose` matches `solve_transpose` (BTRAN) within $10^{-14}$ for 1-nonzero, 5-nonzero, and `sparse_threshold`-crossing dense RHS vectors (forcing the mid-solve switch to dense).
   - Deep chain / triangular matrix with length $n = 4{,}096$ verifying iterative DFS runs safely under `wasm32-wasip1` without stack overflow.
4. **Forrest–Tomlin & Permutation Update Suite:**
   - Run **50 consecutive column replacements** on sparse LP-basis-like matrices (including `m == p` no-spike replacements, Schork–Gondzio pure-permutation updates with `forrest_tomlin_updates` unchanged, and general `m > p` Forrest–Tomlin updates).
   - At **every** step `t = 1..50`, verify dense FTRAN, dense BTRAN, hyper-sparse FTRAN, and hyper-sparse BTRAN against a fresh `BasicLu::factorize` + `crate::lapack::dgesv` solve (relative error $< 10^{-11}$).
   - Verify `SingularUpdate` rollback leaves existing factors completely intact and producing accurate solves for the pre-update matrix.
   - Verify `MaxUpdatesReached` and `InvalidColumn`.
   - Per `AGENTS.md` ("make a deliberate mistake and watch it fail"), confirm a perturbed factor/update triggers test failure.
5. **Performance Benchmark (`#[test] #[ignore]`):**
   - Add `#[test] #[ignore] fn bench_basiclu()` matching `bench_sparse_lu` in `src/sparse/sparse_lu.rs` (factorize, FTRAN/BTRAN sparse vs dense, 50-update loop).
6. **WASM Browser Smoke Test (`tests/wasm_browser.rs`):**
   - Add `#[wasm_bindgen_test] fn basiclu_runs_in_the_browser_target()` testing `factorize`, `solve`, `solve_sparse`, `update`, and post-update `solve` + `solve_transpose`.

### Final Verification Commands
Run all gates from `AGENTS.md` and `.github/workflows/ci.yml`:
```bash
cargo fmt --all -- --check
cargo clippy --all-targets
cargo test
cargo doc --no-deps
cargo build --lib --target wasm32-unknown-unknown
cargo test --lib --target wasm32-wasip1
cargo llvm-cov --lib --lcov          # verify >= 92% overall and >= 95% on src/sparse/basiclu/
cargo publish --dry-run              # verify port-basiclu/ is excluded and package builds standalone
```
