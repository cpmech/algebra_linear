# Implementation Plan: Porting QDLDL to Pure Safe Rust in `algebra_linear`

## 1. Project Metadata, Role in `algebra_linear`, & Licensing Gate

- **Target Integration:** Internal submodule `algebra_linear::sparse::qdldl` (`src/sparse/qdldl/`, re-exported via `pub mod qdldl;` in `src/sparse/mod.rs`), paralleling `src/sparse/amgcl/` and [`AMGCL_PORTING.md`](../AMGCL_PORTING.md). This is **not** a standalone crate.
- **Role Alongside Existing Solvers:**
  - `SparseLu` (`src/sparse/`): general unsymmetric real (`f64`) and complex (`Complex64`) CSC sparse LU with COLAMD ordering, partial pivoting, and simplicial/supernodal paths.
  - `sparse::amgcl`: real (`f64`) CSR algebraic multigrid and Krylov iterative solvers.
  - `sparse::qdldl`: real (`f64`) upper-triangular CSC unpivoted $L D L^T$ factorization for **Symmetric Positive Definite (SPD)** and **Symmetric Quasi-Definite (SQD)** matrices (such as convex QP KKT saddle-point systems), with zero-allocation numeric refactorization on fixed sparsity patterns:
    $$A = L D L^T$$
    where $L$ is unit lower triangular (diagonal ones implicit) and $D = \operatorname{diag}(d_0, \dots, d_{n-1})$ with finite $d_i \neq 0$.
- **Explicit Scope Boundaries:**
  - **In scope:** Upper-triangular CSC structural validation, elimination tree and column nonzero counting, zero-allocation numeric refactorization on a fixed pattern, in-place triangular and full solves ($L x = b$, $L^T x = b$, $L D L^T x = b$), inertia reporting (`positive_inertia`), and diagonal inspection (`d()`).
  - **Out of scope:** Numerical pivoting (by definition of QDLDL), fill-reducing symmetric ordering (callers permute beforehand, e.g., with symmetric AMD or RCM; `src/sparse/ordering/` currently provides nonsymmetric `colamd` only), dynamic diagonal regularization inside the kernel, `f32`/`Complex64` or generic integer widths, and CSC↔CSR conversion (`AGENTS.md`).
- **Upstream Reference:** `osqp/qdldl` (`include/qdldl.h`, `include/qdldl_types.h`, `src/qdldl.c`), latest release **`v0.1.9`**. Pin the exact upstream commit SHA when transcription begins, and name the upstream file and routine in every function doc comment per `AGENTS.md`.
- **Upstream License & Explicit SPDX Decision Gate (CRITICAL):**
  - Upstream `osqp/qdldl` is distributed under the **Apache License 2.0** (copyright OSQP Developers: Paul Goulart, Bartolomeo Stellato, Goran Banjac, Ian McInerney). Note that `osqp/qdldl` has **no** root `NOTICE` file (unlike `osqp/osqp`).
  - `algebra_linear` currently uses `license = "BSD-3-Clause"` in `Cargo.toml`. Unlike the MIT-licensed `faer` and `AMGCL` ports already in `LICENSE`, **Apache-2.0 cannot be silently folded into `BSD-3-Clause` or relicensed**: Apache-2.0 carries patent grant and retaliation clauses (§3), requires distributing the full verbatim Apache-2.0 license text (§4(a)), and requires prominent modification notices in every derived source file (§4(b)).
  - Before writing code, the maintainer must explicitly choose one of two paths:
    1. **Option A — Literal transcription of `osqp/qdldl` (`v0.1.9`):** Update `Cargo.toml` to `license = "BSD-3-Clause AND Apache-2.0"`, append the full verbatim Apache-2.0 text plus `osqp/qdldl` `v0.1.9` attribution and pinned commit SHA to `LICENSE`, and add Apache-2.0 §4(b) modification headers at the top of `src/sparse/qdldl/*.rs`.
    2. **Option B — Keep `algebra_linear` strictly `BSD-3-Clause`:** Implement the ~80-line up-looking sparse $L D L^T$ algorithm clean-room from published textbook algorithms (Liu 1990 elimination tree; Davis, *Direct Methods for Sparse Linear Systems*, Ch. 4, 2006) without transcribing `osqp/qdldl` source files or copying its comments/structure.
- **Engineering Constraints (`AGENTS.md`, `Cargo.toml`, `.github/workflows/ci.yml`):**
  - `#![forbid(unsafe_code)]` in `src/sparse/qdldl/mod.rs` (matching `src/sparse/amgcl/mod.rs`).
  - `#![warn(missing_docs)]` (inherited from `src/lib.rs`) — every `pub` item must carry clear documentation so `cargo doc --no-deps` builds cleanly.
  - **Zero new dependencies** (`Cargo.toml` allows only `num-complex`, `web-time` on `wasm32-unknown-unknown`, and `wasm-bindgen-test` as a `wasm32` dev-dependency; CI rejects any `path =` dependency).
  - **0-based `usize` indices and `f64` scalars** (`AGENTS.md`, `SparseLu`, `sparse::amgcl`).
  - **No workspace arguments in the public API** (`AGENTS.md`, `src/lib.rs`). The opaque `Qdldl` handle owns its symbolic arrays, numeric factors, and reusable solve/factorization scratch internally.
  - **Zero runtime heap allocations** inside `factorize()` (after `analyze()`), `solve()`, `solve_l()`, and `solve_lt()`.
  - **Full WebAssembly compatibility:** Compiles and passes tests on both `wasm32-unknown-unknown` and `wasm32-wasip1`. If factorization timing is ever added to `QdldlStats`, use the `web_time::Instant` cfg shim from `src/sparse/sparse_lu.rs`.
  - **Arithmetic performance trap:** Do **not** call `f64`'s fused multiply-add method in inner loops. On default x86-64 builds (SSE2, without `-C target-feature=+fma`) and on `wasm32`, Rust's FMA method dispatches to software `libm::fma`, which is 10–50× slower than standard `a - b * c` and changes rounding behavior.
  - **American English everywhere** (`AGENTS.md`) — normalize upstream `osqp/qdldl`'s British spelling (e.g., `factorization`, not the British variant) in all comments, docs, identifiers, and commit messages.
  - **Style & Hygiene:** Rust edition 2024, `rustfmt.toml` (`max_width = 150`), `cargo clippy --all-targets` warning-free, and zero developer filesystem paths in source.

---

## 2. Upstream C to `algebra_linear` Module Mapping

Upstream `osqp/qdldl` exports five functions (`QDLDL_etree`, `QDLDL_factor`, `QDLDL_Lsolve`, `QDLDL_Ltsolve`, `QDLDL_solve`) and three typedefs (`QDLDL_int`, `QDLDL_float`, `QDLDL_bool`). There is **no** separate `DL` solve function in upstream `qdldl.h` / `qdldl.c`.

| Upstream C File / Symbol | Target Module (`src/sparse/qdldl/`) | Rust Types & Functions |
| :--- | :--- | :--- |
| `qdldl_types.h` (`QDLDL_int`, `QDLDL_float`, `QDLDL_bool`) | Concrete `usize`, `f64`, `bool` | Matches `algebra_linear` conventions (`AGENTS.md`); avoids generic monomorphization bloat in `llvm-cov` and eliminates signed-index wraparound bugs. |
| `qdldl.h` (input CSC preconditions) | `csc.rs` | `UpperTriCscPattern<'a>`, `UpperTriCscView<'a>`, `validate_upper_csc(...)`, FNV-1a `fingerprint(...)` (matching `src/sparse/symbolic.rs`). |
| `qdldl.c` (`QDLDL_etree`, `QDLDL_UNKNOWN = -1`) | `etree.rs` | `NONE = usize::MAX`, `SymbolicFactor`, `compute_etree(...)`. |
| `qdldl.c` (`QDLDL_factor`, `bwork`, `iwork` of len `3n`, `fwork`) | `factor.rs` | `FactorWorkspace`, `NumericFactor`, `factor_in_place(...)`. |
| `qdldl.c` (`QDLDL_Lsolve`, `QDLDL_Ltsolve`, `QDLDL_solve`) | `solve.rs` | `solve_l(...)`, `solve_lt(...)`, `solve(...)`. |
| High-level handle, stats, errors | `mod.rs`, `error.rs` | `Qdldl`, `QdldlStats`, `Error`, `Result<T>`. |

---

## 3. Repository Integration & Module Layout

```text
src/
├── lib.rs                      # Update crate docs to mention `sparse::qdldl`
└── sparse/
    ├── mod.rs                  # Add `pub mod qdldl;`
    └── qdldl/
        ├── mod.rs              # `#![forbid(unsafe_code)]`, `Qdldl`, `QdldlStats`, re-exports `Error`, `Result`
        ├── error.rs            # `Error` & `Result<T>` (`Display`, `std::error::Error`)
        ├── csc.rs              # Borrowed upper-triangular CSC validation + FNV-1a pattern fingerprint
        ├── etree.rs            # Elimination tree construction (`QDLDL_etree`) & `SymbolicFactor`
        ├── factor.rs           # `FactorWorkspace`, `NumericFactor`, up-looking `LDL^T` (`QDLDL_factor`)
        ├── solve.rs            # In-place `solve_l`, `solve_lt`, `solve` (`QDLDL_*solve`)
        └── tests.rs            # `#[cfg(test)]` unit, analytical, SQD KKT, randomized accuracy, & `bench_qdldl`
tests/
└── wasm_browser.rs             # Add `#[wasm_bindgen_test] fn qdldl_runs_in_the_browser_target()`
```

Additional repository files touched upon implementation:
- `Cargo.toml`: add `port-qdldl/` to `exclude` (or remove the folder once implementation lands), and update `license` to `"BSD-3-Clause AND Apache-2.0"` if Option A is chosen.
- `LICENSE`: append verbatim Apache-2.0 text and `osqp/qdldl` `v0.1.9` provenance + commit SHA if Option A is chosen.
- `README.md`: document `sparse::qdldl` alongside `SparseLu` and `sparse::amgcl`.
- `AGENTS.md`: record any genuinely unreachable coverage lines (e.g., `Error::IndexOverflow` guards) in the coverage table in the same PR.

---

## 4. Data Structures, Error Type, & Public API Specifications

### 4.1 Module-Local Error Type (`src/sparse/qdldl/error.rs`)
`SparseError` in `src/sparse/mod.rs` is not marked `#[non_exhaustive]`, so adding new variants (`LowerTriangularEntry`, `EmptyColumn`, `ZeroDiagonal`) to `SparseError` would be a semver-breaking change. Follow the precedent set by `sparse::amgcl::Error` and define a dedicated error enum for `sparse::qdldl`:

```rust
/// Errors reported by upper-triangular CSC validation, symbolic analysis,
/// numeric factorization, and triangular solves in `sparse::qdldl`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// `col_ptr` length, `row_idx` length, `values` length, or RHS vector length
    /// does not match the matrix dimension or entry count.
    BadShape,
    /// `col_ptr` does not start at 0, is not non-decreasing, or a row index is `>= n`.
    InvalidStructure,
    /// A column's row indices are not strictly increasing.
    UnsortedRows,
    /// A column contains duplicate row indices.
    DuplicateEntry,
    /// An entry has `row > col` (QDLDL accepts only the upper triangle `row <= col`).
    LowerTriangularEntry { row: usize, col: usize },
    /// Column `col` of the upper triangle is completely empty, making pivot `D[col] = 0`.
    EmptyColumn { col: usize },
    /// Total nonzeros in `L` (`l_col_ptr[n]`) overflowed `usize`.
    IndexOverflow,
    /// The CSC pattern passed to `factorize` differs from the pattern passed to `analyze`.
    StructureMismatch,
    /// An input entry or computed diagonal pivot is NaN or infinite.
    NotFinite,
    /// Numerical breakdown at zero-based index `index` (`D[index] == 0.0` or `1.0 / D[index]` overflows).
    ZeroDiagonal { index: usize },
    /// `solve`, `solve_l`, or `solve_lt` was called before a successful `factorize`.
    NotFactorized,
}

pub type Result<T> = core::result::Result<T, Error>;
```
Implement `core::fmt::Display` and `std::error::Error` for `Error`, and test every `Display` arm in `tests.rs` (matching `src/sparse/mod.rs` and `src/sparse/amgcl/error.rs`).

### 4.2 Public Statistics & Opaque Solver Handle (`src/sparse/qdldl/mod.rs`)

```rust
/// Summary statistics of the latest successful QDLDL factorization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QdldlStats {
    /// Order of the symmetric matrix (`n x n`).
    pub n: usize,
    /// Number of stored upper-triangular entries in `A`.
    pub nnz_a: usize,
    /// Number of strictly lower-triangular entries stored in `L`.
    pub nnz_l: usize,
    /// Number of strictly positive diagonal entries in `D` (for SPD/SQD inertia checks).
    pub positive_inertia: usize,
}

/// Pure-Rust sparse $L D L^T$ solver for symmetric positive-definite and
/// symmetric quasi-definite matrices in upper-triangular CSC form.
#[derive(Clone, Debug)]
pub struct Qdldl {
    symbolic: SymbolicFactor,
    numeric: NumericFactor,
    workspace: FactorWorkspace,
    factorized: bool,
    stats: Option<QdldlStats>,
}
```

**Public API Contract:**
- `pub fn analyze(col_ptr: &[usize], row_idx: &[usize]) -> Result<Self>`
  - Infers `n = col_ptr.len().checked_sub(1).ok_or(Error::BadShape)?` (matching `SparseLu::analyze` in `src/sparse/sparse_lu.rs`).
  - Validates upper-triangular CSC structure (supports `n == 0` where `col_ptr = &[0]`, `row_idx = &[]`).
  - Computes `SymbolicFactor` (`etree`, `lnz`, `l_col_ptr`, `nnz_a`, `fingerprint`), pre-allocates `NumericFactor` (`l_row_idx`, `l_values`, `d`, `d_inv`) and `FactorWorkspace`.
- `pub fn factorize(&mut self, col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> Result<()>`
  - **Zero heap allocations** after `analyze()`.
  - **Pre-mutation validation:** Validates `UpperTriCscView` (`BadShape`, `InvalidStructure`, `LowerTriangularEntry`, `DuplicateEntry`, `UnsortedRows`), verifies `(n, nnz_a, fingerprint)` matches `self.symbolic` (`StructureMismatch`), and checks `values.iter().all(|v| v.is_finite())` (`NotFinite`) **before** touching `self.numeric` or `self.factorized`. Thus validation errors preserve the previous valid factorization intact (matching `SparseLu::factorize`).
  - **Numeric invalidation guard:** Before writing into `self.numeric`, sets `self.factorized = false` and `self.stats = None`. If `factor_in_place` fails mid-matrix with `ZeroDiagonal` or `NotFinite`, `solve*` calls return `Err(Error::NotFactorized)` instead of reading half-overwritten factors, and a subsequent valid `factorize` reuses the already-allocated buffers cleanly.
- `pub fn solve(&self, b: &mut [f64]) -> Result<()>` — solves $L D L^T x = b$ in place (requires `&self`, uses zero scratch memory, and is safe to share across threads for concurrent solves).
- `pub fn solve_l(&self, b: &mut [f64]) -> Result<()>` — solves $L x = b$ in place (`&self`).
- `pub fn solve_lt(&self, b: &mut [f64]) -> Result<()>` — solves $L^T x = b$ in place (`&self`).
- `pub fn d(&self) -> Option<&[f64]>` — returns the diagonal factor $D$ if currently factorized.
- `pub fn stats(&self) -> Option<QdldlStats>` — returns `QdldlStats` if currently factorized.

### 4.3 Upper-Triangular CSC Validator (`src/sparse/qdldl/csc.rs`)
Enforced rules for `UpperTriCscPattern::new(col_ptr, row_idx)`:
1. `col_ptr.len() >= 1`; `n = col_ptr.len() - 1`.
2. `col_ptr[0] == 0` and `col_ptr[n] == row_idx.len()` (`InvalidStructure` / `BadShape`).
3. Monotonicity: `col_ptr[j] <= col_ptr[j + 1]` for all `j in 0..n` (`InvalidStructure`).
4. For every column `j in 0..n` with `start = col_ptr[j]`, `end = col_ptr[j + 1]`:
   - Every `row = row_idx[p]` satisfies `row < n` (else `InvalidStructure`) and `row <= j` (else `LowerTriangularEntry { row, col: j }`).
   - Strictly increasing order within the column (`row_idx[p - 1] == row` → `DuplicateEntry`, `row_idx[p - 1] > row` → `UnsortedRows`).
5. **Crucial SQD distinction:** An empty upper-triangular column (`start == end`) is rejected in `compute_etree` with `Error::EmptyColumn { col: j }`, because $A_{0..=j, j} = 0$ guarantees $D_j = 0$ in unpivoted $L D L^T$. However, a **missing diagonal entry** (`start < end` and `row_idx[end - 1] < j`) **MUST be accepted**, because symmetric quasi-definite KKT matrices $\begin{pmatrix} P & G^T \\ G & 0 \end{pmatrix}$ routinely omit zero diagonals in the lower-right dual block while having nonzero coupling entries above the diagonal!
6. Compute a 64-bit FNV-1a `fingerprint` over `col_ptr` and `row_idx` (same constants as `src/sparse/symbolic.rs`) so `factorize` verifies structural equivalence in $O(\text{nnz}_A)$ with zero allocations.

### 4.4 Internal Symbolic, Numeric, and Workspace Structs (`etree.rs`, `factor.rs`)

```rust
pub(crate) const NONE: usize = usize::MAX;

#[derive(Clone, Debug)]
pub(crate) struct SymbolicFactor {
    pub n: usize,
    pub nnz_a: usize,
    pub fingerprint: u64,
    pub etree: Vec<usize>,     // len == n; parent of node i is etree[i], or NONE if root
    pub lnz: Vec<usize>,       // len == n; strictly lower-triangular nonzeros in column i of L
    pub l_col_ptr: Vec<usize>, // len == n + 1; CSC column pointers of L
}

#[derive(Clone, Debug)]
pub(crate) struct NumericFactor {
    pub l_row_idx: Vec<usize>, // len == l_col_ptr[n]
    pub l_values: Vec<f64>,    // len == l_col_ptr[n]
    pub d: Vec<f64>,           // len == n
    pub d_inv: Vec<f64>,       // len == n
}

#[derive(Clone, Debug)]
pub(crate) struct FactorWorkspace {
    pub y: Vec<f64>,           // len == n; upstream `fwork` (dense accumulator)
    pub y_markers: Vec<bool>,  // len == n; upstream `bwork` (visited flags: true = USED, false = UNUSED)
    pub y_idx: Vec<usize>,     // len == n; upstream `iwork[2n..3n]` (ordered nonzero reachability indices)
    pub elim_buffer: Vec<usize>, // len == n; upstream `iwork[0..n]` (single-branch reversal stack)
    pub l_next: Vec<usize>,    // len == n; upstream `iwork[n..2n]` (next write cursor per column of L)
}
```
**Why BOTH `elim_buffer` AND `y_idx` are mandatory:** Upstream `QDLDL_factor` allocates `iwork` of length `3 * n` partitioned into `elimBuffer`, `Lnext`, and `yIdx`. Attempting to collapse `elim_buffer` and `y_idx` into a single stack produces an invalid topological order on multi-branch elimination trees (see Section 5, Step 2a).

---

## 5. Exact Step-by-Step Algorithm Specification

### Step 1: Symbolic Analysis (`compute_etree` in `etree.rs`, transcription of `QDLDL_etree`)

1. Handle `n == 0`: return `SymbolicFactor` with `etree = []`, `lnz = []`, `l_col_ptr = [0]`.
2. Initialize `etree = vec![NONE; n]`, `lnz = vec![0; n]`, `work = vec![NONE; n]`.
3. Loop `j` from `0` to `n - 1`:
   - **Empty column guard:** If `col_ptr[j] == col_ptr[j + 1]`, return `Err(Error::EmptyColumn { col: j })`.
     *(Note on disconnected components: Multiple roots ending with `etree[r] == NONE` in block-diagonal matrices are completely valid, provided every column is non-empty.)*
   - Set `work[j] = j`.
   - For `p` in `col_ptr[j]..col_ptr[j + 1]`:
     - `let mut curr = row_idx[p];`
     - Walk upward along the elimination tree path toward root until hitting an already-marked node (`curr == j` stops immediately because `work[j] == j`):
       ```rust
       while work[curr] != j {
           if etree[curr] == NONE {
               etree[curr] = j;
           }
           lnz[curr] += 1; // Safe: lnz[curr] <= n - 1 - curr < n
           work[curr] = j;
           curr = etree[curr];
       }
       ```
4. Build `l_col_ptr` via **overflow-checked** prefix sum (matching upstream `QDLDL_etree`'s `-2` overflow check, since $\text{nnz}(L) \le n(n-1)/2$ can exceed `usize::MAX` even when every `lnz[curr] < n`):
   ```rust
   let mut l_col_ptr = vec![0usize; n + 1];
   for j in 0..n {
       l_col_ptr[j + 1] = l_col_ptr[j]
           .checked_add(lnz[j])
           .ok_or(Error::IndexOverflow)?;
   }
   ```

### Step 2: Up-Looking Numeric Factorization (`factor_in_place` in `factor.rs`, transcription of `QDLDL_factor`)

1. Handle `n == 0`: return `positive_inertia = 0` immediately.
2. Initialize workspace once per factorization call:
   - `y.fill(0.0);`
   - `y_markers.fill(false);`
   - `l_next.copy_from_slice(&symbolic.l_col_ptr[0..n]);`
   - `let mut positive_inertia = 0usize;`
3. For `k` in `0..n`:
   - **2a. Scatter column `k` of $A$ and construct topological reachability in `y_idx[0..nnz_y]`:**
     - `y_markers[k] = true;`
     - `let mut nnz_y = 0usize;`
     - For `p` in `col_ptr[k]..col_ptr[k + 1]`:
       - `let i = row_idx[p];`
       - `y[i] = values[p];` (sets `y[k]` if diagonal is present; otherwise `y[k]` remains `0.0`).
       - Walk up `etree` from `i` to find unmarked ancestors along this branch:
         ```rust
         let mut curr = i;
         let mut nnz_e = 0usize;
         while curr != NONE && !y_markers[curr] {
             elim_buffer[nnz_e] = curr;
             nnz_e += 1;
             y_markers[curr] = true;
             curr = etree[curr];
         }
         while nnz_e > 0 {
             nnz_e -= 1;
             y_idx[nnz_y] = elim_buffer[nnz_e];
             nnz_y += 1;
         }
         ```
       - **Topological invariant:** When a branch is walked from leaf `i` up to the first already-marked ancestor (or root), `elim_buffer` collects `[leaf, ..., highest_unmarked_ancestor]` bottom-up. Draining `elim_buffer` in reverse appends `[highest_unmarked_ancestor, ..., leaf]` onto `y_idx`. Therefore, **every ancestor strictly precedes all of its descendants in `y_idx[0..nnz_y]`** across every branch! Iterating `(0..nnz_y).rev()` then visits every descendant strictly before its parent — the exact lower-triangular dependency order even on multi-branch trees. (A single stack without per-branch reversal visits `2` before `0` on a simple chain `0 -> 1 -> 2 -> 3`, corrupting both $L$ and the `y` reset!)
   - **2b. Sparse triangular solve and column `k` update in reverse `y_idx` order:**
     - Extract accumulated diagonal initializer and clear `y[k]`:
       ```rust
       let mut dval = y[k];
       y[k] = 0.0;
       ```
     - Iterate `(0..nnz_y).rev()`:
       ```rust
       for idx in (0..nnz_y).rev() {
           let c = y_idx[idx];
           let yc = y[c];
           y[c] = 0.0;
           y_markers[c] = false;

           let col_start = l_col_ptr[c];
           let col_end = l_next[c];
           for p in col_start..col_end {
               y[l_row_idx[p]] -= l_values[p] * yc;
           }

           let l_val = yc * d_inv[c];
           l_row_idx[col_end] = k;
           l_values[col_end] = l_val;
           l_next[c] = col_end + 1;

           dval -= yc * l_val;
       }
       y_markers[k] = false;
       ```
       Note that `y` and `y_markers` are completely reset to `0.0` / `false` on the fly during this loop (zero $O(n)$ cleanup needed inside the `k` loop).
   - **2c. Extract and validate diagonal pivot $D_k$:**
     - If `!dval.is_finite()`, return `Err(Error::NotFinite)`.
     - If `dval == 0.0`, return `Err(Error::ZeroDiagonal { index: k })`.
     - `let d_inv_k = 1.0 / dval;`
     - If `!d_inv_k.is_finite()`, return `Err(Error::ZeroDiagonal { index: k })` (guards subnormal reciprocal overflow to `inf`).
     - `d[k] = dval;`
     - `d_inv[k] = d_inv_k;`
     - If `dval > 0.0`, `positive_inertia += 1`.

### Step 3: In-Place Triangular & Full Solves (`solve.rs`, transcription of `QDLDL_Lsolve`, `QDLDL_Ltsolve`, `QDLDL_solve`)

All routines validate `b.len() == n` (returning `Err(Error::BadShape)` otherwise), require `self.factorized` (else `Err(Error::NotFactorized)`), and operate strictly in place on `&mut [f64]` with zero allocations:

- **`solve_l(x)` ($L x = b$):**
  ```rust
  for j in 0..n {
      let xj = x[j];
      if xj != 0.0 {
          for p in l_col_ptr[j]..l_col_ptr[j + 1] {
              x[l_row_idx[p]] -= l_values[p] * xj;
          }
      }
  }
  ```
- **`solve_lt(x)` ($L^T x = b$):**
  ```rust
  for j in (0..n).rev() {
      let mut sum = x[j];
      for p in l_col_ptr[j]..l_col_ptr[j + 1] {
          sum -= l_values[p] * x[l_row_idx[p]];
      }
      x[j] = sum;
  }
  ```
- **`solve(x)` ($L D L^T x = b$):**
  ```rust
  solve_l(x);
  for j in 0..n {
      x[j] *= d_inv[j];
  }
  solve_lt(x);
  ```
  Note the algebraic order: $L (D (L^T x)) = b \implies y = L^{-1} b$, then $z = D^{-1} y$, then $x = L^{-T} z$. Scaling by $D^{-1}$ *after* `solve_l` computes $(L D)^{-1} b$ and therefore pairs with `solve_lt` to invert $L D L^T$.

---

## 6. Phased Execution Schedule, Verification Suite, & CI Gates

### Implementation Phases
| Phase | Deliverable | Gate |
| :--- | :--- | :--- |
| **1** | SPDX license choice (Option A vs Option B), module skeleton (`src/sparse/mod.rs`, `src/sparse/qdldl/{mod,error,csc}.rs`) | All CSC validation error variants and `Display` strings verified with unit tests; `cargo doc --no-deps` clean. |
| **2** | `src/sparse/qdldl/etree.rs` | `etree`, `lnz`, and `l_col_ptr` verified on chains, multi-branch trees, arrow, tridiagonal, grid Laplacian, and disconnected block-diagonal matrices. |
| **3** | `src/sparse/qdldl/{factor,solve,mod}.rs` (`Qdldl` handle) | Exact $A = L D L^T$ reconstruction, zero-allocation `factorize()` reuse, `solve_l`, `solve_lt`, `solve`, and inertia checks pass. |
| **4** | Full `src/sparse/qdldl/tests.rs`, `tests/wasm_browser.rs`, and `bench_qdldl` | Randomized SPD + SQD KKT sweeps verified against `crate::lapack::dgesv`, WASM browser test passes, deliberate-mistake mutation test confirmed. |
| **5** | `README.md`, `src/lib.rs`, `LICENSE`, `Cargo.toml` (`exclude`), `AGENTS.md` coverage table | All CI, clippy, fmt, doc, WASM, coverage ($\ge 92\%$ crate, $\ge 95\%$ new module), and `cargo publish --dry-run` checks pass. |

### Verification Suite (`src/sparse/qdldl/tests.rs` & `tests/wasm_browser.rs`)

All unit, regression, and randomized accuracy tests live inside `src/sparse/qdldl/tests.rs` (under `#[cfg(test)]`) rather than `tests/*.rs`, because `.github/workflows/ci.yml` gates coverage with `cargo llvm-cov --lib` (`min-lines: 92`) and runs `cargo test --lib --target wasm32-wasip1`. Only the browser smoke test goes in `tests/wasm_browser.rs`.

Required test coverage:
1. **Error & CSC Validation Suite:**
   - Every `Error` variant (`BadShape`, `InvalidStructure`, `UnsortedRows`, `DuplicateEntry`, `LowerTriangularEntry`, `EmptyColumn`, `IndexOverflow`, `StructureMismatch`, `NotFinite`, `ZeroDiagonal`, `NotFactorized`) plus exact `Display` formatting and `&dyn std::error::Error`.
   - `n == 0` and `n == 1` edge cases (`analyze`, `factorize`, `solve`, `d()`, `stats()`).
2. **Graph & Structure Edge Cases:**
   - **Multi-branch elimination tree regression test** (catches broken single-buffer topological sort when two child subtrees share a common ancestor in column `k`).
   - Arrow matrices, tridiagonal SPD matrices, 2D 5-point Laplacian grids, and **disconnected block-diagonal matrices** (multiple roots with `etree[r] == NONE`, all columns non-empty).
   - **Negative-definite** matrix (`positive_inertia == 0`) and **SQD KKT saddle-point** matrix $\begin{pmatrix} P & G^T \\ G & -C \end{pmatrix}$ including **SQD with $C = 0$ (missing diagonals in the lower-right dual block)** where columns remain non-empty via $G^T$.
3. **Refactorization & State-Safety Guarantees:**
   - Reuse one `Qdldl` handle across multiple numerical value sets on the same pattern without reallocation.
   - Verify that a failed `factorize` due to `StructureMismatch` or `NotFinite` leaves the previous valid `factorize` factors intact and usable by `solve`.
   - Verify that a failed numeric `factorize` (`ZeroDiagonal`) transitions the handle to `NotFactorized` (returning `Err(Error::NotFactorized)` on `solve` and `None` on `d()`/`stats()`), and recovers cleanly on a subsequent valid `factorize`.
   - Verify subnormal pivot causing `1.0 / dval` overflow is caught cleanly as `Error::ZeroDiagonal`.
4. **Randomized Accuracy Sweep (Zero External Crates):**
   - Implement a deterministic in-tree PRNG (`SplitMix64`, zero external deps) generating 500–1,000 random SPD and SQD KKT matrices across sizes $n \in [2, 64]$, plus symmetric random matrices generated via `crate::lapack::families::dlatms`.
   - Compare $L D L^T$ solve outputs against `crate::lapack::dgesv`, verifying:
     - Direct factorization residual: $\|A - L D L^T\|_\infty / \|A\|_\infty < 10^{-12}$.
     - Scaled solve residual: $\|A x - b\|_\infty / (\|A\|_\infty \|x\|_\infty + \|b\|_\infty) < 10^{-12}$.
     - Individual `solve_l` and `solve_lt` checks against dense forward/backward multiplication.
   - Per `AGENTS.md` ("make a deliberate mistake and watch it fail"), verify the test assertions fail if a value or sign in $L$ or $D$ is perturbed.
5. **Performance Benchmark (`#[test] #[ignore]`):**
   - Add `#[test] #[ignore] fn bench_qdldl()` following `bench_sparse_lu` in `src/sparse/sparse_lu.rs` (no external benchmark crates).
6. **WASM Browser Smoke Test (`tests/wasm_browser.rs`):**
   - Add `#[wasm_bindgen_test] fn qdldl_runs_in_the_browser_target()` covering `analyze`, `factorize`, `refactorize`, `solve`, and `positive_inertia` on an SQD saddle-point system.

### Final Verification Commands
Run all gates from `AGENTS.md` and `.github/workflows/ci.yml`:
```bash
cargo fmt --all -- --check
cargo clippy --all-targets
cargo test
cargo doc --no-deps
cargo build --lib --target wasm32-unknown-unknown
cargo test --lib --target wasm32-wasip1
cargo llvm-cov --lib --lcov          # verify >= 92% overall and >= 95% on src/sparse/qdldl/
cargo publish --dry-run              # verify port-qdldl/ is excluded and package builds standalone
```
