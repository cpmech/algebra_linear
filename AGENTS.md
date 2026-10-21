# AGENTS.md

Notes for automated agents and contributors working in this repository.

## What this crate is

A pure-Rust implementation of BLAS and LAPACK. It must build for any target,
including `wasm32-unknown-unknown`, and the default build compiles no C and links
no library. External libraries appear only in the development-only reference
harness described below.

## Hard constraints

- **This repository is standalone and must stay that way.** Nothing here may name,
  import, or otherwise reference a project that depends on this crate — not in code,
  not in docs, not in tests, not in CI. The dependency goes one way only. When a
  comment needs to explain a call site, say *the caller*.
- **Everything is column-major**, indexing `data[i + j * nrow]`.
- **Never use `LAPACKE_*`** — it is the row-major API and silently transposes every
  matrix. `lapack.h` binds directly to the Fortran routines and is correctly
  column-major. This applies to the reference harness too.
- **0-based indices, `usize` lengths, no workspace argument.** LAPACK's `lda`,
  `lwork` and 1-based indexing are artifacts of Fortran; convert at the boundary of
  the routine and say so in the comment. Pivot vectors, where a caller sees them,
  are the crate's 0-based indices.
- **Errors, not `INFO`.** A driver returns `Result<(), LapackError>`, with
  `INFO < 0` mapped to `BadShape` and `INFO > 0` to `Singular` or `Convergence`.
  Internal helpers may keep returning `INFO` when they are literal transcriptions.
- **Every routine names the reference source** it follows (file and routine), so a
  reader can check it against LAPACK 3.12.1.
- **No developer paths in source** — not in code, not in defaults, not in doc
  comments. Use environment variables, and prefer *requiring* one over supplying a
  default.
- **American English everywhere** — comments, docs, identifiers, test names, commit
  messages: `behavior`, `normalize`, `factorization`, `center`, `toward`. The
  British variants are not used.
- **Keep the tree rustfmt-clean** — `rustfmt.toml` holds the settings, so do not restate them here.
  CI enforces this.
- **Keep `cargo clippy --all-targets` warning-free**, as a review requirement — CI does not run
  clippy. Its lint set moves with the toolchain, so a new stable would fail the build on style the
  transcription inherits from LAPACK (1.99's `needless_late_init` on a variable the reference
  declares at the top of a block and assigns in each branch is the case in point). Judge such a hit
  on its merits: allow the lint in `[lints.clippy]` with the reason, or restructure — but do not
  leave it sitting as a warning.
- **Never change production code to satisfy a coverage tool's limits.** Add real
  tests or leave the gap documented.

## The reference-parity harness

`dev_lapack` is a development-only feature: `build.rs` compiles
`c_code/reference_lapack.c` against a locally built LAPACK 3.12.1 (point
`ALGEBRA_LINEAR_DEV_LAPACK_ROOT` at a CMake build directory holding
`include/lapack.h` and `lib/liblapack.so`) and links it, adding an `rpath` so the
test binary loads that build rather than whichever `liblapack.so` the loader finds
first. `src/parity.rs` compares the kernels against it.

- **Private by construction:** `mod dev_lapack` and `mod parity`, both gated, are not
  `pub`. Nothing of the harness reaches `cargo doc`, docs.rs or semver — and adding
  a `tests/` integration target would force every symbol into the public API, so
  keep it a unit-test module.
- **Prototypes live in `c_code/reference.h`**, and `build.rs` tracks it: `cc` does
  not track includes on its own, so a header edit otherwise leaves a stale object.
  A definition that drifts from its declaration is a compile error, which is the
  point of the header.
- **12 entry points, no name prefixing.** There is no second copy of anything here,
  so the reference bindings keep their plain names.
- **Tolerances:** `1e-10` relative for anything involving an elimination or an
  iteration. Results that are unique (an inverse, a Cholesky factor, singular
  values) are compared element-wise; results that are not (eigenvectors, eigenvalue
  order) are checked against their defining equation or a canonical order. An FMA
  kernel reassociates sums, so a tolerance change is a decision to state in the PR,
  never a constant to nudge.
- **Run it alone.** Combining `--features dev_lapack` with anything that links
  another BLAS implementation loads two of them, which is a smoke test at best.

## Traps this port has already paid for

- **An index, extent, or length taken from the wrong axis** is the recurring bug:
  `zherk_rank` conjugated the second index, `dsyrk_rank` contracted over the wrong
  dimension, `zgebal` scaled the wrong slice, `norm_char` mapped discriminants by
  position. Check a twin routine against its original and compare.
- **When `dgeev` looks inaccurate, check `dgebal` before suspecting the QR
  iteration.** The missing balancing step once looked like a `dlahqr` bug.
- **Port the reference literally.** Do not re-derive an algorithm whose source is
  available; two hand-derived attempts (`dgehrd`, `dlahqr`) compiled, ran, looked
  orthogonal and were wrong, and every success came from transcription.
- **One implementation per routine.** Two `dlaln2` copies drifted and disagreed in a
  subnormal branch; they are one function in `auxiliary.rs` now. Do not add a second
  copy of anything that computes.
- **`dlarfg`'s underflow branch** scales *up* by `RSAFMN = 1/SAFMN` where
  `SAFMN = DLAMCH('S')/DLAMCH('E')` (1e-292, not `DLAMCH('S')`), and takes `BETA`
  back by repeated multiplication because the power overflows.
- **`dlange`/`zlange` accept `'O'`** as well as `'1'` for the one-norm; rejecting it
  is a panic, not a wrong number.
- **`dpotrf`'s triangle:** LAPACK's `'U'` *reads* only the upper triangle (so the
  other half may be zeros and must be mirrored) and *writes* only the upper triangle
  (so the rest must be left alone).
- **A test that cannot fail is worse than no test.** Four times during this port a
  test passed while checking nothing — decomposing a zeroed matrix, reading back the
  matrix the routine had already overwritten, never inspecting the output it claimed
  to check. When adding a test, make a deliberate mistake and watch it fail.
- **LAPACK's own fixtures can be stale.** `TESTING/dbal.in`'s fifth example still
  states an older `dgebal` permutation order than 3.12.1 produces, and `dchkbl` only
  *reports* its largest error instead of asserting it. Verify a fixture against the
  reference binary before embedding it.

## Measured state

```bash
cargo test --lib                                  # the kernels and their fixtures
ALGEBRA_LINEAR_DEV_LAPACK_ROOT=/path/to/lapack/build \
  cargo test --lib --features dev_lapack          # the same, plus the parity tests
cargo build --lib --target wasm32-unknown-unknown # builds
cargo test --lib --target wasm32-wasip1           # the same tests on wasm
```

The `dev_lapack` run is the ordinary suite plus the reference-parity tests; it adds
tests, it does not replace any. `fmt`, `clippy --all-targets` (both configurations)
and `cargo doc` are clean. The counts are not quoted here on purpose: read them off
the run, which is the only thing that cannot go stale.

Every driver the crate exposes is implemented natively — `dgesv`, `dgetrf`,
`dgetri`, `dpotrf`, `dsyev`, `dgesvd`, `dgeev` (with balancing and left
vectors -- there is no eigenvalues-only entry: the transcription computes the
vectors, so a caller that only wants the spectrum pays for them, at no measurable
penalty), `dggev`, and the complex `zgesv`, `zgetrf`, `zgetri`, `zpotrf`, `zheev`,
`zgeev`, `zgesvd`, `zggev` — with no `not_implemented` guards left.

## Performance: the blocking boundary

Worth knowing before optimizing, because the obvious move is the wrong one.

`dgemm` is a Goto-style blocked kernel with a 4x4 AVX2+FMA micro-kernel
(MC=128, KC=256, NC=256). There is no `portable` module and no `#[cfg]` alternate:
`wasm32` takes the same kernel and gets its scalar fallback, and the suite runs
there under `wasmtime`.

**The LAPACK layer is BLAS-3-free end to end**, so the blocked `dgemm` does not
accelerate any decomposition:

| Routine | Kernel used | BLAS-3? |
|---|---|---|
| `dsyev` | `dsytd2`, reflectors via a hand-written `dgemv` loop | no |
| `dgesvd` / `zgesvd` | `dgebd2` / `zgebd2` | no |
| `dgeev`, `dhseqr` | `dlahqr` (`dlarfg`/`dlanv2`/`dlartg` only) | no |
| `dggev` / `zggev` | `dhgeqz` | no |

Measured single-core, release, through the blocked `dgemm`: a matrix product went
2.2 -> 21 Gelem/s, while `dsyev` runs at 0.4 ns/flop (~2.8 GFLOP/s) and `dgesvd` at
0.2 ns/flop, against the kernel's 0.05 ns/flop.

`dsytrd` was ported anyway and delivered far less than that headroom suggested:
~8-10% on `dsyev` at n >= 400, because the reduction is only ~29% of `dsyev` and
the blocked algorithm still performs the same n^3/3 symmetric product -- only the
rank-2 update becomes BLAS-3. Real headroom for `dsyev` needs the two-stage
`sy2sb` + `sb2st` reduction, which is a different algorithm, not a blocking of
this one. `dgesvd` remains unblocked. The rules of thumb:

* Port the blocked variant only when the target routine's flop count is dominated
  by panel updates that can become `dgemm`. `dlahqr` can never benefit -- it has
  no BLAS-3 to give.
* **A blocked routine can be slower than the unblocked one for small input.**
  `dsytrd` measured slower below n ~ 300 and faster above, so `dsyev` picks by
  size (threshold 128) rather than always paying for the blocking. Always
  blocking would have regressed every small case.
* **Some quantities are not invariants.** The first `dsytrd` test compared the
  reduced `d`/`e`/`tau` entries element-wise against the unblocked reduction, to
  1e-12. That is unsatisfiable and the test was at fault: the two reductions use
  different Householder sign conventions, so their off-diagonals are not
  comparable. Against reference LAPACK's own `dsytrd` and `dsytd2` on the same
  problem, `|e|` differs by 4e-1 at n=128 while the eigenvalues agree to 6e-14,
  and the reference's blocked-vs-unblocked `|d|` alone reaches 3.2e-12. Check the
  spectrum, which is invariant, not the intermediate factors.
* `dlaqr0` has one benefit that is *not* about throughput: aggressive early
  deflation detects converged eigenvalues many sweeps earlier, cutting the
  iteration count independently of kernel speed. It is the only part with no
  `dlahqr` analog.

## Loose ends

- `dhseqr`/`zhseqr` implement the unblocked (`dlahqr`/`zlahqr`) path only, which
  is what the reference uses for n <= 15; the blocked `dlaqr0`/`zlaqr0` fallback
  for large matrices is not ported. Non-convergence `INFO` is *reported*, not
  asserted: `dsyev`/`zheev` (via `dsteqr`/`zsteqr`), `dhseqr` (via `dlahqr`) and
  `dgesvd`/`zgesvd` (via `dbdsqr`/`zbdsqr`) return `LapackError::Convergence`.
  `dhseqr`, `dgeev` and `dgesvd` are exercised through NaN-matrix tests;
  `dsyev`/`zheev` are not, because a NaN matrix reads as zero under the norm gate
  and takes the reference's `ANORM == 0` early return.
- `dgesvd`/`zgesvd` use the direct bidiagonalization path for both shapes, not
  the QR/LQ pre-reduction the reference uses for very tall or wide inputs.
  Agreement with the reference is ~1e-15.
- `--features dev_lapack` is a local command, never a CI job: it needs a LAPACK
  build on the machine and must not be combined with a linked BLAS.

## Coverage: deliberately uncovered lines

The gate is `cargo llvm-cov --lib --lcov` followed by `zscripts/check-coverage.bash`;
CI runs both in its `coverage` job. It counts each source line once — the `DA:`
records — because `--fail-under-lines` counts a line per monomorphization and would
report dimension-dispatching code as partly uncovered.

What follows records *why* a line cannot be covered, not how many there are: the
number belongs to the gate, and this list is what a reader needs when the gate names
a file. **A PR that adds code no test can reach adds it here in the same PR** —
otherwise the next person spends a day rediscovering why a file sits below the bar.

| where | why |
|---|---|
| `src/lapack/qz.rs` | `dhgeqz`'s and `dggbal_p`'s rescue and permutation arms, `dtgevc`'s guard paths. Sweeping `dgeev` and `dggev` over the reference's own spectra — including a singular `B` — moved this file by zero lines, because those families are built to be *solvable* and cannot reach code whose job is to refuse or rescue. `dget54`, the checker for those factors, needs `Q` and `Z` out of `dggev`; the API does not grow for it, so that check is not written here |
| `src/lapack/zeig.rs` | `zsteqr`'s remaining deflation arms, `zlarfg`'s underflow rescue, `zlatrs_upper`'s rescaling |
| `src/lapack/eig.rs` | `dlaln2` and `dlanv2`: 2x2 blocks whose entries force a rescale |
| `src/lapack/sym.rs`, `svd.rs`, `zsvd.rs` | `dsteqr`'s remaining arms, `dbdsqr`/`zbdsqr` convergence for repeated and zero singular values |
| `src/lapack/families.rs` | harness argument checks the tests do not exercise, the `wn == 0` arm of a reflector built from a zero random vector in `dlagge`/`dlagsy` (probability zero), `dlatms`'s `INFO = 2` guard for an all-zero spectrum (unreachable: `latm1` always scales the maximum to one), and the assertion-failure lines of the fixture tests |
| `src/lapack/checkers.rs` | the `TESTING/LIN` checkers' structural-violation penalty and a row-swap arm no failing fixture reaches, plus assertion-failure lines |
| `src/lapack/examples.rs` | short-circuited false arms of the two example checkers. `DGET33`'s structural-violation penalty and `DGET31`'s `scale > 1` and `INFO ∉ {0, 1}` penalties are unreachable for a correct routine (`DGET31`'s own run reports `NINFO(1) = 0`); the tails of the `&&` chains in the `RES = ZERO` tests never evaluate; and the 2x2 complex arm of the `INFO` transcription is never selected by this grid, whose largest pivoted entry is always a diagonal one |
| `src/lapack/auxiliary.rs` | the shared helpers' rescue arms: `dladiv`'s rescale-branch lines are constant-folded out when the fixture's literals reach the inliner (its twelve cases match the reference bit for bit, but the compiler removes the branches it can decide), `lascl_g`'s guards for a zero `cfrom` and for overflow, and the degenerate blocks of `dlae2`/`dlaev2` that the symmetric sweeps never produce |
| `src/lapack/mod.rs` | the shape guards no test constructs (a zero dimension, a zero right-hand side), and `not_implemented`, which is retained so a reverted routine fails with a message naming the feature instead of a link error — no routine returns it |
| `src/blas.rs` | one skipped-column arm (`zgemv` with a zero scalar) and the argument lines of two assertion messages |
| `src/sparse/ordering/colamd.rs` | the garbage-collection and super-column repair arms are reached only by fill patterns larger than the fixtures; `SparseError::IndexOverflow` is unreachable at test sizes |
| `src/sparse/supernodal/lu.rs` | the L/U diagonal-structure guards are drop-invariant checks: a correct supernode partition cannot reach them |
| `src/sparse/mod.rs`, `src/sparse/solver.rs`, `src/sparse/csc.rs`, `src/sparse/lu.rs` | accessor bodies and the `SparseError` `Display` arms that no test calls; they are not refusal arms |

So the remainder is rescue and refusal code, by design: a correct routine cannot
reach the arms whose job is to reject bad input or survive an iteration that gave
up. Three rules follow.

- **Do not refactor production code to satisfy the tool.** A line that reads as
  uncovered may be an inlined region boundary; judge a refactor by the hit counts
  of the lines that work, not by the total.
- **The percentage moves when function layout changes, and that is the tool, not the
  tests.** llvm-cov's `DA:` records include lines that carry no code — comments, a
  `const` declared inside a function, a literal argument on its own line — whenever
  inlining puts a region boundary there. Consolidating two copies of `dlaln2` into
  `auxiliary.rs` moved the ratio while every executed line kept its hit count (the
  `dgebal` entry went from 139 to 139): of `eig.rs`'s 586 missed lines, 416 had no
  code in them. Say so in the PR instead of hunting the total.
- **When a gap is called unreachable, say what was tried.** The LAPACK rows above are
  trustworthy because five generators were ported and verified against LAPACK's own
  build (`families.rs`) before the sweeps were run.

## Invariants the kernels depend on

These are the rules the code depends on. Each one has been violated at least once.

- **Column-major, always.** Element `(i, j)` of an `n`-row matrix is `data[i + j * n]`;
  `lapack.h` binds the Fortran routines and is column-major too. **`LAPACKE_*` is the
  row-major API and would silently transpose every matrix** — it is never used, and
  the reference harness binds the Fortran symbols for the same reason.
- **A routine that writes one triangle leaves the other alone.** `dpotrf` with `'L'`
  writes only the lower triangle and `'U'` only the upper; so do `dsyrk`, `zherk` and
  their `_rank` twins. A caller's other half may legitimately hold zeros, so mirror
  the triangle you *read* and write back only the requested one.
- **Check the contraction length, not just the output shape.** `C := A Aᵀ` and
  `C := Aᵀ A` are both `(n,n)` but contract over different lengths — `k` from `A`'s
  columns in one case and from its rows in the other. This bit `dsyrk_rank`.
- **`idamax` is 1-based in BLAS and 0-based here.** The first maximum wins.
- **Scale before squaring.** A naive sum of squares overflows for entries around
  `1e200` and underflows around `1e-200`. `dnrm2` and `dznrm2` use Blue's scaling.
- **Non-convergence is an error, not a debug assertion.** An `INFO > 0` from
  `dsteqr`, `zsteqr`, `dlahqr`, `dbdsqr`, `zbdsqr` or the QZ iterations becomes
  `LapackError::Convergence`, never a `debug_assert`: a debug assertion only fires in
  a debug build, so a release build handed back partly converged eigenvalues and
  singular vectors as if they had converged.
- **Conjugation is on the first contracted index in `zherk_rank`** (`sum_l conj(a[l][i])
  * a[l][j]`), and on the second elsewhere (`zherk` uses `a[i][l] * conj(a[j][l])`).
  The wrong choice changes only the imaginary parts of the off-diagonals, which is why
  the test data has to be genuinely complex.

## Releasing

1. `cargo publish --dry-run` -- the tarball is complete and builds on its own. `exclude` in
   `Cargo.toml` keeps the process files (this one, `.github/`, `.gitignore`, `rustfmt.toml`)
   out of it; code, README, LICENSE and the reference harness are what ship.
2. Bump `version`, commit, tag `vX.Y.Z`, push the tag -- CI then repeats the dry run in its
   `Package` step, so a broken package cannot reach crates.io.
3. `cargo publish`.
4. Only after the crate is on crates.io may a dependent crate raise its requirement to the new
   version. Nothing, here or downstream, points at a version that is not published.

## Licence

BSD 3-Clause. The kernels derive from LAPACK 3.12.1 (BSD 3-Clause); `LICENSE`
reproduces that notice verbatim and it must stay verbatim. A kernel taken from
another project (OpenBLAS, say) brings its own notice into the same file — and, for
anything descending from GotoBLAS, the University of Texas at Austin line as well.
