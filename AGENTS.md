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
- **Keep the tree rustfmt-clean** (`rustfmt.toml` here sets `max_width = 120`); CI enforces this.
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

## Measured state (2026-10-21)

```bash
cargo test --lib                                  # 139 passed, 0 failed
ALGEBRA_LINEAR_DEV_LAPACK_ROOT=/path/to/lapack/build \
  cargo test --lib --features dev_lapack          # 152 passed, 0 failed
cargo build --lib --target wasm32-unknown-unknown # builds
```

The second run is the same 139 plus the 13 reference-parity tests. `fmt`, `clippy
--all-targets` (both configurations) and `cargo doc` are clean. Re-measure rather
than trusting these numbers when you touch the kernels.

Every driver the crate exposes is implemented natively — `dgesv`, `dgetrf`,
`dgetri`, `dpotrf`, `dsyev`, `dgesvd`, `dgeev` (with balancing, left vectors, and an
eigenvalues-only entry), `dggev`, and the complex `zgesv`, `zgetrf`, `zgetri`,
`zpotrf`, `zheev`, `zgeev`, `zgesvd`, `zggev` — with no `not_implemented` guards left.

## Licence

BSD 3-Clause. The kernels derive from LAPACK 3.12.1 (BSD 3-Clause); `LICENSE`
reproduces that notice verbatim and it must stay verbatim. A kernel taken from
another project (OpenBLAS, say) brings its own notice into the same file — and, for
anything descending from GotoBLAS, the University of Texas at Austin line as well.
