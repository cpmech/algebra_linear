# algebra_linear -- Pure-Rust BLAS and LAPACK

No external library, no C in a normal build, nothing to link — it compiles for any
target, including `wasm32-unknown-unknown`.

## What is here

| module | contents |
|---|---|
| `blas` | level 1–3: `dgemm` (+`tn`/`nt`/`tt`), `dgemv` (+trans), `ddot`, `dnrm2`, `dasum`, `idamax`, `daxpy`, `dlange`, `dger`, `dsyrk`, and the complex twins `zgemm` (+transposes), `zgemv` (+trans), `zger`, `zheev`-support kernels, `zherk`, `zsyrk`, `zaxpy`, `zscal`, `dz nrm2` |
| `lapack` | drivers: `dgesv`, `dgetrf`, `dgetri`, `dpotrf`, `dsyev`, `dgesvd`, `dgeev` (+left vectors, eigenvalues only), `dggev`, and the complex `zgesv`, `zgetrf`, `zgetri`, `zpotrf`, `zheev`, `zgeev`, `zgesvd`, `zggev`, plus the auxiliary routines, the `TESTING/LIN` accuracy checkers and the `TESTING/MATGEN` families the tests use |

Not every BLAS or LAPACK routine is here: the port covers the drivers a dense
linear-algebra caller needs. Each function names the reference source it follows (for example
"transcription of `dsytrd`/`dlatrd`") so it can be checked against LAPACK 3.12.1.

## Conventions

- **Column-major**, so element `(i, j)` of an `nrow`-row matrix is `data[i + j * nrow]`.
- **0-based indices** and `usize` lengths, unlike the Fortran the routines are derived
  from.
- **No workspace arguments.** The workspace a Fortran routine asks the caller for is
  allocated internally here.
- Functions return `Result<(), LapackError>` rather than LAPACK's `INFO`, with
  `INFO < 0` mapped to `BadShape` and `INFO > 0` to `Singular` or `Convergence`.
- An AVX2+FMA micro-kernel is selected at run time on x86-64; every other target
  (including `wasm32`) takes the portable blocked kernel.

## Tests

```bash
cargo test                      # 139 tests: the kernels against closed-form results
```

The reference-parity harness is development-only and lives behind a feature: it
compiles `c_code/reference_lapack.c` against a locally built LAPACK 3.12.1 and
compares the kernels against it.

```bash
ALGEBRA_LINEAR_DEV_LAPACK_ROOT=/path/to/lapack/build \
  cargo test --features dev_lapack            # 152 tests
```

`ALGEBRA_LINEAR_DEV_LAPACK_ROOT` must be a CMake build directory holding
`include/lapack.h` and `lib/liblapack.so`. There is no default: recording one would
put a developer's directory layout into released source.

## License

BSD 3-Clause. The kernels are a transcription of LAPACK 3.12.1 and its reference
BLAS, whose copyright notice and conditions are reproduced verbatim in
[`LICENSE`](LICENSE) and apply to the derived code; the Rust transcription is
copyright Dorival Pedroso. If you redistribute a binary that statically links this
crate, that notice must travel with it (BSD 3-Clause, clause 2).
