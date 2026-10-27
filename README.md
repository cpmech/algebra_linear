# algebra_linear - Pure-Rust BLAS, LAPACK, sparse LU, and AMG

<p align="left">
  <img src="docs/web-assembly-logo.png" width="96" alt="WebAssembly logo"><br>
  <sub>Compiles for <code>wasm32-unknown-unknown</code> and <code>wasm32-wasip1</code></sub>
</p>

No external library, no C in a normal build, nothing to link — it compiles for any
target, including `wasm32-unknown-unknown`.

## What is here

| module | contents |
|---|---|
| `blas` | level 1–3: `dgemm` (+`tn`/`nt`/`tt`), `dgemv` (+trans), `ddot`, `dnrm2`, `dasum`, `idamax`, `daxpy`, `dlange`, `dger`, `dsyrk`, `dtrsv` (+transpose), `dtrsm`, and the complex twins `zgemm` (+transposes), `zgemv` (+trans), `zger`, `zherk`, `zsyrk`, `zaxpy`, `zscal`, `ztrsv` (+transpose), `ztrsm`, `dznrm2` |
| `lapack` | drivers: `dgesv`, `dgetrf`, `dgetrf_rect` (the `nrows x ncols` panel), `dgetri`, `dpotrf`, `dsyev`, `dgesvd`, `dgeev` (+left vectors), `dggev`, and the complex `zgesv`, `zgetrf`, `zgetrf_rect`, `zgetri`, `zpotrf`, `zheev`, `zgeev`, `zgesvd`, `zggev`, plus the auxiliary routines; the `TESTING/LIN` accuracy checkers and the `TESTING/MATGEN` families are test-only and are not compiled into the crate |
| `sparse` | `SparseLu`, an opaque sparse LU over caller-owned CSC arrays, with `LuMethod` choosing the simplicial or supernodal factorization, one analysis reused across value updates, `SparseLuStats` reporting structure, cost and timing, real (`f64`) and complex (`Complex64`) scalars, and the COLAMD ordering |
| `sparse::amgcl` | array-based entry points `amg_setup(row_ptr, col_idx, values, params)` and `krylov_solve(method, &mut amg, rhs, x, params)` for real `f64` CSR arrays, classical Ruge–Stüben and aggregation-based AMG, Gauss–Seidel, damped Jacobi, ILU(0), Chebyshev, SPAI(0/1), AMG-preconditioned CG, BiCGStab, restarted GMRES, and optional native scoped-thread parallelism |

- `sparse` (the sparse LU and its COLAMD ordering) is ported from [faer](https://codeberg.org/sarah-quinones/faer) (MIT License)
- `sparse::amgcl` is ported from [AMGCL](https://github.com/ddemidov/amgcl) (MIT License)

Not every BLAS or LAPACK routine is here: the port covers the drivers a dense
linear-algebra caller needs, sparse LU with a COLAMD ordering, and the core CPU
AMGCL algorithms. Each function names the reference source it follows (for example
"transcription of `dsytrd`/`dlatrd`") so it can be checked against LAPACK 3.12.1.

## Conventions

- **Column-major**, so element `(i, j)` of an `nrow`-row matrix is `data[i + j * nrow]`.
- **0-based indices** and `usize` lengths, unlike the Fortran the routines are derived
  from.
- **No workspace arguments.** The workspace a Fortran routine asks the caller for is
  allocated internally here.
- **Solver-specific sparse formats.** Sparse LU callers provide CSC; AMGCL callers
  provide CSR. The crate does not convert between CSC and CSR.
- **Dense drivers** return `Result<(), LapackError>` rather than LAPACK's `INFO`, with
  `INFO < 0` mapped to `BadShape` and `INFO > 0` to `Singular` or `Convergence`. The sparse
  routines return `SparseError`; the iterative ones return `amgcl::Error`.
- An AVX2+FMA micro-kernel is selected at run time on x86-64; every other target
  (including `wasm32`) takes the portable blocked kernel.

## Sparse LU

`SparseLu` analyzes a compressed sparse column (CSC) pattern once and factorizes
new values on it; `LuMethod` chooses the simplicial or supernodal factorization,
and `stats()` reports fill, flop estimate and timings after a factorization.

```rust
use algebra_linear::sparse::{LuMethod, SparseLu};

fn main() -> Result<(), algebra_linear::sparse::SparseError> {
    // A = [4 1; 1 3] in CSC form.
    let col_ptr = [0, 2, 4];
    let row_idx = [0, 1, 0, 1];
    let values = [4.0, 1.0, 1.0, 3.0];

    let mut lu = SparseLu::analyze(&col_ptr, &row_idx, LuMethod::auto())?;
    lu.factorize(&col_ptr, &row_idx, &values)?;

    let mut b = [1.0, 2.0];
    lu.solve(&mut b)?;
    assert!((b[0] - 1.0 / 11.0).abs() < 1e-14);
    assert!((b[1] - 7.0 / 11.0).abs() < 1e-14);

    // The same pattern with new values reuses the analysis.
    let new_values = [5.0, 2.0, 2.0, 2.0];
    lu.factorize(&col_ptr, &row_idx, &new_values)?;
    let mut b = [1.0, 1.0];
    lu.solve(&mut b)?;
    assert!((b[1] - 0.5).abs() < 1e-14);
    assert!(lu.stats().unwrap().fill_ratio > 0.0);
    Ok(())
}
```

## AMG and Krylov solvers

The `sparse::amgcl` module accepts raw compressed sparse row (CSR) arrays. This
example builds the one-dimensional Laplacian `tridiag(-1, 4, -1)` on three
unknowns and solves it with CG. At that size the hierarchy is a single level that
the coarse solve handles, so the multigrid path appears in the examples below
rather than here:

```rust
use algebra_linear::sparse::amgcl::{AmgParams, KrylovMethod, SolverParams, amg_setup, krylov_solve};

fn main() -> Result<(), algebra_linear::sparse::amgcl::Error> {
    let row_ptr = [0, 2, 5, 7];
    let col_idx = [0, 1, 0, 1, 2, 1, 2];
    let values = [4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];

    let mut amg = amg_setup(&row_ptr, &col_idx, &values, Some(AmgParams::default()))?;
    let b = [1.0, 2.0, 3.0];
    let mut x = [0.0; 3];
    let report = krylov_solve(
        KrylovMethod::Cg,
        &mut amg,
        &b,
        &mut x,
        SolverParams {
            tolerance: 1e-10,
            max_iterations: 100,
            ..SolverParams::default()
        },
    )?;
    assert!(report.converged);
    println!("{} iterations, x = {x:?}", report.iterations);
    Ok(())
}
```

Execution is serial by default. On native targets, set `Parallelism` independently
in `AmgParams` and `SolverParams` to enable standard-library scoped threads. There
is no persistent worker pool or external runtime. WebAssembly always uses the
serial path. Gauss–Seidel, ILU(0) factorization and triangular solves, dense
coarse LU, and sequential coarsening steps remain serial.

Run the examples with:

```bash
cargo run --release --example amgcl_poisson -- 32
cargo run --release --example amgcl_aggregation
cargo run --release --example amgcl_smoothers
cargo run --release --example amgcl_nonsymmetric
cargo run --release --example amgcl_parallel -- 4
```

The module is a scoped core-CPU port, not complete feature-for-feature AMGCL
parity. MPI, GPU backends, block and complex values, user-supplied near-nullspace
vectors, and upstream C++ API compatibility are outside its scope. See
[`AMGCL_PORTING.md`](AMGCL_PORTING.md) for the source map and numerical notes.

## Performance

The `Native` solver of [`russell_sparse`](https://crates.io/crates/russell_sparse) 4.x uses the sparse LU from this crate. The table below compares it with UMFPACK on the benchmark matrices that completed; the data is from [benchmark-russell](https://github.com/cpmech/benchmark-russell).

| Matrix | NROW | NNZ | Native | UMFPACK | UMFPACK / Native |
| --- | ---: | ---: | ---: | ---: | ---: |
| bwm2000 | 2,000 | 7,996 | 1.11 ms | 0.750 ms | 1.47× faster |
| mhd1280b (c) | 1,280 | 22,778 | 0.717 ms | 0.383 ms | 1.87× faster |
| rdb5000 | 5,000 | 29,600 | 39.0 ms | 13.6 ms | 2.88× faster |
| mplate (c) | 5,962 | 142,190 | 3.13 s | 627 ms | 4.99× faster |
| Goodwin_040 | 17,922 | 561,677 | 449 ms | 107 ms | 4.19× faster |
| fp | 7,548 | 848,553 | 904 ms | 154 ms | 5.85× faster |
| xenon1 | 48,600 | 1,181,120 | 55.2 s | 598 ms | 92× faster |

`(c)` marks a complex-valued matrix; rows are sorted by NNZ. **NNZ is the matrix-file non-zero count.** Times are five-run averages from the benchmark on the benchmark machine (Arch Linux, Intel Core i9-13900KF).

That 4.x `Native` solver is fast on small systems: up to about 30,000 non-zeros it takes at most a few tens of milliseconds. It becomes very slow for matrices with more than **1 million** non-zeros: the `xenon1` case (NROW=48,600, 1,181,120 non-zeros) takes **55 s**, and the next real case (`twotone`, NROW=120,750, 1,224,224 non-zeros) did not finish within the 3-minute probe. For matrices with more than **1 million** non-zeros, expect **nearly a minute** or more.

The threshold is not a function of NNZ alone. The complex `mplate` (NROW=5,962, 142,190 non-zeros) takes **3.13 s**, and `RFdevice` (NROW=74,104, 365,580 non-zeros) did not finish in **15 minutes**; sparsity structure and numerical type matter as much as the non-zero count.

UMFPACK is faster than the Native solver on every matrix in the table; the gap grows with the fill-in, reaching **92×** on `xenon1`. The figure below shows the same ratios on a log scale (the dashed line is 1×):

![UMFPACK speed-up over the Native solver](docs/plot_speedup_umfpack.png)

## Tests

```bash
cargo test                      # the kernels against closed-form results and invariants
```

The reference-parity harness is development-only and lives behind a feature: it
compiles `c_code/reference_lapack.c` against a locally built LAPACK 3.12.1 and
compares the kernels against it.

```bash
ALGEBRA_LINEAR_DEV_LAPACK_ROOT=/path/to/lapack/build \
  cargo test --features dev_lapack            # the same, plus the parity tests
```

Do not reach for `--all-features` here: `dev_lapack` is the only feature, so it
selects the harness, which cannot build without a local LAPACK.

`ALGEBRA_LINEAR_DEV_LAPACK_ROOT` must be a CMake build directory holding
`include/lapack.h` and `lib/liblapack.so`. There is no default: recording one would
put a developer's directory layout into released source.

## License

BSD 3-Clause. The dense kernels are based on LAPACK 3.12.1 and BLAS. Their licenses are in the [`LICENSE`](LICENSE) file. The sparse LU and its COLAMD ordering are based on [faer](https://codeberg.org/sarah-quinones/faer) (MIT) and SuiteSparse COLAMD (BSD 3-Clause). The `sparse::amgcl` module follows [AMGCL](https://github.com/ddemidov/amgcl) (MIT). Their notices are also in the license file. The Rust code is by Dorival Pedroso. If you share a binary that statically links this crate, you must include the [`LICENSE`](LICENSE) file (BSD 3-Clause, clause 2).
