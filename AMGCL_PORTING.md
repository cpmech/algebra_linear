# AMGCL porting and integration notes

## References

- Upstream: <https://github.com/ddemidov/amgcl>
- Upstream reference revision: `1fbc24e687b404962c31bc4db598f6c73c0f4333`
- Integrated module: `algebra_linear::sparse::amgcl`
- Scalar/index model: `f64` values and `usize` dimensions/indices
- Execution: serial by default, with optional native standard-library scoped threads

This is a core CPU port, not feature-for-feature parity with upstream AMGCL. It
contains classical Ruge–Stüben and aggregation-based coarsening, ILU(0),
Chebyshev, SPAI(0), SPAI(1), CG, BiCGStab, restarted GMRES, and safe
shared-memory kernels. MPI, distributed memory, GPU backends, block and complex
values, user-provided near-nullspace vectors, and C++ API compatibility are not
included.

## Sparse format contract

AMGCL accepts raw compressed sparse row (CSR) arrays through
`amg_setup(row_ptr, col_idx, values, params)`, which copies and validates them
into the internal `CsrMatrix`. Existing sparse LU solvers accept caller-owned
compressed sparse column (CSC) arrays through `SparseLu`. There is deliberately
no CSC-to-CSR or CSR-to-CSC adapter: the caller provides the format required by
the selected solver. CSR is a row-oriented sparse representation and does not
change the crate's column-major dense-matrix convention.

The only dense conversion in the AMG setup transforms the small coarse CSR
operator into column-major dense storage for the shared LAPACK LU implementation.

## Source map

| Upstream component | Integrated Rust component | Notes |
| --- | --- | --- |
| `backend/builtin.hpp` CSR operations | `src/sparse/amgcl/matrix.rs` | Owned CSR, checked constructors, serial/parallel SpMV, transpose, SpGEMM |
| OpenMP row/vector parallelism | `src/sparse/amgcl/parallel.rs` | Safe scoped threads, deterministic partitions, serial WebAssembly path |
| `coarsening/ruge_stuben.hpp` | `src/sparse/amgcl/coarsening.rs` | Strength graph, C/F split, direct interpolation |
| `coarsening/plain_aggregates.hpp` | `src/sparse/amgcl/coarsening/aggregation.rs` | Symmetric strength graph and greedy aggregate construction |
| `coarsening/tentative_prolongation.hpp` | `src/sparse/amgcl/coarsening/aggregation.rs` | Scalar piecewise-constant tentative prolongation |
| `coarsening/aggregation.hpp` | `Coarsening::Aggregation` | Non-smoothed transfers and over-interpolation scaling |
| `coarsening/smoothed_aggregation.hpp` | `Coarsening::SmoothedAggregation` | Filtered Jacobi smoothing and fixed/estimated damping |
| `detail/spgemm.hpp` | `CsrMatrix::matmul` | Row accumulators with sorted deterministic output |
| `amg.hpp` | `src/sparse/amgcl/amg.rs` | Hierarchy, Galerkin products, cycling, complexities |
| `relaxation/gauss_seidel.hpp` | `Smoother::GaussSeidel` | Serial forward/backward sweeps |
| `relaxation/damped_jacobi.hpp` | `Smoother::DampedJacobi` | Configurable damping |
| `relaxation/ilu0.hpp` | `Smoother::Ilu0` | Zero-fill factors and canonicalized input pattern |
| `relaxation/chebyshev.hpp` | `Smoother::Chebyshev` | Polynomial recurrence and spectral estimates |
| `relaxation/spai0.hpp` | `Smoother::Spai0` | Least-squares diagonal approximate inverse |
| `relaxation/spai1.hpp`, `detail/qr.hpp` | `Smoother::Spai1` | Pattern-matched inverse with local Householder QR |
| Backend coarse direct solver | `src/sparse/amgcl/coarse.rs` | Shared `dgetrf`/`dgetrs` implementation from `algebra_linear::lapack` |
| `solver/cg.hpp` | `src/sparse/amgcl/solver/cg.rs` | Preconditioned CG with SPD breakdown checks |
| `solver/bicgstab.hpp` | `src/sparse/amgcl/solver/bicgstab.rs` | Right-preconditioned BiCGStab |
| `solver/gmres.hpp` | `src/sparse/amgcl/solver/gmres.rs` | Restarted right-preconditioned GMRES |
| Backend vector operations | `src/sparse/amgcl/vector.rs` | Dot, stable scaled norm, axpy, finite checks |

## Intentional API differences

AMGCL is header-only C++ with compile-time backends and value types. This module
has an array-based public API: `amg_setup` takes the CSR arrays and returns an
opaque `Amg`, and `krylov_solve` takes the method, the `Amg` state, the
right-hand side, and the solution. `CsrMatrix`, `CsrRow`, the Krylov solver
types, and the `Preconditioner` trait are internal. The identity preconditioner
is `amg_setup(..., None)`, whose matrix-only state copies the right-hand side to
the solution. Configuration uses plain Rust structs rather than Boost property
trees. The module retains its detailed `Error` type so invalid matrices,
dimensions, parameters, singular pivots, non-finite arithmetic, and Krylov
breakdowns remain distinguishable.

AMG owns its scratch space; mutability avoids hidden shared workspaces and avoids
allocation during serial cycles. `Parallelism` is configured independently on
`AmgParams` and `SolverParams`, and both default to one thread.

Native parallelism uses `std::thread::scope` with fixed balanced partitions.
Workers own disjoint mutable output chunks or return partition-local results.
There is no persistent pool, unsafe lifetime extension, or external runtime.
Small operations use the serial path. WebAssembly always executes serially and
`Parallelism::auto()` returns one thread there.

Gauss–Seidel, ILU(0) factorization and triangular solves, coarse LU, aggregate
formation, C/F splitting, and sparse transpose remain serial. ILU(0) still uses
parallel residual and correction kernels around its serial triangular solve.

## Numerical notes

Restriction is the prolongation transpose and coarse operators are computed as
`R * A * P`. Non-smoothed aggregation divides the coarse Galerkin operator by
`over_interp`. Smoothed aggregation filters weak off-diagonal entries, adjusts
the filtered diagonal, and applies damped-Jacobi interpolation. Power-based
spectral estimates use a deterministic initial vector.

Parallel SpMV and SpGEMM preserve each row's arithmetic order and match their
serial counterparts. Dot products and norms combine partition-local results in
fixed partition order. Repeated runs with identical inputs and configurations
are deterministic; changing thread counts or thresholds can change reduction
roundoff and Krylov iteration counts. SPAI(1) assigns whole rows to workers and
preserves each local Householder QR operation.

The coarse solver reuses the crate's column-major `dgetrf` factorization and the
same `dgetrs` solve phase used by `dgesv`. The factorization is computed once
during setup and reused during cycles. Consequently, coarse-pivot semantics
follow the crate's LAPACK implementation rather than maintaining a second AMG-
specific LU routine.

Floating-point operation order and sparse kernels differ from upstream C++.
Tests assert mathematical convergence and true residuals rather than upstream-
specific floating-point traces or bitwise equality with C++ AMGCL.
