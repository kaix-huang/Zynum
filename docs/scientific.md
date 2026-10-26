# Scientific capability and ownership map

This is the entry point for the `scientific` branch, consolidated on 2026-10-26.
It distinguishes implementation, finite external interfaces and application
adoption. Dated tests qualify their own source and SDK; they do not certify
every later combination or target.

## Code ownership

```text
Zig facades / selected generated C and Fortran exports
                      |
          checked / LAPACKE / Sparse / FFTW adapters
                      |
           one configured src/zynum.zig Core
                      |
       LAPACK / Sparse / FFT / common operator algorithms
                      |
          native storage, Context and scratch leases
                      |
           shared execution and resource modules
```

| Responsibility | Source owner | Boundary |
| --- | --- | --- |
| Public Zig imports | `src/facades` | Stable aliases, no numerical state |
| Implementation graph | `build.zig: configure`, `src/zynum.zig` | One configured Core and shared named modules per artifact |
| Scalar/dense/sparse storage | `src/blas/types.zig`, `src/numerics/dense.zig`, `storage.zig`, `src/sparse` | Native internal dimensions and independent sparse offset/ordinal widths |
| Admission and workspace | `src/numerics/execution/context.zig`, `src/numerics/resources.zig` | Captured allocator, operation and pin accounting, borrowed or leased storage |
| Threads and policy | `src/numerics/execution` | Shared executor; BLAS cache-domain bridge stays in BLAS |
| Dense mathematics | `src/lapack/core` | Borrowed numerical kernels; managed wrappers own admission and translation |
| Tensor transforms | `src/fft/core.zig`, `managed.zig` | Internal plan/workspace representation independent of FFTW wire types |
| Sparse mathematics | `src/sparse/core`, `operations.zig`, `triangular.zig` | Checked products/solves and native sparse objects |
| Reusable algorithms | `src/numerics` | Operator-based polynomial, Krylov, projection and contour composition |
| Foreign contracts | `src/interop`, `src/checked` | Wire conversions, callbacks, workspace/query and error behavior |
| Interface generation | `tools/generate_interfaces.py` and registered renderers | Optional compile-time declarations/forwarders and provenance |

Within an artifact, compatible facades and adapters share the actual Core,
Execution, Resources and options module objects. A separately linked library
has its own globals. The unified artifact is the integration product for calls
that must share runtime state; opaque handles cannot cross independent owners.

GESVD and complex GELSD now reuse the shared bidiagonal SVD implementation.
The partial-factor helper owns upper-bidiagonal orientation and representability
checks over borrowed buffers; the LAPACKE adapter owns public S/U/VT and
WORK/RWORK copyback. Full-U staging reuses an existing matrix lease. These
paths introduce no new Context, allocator, executor or handle owner.

## Capability map

| Area | Implemented scope | External/application qualification boundary |
| --- | --- | --- |
| BLAS | Level 1–3 real/complex; checked views; portable and dispatched kernels | CBLAS/Fortran and existing native suites; platform evidence remains target-specific |
| LAPACK foundation | Four-scalar Cholesky/LU/QR, solves, tridiagonal eigenproblems, SVD and least-squares families | Finite LP64/ILP64 selections; ABPLaS directly uses `dgesv`, `zpotrf`, `dstev`, `dgelsd` |
| LAPACK extensions | Explicit Q, GELS/GELSS, complex GELSD, GESVD, triangular/inverse and symmetric/Hermitian drivers | Optional generated components; installed foreign consumers are separate from application adoption |
| GESVD partial convergence | Upper bidiagonal diagonal/edges and requested factors, actual unfinished-edge INFO; finite/range faults separate | Three native modes exercise failure paths; production C tests exercise normal high/work/query calls, not an injected iteration budget |
| Sparse | Real/complex CSR SpMV/SpMM, independent indices, CSC/COO conversion, triangular solves and operators | Checked mixed-index exports and finite MKL-style same-width profiles; application batching is qualified separately |
| FFT | f32/f64 C2C, real forward/inverse, real-to-real, tensor, strided/batched/guru and split-plane routes | Finite 48-function FFTW profiles per precision; ABPLaS uses its f64 route; complete FFTW and wisdom/planner equivalence remain open |
| Eigenproblems | Complex Hermitian standard/generalized and real symmetric routes | Finite generated drivers and application selection; ELPA remains a separately qualified provider |
| Common operators | Chebyshev, normalized recurrence, Lanczos/chain, subspace projection, BiCGStab, rational filtering and contour composition | Native algorithms and optional checked callback/agreement interfaces; full FEAST compatibility is unqualified |

The exact callable selection is its generated manifest plus the build flags.
There is no universal full-MKL, full-LAPACK, full-Sparse or full-FFTW claim.
Core capability also does not imply that a public Zig facade exposes every
internal algorithm.

## Build and integration

Start with the [interface guide](interop/README.md) for standalone generation,
LP64 scientific installation and profile selection. Keep the existing default
BLAS installation and optional interfaces independently selectable. The
[architecture reference](architecture.md) describes contracts in detail.

The updated ABPLaS integration has used Zynum and ELPA configurations. Earlier
qualified snapshots passed the original 1,289-case regression plus 12 focused
cases, retaining four original skips, with actual SDK image checks. Those
results belong to their recorded source. Static direct-LAPACKE audit found
eight call sites across six names: `dgesv`, `dgelsd`, `zpotrf`, `dstev`, `zheev`
and `zhegv`. It found no direct complex GELSD or GESVD call. This static audit
is not a claim about indirect runtime calls.

The interrupted newer material pipeline and source-closeout job did not leave
complete terminal acceptance receipts. They remain unqualified. Earlier
qualified material runs converged the compact C8 cold SCF in 11 iterations and
passed their density comparisons. The case named `time_evolve` used Moment and
two fixed samples; full TimeEvolution and statistical convergence remain open.

## Measurements and open work

The local 2026-10-15 module comparison measured 72 shapes, thread caps 1/4,
1,276 fresh processes and 17,588 numerically checked timed calls. It compared
the then-current SDK with available OpenBLAS, Accelerate/vDSP, FFTW and Eigen
routes. ABPLaS shapes were source/trace-derived models with independently
constructed inputs. The observations include major dense/FFT performance gaps;
they are not measurements of this later shared-SVD revision or full SCF time.
The [performance guide](performance/README.md#historical-lapack-sparse-and-fft-comparison-2026-10-15)
retains the 15 ABP model comparisons. Existing public BLAS charts retain their
own 2026-10-10 source identity. No
optimization or new benchmark campaign accompanies this consolidation.

See the [roadmap](roadmap.md), [contributor checks](contributors/README.md),
[historical interface checkpoints](interop/checkpoints.md) and
[development history](development_history.md). Foreign native inventory rows,
broader routine/profile coverage and fresh application/material integration
must be closed on their actual configurations.
