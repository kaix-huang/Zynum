# Roadmap

Zynum `0.0.1-beta` establishes the first public foundation for the project:
full BLAS Level 1-3 coverage, typed Zig vector/matrix views, C/CBLAS/Fortran ABI
compatibility, generated headers/modules, examples, tests, benchmark tooling, and
selected architecture-aware GEMM optimization work.

The long-term goal is a high-performance runtime that can directly replace BLAS,
LAPACK, FFT, sparse linear algebra, CNN, and Transformer acceleration libraries
across portable and architecture-specific CPU kernels while preserving Zig
ergonomics and C/Fortran interoperability.

## Beta Line Priorities

| Area | Goal |
| --- | --- |
| API | Stabilize `zynum` and `zynum-blas` package boundaries, typed view names, workspace APIs, and aliasing rules. |
| ABI | Keep standard BLAS and CBLAS symbols compatible, and keep generated headers/modules synchronized with exports. |
| Correctness | Expand randomized compatibility tests, edge-case coverage, complex operations, packed/banded storage, and row-major CBLAS paths. |
| GEMM | Continue architecture-specific SGEMM/DGEMM/CGEMM/ZGEMM optimization while keeping dispatch policy evidence-based. |
| Tooling | Keep CI fast, reproducible, and strict about formatting and generated-header drift. |
| Documentation | Keep usage docs rich, concrete, and conservative about performance claims. |

## 0.1.x Release Line

The `0.1.x` line is the BLAS-completion and performance line. It should remain
focused on Zynum BLAS rather than expanding into future numerical modules.

Release goals:

- Complete practical support for the full BLAS surface: every Level 1, Level 2,
  and Level 3 routine, all real and complex scalar types, packed/banded/triangular
  and symmetric/Hermitian variants, CBLAS row-major wrappers, Fortran ABI entry
  points, generated C headers, and generated Fortran module declarations.
- Keep typed Zig APIs aligned with the BLAS surface while preserving checked
  views, workspace APIs, and explicit aliasing rules.
- Support ARM and x86 CPU families with portable fallbacks plus feature-aware
  kernels for the important architecture tiers:
  - AArch64 ASIMD, SVE/SVE2 where available, Apple AMX, and SME/SME2 paths on
    Apple Silicon and other capable ARM systems.
  - x86_64 baseline/SSE-family, AVX, AVX2/FMA, AVX512/FMA, and current Intel and
    AMD desktop/server CPUs.
- Make Apple's latest production silicon the primary native performance gate.
  The target for 0.1 is that Zynum beats Accelerate across the documented BLAS
  benchmark suite on the latest Apple chips, with `ZYNUM_MAXIMUM_THREADS` unset
  unless a single-thread gate is being measured.
- Treat any performance claim against Accelerate, OpenBLAS, MKL, or another
  comparator as invalid unless it has fresh-process isolation, exact commands,
  CSV artifacts, runtime thread counts, target features, and environment records.
- Keep optimization policy in descriptors, tuning files, dispatch, and
  documented gate records; micro-kernels should stay as small and direct as the
  target ISA allows.

Non-goals for 0.1.x:

- No external BLAS, LAPACK, Accelerate, OpenBLAS, MKL, BLIS, MPS, cuBLAS, or
  accelerator library may become a Zynum compute path.
- Do not add new public environment-variable controls beyond
  `ZYNUM_MAXIMUM_THREADS`.
- Do not broaden dispatch gates based on single-point local wins.

## Near Term

- Stabilize public Zig API naming around typed `Vector`/`Matrix` views and
  descriptive operation names.
- Keep the top-level `zynum` facade small and predictable while allowing
  BLAS-only consumers to import `zynum-blas` directly.
- Expand CI across native macOS, native Linux, explicit AArch64, and x86_64
  cross-target checks.
- Add more focused tests for aliasing, workspace APIs, invalid dimensions,
  complex scalar behavior, and generated compatibility headers.
- Keep fresh-process GEMM sweeps as the path for reportable comparator data;
  in-process sweeps remain smoke checks unless no comparator libraries are
  loaded.
- Define a target/performance matrix that distinguishes build coverage,
  correctness coverage, and native throughput evidence.
- Add small benchmark fixtures that validate benchmark tools without committing
  large generated result files.

## BLAS And GEMM Work

- Preserve full BLAS Level 1-3 coverage while improving edge-case compatibility.
- Improve small and medium SGEMM/DGEMM latency without broad shape regressions.
- Build dedicated complex GEMM paths instead of relying only on repeated real
  transformations where target-specific kernels can do better.
- Keep worker-pool, non-default GEMM IO, SME, AMX, AVX2, and AVX512 experiments
  opt-in until native fresh-process evidence justifies a default rule.
- Record retained dispatch gates with target features, exact shape predicates,
  runtime environment variables, commands, CSV paths, and rollback criteria.
- Add better x86_64 validation on real Intel and AMD hardware.
- Keep shape policy in `src/blas/gemm/dispatch.zig`; keep instruction details
  under `src/blas/kernels/`.

## Future Modules

Candidates for future modules:

- `zynum-lapack` for dense factorizations, solvers, eigenvalue/SVD routines, and
  LAPACK-compatible entry points.
- `zynum-fft` for transform routines and compatibility layers.
- `zynum-sparse` for sparse storage, sparse BLAS, and solver-oriented kernels.
- `zynum-cnn` for convolution and neural-network kernels.
- `zynum-transformer` for attention, matmul, normalization, and transformer
  inference/training primitives.
- `zynum-random` for random number generation.
- `zynum-tensor` for shared tensor abstractions that do not belong to BLAS.

Each module should get its own source root, tests, benchmark tooling,
documentation, and compatibility headers/modules when the domain needs a C or
Fortran surface.

## 1.0 Release Goals

Before a stable 1.0 release:

- Public API naming must be reviewed and documented as stable.
- ABI compatibility policy must be written and tied to release notes.
- Threading and environment variable semantics must be stable.
- Supported target matrix must be explicit for ARM, x86, operating systems, and
  compiler target-feature profiles.
- Native performance claims must be tied to reproducible benchmark data with
  recorded fresh-process isolation when comparators are involved.
- Binary distribution guidance must explain LGPL dynamic/static linking and
  relinking considerations.
