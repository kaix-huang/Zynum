# Roadmap

Zynum `0.0.1-beta` establishes the first public foundation for the project:
full BLAS Level 1-3 coverage, typed Zig vector/matrix views, C/CBLAS/Fortran ABI
compatibility, generated headers/modules, examples, tests, benchmark tooling, and
selected architecture-aware GEMM optimization work.

The long-term goal is a high-performance runtime that can directly replace BLAS,
LAPACK, FFT, sparse linear algebra, CNN, and Transformer acceleration libraries
across ARM, x86, GPU, and NPU backends while preserving Zig ergonomics and
C/Fortran interoperability.

## Beta Line Priorities

| Area | Goal |
| --- | --- |
| API | Stabilize `zynum` and `zynum-blas` package boundaries, typed view names, workspace APIs, and aliasing rules. |
| ABI | Keep standard BLAS and CBLAS symbols compatible, and keep generated headers/modules synchronized with exports. |
| Correctness | Expand randomized compatibility tests, edge-case coverage, complex operations, packed/banded storage, and row-major CBLAS paths. |
| GEMM | Continue architecture-specific SGEMM/DGEMM/CGEMM/ZGEMM optimization while keeping dispatch policy evidence-based. |
| Tooling | Keep CI fast, reproducible, and strict about formatting and generated-header drift. |
| Documentation | Keep usage docs rich, concrete, and conservative about performance claims. |

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
- Supported target matrix must be explicit for ARM, x86, GPU, and NPU targets.
- Native performance claims must be tied to reproducible benchmark data with
  recorded fresh-process isolation when comparators are involved.
- Binary distribution guidance must explain LGPL dynamic/static linking and
  relinking considerations.
