# Changelog

All notable changes to Zynum will be documented in this file.

Zynum uses Semantic Versioning after the first stable release. Before 1.0, the
Zig API, package layout, runtime controls, and performance policy may still
change. Standard BLAS ABI compatibility is a core project goal.

## 0.0.1-beta - 2026-06-22

First public beta release for <https://github.com/kaix-huang/Zynum>.

### Added

- Full Zynum BLAS (`zynum-blas`) module with BLAS Level 1, Level 2, and Level 3
  coverage.
- Typed Zig vector and matrix view APIs with checked dimensions, storage, strides,
  and aliasing in safe build modes.
- Descriptive Zig operations including `matrixMultiply`, `matrixVectorMultiply`,
  `addScaledVector`, `scaleVector`, `scaleVectorInto`, and workspace-oriented
  matrix multiplication helpers.
- Standard Fortran BLAS ABI exports such as `dgemm_`, `zgemm_`, `daxpy_`, and
  `zaxpy_`.
- Standard CBLAS exports such as `cblas_dgemm`, `cblas_zgemm`,
  `cblas_daxpy`, and `cblas_zdotc_sub`.
- Generated compatibility files under `include/zynum/blas/`:
  - `cblas.h`
  - `blas.h`
  - `blas.f90`
- Generated Fortran 2003 `iso_c_binding` module named `zynum_blas_fortran`.
- Portable reference implementations for BLAS correctness and compatibility.
- Architecture-aware GEMM dispatch and optimized no-transpose GEMM paths for
  selected AArch64 and x86_64 targets.
- AArch64 GEMM experiments covering ASIMD, SVE2, SME-oriented assembly paths, and
  Apple AMX controls where available.
- x86_64 GEMM experiments covering baseline/SSE-family paths and AVX/AVX2/AVX512
  planning notes.
- Runtime controls for thread limits and experimental GEMM worker strategies via
  `ZYNUM_BLAS_*` environment variables.
- Benchmark executables for quick BLAS comparisons and full GEMM shape sweeps.
- Fresh-process GEMM sweep helper for cleaner comparisons against Accelerate,
  OpenBLAS, MKL, and other BLAS libraries.
- Python plotting tool for GEMM sweep CSV output.
- Zig, C/CBLAS, and Fortran matrix multiplication examples.
- CI workflow covering formatting, generated-header drift, tests, install build,
  and example smoke checks.
- Compatibility tests for typed Zig APIs, CBLAS wrappers, Fortran wrappers,
  generated headers, row-major CBLAS behavior, complex operations, packed/banded
  cases, triangular routines, and randomized small-matrix reference checks.
- Public documentation for architecture, development and usage, C/Fortran
  compatibility, benchmarking methodology, GEMM optimization notes, release
  preparation, and project roadmap.

### Optimized

- GEMM implementation structure for aggressive optimization while keeping policy,
  dispatch, target-feature detection, packing, and instruction-specific kernels
  separated.
- Selected real no-transpose SGEMM/DGEMM paths on supported AArch64/x86_64 targets.
- Benchmark workflow for reproducible shape sweeps, isolated comparator runs, and
  conservative performance evidence.

### Documented

- Project repository: <https://github.com/kaix-huang/Zynum>.
- LGPL-3.0-or-later licensing and downstream linking considerations.
- Beta stability expectations and pre-1.0 compatibility policy.
- BLAS ABI compatibility goals and generated-header regeneration workflow.
- Target/performance claim rules: native throughput evidence is required for
  reportable performance statements.

### Known limitations

- `0.0.1-beta` is not a stable 1.0 API contract.
- Experimental runtime switches, worker strategies, dispatch thresholds, and
  benchmark output formats may change.
- Architecture-specific GEMM performance depends on target features, CPU, OS,
  thread counts, comparator defaults, and thermal state.
- Future modules such as LAPACK, FFT, sparse, CNN, Transformer, tensor, and random
  number generation are planned but not included in this release.
