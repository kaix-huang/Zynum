# User Guide

This layer is for people evaluating or integrating Zynum. It avoids internal
dispatch details unless they affect observable behavior.

## Start Here

| Goal | Read |
| --- | --- |
| Build the project and run a first program | [`../development_and_usage.md`](../development_and_usage.md) |
| Use the Zig API | [`../development_and_usage.md#typed-zig-api`](../development_and_usage.md#typed-zig-api) |
| Call BLAS from C, C++, or Fortran | [`../fortran_compatibility.md`](../fortran_compatibility.md) |
| Run examples | [`../../examples/README.md`](../../examples/README.md) |
| Understand beta stability | [`../../README.md#stability`](../../README.md#stability) |
| See the planned module roadmap | [`../roadmap.md`](../roadmap.md) |

## User-Facing Contract

- `zynum` is the top-level Zig package facade.
- `zynum-blas` is the BLAS-only Zig package module.
- The installed library is named `zynum_blas`.
- Standard BLAS and CBLAS ABI symbols keep their conventional names.
- `ZYNUM_MAXIMUM_THREADS` is the only Zynum-specific environment variable.

Avoid depending on files under `src/blas/core/`, `src/blas/kernels/`, or
`src/blas/gemm/` from downstream projects. Those are implementation details
during the beta line.
