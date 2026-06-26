# Internal Design Guide

This layer is for maintainers and contributors who need implementation detail:
module boundaries, call paths, ABI layering, dispatch ownership, and file split
rules.

## Core Maps

| Area | Primary docs |
| --- | --- |
| Module boundaries and ownership | [`../architecture.md`](../architecture.md) |
| Public API checks and aliasing model | [`../development_and_usage.md`](../development_and_usage.md) |
| ABI and generated compatibility files | [`../fortran_compatibility.md`](../fortran_compatibility.md) |
| Zig 0.16 `std.Io` threading notes | [`../common/zig_0_16_std_io_threading.md`](../common/zig_0_16_std_io_threading.md) |
| BLAS Level 1 implementation notes | [`../common/level1_optimization_notes.md`](../common/level1_optimization_notes.md) |
| BLAS Level 2 implementation notes | [`../common/level2_optimization_notes.md`](../common/level2_optimization_notes.md) |
| GEMM implementation rules | [`../common/gemm_optimization_notes.md`](../common/gemm_optimization_notes.md) |

## Facade Pattern

Stable import roots should stay as facades:

- `src/zynum.zig`
- `src/blas.zig`
- `src/blas/api.zig`
- `src/blas/core.zig`
- `src/blas/core/level1.zig`
- `src/blas/core/level2.zig`
- `src/blas/core/level3.zig`

Concrete implementations should live below those facades, grouped by operation
family or backend responsibility. Preserve facade exports when moving code so
typed APIs, ABI wrappers, and tests do not learn implementation paths.
