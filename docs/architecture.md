# Architecture

Zynum is organized as a top-level project with numerical submodules. The first
submodule is Zynum BLAS (`zynum-blas`).

## Module Boundary

Zynum keeps Zig package facades, implementation modules, and ABI export roots
separate so each surface has a narrow responsibility:

- `src/zynum.zig` is the top-level package facade. It exposes `pub const blas`
  as the explicit BLAS namespace and currently re-exports the BLAS API at the
  top level for convenience. It should not contain BLAS implementation details.
- `src/blas.zig` is the Zynum BLAS (`zynum-blas`) Zig module root. Import it
  directly when a build only wants the BLAS submodule, or through `zynum.blas`
  when importing the top-level package. It exposes typed Zig API names, shared
  types, and runtime controls, but it is not the native ABI export root.
- `src/blas/api.zig` is the checked public Zig BLAS API facade.
- `src/blas/` contains all implementation files for the BLAS module.
- `src/blas/compat.zig` is the ABI export root used to build the `zynum_blas`
  shared and static libraries. It imports the Fortran and CBLAS ABI modules so
  their `pub export` symbols are present in the final artifact.
- `src/blas/compat_fortran.zig` and `src/blas/compat_cblas.zig` are build-module
  roots for testable Zig compatibility imports.
- `src/blas/compat/fortran.zig` and `src/blas/compat/cblas.zig` are leaf facades
  that re-export ABI functions and constants as ordinary Zig declarations for
  tests and compatibility-focused consumers.
- `include/zynum/blas/` contains generated compatibility files for C, CBLAS,
  and Fortran users.

Future modules should follow the same shape:

```text
src/<module>.zig
src/<module>/
include/zynum/<module>/
docs/<module or platform>/
```

## Public Zig API

- `src/blas/api.zig` is the public BLAS API facade.
- `src/blas/api/views.zig` owns checked vector and matrix views.
- `src/blas/api/aliasing.zig` owns Debug/Safe alias checks.
- `src/blas/api/operations.zig` translates user-facing operations into core
  BLAS calls.

Public Zig names should be descriptive. Use names such as `matrixMultiply`
instead of exposing BLAS abbreviations unless the abbreviation is the domain
term itself.

Default output operations use a no-alias contract. If input/output aliasing is
needed, expose it explicitly through `Into` or `WithWorkspace` APIs.

## Core Reference Layer

- `src/blas/core.zig` is the internal BLAS semantics facade.
- `src/blas/core/scalar.zig` owns scalar arithmetic, complex helpers, BLAS
  character parsing, and enum aliases.
- `src/blas/core/indexing.zig` owns vector, dense, packed, and banded indexing.
- `src/blas/core/operands.zig` owns unchecked internal operand structs.
- `src/blas/core/operations.zig` owns readable execution entry points.
- `src/blas/core/level1.zig` owns portable BLAS Level 1 behavior.
- `src/blas/core/level2.zig` and `src/blas/core/level3.zig` are stable internal
  facades; `src/blas/core/level2/` and `src/blas/core/level3/` group portable
  implementations by operation family.

The typed Zig API validates inputs. Core operands do not validate inputs and
should remain small data carriers.

## File Ownership And Split Rules

Prefer small, purposeful files over broad utility buckets, but do not split a
file just because it is long. Split when ownership becomes ambiguous or when a
new behavior creates an independently testable unit.

Recommended split triggers:

- A BLAS level file grows multiple unrelated operation families with different
  invariants. Split by operation family, for example `level2/general.zig`,
  `level2/triangular.zig`, `level3/gemm.zig`, or `level3/symmetric.zig`.
- An ABI file accumulates generated-looking wrapper groups that are easier to
  review by BLAS level. Split into level-specific ABI leaves while keeping the
  public export root stable.
- A GEMM kernel file mixes target-feature detection, packing, micro-kernels, and
  dispatch policy. Keep feature detection in `features.zig`, task policy in
  `gemm/dispatch.zig`, and instruction details under `kernels/<arch>/`.
- A test file starts combining unrelated public API, ABI, and kernel behaviors.
  Split tests by the surface being validated: API shape/aliasing, CBLAS ABI,
  Fortran ABI, generated headers, and GEMM correctness. Keep surface-level test
  roots under `test/api/`, `test/abi/`, or `test/headers/` as appropriate.

When splitting, preserve import roots and public module names unless the change is
explicitly documented as breaking. Prefer facade files that re-export smaller
leaves over moving public names directly.

## GEMM Fast Path

- `src/blas/core/level3.zig` is the Level 3 facade. `level3/gemm.zig` detects
  no-transpose real GEMM and delegates.
- `src/blas/gemm/dispatch.zig` owns shape policy, task splitting, and threading
  choices.
- `src/blas/gemm/pool.zig` owns the optional persistent worker pool experiment.
- `src/blas/kernels/backend.zig` selects the backend by target features.
- `src/blas/kernels/gemm_task.zig` defines shared task shapes.
- `src/blas/kernels/generic/gemm.zig` is the portable backend.
- `src/blas/kernels/aarch64/` and `src/blas/kernels/x86_64/` contain
  architecture-specific kernels.

Keep shape policy in `gemm/dispatch.zig`. Keep instruction details in
`kernels/`.

## ABI And Compatibility

Compatibility is intentionally layered:

1. ABI implementation files mirror external symbol contracts:
   - `src/blas/abi/fortran.zig` exports classic Fortran BLAS symbols.
   - `src/blas/abi/cblas.zig` exports CBLAS symbols.
2. `src/blas/compat.zig` is the native library export root for `libzynum_blas`.
   It exists to pull both ABI modules into the shared/static library build and
   should stay minimal.
3. `src/blas/compat_fortran.zig` and `src/blas/compat_cblas.zig` are testable
   Zig module roots. They delegate to `src/blas/compat/fortran.zig` and
   `src/blas/compat/cblas.zig`, which re-export the same ABI-backed functions as
   Zig declarations. This two-step wrapper is intentional: build roots provide
   importable module names, while leaf facades keep the Fortran and CBLAS
   compatibility namespaces focused.
4. `tools/generate_compat_headers.zig` regenerates headers from exported ABI
   signatures. It reads explicit ordered ABI source lists and checks expected
   export counts so future ABI file splits do not silently drop declarations.

ABI and compat files may use BLAS names because they mirror external contracts.
They should not contain architecture-specific tuning policy or descriptive Zig
API aliases; those belong in `src/blas/api.zig` and its submodules. When adding
or moving exported ABI functions, update the generator source lists and expected
export counts in `tools/generate_compat_headers.zig` intentionally.

## Naming Rules

- Project: `Zynum`.
- Repository/package slug: `zynum`.
- Current module slug: `zynum-blas`.
- Link library: `zynum_blas`.
- Internal C-visible helper symbols: `zynum_blas_*`.
- Module-scoped environment variables: `ZYNUM_BLAS_*`.
- Standard BLAS ABI symbols remain unchanged, for example `dgemm_` and
  `cblas_dgemm`.
