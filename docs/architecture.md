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

- `src/blas/core.zig` is the internal BLAS semantics facade. It extends the
  unchecked ABI facade with checked operands and structured operation entry
  points.
- `src/blas/core/shared/scalar.zig` owns scalar arithmetic, complex helpers, BLAS
  character parsing, and enum aliases.
- `src/blas/core/shared/indexing.zig` owns vector, dense, packed, and banded indexing.
- `src/blas/core/unchecked.zig` is the narrow unchecked facade for ABI wrappers. It
  re-exports scalar helpers, indexing helpers, and unchecked vector,
  matrix-vector, and matrix-matrix BLAS entry points, but it does not expose
  checked public views or structured operations.
- `src/blas/core/checked/operands.zig` owns structured internal operand
  carriers.
- `src/blas/core/checked/operations.zig` owns readable checked execution entry
  points.
- `src/blas/core/vector.zig`, `src/blas/core/matrix_vector.zig`, and
  `src/blas/core/matrix_matrix.zig` are stable semantic facades.
- `src/blas/core/vector/`, `src/blas/core/matrix_vector/`, and
  `src/blas/core/matrix_matrix/` group portable implementations by operand
  category and storage family. Vector operations currently keep BLAS entry
  semantics and contiguous fast paths in `vector/operations.zig`; split it
  further by operation family when another independently testable group is
  added or when reviewability suffers.

The typed Zig API validates inputs. Core operands do not validate inputs and
should remain small data carriers. ABI wrappers should import
`src/blas/core/unchecked.zig` instead of the wider `src/blas/core.zig` facade so the
external ABI path stays independent of checked public API conveniences.

Level 1 and Level 2 keep BLAS argument semantics, stride handling, complex
fallbacks, and portable scalar loops in the core operation files. Common
contiguous real fast paths first call the operation-level facades under
`src/blas/kernels/dispatch/`: `vector_unary.zig`, `vector_binary.zig`, and
`matrix_vector.zig`. If no target-specific kernel accepts the shape,
core falls back to shared Zig vector loops for scaling, axpy, reductions, GEMV,
SYMV, and GER helpers. Higher-level portable paths should reuse lower-level
unit-stride helpers where practical; for example GEMV and GER fallback code
calls Level 1 `scal`, `axpy`, and `dot` helpers instead of maintaining separate
vector loops. General strides, complex values, packed/banded storage, and
conjugating variants stay on the scalar portable loops.

`src/blas/core/execution/thread_pool.zig` owns the single optional `std.Io.Threaded` helper
lifecycle used by large contiguous Level 1 kernels, selected Level 2 kernels,
and GEMM task execution. The normal path uses `std.Io.Group.concurrent`; narrow
measured paths may use internal low-latency helper publication while still
relying on `std.Io.Threaded` for helper lifecycle. On Linux/x86_64, persistent
helpers may also pin themselves inside the scheduler-provided affinity mask when
that is part of the retained dispatch evidence. On Apple Silicon/macOS, public
affinity APIs must not be treated as CPU or P/E-core pinning; use the dedicated
affinity notes before changing split policy. The pool also owns the explicit
shutdown path used by dynamic library probes before unloading Zynum BLAS. Level 2
direct parallelism is only for column-disjoint work such as real unit-stride GER
and large transposed GEMV, or for kernels such as SYMV that use explicit
per-task reduction storage before writing the shared output. GEMM planning must
submit planned tasks to this core pool rather than introducing a second worker
pool.

Level 1/2 architecture kernels use the same operand categories in
`src/blas/kernels/arch/<arch>/vector/` and
`src/blas/kernels/arch/<arch>/matrix_vector.zig`. For example,
`src/blas/kernels/arch/aarch64/vector/unary.zig` owns single-vector feature
gates such as scale and reductions,
`src/blas/kernels/arch/aarch64/vector/binary.zig` owns two-vector kernels such
as copy, axpy, and dot, and
`src/blas/kernels/arch/aarch64/matrix_vector.zig` owns GEMV and GER
dispatch. Vector wrappers live in `asm/vector.zig`, GEMV-transpose wrappers
live in `asm/matrix_vector.zig`, and the ASIMD GER fast path is kept as Zig
`@Vector` column-block code in `matrix_vector.zig` instead of a NEON assembly
wrapper.
Keep experimental ASIMD/SVE/SME kernels behind shape and feature predicates
until focused benchmarks show they beat the shared Zig vector fallback. SME
kernels that mutate existing matrices must preserve BLAS additive semantics;
for example GER needs `A += alpha*x*y^T`, not the overwrite-only GEMM
direct-store epilogue.

Shared fixed-width microkernels live under operation-family directories before
they become architecture files. `src/blas/kernels/shared/vector/fixed_simd.zig` owns the
parameterized Level 1 SIMD skeletons for copy, swap, scale, AXPY-like updates,
reductions, rotations, and complex vector operations.
`src/blas/kernels/shared/matrix_vector/fixed_simd.zig` owns the parameterized Level 2
GEMV and GER skeletons. Architecture wrappers should pass comptime
configuration such as lane counts, unroll factors, copy lane counts, and maximum
work gates into these shared skeletons before adding a new target-specific loop
body.

AArch64 whole-function inline assembly builders are centralized in
`src/blas/kernels/arch/aarch64/asm/builders.zig`; SVE/SME callers should build
variants through those comptime string builders instead of duplicating lane,
unroll, reduction, prologue, or epilogue text. Operation-family wrapper files
under `src/blas/kernels/arch/aarch64/asm/` own the naked function boundaries.
Apple AMX opcode emission and
operand encoding live in `src/blas/kernels/arch/aarch64/matrix_matrix/amx_ops.zig`, leaving
`src/blas/kernels/arch/aarch64/matrix_matrix/amx.zig` responsible for GEMM algorithm
structure and gate checks. Shared
packed SIMD descriptor parameters for ASIMD and x86_64 belong in
`src/blas/kernels/shared/matrix_matrix/packed_params.zig`.

Current Level 1/2 tuning keeps the common arithmetic in core Zig vector loops
and uses architecture code only for narrow, measured cases. GEMV has two facade
entry styles: `gemvNoTransFullUnitReal` and `gemvTransFullUnitReal` accept a
complete BLAS call before core scales `y` or splits work, while the panel
facades accept a row/column slice after shared core setup. On Apple M-series
targets with SME2, the full-shape f64 GEMV-N gate may use a 256-row ZA kernel
that owns the `beta*y + alpha*A*x` epilogue, and the full-shape f64 GEMV-T gate
may use an 8-column by 32-row ZA kernel that owns the
`beta*y + alpha*A^T*x` epilogue. Both gates require 64-byte streaming vector
length and measured medium-matrix shape constraints. Rejected or partial shapes
fall back to core beta scaling, the parallel splitter, and AMX/SVE/core f64
panel chains. ASIMD Level 1 helpers remain split behind per-routine gates: only
measured narrow copy shards are enabled by default, while axpy/reduction/GER
helpers stay off unless their focused sweep beats the shared fallback. Level 1
SVE/SME scal candidates may be enabled only behind a length and feature gate
with fresh focused data; streaming SME helpers must use the same ABI prologue
and epilogue discipline as GEMM/GEMV. They must preserve only the ABI-visible
callee-saved FP lanes they actually use, such as the low 64 bits of touched
`v8`-`v15`/`z8`-`z15`, and scalar FP values that must survive `SMSTOP` should be
ferried through GPRs and restored after leaving streaming mode. SVE copy and
SVE GER helpers remain candidate kernels only; keep them disabled by default on
Apple M-series unless fresh focused data shows a repeatable win over ASIMD copy
or the core Zig GER panel.
Level 1 parallelism is intentionally chunked more coarsely than GEMV/GER; it is
a bandwidth path, not a latency path.

## File Ownership And Split Rules

Prefer small, purposeful files over broad utility buckets, but do not split a
file just because it is long. Split when ownership becomes ambiguous or when a
new behavior creates an independently testable unit.

Recommended split triggers:

- A semantic core file grows multiple unrelated operation families with
  different invariants. Split by operation family, for example
  `matrix_vector/general.zig`, `matrix_vector/triangular.zig`,
  `matrix_matrix/gemm.zig`, or `matrix_matrix/symmetric.zig`.
- An ABI file accumulates generated-looking wrapper groups that are easier to
  review by BLAS level. Split into level-specific ABI leaves while keeping the
  public export root stable.
- A GEMM kernel file mixes target-feature detection, candidate metadata,
  tuning, packing, micro-kernels, and dispatch policy. Keep feature detection in
  `features.zig`, task/thread policy in `core/matrix_matrix/planner.zig`, shared candidate
  metadata, matching, reusable prologue/epilogue helpers, and parameterized
  kernel bodies under `kernels/shared/matrix_matrix/`, and instruction details under
  `kernels/arch/<arch>/`. Architecture files should pass comptime parameters into
  shared kernels before forking a new loop body.
- A test file starts combining unrelated public API, ABI, and kernel behaviors.
  Split tests by the surface being validated: API shape/aliasing, CBLAS ABI,
  Fortran ABI, generated headers, and GEMM correctness. Keep surface-level test
  roots under `test/api/`, `test/abi/`, or `test/headers/` as appropriate.

When splitting, preserve import roots and public module names unless the change is
explicitly documented as breaking. Prefer facade files that re-export smaller
leaves over moving public names directly.

## GEMM Fast Path

- `src/blas/core/matrix_matrix.zig` is the matrix-matrix facade.
  `matrix_matrix/gemm.zig` detects no-transpose real GEMM and delegates.
- `src/blas/core/matrix_matrix/planner.zig` owns shape policy, task splitting, and threading
  choices.
- `src/blas/core/execution/thread_pool.zig` owns the single BLAS helper pool used by Level 1,
  selected Level 2 paths, and GEMM task execution. GEMM does not keep a separate
  worker pool; `planner.zig` only builds tasks and submits them to the core
  pool.
- `src/blas/kernels/dispatch/matrix_matrix.zig` selects the target-feature candidate set.
- `src/blas/kernels/shared/matrix_matrix/catalog.zig` describes available kernels using stable
  `KernelId`, tile, packing, and minimum-work metadata.
- `src/blas/kernels/shared/matrix_matrix/tuning.zig` scores candidate combinations for the
  current shape and scalar epilogue, and fills execution-plan parameters such as
  pack mode, AMX/SME sub-kernel, panel batching, and workspace budgets.
- `src/blas/kernels/shared/matrix_matrix/executor.zig` maps the selected `KernelId` to the
  implementation module.
- `src/blas/kernels/shared/matrix_matrix/packed_simd.zig` owns the reusable fixed-width packed-B
  SIMD panel prologue, K loop, row/tail handling, and write-back wrapper used by
  ASIMD and x86 SIMD backends.
- `src/blas/kernels/shared/matrix_matrix/epilogue.zig` owns shared real-GEMM alpha/beta
  write-back formulas for scalar and vector kernels.
- `src/blas/kernels/shared/matrix_matrix/task.zig` defines shared task shapes and carries the
  selected runtime kernel id plus `ExecutionPlan`.
- `src/blas/kernels/shared/matrix_matrix/generic.zig` is the portable backend.
- `src/blas/kernels/arch/aarch64/` and `src/blas/kernels/arch/x86_64/` contain
  architecture-specific feature gates, feasibility checks, state handling, and
  parameter choices for shared kernel bodies.

Keep shape policy in `core/matrix_matrix/planner.zig` and `kernels/shared/matrix_matrix/tuning.zig`. Keep
instruction details in `kernels/arch/<arch>/`, but keep reusable loop bodies,
packed-panel setup, and scalar/vector epilogues in `kernels/shared/matrix_matrix/` when they can
be expressed by comptime parameters. Architecture-specific kernel files may
branch on feasibility and correctness only: ISA/OS support, alignment, tile
availability, alpha/beta constraints, state cleanup, and whether the caller's
`ExecutionPlan` requested a variant. Future GEMM tuning should first adjust
descriptor parameters, `ExecutionPlan` fields, shared `Config` parameters, or
`kernels/shared/matrix_matrix/tuning.zig` matching rules before changing micro-kernel code.

## ABI And Compatibility

Compatibility is intentionally layered:

1. ABI implementation files mirror external symbol contracts:
   - `src/blas/abi/fortran.zig` exports classic Fortran BLAS symbols.
   - `src/blas/abi/cblas.zig` exports CBLAS symbols.
   These files should call through `src/blas/core/unchecked.zig` for unchecked BLAS
   semantics instead of importing the wider core facade.
2. `src/blas/compat.zig` is the native library export root for `libzynum_blas`.
   It exists to pull both ABI modules into the shared/static library build and
   should stay minimal.
3. `src/blas/compat_fortran.zig` and `src/blas/compat_cblas.zig` are testable
   Zig module roots. They delegate to `src/blas/compat/fortran.zig` and
   `src/blas/compat/cblas.zig`, which re-export the same ABI-backed functions as
   Zig declarations. This two-step wrapper is intentional: build roots provide
   importable module names, while leaf facades keep the Fortran and CBLAS
   compatibility namespaces focused.
4. `tools/generate_compat_headers.zig` regenerates headers, the Fortran module,
   and the ABI manifest from exported ABI signatures. It reads explicit ordered
   ABI source lists and checks expected export counts so future ABI file splits
   do not silently drop declarations.

ABI and compat files may use BLAS names because they mirror external contracts.
They should not contain architecture-specific tuning policy or descriptive Zig
API aliases; those belong in `src/blas/api.zig` and its submodules. When adding
or moving exported ABI functions, update the generator source lists and expected
export counts in `tools/generate_compat_headers.zig` intentionally, regenerate
`include/zynum/blas/abi_manifest.json`, and verify the manifest against built
static and dynamic libraries.

## Kernel Layer

`src/blas/kernels/dispatch/matrix_matrix.zig` reports the active CPU instruction level and
exposes a candidate list to `src/blas/core/matrix_matrix/planner.zig`. The candidate list is
metadata only; it does not execute kernels.

`src/blas/kernels/shared/matrix_matrix/catalog.zig` is the shared descriptor schema. Descriptors
are intentionally parameter-oriented: tile sizes, K unroll, packing kind, stack
pack budget, minimum useful work, ISA level, family, and stable `KernelId`.
`src/blas/kernels/shared/matrix_matrix/tuning.zig` is the first place to adjust matching rules
for shape classes and performance-related sub-kernel parameters. It produces
the execution plan carried by `src/blas/kernels/shared/matrix_matrix/task.zig`, including
pack-layout choices, AMX/SME variants, panel batching, and workspace budgets.
`src/blas/kernels/shared/matrix_matrix/executor.zig` is the only shared runtime bridge from
`KernelId` to architecture modules.

Reusable real-GEMM kernel mechanics live beside the descriptor/tuning layer:
`src/blas/kernels/shared/matrix_matrix/packed_simd.zig` implements a comptime-configured
packed-B SIMD skeleton with panel preparation, main accumulation, tail handling,
and write-back hooks, while `src/blas/kernels/shared/matrix_matrix/epilogue.zig` centralizes
alpha/beta write-back rules. ASIMD and x86 SIMD modules should provide only
feature gates, lane counts, panel widths, row-group counts, stack-pack limits,
and fallback policy for this shared skeleton. Generic fallback kernels may also
reuse the shared epilogue helpers.

CPU GEMM task splitting and thread selection remain in
`src/blas/core/matrix_matrix/planner.zig`, while instruction details remain under
`src/blas/kernels/arch/<arch>/`. Do not add performance dispatch inside
instruction-set files. They may keep hard safety caps for fixed-size stack
frames, compatibility ABI aliases, and mandatory hardware state prologue/cleanup
such as SME streaming mode or Apple AMX state. The choice to use a performance
path must come from descriptors, tuning, planner-selected policy, or
`ExecutionPlan`. Do not add another backend dispatch layer unless it has a concrete, tested
implementation boundary.

For Level 1/2, the equivalent dispatch boundary is intentionally narrower:
core files decide whether a BLAS call is a real unit-stride fast path, the
operand-category facade selects an architecture module, and the architecture
module may accept only shapes proven by benchmark gates. Architecture modules
must return `false` rather than partially handling unsupported tails unless the
caller explicitly owns the remaining work.

## Naming Rules

- Project: `Zynum`.
- Repository/package slug: `zynum`.
- Current module slug: `zynum-blas`.
- Link library: `zynum_blas`.
- Internal C-visible helper symbols: `zynum_blas_*`.
- Project-specific runtime environment variables: only `ZYNUM_MAXIMUM_THREADS`.
  Do not add other `ZYNUM_*` environment variables; instruction-set, backend, and
  worker policy belong in internal dispatch or explicit APIs/build options.
- Standard BLAS ABI symbols remain unchanged, for example `dgemm_` and
  `cblas_dgemm`.
