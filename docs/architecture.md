# Architecture

Zynum is a Zig-native numerical runtime organized as a top-level package and
independent numerical submodules. Zynum BLAS (`zynum-blas`) is the first
shipping submodule. Public API, BLAS semantics, compatibility ABI, kernel
contracts, tuning policy, and low-level execution have separate owners.

## Module Boundary

- `src/zynum.zig` is the top-level facade and must not contain BLAS
  implementation details.
- `src/blas.zig` is the `zynum-blas` module root.
- `src/blas/api.zig` is the checked Zig API facade.
- `src/blas/compat.zig` is the shared/static library export root.
- `src/blas/compat_fortran.zig` and `src/blas/compat_cblas.zig` expose the
  compatibility modules used by tests and Zig consumers.
- `include/zynum/blas/` contains generated C, CBLAS, Fortran, and ABI metadata.

`zig build install-libraries` installs only the dynamic and static libraries.
ELF and Mach-O use their conventional `zig-out/lib/` layout. Windows installs
`bin/zynum_blas.dll`, the import library at `lib/zynum_blas.lib`, and the static
archive at `lib/static/zynum_blas.lib`. Static Windows consumers name that
archive explicitly and must not add `lib/static` to a normal library-search
path where it could shadow the import library.

Future modules should follow the same shape:

```text
src/<module>.zig
src/<module>/
include/zynum/<module>/
docs/<module or platform>/
```

## Public Zig API

`src/blas/api/views.zig` owns checked vector and matrix views,
`src/blas/api/aliasing.zig` owns checked-build alias validation, and
`src/blas/api/operations.zig` translates descriptive operations into the core.

Public names describe operations rather than ABI abbreviations. Default output
APIs use a no-alias contract. Supported overlap is explicit through an in-place,
`Into`, or `WithWorkspace` form with documented ownership. The checked API
validates dimensions, strides, storage, and aliasing; it must not import
architecture-specific dispatch or instruction modules.

## Core Semantics

- `src/blas/core.zig` is the checked internal facade.
- `src/blas/core/unchecked.zig` is the narrow ABI-facing facade.
- `src/blas/core/shared/` owns scalar arithmetic and indexing.
- `src/blas/core/checked/` owns validated operands and checked execution.
- `src/blas/core/vector/`, `matrix_vector/`, and `matrix_matrix/` own portable
  semantics and fallbacks by operation family.

The core owns argument normalization, traversal, alpha/beta behavior,
conjugation, task composition, workspace acquisition, and whole-operation
fallback. The portable implementation is total. An optimized route may reject
a call only before caller-visible mutation; it cannot partially update output
and restart through the fallback.

## Compatibility ABI

Compatibility is layered deliberately:

1. `src/blas/abi/fortran.zig` exports classic Fortran symbols.
2. `src/blas/abi/cblas.zig` exports CBLAS symbols and normalizes C layouts.
3. `src/blas/compat.zig` imports both into the native libraries.
4. Leaf facades under `src/blas/compat/` support Zig compatibility tests.
5. `tools/generate_compat_headers.zig` generates C headers, the Fortran module,
   and the ABI manifest from the ordered export sources.

ABI wrappers mirror external names and calling conventions and call
`core/unchecked.zig`. They do not contain target selection, tuning, or
descriptive Zig aliases. After moving or changing exports, regenerate
`include/zynum/blas/` and verify both shared and static libraries.

## Kernel Contract Layer

`src/blas/kernels/contract.zig` defines the shared catalog vocabulary:

- stable semantic kernel identity;
- operation, scalar, layout, and entry surface;
- required ISA capability and architecture state;
- lifecycle (`production`, `experimental`, `rejected`, or unavailable);
- stride, alignment, alias, tail, and epilogue behavior;
- whole-operation or sub-operation ownership;
- packing and bounded-workspace requirements; and
- total fallback.

Catalogs describe executable facts. Coverage enumerates supported and missing
cells. Tuning records measured preference. Executors map stable IDs to bodies.
Planners compose tasks, packing, workspace, and fallback.

Build, native correctness, and native performance are separate evidence axes.
Cross-build or emulated results cannot promote native performance support.
Kernel IDs are lowercase dot-separated semantic names, not source paths,
benchmark revisions, or processor product names.

## Level 1 And Level 2 Kernels

Operation dispatch lives under `src/blas/kernels/dispatch/`; reusable loops live
under `src/blas/kernels/shared/vector/` and
`src/blas/kernels/shared/matrix_vector/`. Architecture wrappers add capability
checks and instruction-specific geometry only when the body is genuinely
different.

Level 1 contracts distinguish scalar, contiguous, fixed-width, streaming, and
isolated-object routes plus total fallbacks. Level 2 contracts distinguish
complete calls from panels, columns, private deltas, and dependency steps. A
sub-operation must state output ownership and merge obligations and cannot
impersonate a complete BLAS call.

Private fixed-layout objects may isolate architecture experiments, but they
must keep symbols hidden, verify hard predicates before writing, submit work
through the shared task runtime, and leave non-applicable targets on the total
fallback. The positive-only `-Dlevel2-width-candidates` option selects its
experimental profile; build-only evidence does not establish native
correctness.

See [`common/level1_optimization_notes.md`](common/level1_optimization_notes.md)
and [`common/level2_optimization_notes.md`](common/level2_optimization_notes.md).

## GEMM Fast Path

The matrix-matrix control flow is:

1. Core code normalizes BLAS semantics and requests a whole-operation plan.
2. Dispatch exposes candidates compiled for the active capability tier.
3. Tuning filters hard feasibility and applies measured shape preferences.
4. The planner composes packing, bounded workspace, epilogue, and task topology.
5. The executor maps the selected stable ID to a body.
6. Any pre-compute rejection follows the catalog's total fallback.

Key shared files are `catalog.zig`, `structured_catalog.zig`, `tuning.zig`,
`task.zig`, `executor.zig`, `coverage.zig`, and `generic.zig`. Architecture
directories own feature checks, state handling, and instructions. Shape policy,
packing, tails, and epilogues remain shared when they can be parameterized.

Complex descriptors are distinct because plane materialization, conjugation,
combination, and scalar restrictions are whole-call contracts. Structured
descriptors additionally record side, triangle, diagonal, output ownership,
dependency order, and merge behavior.

See [`common/gemm_optimization_notes.md`](common/gemm_optimization_notes.md).

## AArch64 State Boundaries

ASIMD, non-streaming SVE, and streaming SME are independent capabilities. SME
availability does not prove ordinary SVE availability. Streaming descriptors
declare SM/ZA ownership, required vector length and features, and balanced
entry/exit behavior. Architecture wrappers preserve ABI-visible state, and
transition cost is part of route selection.

Apple AMX encoding stays in a narrow architecture module; algorithm structure
and hard gates remain in its architecture wrappers. The AMX route is disabled
by default and may be compiled only for an explicitly validated
`aarch64-macos` deployment with `-Dapple-amx=true`. CPU family, ASIMD, and SME
features do not prove that the private AMX instruction set is executable. An
unauthorized build takes the ordinary fallback without issuing an AMX raw
opcode. AMX and SME remain internal implementation details rather than public
BLAS API modes.

## Shared Task Runtime

`src/blas/core/execution/thread_pool.zig` owns the optional `std.Io.Threaded`
lifecycle shared by BLAS Levels 1-3. Normal task composition uses
`std.Io.Group.concurrent`.

Parallel paths must:

- derive concurrency from CPU capacity available to the process;
- use disjoint output ownership or bounded private reductions;
- acquire workspace before caller-visible mutation;
- finish unsubmitted work synchronously after partial submission;
- provide explicit shutdown before dynamic-library unloading; and
- avoid nested or architecture-specific worker pools.

Affinity and heterogeneous scheduling are platform constraints, not kernel
semantics. See
[`common/cpu_affinity_and_heterogeneous_scheduling.md`](common/cpu_affinity_and_heterogeneous_scheduling.md)
and [`common/zig_0_16_std_io_threading.md`](common/zig_0_16_std_io_threading.md).

## Runtime Configuration

`ZYNUM_MAXIMUM_THREADS` is the only project-specific environment variable.
When unset, concurrency derives from the execution environment; a positive
value caps it. Instruction tier, backend, task strategy, and tuning profile are
internal policy or explicit build/API choices, not environment variables.

## File Ownership And Split Rules

Split files when a new independently testable responsibility appears:

- semantic code splits by operation or storage family;
- ABI exports split by BLAS level while stable roots remain facades;
- kernel metadata, tuning, packing, execution, and instructions stay separate;
- tests split by public API, ABI, generated artifacts, registries, and numerical
  behavior.

Prefer precise names such as `catalog.zig`, `coverage.zig`, `tuning.zig`,
`executor.zig`, `planner.zig`, and `asm/<family>.zig`. Preserve public import
roots when moving implementation leaves.

## Validation Boundaries

Registry tests may bypass preference thresholds to force an executable ID, but
they still enforce capability, layout, epilogue, state, workspace, and
failure-before-write constraints. Ordinary API and ABI tests validate selected
production routes and the fallback chain.

The public test inventory owns logical roots, ordered compiler-enumerated sets,
environment predicates, modes, native-enumeration state, and evidence joins.
Official test runners validate it before executing test bodies. Inventory
commands accept only the exact `-Dcpu=baseline` request for a declared
environment. Cross-linking and emulation never fill native evidence, and an
undeclared class cannot infer results from another OS, libc, object format, or
CPU profile.

`test-native-feature` is a separate correctness-only path for explicit
non-baseline CPU profiles. It reuses the official test bodies, requires the
target ABI and object format to match the host, and requires the host feature
set to cover the requested profile. It neither invokes the inventory runner nor
creates inventory evidence; the default `test` step remains inventory-certified
and fail-closed.

The build and test inventory checkers use bounded, fail-closed file admission
and code-reviewed digests. Those controls establish repository consistency;
they are not signatures, remote provenance, or authentication. Refresh and
publication validate complete candidates before replacement and preserve
uncertain recovery material. The authoritative schema, resource bounds, and
exit statuses live in `tools/check_build_inventory.py`,
`tools/check_test_inventory.py`, and the runner sources.

In GitHub Actions, the same-SHA Linux source, build-inventory-security, and
test-inventory-security jobs own inventory and repository-security evidence.
The Windows `x86_64 build-link-native-smoke` job instead proves cross-target
builds, Debug/ReleaseSafe/ReleaseFast inventory linking, library layout, the
canonical DLL's complete manifest export surface, and native Python tooling
compatibility. It is not inventory certification or attestation. At this
revision, Windows native-enumeration observations remain explicitly pending,
and the complete native matrix remains incomplete. The structure checker is
the authoritative source for the current pending set.

Run these security and consistency gates in a process with no inherited
`GIT_*` variables:

```sh
env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
  sh <<'ZYNUM_INVENTORY_CHECKS'
python3 -B tools/check_build_inventory.py --root .
python3 -B tools/check_test_inventory.py --structure-only
zig build test-build-inventory --summary failures
zig build test-test-inventory --summary failures
ZYNUM_INVENTORY_CHECKS
```

Performance changes require correctness-checked native evidence for
representative shapes on the advertised AArch64 or x86_64 capability tier.
Raw reports and host-specific records remain outside the public repository.

## Naming Rules

- Project: `Zynum`.
- Repository/package slug: `zynum`.
- Shipping module slug: `zynum-blas`.
- Link library: `zynum_blas`.
- Internal C-visible helpers: `zynum_blas_*`.
- Standard BLAS ABI symbols remain unchanged, such as `dgemm_` and
  `cblas_dgemm`.
- `portable_scalar` names a terminal complete fallback. Architecture names are
  reserved for distinct executable bodies or independently compiled tiers.
