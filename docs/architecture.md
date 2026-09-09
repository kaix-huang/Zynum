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
`src/blas/api/operations.zig` translates descriptive operations through
`src/blas/core/checked.zig`. This narrow facade exposes only scalar helpers,
validated operands, and descriptive operations. A compile-time guard rejects
raw BLAS and scheduling declarations at the API import boundary.

Public names describe operations rather than ABI abbreviations. Default output
APIs use a no-alias contract. Supported overlap is explicit through an in-place,
`Into`, or `WithWorkspace` form with documented ownership. The checked API
validates dimensions, strides, storage, and aliasing; it must not import
architecture-specific dispatch or instruction modules.

## Core Semantics

- `src/blas/core.zig` is the internal facade for checked callers; it currently
  also re-exports unchecked operations, so the import boundary alone does not
  enforce validation.
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

In GitHub Actions, the same-SHA Linux source and test-inventory-security jobs,
together with the Linux/macOS build-inventory-security matrix, own inventory
and repository-security evidence.
Each inventory-certified target row invokes `test-host-tool-smoke` once in a
dedicated step; its Debug, ReleaseSafe, and ReleaseFast test steps explicitly
disable host-tool smoke, while link-only steps do not request it. This keeps
the independent build-inventory suite and the ABI-baseline/Python-tooling host
aggregate out of the three per-mode target invocations.
The Linux ARM link-only row keeps ordinary POSIX structure-gated
`test-inventory-link`. The exact native x86_64 Windows GNU row instead invokes
`test-inventory-link-windows-native-smoke` for Debug, ReleaseSafe, and
ReleaseFast. That compatibility-only step omits the POSIX structure checker and
executes no inventory runner or test body. It does not provide inventory
certification, native enumeration, or correctness evidence.

The Windows job separately checks library layout, the canonical DLL's complete
311-name manifest export surface, and deterministic representative CBLAS Level
1, Level 2, and Level 3 calls (`daxpy`, `dgemv`, and `dgemm`) through that exact
loaded DLL. The three calls do not establish the semantics of all 311 exports,
Fortran compatibility, inventory evidence, attestation, or performance. The
nonce-bound completion record only detects an accidental early exit; a
malicious DLL running on the same runner with the same permissions could still
forge process state, and adversarial attestation would require an isolated
execution boundary. At this revision, all 63 Windows native-enumeration rows
remain explicitly pending, and the complete native matrix remains incomplete.
The structure checker is the authoritative source for the current pending set.

Run these security and consistency gates in a process with no inherited
`GIT_*` variables:

```sh
env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" \
  sh <<'ZYNUM_INVENTORY_CHECKS'
set -eu
python3 -B tools/check_build_inventory.py --root .
python3 -B tools/check_test_inventory.py --structure-only
zig build test-build-inventory --summary failures
zig build test-test-inventory --summary failures
ZYNUM_INVENTORY_CHECKS
```

Performance changes require correctness-checked native evidence for
representative shapes on the advertised AArch64 or x86_64 capability tier.
Raw reports and host-specific records remain outside the public repository.

## Improvement Priorities

The completed items below record the repository review follow-ups. The
subsequent boundary work remains proposed. Preserve numerical
semantics, public imports, ABI symbols, and measured fallback gates while
addressing them incrementally.

### Near Term

1. **Completed: reject conflicting experimental profiles.** `build.zig`
   rejects multiple enabled profile flags before selecting `selected_profile`.
   The six flags are `structured-object-candidates`,
   `structured-object-baseline`, `level1-sve-candidates`,
   `level1-fixed-candidates`, `level2-fixed-candidates`, and
   `level2-width-candidates`. Default and single-profile behavior is unchanged;
   `apple-amx` and `level2-compact-triangular-baseline` remain independent
   controls. `zig build test-build-profiles` checks defaults, every single
   profile, all 57 conflicting combinations, and independent controls. It also
   runs through `test-host-tool-smoke`. These are configuration checks, not
   native kernel correctness or performance evidence.
2. **Completed: remove unreachable tuning rules without enabling new routes.** In
   `src/blas/kernels/shared/matrix_matrix/tuning.zig`, f32 AMX rules now
   contain only predicates reachable under the retained `K <= 512` cap.
   The existing registry test covers K=512/513, fringe, partial-N, square
   exclusions, and an f64 high-K control. No new route is enabled.
3. **Completed: narrow benchmark run dependencies.** Benchmark runs depend on
   their emitted probe and shared library instead of the full install graph.
   The direct GEMM sweep installs only the shared library to create its output
   prefix, and respects `--prefix` for the default CSV. Default installation
   remains compatible; explicit tool-only installation is still future work.
4. **Completed: narrow checked imports.** API views and operations use the
   checked facade described above. Legacy `core.zig` aliases remain available.
5. **Completed: make chart statistics explicit.** All three public plotters
   accept `--stat median`; missing median evidence fails before publication.
   GEMM reports retain ordered per-process median timings alongside aggregates.

6. **Completed: make persistent-worker generations wrap safely.** Submission
   and shutdown increment each addressed worker's own atomic counter. The
   admission lock serializes producers; completion acknowledges that the worker
   has consumed its generation. This removes checked-build overflow and avoids
   assigning an idle worker a global generation equal to its stale value.
   The existing runtime test now covers submission and shutdown rollover,
   exactly-once indices, inactive workers, alternating helper sets, and restart.

7. **Completed: separate isolated Level 2 worker startup.** The frozen worker
   parses its own arguments and finishes before controller-only imports are
   evaluated. It therefore runs from its captured script without repository
   siblings or a `PYTHONPATH` dependency. Regression coverage executes the
   captured script, rather than only mocking subprocess results.

### Subsequent Boundary Work

- **Separate measured preferences from plan feasibility.**
  `src/blas/core/matrix_matrix/planner.zig` embeds shape/type/thread preferences
  alongside task construction, while kernel preferences also live in
  `kernels/shared/matrix_matrix/tuning.zig`. Name and centralize empirical
  policy, retaining workspace, alignment, task coverage, and fallback
  constraints in the planner. Compare selection results before retuning.
- **Explain and test runtime scheduling exceptions.**
  `src/blas/core/execution/thread_pool.zig` shares admission between ordinary
  and persistent execution and contains a `count == 3` helper-index special
  case. Establish evidence for that exception; extend lifecycle coverage for
  mixed modes, concurrent callers, shutdown/restart, and thread-cap changes.
  The new rollover/restart and existing failure tests are useful foundations.
  Add simultaneous-caller and mixed ordinary/persistent admission tests before
  changing that helper mapping. Do not merge the
  two lifecycles or change waiting protocols without independent correctness
  and native performance validation.

Build/test inventories and benchmark report helpers are maintained project
infrastructure, not disposable planning metadata. Keep their safety gates and
shared utilities; delete tools only after checking build, CI, test, and report
consumers. Complete pending native inventory rows on the exact target systems
rather than replacing missing evidence with cross-link results.

## Runtime Kernel Selection

The default build emits one library for a single target architecture, OS and ABI.
The host code uses that target's baseline CPU, while `buildKernelTiers` compiles
ISA objects separately with LTO disabled. x86 objects are absent from ARM builds
and vice versa. The same objects serve the checked Zig API and C/Fortran ABI.

`hardware.zig` detects OS-usable features: macOS AArch64 uses sysctl (including
SME, SME2, SME2p1, FP16 and BF16); Linux AArch64 uses HWCAP/HWCAP2; x86_64 uses
CPUID plus OSXSAVE/XCR0. Compiler-implied dependencies are also required. Unknown
ARM OS capabilities fall back conservatively. In particular, Zig identifying an
M5 as `apple_m1` no longer prevents the dynamic library from using its SME tier.

The admitted tiers are `baseline`, `aarch64_sve2`, `aarch64_sme`,
`aarch64_sme2`, `aarch64_sme2p1`, `x86_avx`, `x86_avx2_fma`, and `x86_avx512`.
SME2/2.1 objects additionally require F64F64. The resolver intersects linked
objects, detected capabilities and optional `ZYNUM_MAX_ISA` ceiling. Unknown or
wrong-architecture ceiling values select baseline. It publishes one immutable
selection atomically; each selected object's existing shape, scalar, workspace
and streaming-vector constraints still choose the actual kernel or fallback.
A lower ceiling is useful for testing, never for enabling unsupported ISA.

The four kernel dispatch facades route through a generated private operation
protocol. The generator derives signatures from those facades, including concrete
scalar variants. Pointer packets stay within one build/compiler/architecture;
they are not a public ABI. Host scheduling, parameter validation, threading and
core workspaces remain single instances. Kernel-local caches are freed through
the same immutable selected object on shutdown. No target-specific initialization
runs before admission, and private tier entry symbols are hidden.

An explicit `-Ddispatch=dynamic` resets the host CPU/features to baseline, which
also supports Zig dependencies that implicitly forward CPU options with their
target. Explicit `-Dcpu` selects specialization in `auto` mode. In a specialized
build, `dynamic_dispatch=false` removes the generated calls and linked tier
objects at compile time. The
ordinary feature constants become valid compile-time facts again. This option
is a deployment requirement chosen by the caller, not a claim that any machine
can execute that library. `-Dthread-limit` is a compile-time concurrency ceiling;
OS capacity, runtime overrides and ordinary operation feasibility still apply.

`test-dynamic-dispatch` exercises native automatic and forced-baseline execution
in separate processes. It is correctness evidence, not a canonical inventory
row. Explicit `-Dcpu=baseline` keeps the original inventory test path. Run
`zig build check-multiversion` to reject stale generated adapters/protocols.

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
