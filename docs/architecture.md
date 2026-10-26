# Architecture

The [scientific capability and ownership map](scientific.md) is the concise
current entry point. This reference describes durable implementation contracts;
dated acceptance records live in [interface checkpoints](interop/checkpoints.md)
and [development history](development_history.md).

Zynum is a Zig-native numerical runtime organized as a top-level package and
independent numerical submodules. Zynum BLAS (`zynum-blas`) is the first
shipping submodule. Public API, BLAS semantics, compatibility ABI, kernel
contracts, tuning policy, and low-level execution have separate owners.

## Module Boundary

- `src/facades/{zynum,blas,lapack,sparse}.zig` are the public Zig roots for
  `zynum`, `zynum-blas`, `zynum-lapack` and `zynum-sparse`. Each exports aliases
  from the same internal implementation owner; none owns separate runtime state.
- `src/zynum.zig` is the single internal Core owner within the artifact.
  It imports `src/blas.zig`, `src/lapack.zig` and `src/sparse.zig` within one
  Zig module, preserving shared scalar, view and runtime identity. Numerical
  implementations remain under those namespaces.
- Public package names and declarations remain stable; their physical roots
  have moved to `src/facades/`. Consumers import the package names rather than
  relying on the former implementation-root paths.
- The build attaches shared resources and isolated kernel objects to the
  implementation owner. Compatible facades reuse that exact `Build.Module`.
  Their core and BLAS compatibility/runtime consumers receive the same internal
  `zynum-execution` module for the artifact's target, mode and configuration.
  They also share one options `Build.Module`, created once and passed through
  `addImport`; sharing only an options-generation step or source path is not
  sufficient. Production and test execution modules remain separate when
  their optimization modes differ.
- `src/blas/abi/root.zig` owns the Fortran/CBLAS adapter modules. Their thin
  aliases borrow unchecked operations, types and runtime from `zynum-core`;
  compatibility roots reuse that adapter Module instead of importing another
  numerical closure or another Fortran adapter owner.
- `src/checked.zig` owns the experimental checked export shims.
  `src/checked/implementation.zig` validates foreign descriptors and borrows
  LAPACK/Sparse from the same `zynum-core`. Its artifact receives the matching
  Core, Execution, Resources and options Module objects. Core does not import
  checked adapters, interop profiles or generated foreign declarations.
- `tools/numerical_operation_schema.py` defines mathematical operation roles;
  `tools/checked_abi_profile.py` binds the checked wire contract to those roles.
  `tools/checked_abi_schema.py` selects and validates that profile, and
  `tools/generate_checked_abi.py` generates its C/Fortran declarations, Zig export shims, fixed layout
  manifest and Windows export list. `include/zynum/checked/` holds the generated
  foreign interfaces.
- `src/blas/api.zig` is the checked Zig API facade.
- `src/blas/compat.zig` is the shared/static library export root.
- `src/blas/compat_fortran.zig` and `src/blas/compat_cblas.zig` expose the
  compatibility modules used by tests and Zig consumers.
- `include/zynum/blas/` contains generated C, CBLAS, Fortran, and ABI metadata.

`zig build install-libraries` installs the BLAS dynamic and static libraries.
`zig build install-checked-abi` installs the separate checked interface;
`-Dchecked-abi=true` also adds it to the default installation.
ELF and Mach-O use their conventional `zig-out/lib/` layout. Windows installs
`bin/zynum_blas.dll`, the import library at `lib/zynum_blas.lib`, and the static
archive at `lib/static/zynum_blas.lib`. Static Windows consumers name that
archive explicitly and must not add `lib/static` to a normal library-search
path where it could shadow the import library.

The four public imports can coexist in one consumer. Exported types and runtime
controls remain aliases of the common owner; root namespaces themselves need
not be identical. Distinct contracts, such as Sparse dense views versus
BLAS/LAPACK dense views, remain distinct types. The existing modern API test
root imports all four facades and checks their aliases, runtime state and
cross-facade numerical calls.

`src/interfaces.zig` defines an optional combined `zynum_interfaces` dynamic
artifact over the existing BLAS and checked export roots. It adds no numerical
owner, API or default installation; the existing `zynum_blas` and
`zynum_checked` library names and installation steps remain.
`build-combined-interfaces` builds and emits the combined library without
installing or executing it. Successful emission or linking does not establish
symbol-set equivalence, ABI conformance or runtime correctness; those require
separate checks bound to the library's source and target configuration.

This owner boundary applies within one compatible linked artifact. The combined
artifact uses one Core and default Context for its existing interfaces;
separately linked BLAS/checked DSOs or duplicate package instances can still
hold separate globals. Sharing source paths or Build.Module objects while
constructing two libraries does not unify their runtime instances, allocators
or handle lifetimes. Cross-DSO sharing would require an explicit single core
product and thin dependent shims; it is not implemented here.

Future modules should follow the same shape:

```text
src/facades/<module>.zig
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

1. `src/blas/abi/root.zig` is the single Fortran/CBLAS adapter owner for the
   artifact's configuration, with a named dependency on `zynum-core`.
2. `src/blas/abi/fortran.zig` exports classic Fortran symbols; `cblas.zig`
   normalizes C layouts and shares that same Fortran adapter module.
3. `src/blas/compat.zig` selects the existing library exports through the named
   ABI Module. The leaf facades under `src/blas/compat/` borrow it for tests.
4. `tools/generate_compat_headers.zig` generates C headers, the Fortran module,
   and the ABI manifest from the ordered export sources.

ABI wrappers mirror external names and calling conventions and call Core's
internal alias of `core/unchecked.zig`. They do not contain target selection,
tuning, or descriptive Zig aliases. After moving or changing exports, regenerate
`include/zynum/blas/` and verify both shared and static libraries.

## Shared LAPACK And Sparse Architecture

The target is one numerical implementation with interchangeable, generated
API/ABI profiles. The first compatibility target is the classic oneMKL
C/Fortran interface, including LAPACK/LAPACKE and Sparse. Reference LAPACK
3.12.1 defines the dense functionality catalog; it does not establish oneMKL
binary compatibility. Other vendors get additional profiles over the same
internal contracts. FFT compatibility starts with FFTW API/ABI. Standalone
compile-time scripts generate selectable external interface artifacts; internal
data, algorithms, memory, threads, workspace and handles remain Zynum-defined
over the same shared Core. Compatible symbols require compatible observable
semantics as well as declarations. The current checked Zig and experimental checked ABI are
existing profiles with their own semantics, not evidence of standard LAPACK
or oneMKL compatibility.

This section is the implementation contract for the ongoing migration.
Shared storage arithmetic, native-width real and complex Cholesky/LU operands,
borrowed pivot codecs, shared complex arithmetic, compressed-pattern validation
and a common compressed sparse product are the first extractions. Regular
borrowed dense panels let the same sparse traversal serve vectors and multiple
right-hand sides without adding a second numerical implementation. Existing
checked frontends preserve their contracts. The internal sparse product accepts
mixed offset/ordinal widths and split ranges; the public checked API retains
its existing signatures. The common Context provides operation admission,
resource pins and Context-bound scratch around the shared execution owner.
Typed and raw internal POTRF/POTRS/GETRF/GETRS adapters and their shared NaN/callback
configuration live behind `src/interop/lapacke_binding.zig`. A Sparse handle
binding supplies typed and raw lifecycle and general or square structured
matrix-vector routes over common CSR/CSC controls and the product implementation
from the same Core.
Interface configuration
and handle representation remain outside Core; allocator, Context, Resources
and execution identities come from Core. Neither production binding is
referenced by standard exported entrypoints. Other dense kernels, standard ABI
exports, symbolic analysis/factors and a public per-instance resource context
remain subsequent work. The layers below distinguish those targets from the
concrete facilities described here; they are not additional build switches.

Raw access belongs to the adapter boundary. LAPACKE acquires each operand only
when its existing validation, NaN scan or copy stage needs it; it does not invent
buffer capacities or move all pointer checks ahead of the documented argument
order. Sparse creation captures the required index prefixes and retains borrowed
array provenance in the existing shell. Export validates all seven destinations
before committing them inside one operation. Matrix-vector calls reuse the
handle's Context, validate borrowed structure and vector overlap on each call,
then invoke the common product without allocating. Both boundaries share the
typed algorithms, captured allocator and resource controls; neither boundary
is itself a standard C export.

`src/numerics_contract_test.zig` owns the internal contract test root. It imports
the common descriptors, Cholesky/LU cores, pivot codecs and compressed pattern,
with boundary regressions against the checked frontends. These tests run without adding
internal declarations to public modules. The `test-numerics-contract` step
participates in both `test-lapack` and
`test-sparse`; the correctness aggregate also includes it in baseline and
native dynamic builds. Baseline enumeration tracks this root separately.
Native dynamic runs check correctness but do not substitute for baseline
enumeration evidence. The root has no BLAS target-object attachments.

### Layers And Dependency Direction

| Layer | Owns | Must not own |
| --- | --- | --- |
| 0. Interface definitions and generated exports | Versioned symbols, declarations, calling convention, integer/logical/complex representation, target data model, export and installation manifests. | Numerical loops, allocators, caches or scheduling. |
| 1. Semantic adapters | External argument order and validation precedence, layout/index conversion, workspace queries, error/callback translation and external handle representation. | A vendor-specific copy of an algorithm, factor, planner or worker pool. |
| 2. Common operation contracts | Typed operands, storage geometry, ownership/access rules, scalar/index traits, operation modes, numerical outcomes and workspace requirements. | Vendor headers, external enum values, symbol names or compiler-specific Fortran conventions. |
| 3. Algorithms and reusable components | Dense drivers/panels, sparse symbolic analysis, numeric factors, solves, preconditioners and iteration; composition through validated internal operations. | Re-entry through a public checked or compatibility wrapper. |
| 4. Plans and execution services | Feasibility, packing, ordered tasks, memory budgets, resource lifetime, dispatch and measured tuning. | ABI selection or independent pools/caches per interface. |
| 5. Portable and target kernels | Arithmetic on precise prevalidated contracts and target instruction/state handling. | Foreign handles, parameter parsing or interface-specific global state. |

Calls flow from exports through adapters to common operations, then algorithms,
plans and kernels. Lower layers never import an adapter. Higher-level algorithms
can compose BLAS operations through internal contracts without calling BLAS
ABI exports. Geometry/scalar facilities are leaf dependencies. Planning policy
and execution mechanisms remain separate even where an initial portable
operation needs no materialized plan.

Public import names and declarations stay stable during migration; physical
root files can move, as the `src/facades/` extraction demonstrates. Common
facilities belong under `src/numerics/`; operation-specific algorithms remain
under `src/lapack/` and `src/sparse/`. Compatibility code will be grouped by
interface profile, outside the algorithm owners. Preserve existing complex/view
type identity through the shared owner and explicit aliases; structurally
similar duplicate Zig types are not interchangeable.

### Uniform Internal Representation And State

The common model separates mathematical shape from storage and access:

- A dense operand records shape, strides/storage layout, scalar type, access
  mode, physical span and N/T/H operation separately. Normalization must retain
  triangle, diagonal, packing and conjugation meaning. Row-major Hermitian or
  triangular input cannot be implemented by blindly swapping two dimensions.
- Sparse operands separate pattern, values and operation. Offset and ordinal
  types are independent; logical indices are zero based. A checked index view
  may translate a foreign base without copying, while owned analysis can use a
  normalized structural snapshot. CSR, CSC, COO, BSR and later formats retain
  their ordering, duplicate and block-layout contracts. Sorting, coalescing or
  dropping zeros is an explicit transformation, not incidental normalization.
- Dimensions and address extents use checked target-sized arithmetic; signed
  foreign counts are validated before conversion. Algorithms support the
  selected index range without passing through the current i32 BLAS view.
  A 64-bit descriptor alone is not ILP64 support. Every narrowing output
  conversion, including pivots and workspace-query results, is checked.
- LU algorithms consume zero-based `usize` row swaps through borrowed pivot
  readers/writers. Encoding width and zero/one base are selected by the adapter;
  the writer stores each pivot directly at the algorithm's original mutation
  point. Capacity, representability and overlap checks remain in the frontend.
  The checked API still uses zero-based i32 pivots. Numerical failure `info` is
  not a count of committed pivot entries, so adapters must not use it to decide
  which entries to convert after a call. No intermediate pivot buffer is needed.
- Scalar storage, accumulator precision and numerical policy are separate.
  Shared types do not imply identical summation, complex division or residual
  policies. Preserve QR/SVD scaling and CG's wide true-residual check when
  composing lower-level primitives.
- A numerical outcome records the operation-specific reason and location,
  completion state and permitted partial mutation. Boundary code maps that to
  INFO, a checked error or Sparse status. Core code does not store external
  parameter numbers or call foreign error callbacks.

The internal `src/numerics/execution/context.zig` owns an
allocator-captured Executor, active-operation count, resource-pin count and
`open → closing → draining → drained → closed` state. `default.zig` stores one
stable, immortal default Context; the existing Executor accessor returns its
embedded executor, so this adds no second default pool. Context initialization,
operation admission and pin accounting allocate no memory and do not initialize
Io. This is not a public per-instance Context, memory budget or plan cache;
Policy still uses the existing shared static callbacks.

Automatic admission covers the thirty direct checked computation entries:
fifteen BLAS, nine LAPACK and six Sparse. Each entry acquires one operation on
the immortal default Context before validation and ends it after all numerical
work, output writes and cleanup, including errors, empty inputs and any
executor-refused serial fallback. Pure workspace queries do not admit work.
The eight checked ABI exports also admit once through the three handwritten
`factor`, `solve` and `spmv` entries. Their Cholesky and SpMV computations borrow
that same token through internal checked dispatchers; foreign callers need no
new token and cannot select or close the default Context.

Composition keeps the operation lifetime at the outer entry. BLAS `Into`
routes borrow their entry's token, and least squares passes its operation to
internal checked QR application. These internal routes keep their numerical
preflight and error order but never acquire or end another operation. A new
public call from a callback is a separate operation; no thread-local nesting
flag silently suppresses admission. Explicit Context algorithms must use that
Context's services; a token from one Context must not authorize work on an
unrelated default Executor.

The 156 legacy Fortran and 150 CBLAS numerical exports also acquire one operation
on that same immortal default Context before argument access or validation and
retain it through numerical work, output writes and early returns. The 150 shared
Fortran bodies in adapter-only `internal_admitted` borrow the outer Fortran or
CBLAS token; the six Fortran-only numerical bodies are wrapped directly. The
nine CBLAS local numerical helper families execute under their caller's outer
token without another admission. Valid, invalid and empty calls incur
allocation-free Context counter updates and locking; this states lifecycle
coverage, not a measured performance result. These internal routes add no public
symbols. They do not establish complete vendor profiles or cross-DSO lifetime
coordination.

The numerical bodies behind the eight checked ABI exports use caller storage
without allocation or an external numerical backend. Admission itself also
allocates nothing. This does not describe every checked Zig BLAS kernel: real
and complex GEMM and GEMV can allocate call-local scratch, and shared task
machinery can initialize when used.

The checked Zig matrix-product entries and shared Fortran/CBLAS GEMM bodies
use `gemmRealAdmitted` for f32/f64 and `gemmComplexAdmitted` for complex scalars.
Their borrowed `CallResources` derives the Executor and scratch allocator from
one live Operation. Complex 3M and expanded-real decompositions pass that borrow
to their real products, packing and merging. GEMM routes that reduce to GEMV
retain it through GEMV's temporary storage and contiguous scal/axpy helpers,
including their real-coefficient fast paths. Standalone real and complex GEMV
uses the same borrow through the checked Zig matrix-vector entries and shared
Fortran/CBLAS GEMV bodies. Checked GEMV workspace copy-in and copyback also pass
it through COPY task selection and submission without another admission.

The checked Zig vector entries and computational Level 1 Fortran/CBLAS routes
also carry the same live Operation through task policy and submission. This
covers SCAL/RSCAL, COPY/SWAP, AXPY/AXPBY, ordinary and mixed-precision DOT,
ASUM/NRM2/IAMAX, and ROT/ROTM, including the four direct complex-return DOT
exports. Checked vector `Into` composition uses that borrow for both COPY and
computation; exact-alias paths retain their private admitted helpers. Nested
SCAL/AXPY/COPY shortcuts and real-coefficient complex paths retain the same
resources. Scalar-only ROTG/ROTMG remain scheduler-neutral under their existing
outer admission.

The twelve real symmetric and complex Hermitian matrix-vector routines
(s/d SYMV, SBMV and SPMV; c/z HEMV, HBMV and HPMV) now pass the outer
Fortran/CBLAS Operation through their internal admitted bodies. Dense and
packed task policy, submission and scratch allocation, and nested Hermitian
SCAL/AXPY/DOT calls, use that Operation's resources. Admitted scratch bypasses
the legacy TLS workspace. A completed scratch-backed path frees it after task
completion and output merging; a refused task submission releases it before
the existing serial fallback. Both finish before the operation ends. Already
admitted work can finish after a close request; close remains busy even while
a scratch free callback is paused. A busy Executor falls back on the same
resources, including nested Level 1 work from one of its real workers, without
another admission. Whole-operation fallback applies beta once. No-token
internal bridges retain their legacy workspace and executor policy.

The resource-bearing triangular routes s/d/c/z TRMV, TRSV, TPMV and TBSV
also borrow the outer Operation. Dense triangular AXPY/DOT helpers carry it
into nested Level 1 work; selected real packed-multiply and band-solve scratch
uses its captured allocator. Scratch release and any staged output copyback
finish before the outer operation ends. Allocation refusal retains the
existing fallback before caller writes.

Shared Fortran GER/GERU/GERC, SYR/SPR/SYR2/SPR2 and HER/HPR/HER2/HPR2 bodies
carry the same resources through dense and packed task selection, submission
and nested AXPY, including rank-two fringes. CBLAS branches that delegate to
these shared bodies inherit that borrow. Existing CBLAS local row-major
helpers remain unchanged under their outer admission. TBMV and TPSV retain
their existing serial resource-neutral leaves. No new admission, allocator,
Context or Executor is introduced; numerical kernels, tuning gates and
reduction order are unchanged.

CBLAS row-major HEMV/HBMV/HPMV normalizes dense, band and packed storage through
a shared borrowed Hermitian view. The opposite physical triangle carries an
explicit matrix-conjugation flag; the core applies reflection and conjugation
and reads only the real diagonal component. This fixes the row-major result
without a temporary matrix or a second operation. Zero beta discards old logical
Y values, including NaNs; ignored imaginary diagonal values do not contribute.
The existing argument checks and stride handling remain at their boundaries.

GEMM, GEMV and symmetric/Hermitian MV scratch obtained through this borrow is
call-local and bypasses legacy TLS caches. Selected triangular scratch also
uses the captured allocator. All such scratch is freed before the outer
operation ends. Tasks, final scatter and checked workspace
copyback remain within that operation; no second admission, Context or pool is
introduced. The Level 1
resource paths add no call-local heap storage: their task and reduction arrays
remain on the synchronous call stack. The shared Executor can still allocate
worker/runtime storage with its captured allocator. Such storage may outlive an
Operation and is released by executor drain/reset or internal Context close,
not necessarily at operation end.

Busy-Executor refusal and failed task allocation retain the existing fallback
on the same resources, without repeating caller-visible updates. Internal
no-token bridges retain their legacy default execution and allocation/cache
policy. Structured Level 3 SYMM/HEMM, SYRK/HERK, SYR2K/HER2K and
TRMM/TRSM paths now carry the same borrowed resources through nested GEMM and
triangular operations. The matched isolated object receives a versioned,
synchronous service descriptor borrowing the operation, captured allocator,
executor and GEMM callback. It does not own a second runtime. Host callbacks
and a separately compiled native object have runtime coverage; x86 object
cross-compilation does not establish x86 runtime acceptance. Other unmodified
paths are not migrated merely by their outer admission. No public per-instance BLAS Context, complete
vendor profile, cross-DSO resource protocol or performance claim is provided.

`src/numerics/resources.zig` supplies the allocator-captured `ScratchLease(T)`
used by BLAS triangular-band solve. `ContextScratch(T)` wraps that same lease
with one pin from an already admitted operation. It pins before allocation,
rolls back the pin on failure, and frees memory with its captured allocator
before unpinning. A lease may outlive the operation, including worker ownership;
zero-length and zero-sized-element leases still hold their pin until release.
These affine values must not be copied or released twice. Typed managed
POTRF/POTRS/GETRF/GETRS calls use ContextScratch for conversion scratch. Their operation
spans validation, callbacks, numerical work, copyback and every free; allocation
uses the admitted Context's captured allocator. This does not change the
allocation-free checked ABI bodies, add a cache/pool or impose a context-wide budget.

Compatible production/test/owner consumers share the same named Resources
Module for their complete target. The fixed-target lifecycle artifact reuses
main or host Resources only when the full target query, object format and
dynamic linker match; otherwise it gets a matching separate Module. This
preserves type identity without claiming a runtime singleton across DSOs.

Only explicitly internal closable Contexts enter terminal close. A close
request refuses new operations; already admitted work may finish and acquire
cleanup pins. Progress reports busy while operations remain, drains the
Executor outside the Context gate, then reports waiting-resources until the
remaining pins are released before publishing closed. Draining first allows worker cleanup to release worker-held
pins. Callbacks run outside the gate; reentrant progress during drain returns
busy. The finalizer may wait synchronously for workers, but does not block
waiting for an active caller or outstanding caller-owned pin.

The external owner must keep the Context address, allocator backing state and
callback/code dependencies alive through close and every API stack's return.
Pins do not make arbitrary borrowed pointers safe or reclaim Context storage.
Legacy shutdown releases the existing caller-local caches and performs
restartable `Executor.drainAndReset`; it is not terminal Context close, global
operation admission, or an unload barrier. Explicit
quiescence and ownership obligations remain necessary. Independent Context
policy state, memory budgets, externally exposed handle/array lifetimes and
cross-DSO unload handling remain future work. Persistent factors and plans
still need distinct typed lifetimes; do not create interface-specific pools.
Queries report requirements without numerical execution. Acquire required
scratch before the mutation point; partial numerical failure follows the
routine's contract rather than an implicit rollback or fallback.

The common LU core has explicit checked and IEEE numerical policies. The
existing checked entrypoints retain their status and partial-write behavior.
The GETRF policy records the first zero pivot while completing later pivot
stores and updates; NaN/Inf are not relabeled as singularity. The GETRS policy
handles N/T/H using the same borrowed factor and pivot codecs, and propagates
exceptional arithmetic without a positive failure INFO. These are internal
numerical contracts, with no external enum values, callbacks, allocation or
Context of their own. The internal GETRF adapter selects the IEEE policy and
one-based signed pivot writer directly. It never reads the old output pivots;
row-major factor copyback and scratch cleanup finish before its operation ends.
The internal GETRS adapter selects that same IEEE solve policy for real and
complex N/T/H with borrowed one-based signed pivots. It rejects unsafe pivot
values separately from INFO and keeps stage-local access and Reference-profile
argument order. Row-major solves acquire factor and RHS conversion scratch in
order and release it in reverse order within the one admitted operation.
Factors and pivots remain read-only. Standard LU exports remain future work;
these internal routes do not imply vendor-identical rounding, exceptional
pivot behavior or ABI compatibility.

The common Sparse Product owns a vendor-neutral `MatrixSemantics` union.
General multiplication retains its existing route. Symmetric, Hermitian,
triangular and diagonal square matrices use one coordinate traversal with
explicit stored-half and unit-diagonal semantics. The adapter translates the
external descriptor; it does not expand a dense matrix or allocate another
object. Hermitian mirroring conjugates once, and its diagonal uses the real
component. Unit diagonals are synthesized without reading stored diagonal
values. Pattern validation and overlap checks still cover the retained storage,
including numerical entries that the selected structure does not use.

`Dense.PanelShape`, `ConstPanel` and `Panel` describe regular row-major or
column-major borrowed storage with native-width dimensions and strides. Their
checks cover geometry and capacity; callers still own lifetime, shape and alias
validation. The common Product accepts these panels and shares its one- or
two-column traversal with existing vector routes. Checked CSR SpMM delegates
to that route after its unchanged column-major preflight. Empty panels avoid
unused offset arithmetic, and only logical output entries are written. The
implementation allocates no workspace and creates no resource or execution
owner. This does not add an MM entry to the internal Sparse handle adapter,
introduce arbitrary-stride panels, or enable a vendor interface profile.

Adapters may keep only externally required configuration and representation
state, such as NaN-check settings, callback bridges and a handle shell. The
shell refers to a common object/control block; it does not duplicate factors,
buffers, scheduling or analysis. Plan identity depends on operation semantics,
pattern identity, scalar/index range, numerical policy, capability and resource
constraints, not the vendor name or exported symbol. Equivalent normalized
calls therefore use the same internal route.

The internal LAPACKE binding imports one non-generic `lapacke_services.zig`
configuration owner for all its scalar and integer specializations. Each
shared managed call admits once, captures the two callback bindings as one
record, and derives allocator authority from that operation. Default routes
use Core's existing immortal Context. NaN policy is read later at the adapter's
high-level scan point; capturing callbacks does not snapshot that flag. Neither
Core nor its execution services import the interface configuration or events.
See [internal interface services](interop/contracts.md#internal-lapacke-services)
for initialization, callback lifetime and the current typed/raw internal boundary.

### Borrowing, Analysis And Handle Lifetime

Borrowed buffers remain caller-owned and must be alive and stable during the
call. A `const` slice, unchanged pointer or generation counter cannot certify
that foreign code has left its metadata unchanged. The checked borrowed Sparse
path continues structural validation on each public call.

`src/sparse/object.zig` supplies internal CSR/CSC control blocks through Core
aliases. `MatrixControl` is allocator-only; `ManagedMatrixControl` uses the
actual `Execution.context` Operation and ResourcePin types. Managed creation
borrows an admitted operation, captures its allocator and acquires one pin per
control. Borrowed arrays remain caller-owned; an owned snapshot owns its copied
storage. Managed data access requires a live same-Context operation. Retain and
release are non-atomic and require external serialization; the final release
frees storage and the control before unpinning. Exported views do not retain a
reference or extend storage lifetime. These controls add no public handle API
or automatic admission to existing Sparse calls.

`src/interop/mkl_sparse_binding.zig` instantiates one typed shell family for
four real/complex scalars and signed i32/i64 indices. Each shell owns one common
control reference; only the control owns a resource pin. Borrowed creation
allocates the control and shell with the admitted Context's captured allocator,
rolling back before unpinning on failure. Destruction frees the shell before
releasing its reference; final common release frees the control before unpin.
The shell preserves mutable input provenance and adds no allocator, pool,
numerical state or Intel private-layout assumption. Its create/export/destroy
slice has a narrower representation domain than the common Pattern; see
[typed Sparse handles](interop/contracts.md#internal-sparse-handle-slice).

Reusable symbolic analysis is tied to an immutable owned pattern snapshot or
an explicitly enforceable mutation/lifetime contract. Store entry mappings
when values remain in a different order; bind numerical factors to the pattern
and values revision that produced them. Internal calls can use ephemeral
validated records for the duration of one stable call, avoiding repeated public
preflight without inventing a persistent validation certificate.

State transitions depend on object kind. Matrix/analysis objects move from a
created pattern through optional analysis to execution/update and release.
Factor objects add symbolic and numeric phases; a failed partial factor is
available only for documented diagnostics or recovery, not ordinary solving.
Iterative sessions separately track initialization, an outstanding
reverse-communication request, resumption, convergence/failure and release.
Only the expected response may resume a suspended session, with required
buffers and callback context still alive.

Failure leaves each object in a documented valid state. Pattern changes
invalidate symbolic and numeric state; value changes invalidate numeric state
while preserving applicable symbolic work. Mutations and destruction require
exclusive ownership or synchronization; read-only solves may share an
immutable factor with separate scratch. Exported borrowed arrays and
reverse-communication requests retain their external lifetime rules. Do not
silently copy or replace storage where an interface promises a particular
borrowed buffer or writeback behavior.

### Profiles, Generation And Switching

Keep three independently versioned inputs: mathematical operation definitions,
external interface profiles and implementation/verification coverage. An
external routine binding joins them by semantic operation identity. It records
argument roles and shapes, mutability, modes, scalar family, index model,
workspace/query rules, error precedence, output translation and ownership.
The profile additionally pins vendor/header version, target/compiler ABI,
symbol spelling, calling convention, character lengths, logical encoding,
complex passing/return rules, enum/struct layouts and callback signatures.
Unknown fields remain unresolved; generators must not guess them from the host
Python/C layout or infer an ABI merely from matching names.

Generate declarations, thin export shims, symbol/layout manifests, dispatch
bindings and the support matrix from those inputs. Reuse handwritten semantic
converters by contract, such as pivot-base conversion or row-major packing.
Adding another vendor should add a profile and necessary semantic conversions,
not another numerical backend. A generator may emit a callable export only
when its required modes have a real implementation and an explicit tested
support record. An absent routine stays a visible coverage gap; a placeholder
returning success or a universal unsupported status is not implementation.

The generator separates operation definitions from external profiles. Default
and explicit checked-v1 generation retain six artifacts and eight exports.
Finite LP64/ILP64 binding sets compose ordinary factorization, solve, eigen,
QR, SVD and least-squares routines; additive components avoid copying another
powerset of complete packages. The selected manifest records exact symbols,
source dependencies, plan fingerprints and generation acceptance for the
explicit `apple-arm64-default-c-v1` contract. A complete vendor profile remains
unavailable; generation does not certify runtime compatibility.

`tools/generate_interfaces.py` is the standalone entry point over the existing
renderers. Its read-only check re-renders committed accepted selections;
optional build-only components retain their existing renderers and selections.
Generated entrypoints import the configured interface binding and numerical
owner. Semantic adapters own layout/index conversion, query rules and error
translation. No generator owns a numerical algorithm, allocator, Context,
Executor or mutable ABI-width switch. Installed C/C++ calls and application
adoption require their own exact-image checks.

LP64/ILP64 packages install separately with width-specific headers,
static/shared libraries and pkg-config files. Default diagnostics retain
separately linked weak C handlers, while configured callbacks take precedence
through the existing shared services state. See the [interface guide](interop/README.md)
for current commands and [capability map](scientific.md) for qualified scope.


Select profiles when building/linking interface artifacts. Keep the core
selection independent. Distinct prefixed surfaces can coexist over one shared
core instance. LP64 and ILP64 interfaces that export the same names require
separate link choices or explicit namespaces; reject collisions rather than
trying to infer integer width from pointer values. Do not add a mutable global
ABI switch. Header, integer profile and linked artifact must agree.

The build configuration exposes an `ImplementationGraph` containing the actual
production Core, Execution, Resources, options and two interface binding Module
objects. The normal build invokes that configuration once. A future export
artifact must borrow those objects, preserving target and optimization settings;
recreating modules from the same source files would create a different owner.
The graph return does not select an ABI or enable a standard library artifact.

A future shared adapter distribution must use one core runtime and type owner.
Static bundles must resolve that owner once; objects from separately embedded
runtime copies cannot exchange handles or deallocation duties.
Record runtime instance and object-kind identity in internal handle contracts.
Compatibility means accepting the documented application ABI and behavior;
it does not imply that Zynum can consume another implementation's private
opaque handles or factor memory. Cross-profile handle conversion, if exposed,
must be an explicit operation with reviewed lifetime and representation rules.

The standard and checked semantic front ends remain distinct. LAPACKE
high-level and `_work` calls have different workspace and NaN-check behavior;
Fortran INFO and pivot conventions differ from the current checked results.
oneMKL also distinguishes ordinary LP64/ILP64 and explicit `_64` C entries.
These facts belong to the profile/binding, while the common implementation
supplies the operation. See the [official LAPACKE interface rules](https://www.netlib.org/lapack/lapacke.html)
and [oneMKL C interface conventions](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/c-interface-conventions-for-lapack-routines.html).

### Migration And Acceptance

Implement in dependency order: shared storage checks; wide internal operands
and outcomes; checked-wrapper/core separation; resource/analysis lifetimes;
generated standard adapters; then complete algorithm families. Move one
responsibility at a time, preserving public type identity, argument precedence,
empty/padding behavior and failure mutation with regression checks.

For every profile, track functionality, source API, binary layout/symbols,
behavioral conformance, native execution and performance separately. Catalog
counts are scope evidence, not coverage. The Sparse scope includes format and
handle operations, analysis/hints, execution, direct and iterative solvers,
preconditioners, Sparse QR, and legacy interfaces as separate families; use
the [oneMKL Sparse operation catalog](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/inspector-executor-sparse-blas-operations.html)
and [solver catalog](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/sparse-solver-routines.html)
to enumerate exact per-routine restrictions. Distributed/offload APIs and
vendor extensions need explicit rows rather than silently entering or leaving
the denominator.

Verification must include independently sourced foreign headers/compiler
calls, exact symbol/layout checks, invalid-argument/query behavior, independent
numerical residuals and lifecycle/concurrency cases. Generated declarations
checked only against generated declarations do not prove compatibility.
Instrumented integration checks should show equivalent profiles reaching the
same operation/plan and sharing resource accounting. Native execution remains
separate from foreign linking. Performance comparisons start after functional
and compatibility coverage for the measured cells, include adapter costs and
bind exact source/binaries; earlier measurements retain their original source
identity throughout the migration.

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

`src/numerics/execution/owner.zig` owns the vendor-independent `Executor`:
admission, ordinary `std.Io.Group.concurrent` submission, persistent workers,
completion and reset. It captures the allocator at initialization and uses that
same provider for release. `default.zig` owns the stable immortal Context using
`std.heap.c_allocator`; its embedded Executor is the existing default execution
owner. `policy.zig` owns thread limits, hardware/environment
caches and worker QoS/affinity policy. `Policy` currently holds static function
pointers, not independent per-context configuration state.

`src/blas/runtime.zig` forwards the existing runtime controls to this policy.
`src/blas/core/execution/thread_pool.zig` is the BLAS adapter: it retains the
BLAS worker-cache release callback/domain, typed task bridges and existing
forwarding signatures. The common execution code does not import BLAS.
Ordinary helpers release their domain's worker-local caches before returning;
persistent helpers release the previous domain on a different/null-domain
transition and release their last domain before exit. Caller task zero and
inline fallback do not perform helper-cache cleanup.

The ordinary and persistent paths still own two separate `Io.Threaded` objects
inside one executor. Initialization order is recorded under the admission lock;
`drainAndReset` releases them in reverse order. A persistent object that accepts
no initial task is released before its failed state is published. Reset waits
for admitted work and permits restart. It requires caller-coordinated quiescence
for destruction or unloading: it neither blocks future submissions permanently
nor tracks a refused caller's serial fallback. It is not global operation
admission or the internal Context's terminal close. Context tokens retain the
lifetime of admitted fallback work, but legacy shutdown does not close that
Context or wait on its operation/pin counters. External Io/signal owners
and separately linked DSOs remain outside this local lifetime contract.

Keep an initialized executor at a stable address. Its allocator state, policy
and callback code, and borrowed cache-domain descriptors must remain valid
until reset completes. Task payloads are borrowed for the synchronous call and
must remain stable until it returns. Do not infer an unload guarantee from a
shared source path or module graph.

`test-execution-contract` builds `owner.zig` directly as a canonical root with
libc and the test target/mode. Its Executor tests use an explicit fixture
policy, with no production options, default executor or ISA objects. The root
has one named Resources dependency and explicitly imports Context and
ContextScratch regressions. The Executor cases cover real zero/partial
`Group.Task` allocation failures,
captured-allocator release, simultaneous-caller refusal and reset, restart,
generation boundaries and worker-local domain cleanup. A named dependency does
not automatically enumerate these tests. The two wrapper tests live in
`thread_pool.zig` and are reached by existing BLAS internal roots; the public
BLAS module test separately checks shared policy identity/state and caller
fallback while the default owner is busy. The shared-facade consumer checks
public aliases and cross-facade calls. Runtime checks and actual linked-artifact
inspection supplement graph identity assertions; none establishes cross-DSO
identity.

`test-runtime-lifecycle` retains six fixed child-process cases for both
initialization orders, each single pool, and restart with reversed order. Each
case also performs four real allocator-failure subchecks: ordinary/persistent
paths with zero or one accepted helper allocation. They check task completion,
helper presence, allocation/release accounting, and signal handler, mask and
flags after cleanup. These are actual `Group.Task` allocation failures, not a
fake submission hook or standard-library patch.

The lifecycle gate runs only on native macOS AArch64 with a matching requested
target, baseline code and the selected test optimization mode. The applicable
correctness aggregate includes it; an explicit unsupported request fails. Its
subprocess evidence does not replace canonical enumeration. Acceptance requires
inventory checks and native results bound to the exact source, target and
configuration. Linking the fixed-target lifecycle artifact does not establish
that its process cases ran, and foreign linking does not establish native
behavior. The test contracts described here do not certify an unverified
configuration.

Parallel paths must:

- derive concurrency from CPU capacity available to the process;
- use disjoint output ownership or bounded private reductions;
- acquire workspace before caller-visible mutation;
- finish unsubmitted work synchronously after partial submission;
- require caller quiescence and reset before destruction; separately satisfy
  callback/object lifetime obligations before unloading; and
- avoid nested or architecture-specific worker pools.

Affinity and heterogeneous scheduling are platform constraints, not kernel
semantics. See
[`common/cpu_affinity_and_heterogeneous_scheduling.md`](common/cpu_affinity_and_heterogeneous_scheduling.md)
and [`common/zig_std_io_threading.md`](common/zig_std_io_threading.md).

## Runtime Configuration

`ZYNUM_MAXIMUM_THREADS` controls the thread ceiling. When unset, concurrency
derives from the execution environment; a positive value caps it. The dynamic
dispatcher also accepts `ZYNUM_MAX_ISA` as a diagnostic ceiling before its first
selection. It can remove capabilities, never enable unsupported instructions;
unknown or wrong-architecture values select baseline. Backend, task strategy,
ABI profile and tuning remain internal policy or explicit build/API choices.

## File Ownership And Split Rules

Split files when a new independently testable responsibility appears:

- semantic code splits by operation or storage family;
- ABI exports split by BLAS level while stable roots remain facades;
- kernel metadata, tuning, packing, execution, and instructions stay separate;
- tests split by public API, ABI, generated artifacts, registries, and numerical
  behavior.

Prefer precise names such as `catalog.zig`, `coverage.zig`, `tuning.zig`,
`executor.zig`, `planner.zig`, and `asm/<family>.zig`. Preserve public import
names and declarations when moving implementation leaves or physical roots.

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
creates inventory evidence. The `test` step with explicit `-Dcpu=baseline`
remains inventory-certified and fail-closed. Without `-Dcpu`, `test` selects
dynamic-dispatch correctness and host-tool smoke.

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
dedicated step; its debug, safe, and fast test steps explicitly
disable host-tool smoke, while link-only steps do not request it. This keeps
the independent build-inventory suite and the ABI-baseline/Python-tooling host
aggregate out of the three per-mode target invocations.
The Linux x86_64 and ARM link-only rows keep ordinary POSIX structure-gated
`test-inventory-link`. The exact native x86_64 Windows GNU row instead invokes
`test-inventory-link-windows-native-smoke` for debug, safe, and
fast. That compatibility-only step omits the POSIX structure checker and
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
execution boundary. Windows native enumeration remains separate from this
link/loaded-library evidence. The structure checker is the authoritative source
for the current pending rows; adding a logical root does not fill those rows.

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
   The owner regression covers submission and shutdown rollover,
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
  `src/numerics/execution/owner.zig` shares admission between ordinary and
  persistent execution. Its `runPersistent` retains the `count == 3`
  helper-index exception. The owner tests cover helper-set/generation
  boundaries, concurrent refusal/reset and real allocation failures; the
  subprocess gate covers both initialization orders and restart. These checks
  do not establish the measured rationale for that helper mapping. Review that
  rationale and thread-cap/mixed-mode behavior before changing it. Do not merge
  the two Io lifecycles or change waiting protocols without independent
  correctness and native performance validation.

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
ARM OS capabilities fall back conservatively. Zig 0.17 recognizes Apple M5
directly for specialized `-Dcpu=native` builds; dynamic dispatch continues to
admit kernels from runtime CPU and OS capabilities.

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
