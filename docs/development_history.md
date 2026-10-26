# Scientific development history

The following records retain their original dates and qualification boundaries. The [current roadmap](roadmap.md) and [scientific capability map](scientific.md) supersede their scheduling statements. Relative evidence paths refer to external local records, not packaged SDK files.

# Roadmap and active development priorities

Updated 2026-10-12 from the local ABPLaS source audit and the owner's direction.
This active plan takes precedence over the historical release outline below.
Performance tuning and competitor benchmark campaigns are deferred.
Existing correctness comparisons remain valid acceptance tools.

## Delivery order

| Stage | Deliverable | Acceptance |
| --- | --- | --- |
| P0: stabilize existing functionality | Finish shared resource ownership across existing BLAS paths; close current LAPACK/Sparse correctness, lifecycle and packaging gaps. | Numerical and boundary regressions, allocator failures, busy/closing execution, nested work and destruction pass. External consumers link and call the declared supported surface. |
| P1: ABPLaS minimum replacement | LP64 LAPACKE `dgesv`, `zpotrf`, `dstev`, `dgelsd`; complex-double CSR SpMV/SpMM, independent offset/ordinal widths and operator integration. | Each replaced application path passes its existing regression and an independent numerical oracle. Unsupported standard semantics remain explicit. |
| P2: FFT | 1D C2C/R2C, 3D R2C/C2R and DCT-I, with plans, workspace, normalization and lifetime contracts. | Cover spectral transforms, Chebyshev coefficients, HPP certification and Kerker mixing. |
| P3: eigenproblems and operators | Complex Hermitian standard/generalized eigenproblems and a shared operator contract; then reusable Chebyshev, Lanczos and subspace operations. | Residuals, orthogonality, generalized metric and failure behavior are validated; integrate a single-node application path first. |
| P4: complete coverage | Extend LAPACK families, Sparse formats, direct/iterative solvers, preconditioners and interval eigensolvers; subsequently distributed capabilities. | Routine/profile coverage is measured per type, mode, index model and target. Catalog entries alone never count as support. |
| Deferred performance | Compare equivalent operations with competitors and optimize measured application bottlenecks. | Start only after functionality/stability milestones and a subsequent priority review. |

## P0 execution sequence

1. Completed locally: Level 2 borrowed-resource integration, including
   numerical regressions, native enumeration and final inventory checks.
2. Completed locally: Level 3 structured resource paths, including nested
   operations and isolated-object service callbacks. Numerical, allocation
   failure and closing-owner regressions passed; Debug/Safe/Fast native
   enumeration and final inventory checks passed after local integration.
   Algorithms and dispatch thresholds are unchanged. Native isolated-object
   runtime evidence is from Apple arm64; x86 cross-compilation is not x86
   runtime verification.
3. Completed locally: finite Apple arm64 LAPACKE packages for LP64 and ILP64,
   each exposing 32 POTRF/POTRS/GETRF/GETRS high/work entrypoints. Independent
   installed C/C++ static/shared consumers and diagnostic contracts passed;
   Debug/Safe/Fast test enumeration and final inventory checks passed after
   integration. This does not establish complete LAPACKE or MKL compatibility.
4. Completed locally: finite Apple arm64 Sparse packages for LP64 and ILP64,
   each exposing 25 CSR/CSC create/export/MV/MM/destroy entrypoints. Generated
   and relocated installed C/C++ static/shared consumers passed, as did the
   170-test Sparse numerical/lifecycle suite and final build/test inventory
   checks. Both packages reuse the configured shared numerical owner. This
   does not establish full MKL compatibility or mixed-index public interfaces.
5. P1 GESV: the finite packages now expose 40 entrypoints per width, adding
   real/complex single/double GESV high/work drivers over the shared LU core.
   The 222-test LAPACK numerical suite and installed C/C++ static/shared
   consumers passed. An independent 36-check harness exercised unmodified
   ABPLaS lattice.cpp with the installed LP64 library; the application's own
   complete test suite has not been run. Debug/Safe/Fast enumeration verifies
   the eight new tests are in the canonical LAPACK root.
6. P1 STEV: the finite packages add single/double real high/work entrypoints,
   bringing each integer-width selection to 44 symbols. The shared tridiagonal
   core supports values and vectors, independent scale blocks, bounded iteration
   and explicit partial-convergence results. All 243 canonical LAPACK tests
   passed in native Safe mode. Installed static/shared C/C++ consumers passed;
   the original ABPLaS Lanczos fixture passed five single-rank and ten two-rank
   cases using Zynum `dstev` and OpenBLAS `zheev` as the independent test oracle.
   This is focused Lanczos integration, not a complete ABPLaS/SCF replacement.
   Native dynamic packaging restricts exports to avoid a C++ `__dso_handle`
   conflict found by the actual application fixture.
7. Current P1 work: the mixed-index/operator bridge. `dgelsd` is delivered
   locally and focused ABPLaS `zpotrf` integration passes, as detailed below.
   Mixed-index CSR application integration remains unverified. Retain
   explicit unsupported cells and meaningful failure evidence, with checks
   proportional to changed sources and unresolved concerns.

## Current delivery and next gates

GELSD is delivered to the local working tree through the shared bidiagonal
SVD/minimum-norm core and LP64/ILP64 high/work adapters. Installed finite
LAPACKE selections now contain 48 entrypoints. The native Safe LAPACK suite
passes 286 tests; Debug/Safe/Fast compiler inventories each contain 1,157 tests
across 24 roots, with exactly 43 LAPACK additions and no removed tests.
Inventory enumeration is distinct from numerical test execution. The wider
cross-environment inventory still has 222 explicitly pending rows.

Eight relocated C/C++ consumers pass across integer widths and static/shared
linkage. Both original ABPLaS density-mixing tests pass against installed LP64
static/shared libraries. ZPOTRF also passes the original CheFSI full-space and
rank-deficient tests with both linkage forms, and the four-rank distributed
full-space test with the shared library. ELPA and other providers remain in the
application fixtures. These are focused replacement boundaries, not a complete
ABPLaS/SCF replacement. Internal GELSD range/invariant faults still terminate
after managed cleanup in finite exports and remain an explicit release limit.

The locally integrated mixed-index Sparse implementation supports separate offset/ordinal
types while preserving the same-width API, with 177 canonical Sparse tests
passing, including three checked adapter regressions. Its checked complex SpMV/SpMM adapters also pass targeted numerical,
pointer and output-preservation tests. All 128 candidate C exports compile on
Apple arm64 and pass independent C11 and C++20 calls covering real/complex
scalars, each offset/ordinal type pairing and both products. Export symbols
match the catalog exactly; rendering is deterministic and rejects malformed
candidates before invoking the formatter. The isolated build now installs static
and shared packages containing the eight existing checked exports and 128 new
Sparse exports through the same internal owner. Four relocated C11/C++20
consumers call all 136 exports and retain the 448 legacy assertions per consumer.
This is Apple arm64 package evidence. The canonical native `test-sparse` build
step and its numerical-resource/execution dependencies completed successfully.
The 40 reviewed implementation, interface and inventory files are integrated
into the local working tree. Public build inventory validation, test inventory
structural validation, nine additive profile tests and deterministic generation
checks pass. The 24-root native enumeration is distinct from numerical test
execution; 222 foreign-environment rows remain pending. The additive ABI remains
experimental, and these changes have not been committed or pushed.

The isolated ABPLaS CSR adapter also passes the original
`NonlocalProjectorSamples.CsrMatchesDirectSamplingAndRestoresK` fixture with
1, 2 and 4 MPI ranks. It preserves application-owned assembly and halo exchange,
replacing local overwrite and ghost accumulation with the shared mixed-index
SpMV adapter. All application sources used by this fixture were rebuilt with
their target's production flags; the final link contains no old ABPLaS archive
members. The test compares the physical matrix action with direct sampling and
checks K-point restoration. This is one focused application fixture; formal
package integration, the general operator bridge and broader application
acceptance remain pending.

A separate extension of that physical fixture now also passes with 1, 2 and
4 MPI ranks. It calls the generic `Hmlt::Apply` boundary and the application's
Lanczos estimator on the assembled physical CSR Hamiltonian, comparing the
estimated endpoints with an independently assembled dense Hamiltonian and
OpenBLAS `zheev`. All 345 inherited source/dependency hashes were checked before
reusing the previously rebuilt production object closure; the new fixture was
compiled with its production flags, and no old application archive members
were linked. Direct sampling and K-point restoration assertions remain in the
extended fixture. OpenBLAS `dstev` remains in this particular executable, and
it does not call the new internal Zynum operator contract. This is focused
physical consumer evidence, not formal package or complete SCF acceptance.
Complete next:

The locally integrated internal operator now connects borrowed CSR actions and
matrix-free callbacks through one synchronous contract. It carries explicit
dimensions and caller workspace, propagates callback errors and borrows one
already-admitted shared execution operation. The native Apple arm64 Safe Sparse
suite passes 181 tests (the existing 177 plus four operator regressions). An
independent Safe compiler-enumeration review confirms 24 roots and 1,168 names:
exactly four Sparse additions, no removals and 23 unchanged roots. Debug and
Fast enumeration passed the same independent review. All 72 Apple arm64 native
bindings are refreshed; 222 foreign-environment rows remain pending. The 22
reviewed source/interface/inventory files are integrated into the local working
tree. Post-integration canonical Safe `zig build test-sparse`, including its
full native enumeration and shared execution prerequisites, now passes. This is not yet a
public operator interface or a generalized CG implementation. Existing CG
widened accumulation and true-residual semantics are unchanged. This evidence
does not advance the FFT/eigensolver stage gate.

A private external bridge also passes the extended physical fixture with
1, 2 and 4 MPI ranks while invoking this internal CSR operator for both local
overwrite and ghost accumulation. It uses one shared admission per action and
managed context scratch for the accumulated ghost product, checking that no
resource pins remain on return. A separate C boundary probe passes 64 repeated
overwrite/accumulate actions and six output-preserving rejection cases. The
shared probe restricts exports to its two private symbols; this avoids the
Apple C++ `__dso_handle` collision observed during its first link. The bridge
supports only normal operation with alpha one and beta zero/one, retains
OpenBLAS `dstev` and `zheev`, and is not an installed public operator ABI or
full SCF acceptance. Formal package selection remains the next external gate.

The installed checked Sparse dynamic package's confirmed Apple C++
`__dso_handle` collision is now closed in the local working tree. Its shared
artifact is linked from the same single static numerical owner and exports
exactly the declared 136 checked/Sparse symbols. The actual MAIN installation
passes four complete C/C++ static/shared consumers, two C++ static-lifetime
regressions, four installed LP64 GELSD C/C++ static/shared consumers, and the
focused physical generic Hmlt/Lanczos fixture with 1, 2 and 4 MPI ranks. The
build and test inventories pass current-only validation; 72 native bindings
remain unchanged and 222 foreign-environment rows remain pending. Six resource
contract tests, four Sparse reachability tests, fixture positive validation,
and a rejected false-gap mutation pass after correcting stale fixture coverage.
Nine malformed linker contracts are rejected before launch without replacing
existing output. Numerical sources are unchanged. The guarded 10-file
integration and installed acceptance receipt are recorded under
`zig-out/module-sparse-shared-export-fix-20261012/main-verification/`.
This closes the focused installed Sparse package gate. The executable retains
OpenBLAS `dstev` and `zheev`; broader application and full SCF acceptance remain
pending, as does an installed public unified operator interface.

The opt-in Apple arm64 `install-abplas-lp64` package now combines the existing
48 finite LP64 LAPACKE, 128 mixed-index checked Sparse and eight checked entries
in one static/shared library with one Core/default Context. All 184 entries
pass relocated C11/C++20 consumers for both linkage modes; nested LAPACKE
error-callback/Sparse actions and C++ static lifetime checks pass. Shared exports
are restricted to those entries plus two diagnostic symbols. Reinstallation
after coalescing common header/manifest nodes preserves every installed byte.

A jointly linked ABPLaS production-source executable resolves the four required
LAPACKE drivers and complex i64/i32 SpMV to that same library. Focused lattice,
density mixing, CheFSI and nonlocal projector regression suites pass at 1/2/4
MPI ranks: 38/38/43 passed, one Debug-only assertion test skipped per run;
five declared four-rank tests run only at four ranks. OpenBLAS BLAS and the
independent `zheev` oracle, FFTW, ELPA and PSolver remain in that application.
This supersedes the separate-package and single-fixture limits above for these
selected regression boundaries. Full SCF, public operator ABI and complete
backend replacement remain pending. The GELSD range/invariant abort limitation
and 222 pending foreign-environment rows remain unchanged. FFT and complex
Hermitian eigensolvers have not passed their stage gates.

A reproduced shared GELSD/SVD scaling failure is now fixed by normalization in
existing widened scratch before bidiagonal iteration, with restoration to the
original units afterward. The installed LP64 package previously returned
positive INFO in 16 large-scale cases where an independent OpenBLAS reference
converged. It now passes all 144 tall/square and 96 additional wide/square
finite-input cases across real precisions, layouts and high/work calls. Analytic
solutions and independently computed rank/singular values agree. A focused
native regression fails on the original core and passes on the updated core;
all 30 SVD/minimum-norm/rank-least-squares root tests pass. Both-width generated
wire outputs are unchanged; only reviewed source-envelope manifests change.
The existing internal range/invariant abort limitation remains explicit. This
numerical correctness fix introduces no extra runtime owner, workspace or
performance work.

The installed checked CSR SpMV path now uses the internal synchronous operator
contract for `y = alpha*A*x + beta*y`, including independent offset/ordinal
widths and complex values. The CSR backend borrows the existing admission and
needs no extra scratch. Matrix-free callbacks use explicit caller workspace
for scaled accumulation; errors from that scratch-backed callback leave the
final output unchanged. Direct overwrite and fused-backend errors retain their
backend output semantics. Preflight rejects dimension, size, workspace and
alias violations without touching output, and beta zero does not read old y.
Zero alpha still validates CSR metadata. No admission, allocation or resource
pin is added. The existing 181-test Safe Sparse suite covers these boundaries;
all installed C/C++ entries and the selected ABPLaS MPI regressions pass through
the updated package. This establishes installed SpMV integration only: At that SpMV acceptance, SpMM
still called the existing shared Sparse kernel, and public operator ABI,
generalized CG and broader matrix-free algorithm integration remain pending.

The installed checked mixed-index CSR SpMM path now also borrows the internal
operator's multi-column action. It uses the existing native dense panels and
Sparse product kernel, one admitted Operation and no additional CSR workspace.
The internal panel contract accepts both row-major and column-major operands;
the existing checked C ABI keeps its column-major dense-view convention and
empty leading-dimension rules. Matrix-free callbacks pack one input column and
retain all product columns in explicit caller scratch before committing output.
Failure in a later column therefore preserves the entire final output. Padding,
storage/workspace tails, zero-coefficient semantics and metadata validation are
covered by the extended existing operator regressions; the complete Safe Sparse
root still passes 181 tests with unchanged test names.

Installed C11/C++20 static/shared consumers additionally pass 196 independent
dense-reference SpMM cases each: complex f32/f64, i64/i32 and u32/u64 indices,
normal/transpose/adjoint, 0/1/2/3/5 right-hand sides, complex scales and zero-scale
NaN nonreads, plus output-preserving rejection checks. All 184 combined entries,
shared-context lifetime checks and the same selected ABPLaS production-object
regression closure pass again at 1/2/4 MPI ranks. That application closure uses
the existing checked SpMV bridge; these runs do not establish an ABPLaS block
SpMM consumer or full SCF. Public operator ABI and reusable operator algorithms
remain pending. This stage adds no allocator, executor, handle or public symbol.

ABPLaS now has an explicit opt-in build selection for this installed package:
add its `lib/pkgconfig` directory to `PKG_CONFIG_PATH` and configure the existing
ABPLaS build with `-DABPLAS_ZYNUM_LP64=ON`, retaining the same compiler, materials,
Python validation environment and other dependency selections. The option is
off by default. Configure rejects a missing package or non-LP64 declaration
contract and checks the four required drivers plus mixed-index complex CSR
SpMV/SpMM linkage. It keeps the selected base BLAS provider and remaining
numerical dependencies. Current acceptance is Apple arm64 with OpenBLAS as that
base provider; a mixed MKL-base build has not been executed.

The actual ABPLaS `hmlt_csr.cpp` now uses checked complex f64 i64/i32 SpMV for
local overwrite and ghost accumulation under this selection, retaining its own
assembly, buffers and halo communication. The original implementation remains
selected when the option is off. LAPACKE includes explicitly use Zynum's finite
LP64 header. The selected finite library is placed before base-provider linker
options as well as ordinary libraries: the initial normal library ordering
still bound the four drivers to OpenBLAS and was rejected. Corrected actual
application and regression executables resolve all four required drivers and
mixed-index SpMV to the one installed `libzynum_abplas_lp64`; the test-only
`zheev` oracle continues to resolve to OpenBLAS.

Build-generated third-party configuration headers now precede source-tree
headers. This prevents a stale source-tree pseudopod config from redirecting
the actual build to an unavailable runtime data directory. The failed main
regression is preserved and excluded from acceptance.

The formal build passes the physical nonlocal-projector CTest at 1/2/4 MPI ranks
(24 passed and one Debug-only skip per run), 49 math tests at four ranks, and the
selected density/CheFSI/Ritz-state/ELPA-driver suites (49 passed at one/two ranks,
61 at four). Declared four-rank tests are excluded only from the lower-rank
selected solver invocations. A relocated ABPLaS runtime/SDK package rediscovers
Zynum with pkg-config and passes an independent linked consumer with verified
symbol origins. This supersedes the replacement-translation-unit limitation
above for these actual build/application boundaries. It does not establish an
ABPLaS block SpMM consumer, full SCF, complete BLAS/backend replacement, or
FFT/Hermitian eigensolver acceptance. The internal GELSD range/invariant terminal
fault boundary remains pending.

The formal opt-in build also passes the existing four-rank C8 cold-start
SCF-DIAG case against its unchanged frozen strict reference. It converges in
11 iterations with the configured joint energy/density stop and final refresh.
Energy and free-energy errors are 8.9e-16 Ha/atom; physical relevant eigenvalues
differ by at most 8.0e-10 Ha. Charge is 31.999999999999968 electrons and the
combined density residual is 3.52e-9 against the 1e-6 threshold. The validator
checks mixing, per-K electronic provenance and committed HDF5/sidecar output.
The frozen reference has no density field, so this is not a density-field or
cross-code comparison. This establishes this C8 full SCF case; other materials,
non-Gamma SCF, MKL-base builds and application block SpMM remain pending.

The same actual solver executable additionally passes 59 selected SCF
lifecycle, convergence, electronic-state and mixing tests at four MPI ranks,
including collective failure, publication and retry boundaries. A separate
installed GELSD probe passes 108 analytic DCT-factor cases across row/column
layout, square/tall/wide matrices, 8/32/64 columns, common scales 2^-1000/1/2^1000
and condition numbers 1/2^24. These finite numerical cases do not close the
internal range/invariant terminal-fault policy gap.

The formal opt-in package also passes the existing C8 non-Gamma DOS/CheFSI
case at crystal k=(0.25,0,0), independently at one and four MPI ranks. The
unchanged validator checks adaptive filtering, target/guard residuals, solve
budget, Hamiltonian application counts and finite band/DOS output. Four-rank
acceptance uses four solves, target residual 6.15e-9 Ha and guard residual
1.68e-7 Ha against 1e-6 Ha. The 24 band energies differ across rank counts by
at most 1.86e-15 Ha. The full-workspace residual remains diagnostic; this case
requires only target and verification-guard convergence. This is one non-Gamma
DOS/eigenpair workflow, not non-Gamma full SCF or a density-field comparison.

The next FFT implementation is grounded in nine current ABPLaS C++ plan sites:
real/complex 1D transforms, real half-spectrum 3D forward/inverse transforms
with x-fastest storage, and DCT-I (REDFT00). Callers retain their existing
normalization and MPI gather/scatter ownership. Chebyshev sampling uses length
32768; time-DOS uses power-of-two lengths of at least 1048576; spectrum lengths
and DCT lengths also require general dimensions. Plans and scratch must use
the shared Zynum data, memory, execution and handle contracts. This dependency
audit establishes requirements only; no Zynum FFT capability is delivered yet.

The shared Core now exposes an internal finite FFT foundation, using the same
complex scalar identity, storage range checks and injected Execution.Context
as the existing numerical modules. It covers arbitrary-length complex forward
and backward 1D transforms, real half-spectrum 1D transforms, DCT-I, and
contiguous complex/real half-spectrum 3D volumes in (z,y,x) order with x fastest.
Both directions are unnormalized; existing application normalization remains
caller-owned. Numerical work uses explicit widened f64/f128 complex scratch.
Output commits only after checked geometry, capacity, alias and finite-result
validation, preserving unused tails and final output on failure.

The internal managed facade admits one operation, or borrows a caller's already
admitted operation, and borrows either explicit workspace or the existing
ContextScratch lease. Captured allocator cleanup precedes resource-pin release
and operation end. Callback close, allocation/numerical failure, owner identity,
closing-context admission and operation/pin counter exhaustion are covered.
Plans remain numerical value descriptors. This foundation creates no separate
default, executor, allocator policy, persistent cache or handle registry.

The five numerical FFT tests and six actual-Core managed tests are registered in
the existing numerical-contract and LAPACK test roots. Compiler inventories and
body execution are recorded separately. The numerical source is identical to
the independently accepted 45-case FFTW comparison, including lengths 32768 and
1048576, odd/general dimensions, DCT-I and arbitrary real inverse half spectra.
This is internal native foundation acceptance. An installable FFTW-first FFT API/ABI,
persistent descriptor/handle lifecycle, public IEEE/stride/in-place conventions
and actual ABPLaS FFT call replacement remain pending; the existing LP64
LAPACKE/Sparse exports retain their finite interfaces.


The internal FFT foundation also provides an affine persistent descriptor over
an existing ContextScratch metadata lease. It holds one pin across operations,
uses its original allocator for final metadata cleanup, and requires one live
same-Context operation for configuration, commit and execution. Invalid
geometry preserves the previous configuration and commit status; valid internal
reconfiguration clears commit. Executing an uncommitted descriptor or the wrong
transform domain fails before acquiring numerical scratch. Input/output/work
prefixes cannot overlap heap metadata or the descriptor's lease bookkeeping.
Explicit workspace adds no resource pin; automatic work uses the existing
ContextScratch mechanism. Deinitialization remains available after close and
frees metadata before its last pin, without a new admission or implicit drain.

Descriptor calls and destruction are externally serialized; copying a live
value does not retain it. No extra Context, allocator policy, thread pool,
handle registry or workspace cache is introduced. Nine transform directions in
both precisions, transactional configuration, another-owner refusal, allocation
and counter failures, callback close and captured-allocator cleanup are tested.
This is internal resource acceptance, not an installed vendor descriptor ABI.
Geometry reconfiguration is an internal facility: an MKL-facing DftiSetValue
must preserve the vendor's creation-only domain, precision, rank and lengths.
Public variadic/configuration/status rules, typed opaque shells, installation
and actual ABPLaS FFT replacement remain pending.

1. Extend the delivered CSR-backed and matrix-free internal operator contract
   into further algorithms only with explicit workspace and one admitted
   execution context. Preserve the existing CG widened accumulator and
   independently recomputed residual semantics.
2. Extend the delivered opt-in ABPLaS package integration into remaining
   application regressions and full SCF acceptance. Preserve the verified int64
   row offsets/int32 column indices, application-owned local/ghost blocks and
   halo communication, and overwrite/accumulate semantics.
3. Extend the internal FFT foundation with an installable FFTW-first FFT
   profile and FFTW plan lifecycle over the existing shared resources. Replace actual ABPLaS FFT call
   sites with numerical, lifecycle and application acceptance before complex
   eigensolver expansion. Performance work remains deferred.

## P1 acceptance checklist

- `dgesv`: row-major layout used by lattice coordinate and reciprocal-lattice
  construction, pivot encoding, singular `info` and multiple right-hand sides.
- `zpotrf`: complex Hermitian triangle semantics and non-positive-definite
  reporting used by CheFSI orthogonalization.
- `dstev`: real symmetric tridiagonal eigenvalues used by Lanczos; implement
  and test every mode advertised by the exported routine, including vectors.
- `dgelsd`: rank-revealing least squares, `rcond`, returned rank/singular values
  and rank-deficient inputs used by density mixing. Existing QR least squares
  or thin SVD alone do not establish this driver contract.
- Sparse: complex-double CSR products, transpose/adjoint, strides/layout,
  duplicate and empty entries, and 64-bit row offsets with 32-bit column
  ordinals. Keep index conversion and ownership explicit.
- ABPLaS integration: switch individual dependency boundaries; preserve
  Hamiltonian assembly, ghost communication and physical workflows in ABPLaS.
  Validate SCF convergence, density/energy and relevant existing physical cases.
  Replacing the direct calls does not replace ELPA/FEAST transitive dependencies.

## Architecture invariants

Dependency direction remains: interface profile, semantic adapter, normalized
operation/data contract, algorithm, shared execution resources, target kernel.

MKL conventions remain the first LAPACK/Sparse compatibility profile; FFT
compatibility starts with FFTW API/ABI. Generate selected external profiles
with standalone compile-time scripts over the same Zynum-defined internals. ABI integer widths, complex representation, layout,
index base, symbol naming and error translation belong at the boundary. Start
ABPLaS with its current LP64 interface; test ILP64 separately. Same-name LP64 and
ILP64 symbols require distinct selected artifacts, not a global runtime switch.

All profiles share internal allocator, thread/executor, workspace, handle and
operation-lifetime semantics. No vendor-specific duplicate runtime or numerical
implementation. Separate shared libraries must not be assumed to share runtime
state merely because their builds reuse a module definition.

Use an operator contract for explicit matrices and matrix-free application.
ABPLaS already has CheFSI, Lanczos and Chebyshev workflows: reuse these as
consumers before extracting generic algorithms. MPI ownership remains explicit.

## Source audit anchors

Paths below are relative to the local ABPLaS repository, not Zynum:

- `src/math/geometry/lattice.cpp`: `LAPACKE_dgesv`.
- `src/solver/eigensolver/chefsi_solver.cpp`: `LAPACKE_zpotrf` and complex BLAS.
- `src/solver/spectrum/lanczos_spectrum_estimator.cpp`: `LAPACKE_dstev`.
- `src/solver/density/density_updater.cpp`: `LAPACKE_dgelsd` and Kerker FFT.
- `src/core/operator/hmlt.h`, `hmlt_csr.cpp`: operator and distributed CSR.
- `src/math/linalg/feast_csr.cpp`: mixed-width input indices and interval solve.
- `src/math/linalg/elpa_dense_eig.cpp`: standard/generalized complex eigenproblems.
- `src/math/spectral/cheby_coeff.h`,
  `src/solver/hpp/hpp_filter_estimator.cpp`: FFT and DCT-I consumers.
- `src/core/poisson/README.md`: BigDFT PSolver remains the production backend;
  local FFT Poisson is diagnostic. FFT completion is not PSolver replacement.

The baseline audit was source inspection, not a new application execution or
performance measurement. Current local Zynum foundations do not imply complete
LAPACK/LAPACKE, MKL Sparse, ELPA, FEAST or FFTW compatibility.

---

## Historical release outline

Zynum `0.0.1-beta` establishes the first public foundation for the project:
BLAS Level 1-3 compatibility coverage, typed Zig vector/matrix views,
C/CBLAS/Fortran ABI compatibility, generated headers/modules, examples, tests,
benchmark tooling, and selected architecture-aware optimization work.

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

The `0.1.x` line continues BLAS completion and performance work while adding
LAPACK and Sparse foundations. Existing BLAS APIs and ABI remain compatible.

Release goals:

- Complete practical support for the full BLAS surface: every Level 1, Level 2,
  and Level 3 routine, all real and complex scalar types, packed/banded/triangular
  and symmetric/Hermitian variants, CBLAS row-major wrappers, Fortran ABI entry
  points, generated C headers, and generated Fortran module declarations.
- Keep typed Zig APIs aligned with the BLAS surface while preserving checked
  views, workspace APIs, and explicit aliasing rules.
- Support ARM and x86 CPU families with portable fallbacks plus feature-aware
  kernels for the important architecture tiers:
  - AArch64 ASIMD, SVE/SVE2 where available, SME/SME2, and experimental
    explicitly enabled Apple AMX paths on independently validated AArch64
    macOS deployments.
  - x86_64 baseline/SSE-family, AVX, AVX2/FMA, AVX512/FMA, and current Intel and
    AMD desktop/server CPUs.
- Maintain representative native gates for supported AArch64 and x86_64
  capability tiers. The target for 0.1 is that Zynum matches or beats the best
  eligible comparator across the documented BLAS benchmark suite, with
  `ZYNUM_MAXIMUM_THREADS` unset unless a single-thread gate is being measured.
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
- Keep non-default threading, SME, AMX, AVX2, and AVX512 experiments opt-in
  until native fresh-process evidence justifies a default rule. All retained
  parallel paths must use the shared core `std.Io.Threaded` helper lifecycle.
- Keep private Apple AMX disabled by default. Its build-time opt-in requires a
  deployment that independently confirms instruction execution because macOS
  exposes no reliable public runtime capability signal for this ISA.
- Record retained dispatch gates with target features, exact shape predicates,
  runtime controls, reproducible commands, curated evidence identities, and
  rollback criteria. Keep raw run artifacts outside the repository.
- Add better x86_64 validation on real Intel and AMD hardware.
- Keep shape policy in `src/blas/core/matrix_matrix/planner.zig`; keep instruction details
  under `src/blas/kernels/`.

## Local validation boundary

On this Apple arm64 Mac, execute only native macOS arm64 tests. Do not run
Linux or x86 tests locally, including through containers, virtual machines,
QEMU or translation layers. Linux/x86 runtime acceptance remains pending until
performed on the corresponding remote or CI environment. Local cleanup of
foreign test environments does not remove platform source support or CI jobs.

## LAPACK and Sparse Development

The portable Zig foundations now provide real/complex Cholesky and LU,
real QR and least squares, real symmetric eigenproblems and thin SVD in
[LAPACK](lapack/README.md). [Sparse](sparse/README.md) provides real/complex
CSR products and CSC/COO conversion, real triangular solves, and real CG.
Each operation uses caller storage with explicit checked contracts. These
initial algorithms do not claim broad LAPACK compatibility or competitive
performance across the full surface.

The separate experimental [checked C/Fortran interface](interop/README.md)
provides eight real Cholesky and CSR SpMV functions with explicit lengths and
statuses. [Module comparison tools](../bench/modules/README.md) provide isolated
real Cholesky/LU and CSR product comparisons, with numerical/storage checks and
source, build, library, setup, indexing and thread-policy records.

The active development target is complete LAPACK and Sparse coverage with
interchangeable interfaces over one internal implementation. The first foreign
compatibility profile follows classic oneMKL C/Fortran conventions, including
LAPACK/LAPACKE and Sparse. FFT starts with FFTW API/ABI. External profiles
are selectable generated boundary artifacts; internal data, algorithms and
resource management remain Zynum-defined and shared across those profiles.
The [layered architecture contract](architecture.md#shared-lapack-and-sparse-architecture)
defines the dependency direction, ownership and acceptance rules.

Work follows the P0–P4 priorities at the beginning of this document:

1. Stabilize and finish existing functionality, shared resource ownership,
   validation, lifecycle, installation and external-consumer contracts. Keep
   API/ABI profiles separate from common storage, algorithms and runtime.
2. Finish the ABPLaS minimum replacement set: LAPACKE `dgesv`, `zpotrf`,
   `dstev`, `dgelsd`, complex CSR SpMV/SpMM with independent offset/ordinal
   widths, and the unified operator bridge. Require focused application
   regressions before calling a migration complete.
3. Add FFT and complex standard/generalized Hermitian eigensolvers, then
   broaden algorithms and LAPACK/Sparse coverage. Maintain exact routine and
   profile matrices for types, modes, formats, indices, workspace and lifecycle;
   catalog entries alone never count as implemented support.
4. Defer competitor benchmarks and performance optimization until functionality
   and stability milestones are accepted and performance work is reactivated.
   Existing benchmark records are historical evidence, not an active work item.

The current implementation is still a foundation. Complete standard symbols,
all LAPACK/Sparse components, ILP64 semantics and oneMKL-compatible opaque
handles have not yet been delivered. No broad compatibility or performance
claim follows from the architecture or from the existing limited benchmarks.

## Future Modules

Candidates for future modules:

- `zynum-fft` for transform routines and compatibility layers.
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

## Selectable FFTW frontend integration (2026-10-13)

FFT compatibility starts with FFTW API/ABI. The finite Apple arm64 f64
ABPLaS profile implements the nine functions listed in
`tools/fftw_profile.json`; complete FFTW coverage remains pending.
`tools/generate_fftw.py` independently generates declarations and export
wrappers from that profile and its authoritative template. It generates no
numerical owner, allocator, executor, registry, workspace or plan cache.
The runtime binding borrows the existing exact Core and shared resources.

The unified LP64 installation selects `-Dfft-interface=none` (default, 186
shared exports) or `-Dfft-interface=fftw3` (195 shared exports). The latter
installs `zynum/fft/fftw3_subset.h` and its manifest. The finite header may be
included alone, or after the original `fftw3.h`. Generator manifest status
describes declaration generation; runtime acceptance is recorded separately.

The isolated integration passed 424 canonical Debug tests (94 numerical,
307 LAPACK/interface and 23 execution tests), three-mode compiler enumeration,
74 independent public FFT numerical cases including million-point transforms,
original/generated-header C/C++ static/shared consumers, all 184 existing
numerical exports and relocated pkg-config callers. Each selectable artifact
has one default Context. The actual ABPLaS FFTW candidate separately passed
305 selected unit tests, Poisson/nonlocal regressions, cold C8 SCF and one
non-Gamma diagnostic. Those application candidate results do not constitute
ABPLaS MAIN integration acceptance. Foreign-target runtime rows, allocation-free
FFT execution and broader FFTW profiles remain pending.

## Actual ABPLaS MAIN FFTW acceptance (2026-10-13)

The direct application FFT boundary is now integrated into ABPLaS MAIN and
reverified against the accepted Zynum MAIN SDK. Enable both
`ABPLAS_ZYNUM_LP64=ON` and `ABPLAS_ZYNUM_FFTW=ON`, and select an installed SDK
built with `-Dfft-interface=fftw3`. Both application options default to OFF.
Its nine direct FFTW functions and five canonical LAPACKE/Sparse entries bind
the same library and Core. FFTW MPI, PSolver and ELPA keep their current
providers. The relocated installed ABPLaS CMake client propagates this explicit
selection and rejects mismatched FFT SDK profiles.

MAIN passed 305 selected math/solver/SCF/HPP/Spectrum/DOS unit tests,
nonlocal and Poisson physical regressions at 1/2/4 MPI ranks, and actual cold
C8 SCF at four ranks in 11 iterations with final density refresh. The density
residual is 3.5184854049210824e-9, charge is 31.999999999999968, independent
reference energy error is 1.7763568394002505e-15 Ha per atom and physical
eigenvalue error is 7.986353800504276e-10 Ha. Reference-density comparison is
not included. The one-K non-Gamma DOS/CheFSI diagnostic also passed its target
and guard residual gates; this is not multi-K cold-SCF acceptance.

Complete FFTW coverage, complex standard/generalized Hermitian solvers,
general operator algorithms, and broader LAPACK/Sparse and foreign-runtime
coverage remain required work. No performance optimization or competitor
benchmark was added. Earlier candidate-only and pending-application notes
describe their respective historical stages.

## Internal Hermitian Core MAIN acceptance (2026-10-13)

`Core.internal.numerics.Hermitian` now provides standard complex Hermitian
eigensystems and all three positive-definite generalized problem types for
the existing ComplexF32/ComplexF64 scalar identities. It uses common borrowed
native dense matrices and widened numerical workspace. Inputs are preserved;
eigenvalue/vector outputs commit only after convergence and narrowing checks.
The managed paths inject the existing Execution, admit exactly one owned
operation or borrow a caller operation, and use the existing ContextScratch.
No additional context, allocator policy, executor, handle owner or cache is
introduced. Scratch acquisition precedes leaf validation.

MAIN passed 431 actually executed canonical Debug tests: 94 numerical,
314 LAPACK/interface and 23 execution tests. The seven new managed groups cover
allocation failure, counter/byte limits, numerical rollback, closing allocation
callbacks, admitted work after close, resource release order and six concurrent
callers on one Context. The byte-identical numerical/managed candidate passed
314 LAPACK-root bodies in Debug/Safe/Fast; the final MAIN test import differs
only in canonical formatting. Final compiler enumeration was recorded in all
three modes with closed CURRENT-only inventory slots. The unchanged leaf's
168 original-LAPACKE comparisons and six exact-input decimal eigenvalue checks
remain applicable; ill-conditioned eigenvector metric acceptance remains pending.

Both selectable installed profiles retain their exact 186/195 shared exports
and one Core per artifact. All 184 existing numerical exports were retested
through C/C++ static/shared callers, together with eight original/generated
FFTW-header clients and global/local lifecycle checks. Hermitian C symbols are
not installed by these profiles yet. Evidence is recorded in
`zig-out/module-hermitian-main-20261013/acceptance-receipt.json`.

Next gates are optional MKL LP64 LAPACKE cheev/zheev/chegv/zhegv generation,
installed C/C++ ABI and overwrite/workspace/status semantics, followed by
actual ABPLaS explicit-communicator distributed integration and MPI 1/2/4
regressions. Existing ELPA/OpenBLAS application paths remain selected. Complete
FFTW coverage, general operator algorithms and wider LAPACK/Sparse coverage
remain pending. The inventory retains 72 native observations and 222 foreign
rows awaiting enumeration; no foreign-runtime or performance claim is made.

## LP64 Hermitian generated-interface MAIN acceptance (2026-10-13)

The finite native Apple arm64 MKL LP64 interface now includes `LAPACKE_cheev`,
`LAPACKE_zheev`, `LAPACKE_chegv`, `LAPACKE_zhegv` and their `_work` forms.
The installed dense set has 56 numerical entrypoints. The unified LP64 package
has exactly 194 shared exports with FFT disabled, or 203 with the optional
FFTW profile enabled. Previous 48-entry dense sets remain reproducible, and
that combined whitelist remains supported by the Darwin linker.

The external Hermitian template is a compile-time generator input. It calls
the existing configured Shared binding, with the same Core scalar identities,
Execution, allocator capture, scratch leases and resource admission. It owns
no additional numerical implementation, default context, handle registry or
thread pool. The checked internal solver retains its transactional finite
contract; the LAPACKE interface selects the separate widened Householder / real
tridiagonal IEEE path and the ordinary overwrite/workspace/INFO protocol.

MAIN passed 456 canonical Debug test bodies: 94 numerical, 339 LAPACK/interface
and 23 execution tests. The final LAPACK-root source passed all 339 bodies in
Debug, ReleaseSafe and ReleaseFast and was compiler-enumerated in all three
modes. Both MAIN installed profiles passed 32 original/generated-header
Hermitian C/C++ clients, all 184 existing numerical exports through four
C/C++ static/shared clients, eight FFTW callers, one-Core identity checks and
combined global/local lifetimes. Four relocated pkg-config callers passed,
including generalized Hermitian plus FFT/Sparse lifetime calls. Eight installed
weak-handler/INFO clients and eight isolated raw-pointer fault processes cover
the generated terminal ABI path; cleanup itself is covered by the shared
adapter tests, rather than directly observed by those subprocesses.

This is finite native LP64 acceptance, not a complete MKL or FFTW profile.
The existing independent 168 numerical comparisons remain applicable to the
byte-identical numerical leaf. Invalid nonfinite generalized B trajectories
can differ from the original vendor (176 of 336 paired boundary INFO results
in the prior access candidate); NaN payloads, floating exception flags,
rounding trajectories and vendor Fortran hidden-length error-handler ABI are
not accepted as identical. The inventory still has 222 foreign rows awaiting
enumeration. No performance or competitor benchmark acceptance is claimed.

Actual ABPLaS distributed Hermitian integration remains the next gate:
borrow the explicit communicator for the entire call, preserve local
column-major block-cyclic storage and collective failure agreement, and run
real MPI 1/2/4 projected/exact-diagonalization regressions. Existing ELPA and
OpenBLAS Hermitian application paths remain selected until that integration
is verified. General operator algorithms and broader LAPACK/Sparse/FFTW
coverage remain pending.

## ABPLaS Hermitian application MAIN acceptance (2026-10-13)

The previous distributed-integration gate is now accepted for the current
native Apple arm64 finite LP64 package. `ABPLAS_ZYNUM_HERMITIAN=ON`, with
`ABPLAS_ZYNUM_LP64=ON`, selects the shared-Core complex-double standard and
generalized Hermitian provider in exact diagonalization and CheFSI. FFTW remains
an independently selected compile-time frontend. Internal implementations and
resource ownership remain Zynum-defined; compatible external declarations and
forwarders remain optional generator inputs.

The application borrows an explicit communicator, agrees rank-local preparation
and numerical failures, gathers block-cyclic matrices for a root native solve,
and distributes staged results before publishing caller outputs. Inputs remain
unchanged on this path. It introduces no additional numerical Context, thread
manager or workspace owner. This accepts distributed transport and lifecycle;
it does not claim a parallel distributed numerical eigensolver.

MAIN was rebuilt and passed 55 mathematics tests, actual MPI 1/2/4 Hermitian
checks, CheFSI run counts 49/49/61, selected symbol binding and a relocated
installed C++ facade client. The byte-identical candidate passed 311 selected
unit run counts, independent original-OpenBLAS spectra, failure/retry and parent
phase/closed-communicator/zero-local-storage checks, nonlocal and Poisson MPI
1/2/4 validation, C8 cold SCF with 11 iterations, and nonGamma DOS/CheFSI
validation. Those physical cases ran on the candidate build and were not
repeated on the separate MAIN build.

Evidence: `zig-out/module-abplas-hermitian-20261013/main-acceptance-receipt.json`.
Next, implement the shared admitted-operator Chebyshev recurrence and Lanczos
algorithm against the recorded actual application dependency contract. Complete
FFTW, broader LAPACK/Sparse and foreign runtime coverage remain pending. No
performance optimization or competitor benchmark was performed.

The Apple arm64 LP64 package now has an installed optional `zynum-v1`
operator frontend. Enable it with `-Doperator-interface=zynum_v1`; FFTW remains
an independent `-Dfft-interface=fftw3` selection. The installed header is
`zynum/operator/operator_v1.h`. It exposes four scalar families for Chebyshev
coefficient series, incremental `next = 2*A*current - previous` steps, Lanczos
Ritz estimates, and borrowed-operation i64-offset/i32-ordinal CSR SpMV/SpMM.
The previous 16 forward declarations remain unchanged; the step adds four.

Each algorithm entry uses the existing Core operation and typed Context scratch.
Callbacks receive a synchronous borrowed token for the admitted Sparse forwards.
The caller supplies affine scaling, communicator and distributed reductions.
The token and callback buffers must remain within their synchronous call scope.
The generator is optional and declares its script as a build-cache input;
changing and restoring the script was verified to regenerate the installed
header with identical build arguments and without deleting the cache.

The MAIN installed/relocated package has passed C/C++ static/shared step,
series/Lanczos, existing LAPACKE/Sparse, Hermitian, FFTW and lifetime consumers.
The candidate profiles contain 223 combined symbols, 214 without FFTW, and
203 without the operator frontend. All use one default Context definition.
The 25 algorithm test groups pass Debug, ReleaseSafe and ReleaseFast; compiler
enumeration registers 364 tests in the full LAPACK root. This is enumeration
of the full root, not a new numerical-body run of all 364 cases. The 222
foreign execution rows remain pending.

Actual ABPLaS streaming recurrence/Lanczos integration and MPI 1/2/4 physical
acceptance remain pending. Its callback/completion protocol must also include
ranks that return before the first callback, including allocation failures.
The generated operator frontend currently exposes vectors; the internal
Chebyshev panel implementation is a separate capability. Complete LAPACK,
Sparse and FFTW coverage remains in the development objective.
