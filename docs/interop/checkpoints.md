# Historical interface checkpoints

These dated records describe their original source, binaries and acceptance scope. They are retained for traceability and do not certify a later branch. See the [current scientific capability map](../scientific.md).

## Accepted finite FFTW f32 checkpoint (2026-10-16)

`-Dfft-f32=true` adds the 48-entry `fftw3.f32.abplas-v1` companion to an
FFTW-enabled unified LP64 build. Its default is false, and enabling it requires
`-Dfft-interface=fftw3`. The separate compile-time generator emits the original
FFTW-compatible declarations, forwarding functions, symbol list and manifest.
It installs `include/zynum/fft/fftw3f_subset.h` and `f32_abi_manifest.json`.
The original f64 generator, schema and four generated payloads are unchanged.
This is a finite interface; it does not claim the complete FFTW API.

The f32 and f64 frontends use one scalar binding over the existing Core,
Execution and Resources. Their allocation registry is shared across precisions;
scratch leases retain the allocator captured when admitted. Plans add no Core,
thread pool or Context owner. Native tests separately cover close during
allocation, allocator replacement, foreign Context rejection, four concurrent
admissions, busy close and eventual release. Standard FFTW execute-only thread
safety applies; planning and destruction remain serialized in the clients.

Canonical ReleaseSafe execution passes 1,571 actual Zig test bodies over 26
roots, with four original skips, all 139 build steps and no cached numerical
root results. The complete selected 56-method generation, owner and fixture
regression passes without skips, retaining every original 51 method. The two
earlier complete failing runs and their three/one stale fixture failures remain
recorded separately. The 72 actual compiler observations qualify enumeration;
222 foreign native execution rows remain pending. ABI tests record 194 tests
with one original skip; benchmark-tooling tests record 492 with nine original
skips. These tool tests do not establish performance results.

Fresh Main installation passes 96 steps and produces 47 files. The recorded
full selection has 754 unified defined exports, adding exactly 48 `fftwf_*`
symbols to the prior 706; standalone LP64/ILP64 LAPACKE sets retain 134/90
exports and Sparse sets are unchanged. Each of ten numerical artifacts has one
default Context. Separate library artifacts retain separate owners.

All 138 fresh Main C/C++ consumers pass: 108 existing consumers, 18 all-function
f32 consumers and 12 mixed-precision/cross-module consumers. They cover original
and generated declarations, C arrays and C complex, C++, static/shared linking
and relocated pkg-config paths. Every f32 function address is checked against
its expected image. Mixed consumers retain both precision plans through 36
BLAS drain/reset cycles with one/four thread limits. Thirty-two separate expected
alignment aborts remain outside successful typed-C numerical inputs. The first
relocation harness failure was a directory-name substring collision; its two
successful clients and failure are preserved, followed by a complete successful
18-client replay using canonical include/library directory checks.

The updated ABPLaS source was completely rebuilt, then its singleton HPP
nonfinite-result exception boundary was corrected. The correction passes all
three selected MPI4 unit groups: 1,049 GTest cases pass and two original nested
skips remain. The fresh Main runtime replay reuses that qualified executable
without further C++ recompilation: three production cases and two preparations
pass, the C8 cold SCF converges in 11 iterations, all 288 compared DOS dataset
payloads match, and six actual new-library rank images are verified. ABPLaS
continues its f64 FFT route and ELPA exact/projected Hermitian route; these runs
establish no f32 FFT or POTRI production adoption. Reference density arrays,
material TimeEvolution accuracy and statistical convergence remain unqualified.

`fftw-f32-fixture-final-20261016/functional-acceptance.json` in the external
`zynum-work` directory records the finite delivery; the separate
`abp-zynum-nonfinite-contract-20261016/functional-qualification.json` preserves
the application correction. Complete LAPACK/Sparse/FFTW interfaces, new ABPLaS
Hermitian routing and foreign execution remain unfinished. Existing benchmark
reports retain their recorded older SDK identities; this interface is not retimed.

## Accepted finite POTRI checkpoint (2026-10-16)

`-Dlapacke-potri=true` adds ordinary Cholesky inverse high/work entries for
four scalars to the selected LP64/ILP64 LAPACKE and unified LP64 packages.
The default is false. Both levels have five parameters; ordinary POTRI work
has no caller WORK array or workspace-query interface. Original MKL
declarations define the external wire contract. Internal inverse-factor and
Hermitian-product work uses the existing Core, native data, Context and
scratch leases. It adds no memory, thread, workspace or handle owner.
Accepted complex-factor cases have the real diagonal produced by Cholesky;
arbitrary nonreal-diagonal bitwise vendor equivalence is not claimed.

The optional component generator emits five files per integer width and
borrows the selected base manifest. It preserves the 238 accepted base sets
without copying a powerset of extended packages. All 727 prior public
header/export/definition payloads remain byte-identical; 242 existing
manifests update source provenance only. Default-off interfaces are preserved.

Canonical ReleaseSafe correctness passed all 139 build steps and 1,556
actual Zig test bodies over 26 roots, with four original skips and no cached
numerical-root results. ABI tests passed 194 cases with one original platform
skip; benchmark-tooling tests passed 492 with nine original skips. These tool
tests are not performance measurements. Seventy-two actual compiler
observations separately qualify the catalog; 222 foreign execution rows
remain pending. All migration windows are closed. The complete selected
51-method generation, ownership and source-fixture regression passed with
no skips. This does not claim a complete current build/test tooling suite.
The earlier failed source and its five reproduced fixture failures remain
separate from the corrected source and complete successful rerun.

Fresh Main installation passed 93 steps and produced 45 SDK files. All 108
freshly compiled C/C++ consumers passed: 14 standalone POTRI, 14 GETRI,
14 TRTRI, 14 TRTRS, 44 previous unified consumers and eight new unified
POTRI consumers. They cover both index widths, layouts, static/shared
linking, original-MKL-header-first declarations and relocated pkg-config
calls. The POTRI callers made 14,080 public calls; GETRI retained its 6,600
calls. Thirty-two separate alignment-abort processes are invalid typed-C
pointer guards and are not successful numerical inputs. Eight POTRI and
eight GETRI unified callers retain an FFT plan across BLAS shutdown.
Public counts are LP64=134, ILP64=90 and unified=706; Sparse sets are
unchanged. Each of ten numerical artifacts has one actual default Context.
Separate libraries retain separate owners.

The existing updated ABPLaS executable passed three regressions against the
fresh Main SDK, with six loaded rank images verified. Four-rank C8 cold SCF
converged in 11 iterations under the unchanged compact reference thresholds;
all 288 compared DOS dataset payloads match the accepted baseline. This
phase does not recompile ABPLaS or establish POTRI adoption. Density arrays
are not compared. The case named time_evolve uses Moment with two fixed
samples; material TimeEvolution accuracy and statistical convergence remain
unqualified.

`potri-main-candidate-20261015/functional-acceptance.json` in the external
`zynum-work` directory records this finite delivery. Standalone, unified,
canonical and source-fixture evidence retain their original snapshots.
At this POTRI checkpoint the single-precision FFTW work was a separate
private candidate. The newer f32 checkpoint above records its subsequent delivery. Complete LAPACK/Sparse/FFTW APIs, other generated
interfaces and foreign execution remain unfinished. Existing benchmark
reports describe their own older SDK snapshot; POTRI is not retimed.

## Accepted finite GETRI checkpoint (2026-10-15)

`-Dlapacke-getri=true` adds the four-scalar LU inverse high/work/query entries
to the selected LP64/ILP64 LAPACKE and unified LP64 packages. The default is
false. High calls have five parameters; work calls have seven and use
`lwork=-1` for query. Original MKL declarations supply the external wire
contract. Generated code forwards once to the shared owner. The pure Core
implementation, native Dense/Pivot data and widened output workspace remain
internal choices. Public WORK and private leases use the existing Context
and scratch allocator, with no added Context, thread, pool or handle owner.

Canonical ReleaseSafe correctness actually passed 1,532 Zig test bodies over
26 roots with four original skips and all 137 build steps. It also passed
194 ABI tests with one original platform skip and 492 benchmark-tooling tests
with nine original skips; these do not establish performance measurements.
The 72 compiler observations qualify discovery separately. Twenty-six finite
generation/selector methods and five original directory fixture methods pass.
All migration windows are closed; 222 foreign execution rows remain pending.

Fresh Main installation passed all 83 build steps and produced 39 SDK files.
All 86 freshly compiled C/C++ client processes passed: 14 standalone GETRI,
14 prior TRTRI, 14 prior TRTRS and 44 unified consumers. They cover both index
widths, row/column layouts, static/shared linking, original-MKL-header-first
compilation and relocated pkg-config use. The 22 GETRI clients made 6,600
checked public GETRI calls. Eight unified GETRI clients retain a live FFT
plan across BLAS shutdown, execute GETRI and re-execute the original plan
after another shutdown. Export counts are LP64=126, ILP64=82 and unified=698;
each of ten numerical artifacts retains one actual default Context. Separate
libraries retain separate owners. Eight earlier candidate clients verify
GETRI-default-off headers, exports and prior TRTRI behavior. These are finite
Apple arm64 interfaces; full vendor profiles remain pending.

The existing updated ABPLaS executable passed three production regressions
against the fresh Main SDK, with six actual loaded rank images checked.
Four-rank C8 cold SCF converged in 11 iterations under the unchanged compact
reference thresholds. All 288 compared DOS dataset payloads match the earlier
accepted baseline. This phase does not recompile ABPLaS or establish GETRI
adoption. Density arrays are not compared, and the case named time_evolve
uses Moment with two fixed samples; material TimeEvolution accuracy and
statistical convergence remain unqualified.

`getri-main-candidate-20261015/functional-acceptance.json` under the external
`zynum-work` directory records this bounded Main delivery. Source lineage is
retained in `getri-standard-20261015`, `getri-integration-20261015`,
`getri-canonical-integration-20261015` and
`getri-fixture-assertion-repair-20261015`. The original failed full 72 and the
separate frozen full 469 tooling run do not establish a complete current
GETRI tooling pass. Existing benchmark artifacts describe their older SDK;
GETRI is not retimed. Broader LAPACK/Sparse/FFTW coverage remains pending.

## Accepted finite TRTRI checkpoint (2026-10-15)

The finite four-scalar triangular inverse is available with
`-Dlapacke-trtri=true`. Its six-parameter high/work boundaries forward to the
existing Core and borrow the same Context, allocator and scratch lifecycle.
The mathematical implementation and widened workspace remain internal choices;
the generator owns declarations and forwarding only.

The canonical ReleaseSafe aggregate actually executed 1,510 successful Zig
test bodies across 26 roots, with four original skips, and passed all 135 build
steps. It also ran 194 ABI tests (one original platform skip) and 492
benchmark-tooling tests (nine original skips). These tooling tests do not
establish new performance measurements. The 72 compiler observations qualify
test discovery separately; 222 foreign execution rows remain pending.

Fresh Main installation passed all 83 build steps and produced 39 SDK files.
All 64 freshly compiled C/C++ client processes passed: 14 standalone TRTRI,
14 prior TRTRS and 36 unified consumers. These cover LP64/ILP64, static/shared
linking, original-MKL-header-first compilation and relocated pkg-config use.
Eight unified TRTRI clients retain a live FFT plan across BLAS shutdown,
perform the inverse and re-execute that plan after another shutdown. The
enabled LAPACKE LP64/ILP64 and unified packages have 118/74/690 public symbols.
Each of ten numerical static/shared artifacts contains one default Context;
separate libraries retain separate owners. Eight earlier installed candidate
clients also verified TRTRI-default-off headers, exports and prior TRTRS calls.

The existing updated ABPLaS executable passed three production regressions
against this fresh Main SDK, with six actual loaded rank images checked.
Four-rank C8 cold SCF converged in 11 iterations at -47.57100851624413 Ha under
the unchanged compact reference thresholds. All 288 compared DOS numeric
dataset payloads matched the earlier accepted baseline. ABPLaS was not
recompiled and uses no TRTRI entry. The density, Moment sampling and material
TimeEvolution limits in the historical checkpoints below still apply.

Current-only build and test-structure gates pass. The old finite-generation
test fixture listed only 23 interface sets; it now independently enumerates
the exact 118 accepted sets and their symbol counts. Its original method and
the full 20-method finite-generation class pass, and seven separate negative
probes reject missing, extra and miscounted interfaces. This fixture-only
repair and the documentation amendment leave all compiled inputs unchanged.
The independent frozen TRTRS full build-tooling run has additional failures;
its results remain separate. A full current 469-method build-tooling pass is
not claimed.

`trtri-main-candidate-20261015/functional-acceptance.json` in the external
`zynum-work` directory records this bounded Main delivery. Standalone,
unified and canonical evidence remains in `trtri-standard-20261015`,
`trtri-integration-20261015` and `trtri-canonical-integration-20261015`.
Complete vendor profiles and wider LAPACK/Sparse/FFTW coverage remain pending.
Older references to the latest or current installation below describe their
named historical checkpoints. The earlier competitor benchmark is unchanged.

## Accepted POSV integration checkpoint (2026-10-15)

The updated ABPLaS consumer has been integrated and built. The latest finite
Apple arm64 Main installation has 31 files and passed 28 C/C++ standalone,
unified and cross-module client processes across static/shared linking and
requested thread caps of one and four. Each of the six installed numerical
artifacts contains one default Context; separately loaded libraries retain
separate owners. Ordinary POSV is qualified through external clients and is
not called by ABPLaS.

The existing installed ABPLaS executable was replayed against that fresh Main
SDK: four-rank C8 cold SCF converged in 11 iterations at
-47.57100851624413 Ha and passed the unchanged compact reference thresholds.
The two DOS cases preserve all 288 compared HDF5 dataset payloads relative to
the previous accepted SDK. This compares dataset names, types, shapes and
values, not HDF5 container bytes or attributes. No ABPLaS source recompilation
was performed during the POSV replay phase. The SCF reference does not compare
a density array. The case named `c8_dos_time_evolve` uses the Moment backend
with two fixed samples and does not establish material TimeEvolution accuracy
or statistical convergence.

The receipt is `posv-standard-20261015/functional-acceptance.json`, SHA256
`0ac6f26e3a5f1fc93455093745dc6efc3c12504cf09a644027729d49b5aedb04`,
in the external `zynum-work` evidence directory. Current-only build and test
structure gates pass. At this functional checkpoint, the broader Python tooling
regression repairs remain in independent snapshots, and their full suites are
not yet accepted in Main.
The 222 foreign native-execution rows remain pending. Full LAPACK/Sparse,
complete FFTW and full vendor API/ABI profiles remain unfinished.

The requested local LAPACK/Sparse/FFT comparison has separately completed:
72 shapes, requested thread caps of one/four, 1,276 qualified processes and
17,588 checked timing calls. Its report is
`module-benchmark-20261015/report/README.md` in the same external evidence
root, with CSV/JSON and raw samples alongside it. Those timings describe the
recorded older benchmark SDK; they are not measurements of the POSV-enabled
SDK. No optimization or new timing run was performed for this checkpoint.

Later sections retain historical phase boundaries. Their earlier pending
Main publication, updated-consumer integration and benchmark statements must
be read with this checkpoint and the corresponding immutable phase receipts.

## TRTRS integration and validation checkpoint (2026-10-15)

The next finite interface adds ordinary four-scalar triangular solves through
`-Dlapacke-trtrs=true`. The generated frontend forwards to the same Core and
captured Context. Internal arithmetic and storage remain independent of the
external MKL-compatible declarations. The optional generation script emits
only the selected boundary; it introduces no numerical implementation or
interface-specific resources.

The canonical ReleaseSafe aggregate passed all 133 build steps. The preceding
run actually executed 1,491 successful Zig test bodies with four original
skips; that run failed only the stale Python ABI observer. After the observer
and its source-archive fixtures were repaired, the successful aggregate reused
those unchanged numerical results and actually ran all 194 ABI observer tests
(one original platform skip) and 492 benchmark-tooling tests (nine original
skips). These tooling tests do not measure numerical performance. The original
failure and both source snapshots remain recorded separately.

Selected standalone LP64/ILP64 candidate packages passed 14 installed C/C++
processes across static/shared linking, original MKL-header-first compilation
and relocated pkg-config consumers. Eight additional processes verify that
disabling TRTRS preserves the prior 102/58 public symbols and header bytes.
The enabled full standalone packages have 110/66 public symbols, exactly eight
more per width. The unified LP64 candidate has 682 public symbols and passed
20 installed consumer processes, including eight TRTRS clients that keep an
FFT plan live across BLAS shutdown/reinitialization. Each artifact retains one
default Context; separate libraries retain separate owners.

The existing updated ABPLaS executable also passed the same three production
regressions against the TRTRS candidate SDK: C8 cold SCF converged in 11 steps
at -47.57100851624413 Ha, and both DOS cases preserved all 288 compared dataset
payloads. This replay did not recompile ABPLaS or introduce TRTRS calls into it.
The density, Moment sampling and material TimeEvolution limits above still
apply.

Fresh Main acceptance has now passed: installation completed all 83 steps and
produced 39 files. All 42 standalone, unified and cross-module C/C++ client
processes passed; ten static/shared numerical artifacts each contain one
default Context. Original-header and relocated consumers load the recorded
Main libraries. The actual updated ABPLaS executable passed the three cases
again against this fresh Main SDK, with six rank images checked. Its SCF
energy/iteration result and all 288 compared DOS dataset payloads remain as
above. No ABPLaS compilation or new TRTRS production adoption occurred.

The full Git-mode build gate also exposed a stale section-wide derived-file
digest hidden by archive-mode fixtures. Updating that one expected digest
records exactly 96 reviewed TRTRS additions; all 201 earlier records are
unchanged, and the original validation logic remains in place. The temporary
repository fixture also now copies the TRTRS source/classification registry.
Its original failing alias test and full positive validation both pass.
Full default-order tooling sweeps remain separate and are not yet accepted.
`trtrs-main-candidate-20261015/functional-acceptance.json` in the external
evidence directory records this bounded Main delivery and source provenance.

Detailed evidence is in `trtrs-canonical-integration-20261015`,
`trtrs-standard-20261015` and `trtrs-integration-20261015` under the external
`zynum-work` directory. The full default-order 467-method build-inventory and
72-method test-inventory suites run against an immutable copy in
`trtrs-full-tooling-20261015`; they are not yet counted as passed. The 72 native
compiler observations describe test discovery, not test-body execution. The
222 foreign execution rows and broader LAPACK/Sparse/FFTW coverage remain
pending. The earlier competitor benchmark has not been repeated for TRTRS.

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

## Accepted ABPLaS Hermitian selection

`ABPLAS_ZYNUM_HERMITIAN=ON` requires the installed finite LP64 package and
`ABPLAS_ZYNUM_LP64=ON`. Its configuration and installed CMake package check the
required Hermitian entries; older packages remain usable with this selection
disabled. Exact diagonalization and CheFSI use the neutral dense-provider facade,
with the existing ELPA provider retained when the selection is disabled. An
explicit independent OpenBLAS oracle is required for LP64 application unit tests
so the new exported `LAPACKE_zheev` cannot silently become its own reference.

The current adapter borrows a valid communicator for the entire call and stages
block-cyclic transport around a root `LAPACKE_zheev`/`LAPACKE_zhegv` call into the
same installed Core as FFT/LAPACK/Sparse. Numerical, collective failure/retry,
subcommunicator, zero-local-shard, active-parent phase and relocated external
calling checks are accepted on native Apple arm64. Candidate C8 SCF/nonGamma
physics passed; MAIN was separately rebuilt from identical source and checked
for affected MPI/solver/external calls. See
`zig-out/module-abplas-hermitian-20261013/main-acceptance-receipt.json` for the
precise separation of candidate physical evidence and direct MAIN checks.

This does not extend the finite interface manifests into complete MKL/FFTW
coverage. General operator algorithms and broader profiles remain pending; all
external frontend generation stays optional and separate from internal data and
resource management.

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
The earlier candidate profiles contained 223 combined symbols, 214 without FFTW, and
203 without the operator frontend. All use one default Context definition.
The 25 algorithm test groups pass Debug, ReleaseSafe and ReleaseFast; compiler
enumeration registers 364 tests in the full LAPACK root. This is enumeration
of the full root, not a new numerical-body run of all 364 cases. The 222
foreign execution rows remain pending.

The earlier protocol-only checkpoint preceded actual ABPLaS integration. The
callback/completion protocol must also include
ranks that return before the first callback, including allocation failures.
The generated operator frontend currently exposes vectors; the internal
Chebyshev panel implementation is a separate capability. Complete LAPACK,
Sparse and FFTW coverage remains in the development objective.


## Admitted FFTW execution and native ABPLaS integration

With both `-Doperator-interface=zynum_v1` and `-Dfft-interface=fftw3`, the
compile-time generator additionally emits `zynum_operator_fftw_execute_admitted`.
This entry borrows the current synchronous callback token and executes a live
FFTW plan's original arrays. The plan must belong to the same injected Core;
null arguments, owner mismatch and workspace failures return checked status.
No nested operation, separate Context or retained token is created. Original
FFTW API/ABI declarations and their IEEE propagation rules remain unchanged.
This profile has 21 operator entries and 224 combined shared exports. With FFTW
disabled it retains 20 operator entries and 214 exports; disabling the operator
profile retains 203 exports with FFTW enabled.

Internal lifecycle tests cover close-after-admission, foreign-owner
rejection, allocation rollback/retry, and padded real-volume roundtrip against
an independent DFT in Debug, ReleaseSafe and ReleaseFast. Installed C/C++
static/shared clients verify execution with an outer Chebyshev callback token;
eight ordinary FFTW clients remain passing. Exact manifest/export equality and
one default Context are checked for all three profiles. The reviewed implementation
and closed inventory transitions have now been integrated into MAIN. Its fresh
installed ReleaseSafe SDK has the exact 224-symbol set and one default Context;
C/C++ static/shared admitted and ordinary FFTW clients pass. The 10 FFT and
25 general-operator numerical/lifecycle groups pass in all three build modes.
Compiler enumeration now registers 367 tests; this is enumeration rather than
numerical execution of the full root. The 222 foreign rows remain pending.

Actual native ABPLaS integration borrows the application's communicator and
uses the same installed Core for streaming Chebyshev and Lanczos. Its split_proj
representation uses CPU finite differences and projector/MPI transport, without
FFT calls. Native CSR, borrowed child communicators, empty shards, real active
halo rejection/drain/retry, and actual installed-Core allocation failure have
passed MPI 1/2/4 acceptance. The opt-in Darwin shared-library OOM fixture rejects
real Core allocations, checks collective failure origin, real frees, transactional
outputs and same-object retries for startup/step/series/Lanczos. It adds no
production fault-control API or second Context. Static/platform fault injection
is not verified. Fresh MAIN production split HPP force np1/np2/np4 and complete
native C8 SCF np4 pass; the relocated installed ABPLaS consumer also passes.
An older protocol-only Debug C8 timed out and is recorded separately from this
native ReleaseSafe physical acceptance. Complete LAPACK, Sparse, FFTW and
foreign-platform coverage remain in the full development objective.

### FFTW real-to-real runtime expansion (2026-10-13)

The existing `fftw_plan_r2r_1d` entry now supports all eleven FFTW transform
kinds. The implementation uses a neutral immutable Core real-transform
descriptor and borrowed workspace under the existing execution and handle
owner. The original cosine-I numerical leaf remains in use. Internal real
scalar families are tested through ComplexF32 and ComplexF64; the installed
FFTW profile continues to expose its existing double-precision declarations.

Thirteen FFT binding test groups pass in Debug, ReleaseSafe and ReleaseFast.
Fourteen managed facade groups pass in Debug. Independent installed FFTW
comparison covers 98 configurations per C/C++ static/shared client, including
in-place calls, guards and execution within the borrowed native callback
operation. All eleven kinds have allocation failure, close, cleanup and
same-operation retry checks. Actual ABP HPP fixtures pass at MPI 1/2/4 with
their declared minimum-rank exclusions (13/16/17 groups respectively).

Public headers and the 224 combined exports remain unchanged. Twelve finite
declaration manifests were regenerated from reviewed source-envelope inputs.
Build source projection and shared ownership are unchanged. Compiler root
enumeration now contains 370 names; this is distinct from numerical body
acceptance. The test inventory retains 72 native observations and 222 foreign
platform rows requiring native enumeration. New-array, many, guru and other
precision FFTW interfaces and complete LAPACK/Sparse coverage remain pending.

### FFTW new-array execution integration (2026-10-13)

The optional double-precision FFTW boundary additionally generates
`fftw_execute_dft`, `fftw_execute_dft_r2c`, `fftw_execute_dft_c2r`,
`fftw_execute_r2r` and `fftw_alignment_of`. Each new-array execution checks
the plan's transform family and original in-place or out-of-place layout
before admission or allocation, then uses the existing shared Core and
invocation-owned scratch. Original plan arrays and metadata remain unchanged.
The scalar implementation requires native double alignment only; all valid
arrays share one alignment class, without an additional SIMD restriction.

Sixteen FFT binding groups pass in Debug, ReleaseSafe and ReleaseFast,
including allocation rollback, close with a retained operation, foreign-owner
rejection, physical real-volume padding and six concurrent workers. Four
C/C++ static/shared clients compare 98 r2r configurations and complex-1D,
real-1D forward and real-3D backward new-array calls against installed FFTW.
Each client also executes one plan on six concurrent independent arrays.
The original header and generated declarations compile together.

The combined package has 229 exports and one default Context. Disabling FFTW
produces 214 exports with no FFTW header or symbols; disabling the unified
operator boundary produces 208 exports. The generator accepts exactly the
reviewed ordered fourteen FFTW symbols; the linker rejects missing or foreign
FFTW symbols. Generated declaration metadata is separate from runtime
acceptance. Actual ABP HPP groups pass at MPI 1/2/4 (13/16/17 groups), and C8
native split MPI4 cold SCF converges in eleven iterations within its frozen
energy/eigenvalue budgets. No reference density comparison was performed.

Compiler enumeration contains 373 names, independently of numerical body
execution. Strict inventories retain 72 native observations and 222 foreign
platform rows pending native enumeration. Split-array execution, many/guru
planning, remaining ranks and precisions, wisdom and thread/MPI APIs, and
complete LAPACK/Sparse coverage remain pending.

### Neutral tensor FFT and arbitrary-rank FFTW planning (2026-10-13)

Core `TensorPlan` borrows immutable neutral `usize` dimensions and uses the
existing one-dimensional numerical leaves with caller workspace. It supports
complex forward/backward and real forward/backward transforms of arbitrary
rank, including zero-rank scalar copies. Last-axis half-spectrum storage and
reflection across every frequency axis define real inverse reconstruction.
Checked execution rejects shape/data/workspace alias, short buffers and
nonfinite input/result before committing output; the same algorithms provide
IEEE execution at the external boundary. No allocator or owner is added to
the numerical plan.

The optional FFTW generator adds complex 2D/3D/general planners, real forward
2D/general planners, and real backward 1D/2D/general planners (22 FFTW
functions total). The adapter copies external dimensions into captured
ContextScratch, retaining a dimension resource and a handle resource under
the same Context. It releases both after close without new admission.
New-array execution and the existing borrowed operation seam share these
immutable plans. Multidimensional out-of-place c2r with PRESERVE_INPUT
returns NULL; in-place c2r accepts that flag, matching the installed FFTW
oracle, including the existing 3D entry.

Twenty FFT groups pass in Debug, ReleaseSafe and ReleaseFast; fourteen
managed groups pass in Debug. Zero through seven ranks and both internal
scalar families are compared with independent direct definitions. Shape
copies, staged allocation failure, close during metadata allocation,
captured allocator replacement, transactional rollback and IEEE behavior
are checked separately. C/C++ static/shared clients compare 112 numerical
configurations each against installed FFTW across ranks zero through eight,
signs, original/new arrays and physical in-place real row padding. Each
client also runs six concurrent workers on one four-dimensional plan.

The combined SDK has 237 exports and one default Context. No-FFT and
no-operator packages have 214 and 216 exports respectively, each with one
Context. Exact generator and finite linker negative cases reject foreign,
missing and reordered declarations as applicable. Twelve MKL finite
manifests were regenerated solely for reviewed Core source envelopes; their
public declarations remain unchanged. ABP HPP groups pass at MPI 1/2/4
(13/16/17 groups). C8 native split MPI4 cold SCF converges in eleven
iterations within frozen energy/eigenvalue budgets, without a reference
density comparison.

Canonical compiler enumeration now contains 377 names. This is distinct
from numerical body execution. Strict inventories retain 72 native
observations and 222 foreign platform rows pending native enumeration.
Multidimensional r2r, many/guru and split-array interfaces, remaining external
precisions, wisdom/thread/MPI APIs and complete LAPACK/Sparse coverage
remain pending.

### Arbitrary-rank separable real transforms (2026-10-13)

Neutral `TensorRealTransformPlan` borrows immutable dimensions and per-axis
`RealTransform` kinds. It applies the existing one-dimensional real leaves
along each axis, with invocation-owned row buffers and unit workspace. Zero
rank copies one real scalar. Halfcomplex and Hartley kinds are separable r2r
products, rather than multidimensional complex DFT substitutes. Checked
execution preserves the complete output on metadata, buffer, nonfinite or
intermediate overflow failure; IEEE execution shares the numerical leaves.

FFTW 2D/3D/general r2r planners are generated from the optional profile.
Wire kinds are explicitly translated for both the existing 1D planner and
the new planners. One captured metadata block holds neutral dimensions and
kind bytes; the same Context owns that block and the opaque handle. Caller
dimensions/kinds may change after planning. New-array and admitted execution
reuse immutable metadata and per-call workspace; teardown works after close.

Twenty-three FFT groups pass in Debug, ReleaseSafe and ReleaseFast, including
all 11x11 mixed-kind inverse normalization in both internal scalar families,
in-place guards, staged OOM, close and captured allocator cleanup. Fourteen
managed groups pass in Debug. Each C/C++ static/shared client compares 3,080
configurations with installed FFTW: every 2D/3D kind product, ranks zero
through seven, original/new and in-place arrays, metadata copies and six
concurrent workers. The package has 25 FFTW functions, 240 combined exports
and one default Context. No-FFT/no-operator configurations have 214/219
exports respectively, each with one Context. Exact generator/linker negative
cases reject drift. Twelve MKL manifests were regenerated only for reviewed
Core source envelopes, preserving their public declarations.

Actual ABP HPP groups pass at MPI 1/2/4 (13/16/17 groups). C8 native split MPI4
cold SCF converges in eleven iterations within frozen energy/eigenvalue
budgets; no reference density comparison was performed. Compiler enumeration
contains 380 names, separately from numerical body execution. Native
observations remain 72, with 222 foreign platform rows pending enumeration.
Many/guru and split-array interfaces, remaining external precisions,
wisdom/thread/MPI APIs and complete LAPACK/Sparse coverage remain pending.

## Strided complex FFT checkpoint (2026-10-13)

Neutral `StridedComplexPlan` borrows transform/batch extents and signed input/
output strides. It gathers all touched values before computation, reuses the
existing Tensor/1D leaves, and commits only touched output positions. Negative
strides, embedded/interleaved batches and overlapping input/output storage
preserve holes. Checked execution validates every result before committing.
Workspace comes from the existing Context; the borrowed plan owns no allocator,
thread pool or secondary manager.

The optional generator adds exactly `fftw_plan_many_dft`,
`fftw_plan_guru_dft` and `fftw_plan_guru64_dft`. Guru wire structs use the
original FFTW tags and checked int/ptrdiff_t layouts. One captured metadata block
holds six neutral extent/stride vectors; the handle and invocation workspace
use the same Context. Inactive strides impose no in-place constraint. Empty
NULL-array plans perform admitted non-accessing execution; nonempty unbound
ESTIMATE plans require new arrays.

Twenty-nine FFT numerical/lifecycle groups pass in Debug, ReleaseSafe and
ReleaseFast. Eight C/C++ static/shared clients using original or standalone
generated headers pass 38 signed-layout, embedding, metadata-copy, in-place,
NULL-planning and direct-DFT/FFTW oracle configurations each. Shared plans also
run through six workers with eight new-array calls each. The candidate package
has 28 FFTW functions, 243 combined exports and one default Context;
no-FFT/no-operator configurations have 214/222 exports.

Actual ABP HPP MPI 1/2/4 groups pass (13/16/17). C8 native/split MPI4 cold SCF
converges in eleven iterations within frozen energy/eigenvalue budgets; no
reference density comparison was performed. Compiler enumeration adds six
names to the previous LAPACK root, separately from numerical body execution.
Native observation bindings remain 72, and 222 foreign platform rows remain
pending. Main publication and fresh Main SDK verification are tracked in phase
receipts. Many real/split transforms, other external precisions,
wisdom/thread/MPI APIs and complete LAPACK/Sparse coverage remain pending.

## Strided real FFT checkpoint (2026-10-13)

Neutral `StridedRealPlan` reuses signed layouts and the existing real/half-spectrum
and per-axis real-transform leaves. It gathers all batches before calculation
and stages results before committing touched outputs. Checked failure preserves
output and holes even for overlapping real/complex views. The same Context
captures immutable half dimensions, kinds and strides with the handle/workspace.

Nine optional generated many/guru/guru64 real/r2r planners bring the candidate
package to 37 FFTW functions, 252 exports and one default Context. Explicit
embeddings use logical minimum extents; omitted in-place real embeddings retain
standard padding. Real/complex element strides need not coincide. Unbound
ESTIMATE plans require new arrays. No-FFT/no-operator exports are 214/231.

Forty FFT groups pass in Debug/ReleaseSafe/ReleaseFast, including metadata/state
OOM cleanup and allocation-triggered close under the original admission.
Eight C/C++ original/standalone-header static/shared clients each pass 108 signed
embedded out-of-place oracle configurations. Shared-library checks cover 216
rank-one in-place configurations (including unsupported NULL cases), sixty
signed guru/guru64 configurations at ranks zero through four, 54 empty/unbound
planner comparisons, and six workers with eight new-array calls each in three
real families. Numeric results use independent analytical references and FFTW.
Exact generator/linker negative symbol sets reject drift.

ABP HPP MPI1/2/4 passes 13/16/17 groups. C8 native/split MPI4 cold SCF converges
in eleven iterations within frozen energy/eigenvalue budgets; no reference
density comparison was performed. Canonical LAPACK enumeration contains 397
names, separately from numerical body execution. Native observation bindings
remain 72, with 222 foreign rows pending. Main publication and fresh Main SDK
verification are tracked separately in phase receipts. Split arrays, other
external precisions, wisdom/thread/MPI APIs and complete LAPACK/Sparse coverage
remain pending.

## Split-plane FFT checkpoint (2026-10-13)

Neutral `ConstPlane`/`Plane` views pair bounded scalar slices with origins.
SplitComplexPlan/SplitRealPlan reuse signed layouts, existing tensor/real
leaves, complete input gathering and transactional output commit. Touched
output-cell collision checks permit overlapping byte spans for interleaved
planes without permitting conflicting output cells. Metadata, handles and
invocation workspaces use the existing Context and no extra owner/registry.

Six optional guru/guru64 split planners plus three new-array execute functions
bring the candidate to 46 FFTW functions and 261 exports with one Context.
Bindings preserve original plane gaps and alias relations, support exchanged
planes for complex backward execution, and bridge corresponding packed/split
new-array formats. Native FFTW 3.3.11 accepts the tested multidimensional paired
in-place real layouts despite narrower manual guarantees; the candidate follows
the observed numerical behavior. Imaginary/output cross-plane real aliases
remain rejected where native planning rejects them.

Fifty-three FFT groups pass Debug/ReleaseSafe/ReleaseFast. Eight C/C++,
original/standalone-header static/shared clients pass sixteen split original/new
configurations each. Shared-library independent/native checks cover 64 signed,
embedded, two-dimensional split configurations; sixteen packed-real cross-format
configurations; complex format bridges; three families with six workers and
eight calls per worker; and 36 rank-zero/one/two empty or paired-in-place boundary
configurations. Original-unbound access, OOM, touched output collision, metadata
copying and allocation-triggered close have separate numerical/lifetime checks.

No-FFT/no-operator profiles have 214/240 exports. Exact 46-symbol generator and
linker negative cases reject drift. HPP MPI1/2/4 passes 13/16/17 groups; C8
native/split MPI4 cold SCF passes frozen energy/eigenvalue budgets in eleven
iterations, with no reference density comparison. Canonical LAPACK enumeration
contains 410 names, separately from numerical execution. Observation bindings
remain 72 with 222 foreign rows pending. Main publication/fresh SDK checks are
recorded in phase receipts. Other external precisions, planning resource/
wisdom/thread/MPI APIs and complete LAPACK/Sparse coverage remain pending.

### FFTW aligned byte/real allocation and unbound ESTIMATE checkpoint (2026-10-13)

The optional f64 FFTW profile now generates 48 functions, including `fftw_malloc` and `fftw_alloc_real`; the installed full LP64 SDK exports 263 symbols and retains one shared default numerical context. Byte, real and complex allocation use the same captured-context aligned allocation/free lifecycle. Zero-size blocks are freeable; overflow and allocator failure return NULL. Basic and generic tensor/r2r planners accept NULL/NULL or one-sided NULL under ESTIMATE, retain in-place/out-of-place metadata, and require valid typed new arrays for execution. This does not certify MEASURE planning with missing arrays, wisdom, float/long-double profiles, or complete FFTW compatibility.

This checkpoint executed 56 FFT numerical, boundary and lifecycle test bodies in Debug, ReleaseSafe and ReleaseFast. Eight C/C++ clients exercised original FFTW/standalone headers and static/shared installed libraries. Twenty-four both-null/one-sided-null generic new-array configurations passed native FFTW and analytical numerical comparison. Captured allocator replacement, close callbacks, already admitted work, rejected new work and cleanup were checked. Canonical LAPACK-root enumeration contains 413 test names; enumeration is distinct from executing all bodies. The inventory retains 72 native bindings and 222 foreign rows pending.

ABPLaS HPP fixtures passed at MPI 1/2/4 ranks (13/16/17 groups). Native/split C8 MPI4 cold SCF converged in 11 iterations and passed the frozen compact energy/eigenvalue reference; no reference density comparison was performed. Main publication and fresh Main SDK acceptance are recorded separately in `zig-out/module-fft-allocation-20261013/MAIN-acceptance-receipt.json`. The earlier benchmark remains tied to its recorded ReleaseFast SDK; these allocation/planning additions were not benchmarked. Optional no-FFT and no-operator profiles retain 214 and 242 exports respectively.

### Optional QR unified SDK candidate (2026-10-13)

`-Dlapacke-qr=true` selects the generated Apple arm64 LP64 GEQRF/ORMQR/UNMQR frontend in the finite LAPACKE and unified ABPLaS packages. The default remains the earlier 56-entry LAPACKE profile. QR adds 16 functions for four scalar families, high/work/query, real N/T and complex N/C, left/right application and row/column layouts. The numerical implementation remains in the same Core with checked and own IEEE projection contracts; vendor nonfinite trajectories, payloads, rounding and exception flags are not equated.

The candidate installed unified SDK has 279 exports, 72 LAPACK and 48 FFTW functions, with one default Context. Disabling QR retains 263 exports and no QR C symbols. Sixteen C/C++ original/installed-header static/shared clients passed across original and copied/relocated install prefixes, including queries with NULL data pointers and Q roundtrip checks. Shared-loader logs confirm the selected prefix. Three malformed 72-symbol manifests were rejected before linking output. Standard and checked Sparse provenance refreshes preserved the full existing declaration/type/semantic contracts.

ABPLaS HPP MPI 1/2/4 passed 13/16/17 groups with this candidate installed SDK. C8 native/split MPI4 cold SCF passed the frozen energy/eigenvalue reference in 11 iterations; no reference density comparison was performed. QR is not an actual ABPLaS call dependency in these fixtures. These results are candidate SDK/application evidence, not Main publication, canonical inventory migration or full library coverage. See `zig-out/module-complex-qr-20261013/sdk-installation-acceptance-receipt.json` for the current boundary. The earlier benchmark remains tied to its recorded SDK.

### QR Main acceptance (2026-10-13)

The optional QR frontend is now published in Main. A fresh Main `-Dlapacke-qr=true` unified SDK exposes 279 symbols (72 LAPACK, 48 FFTW) and one default Context. Main executed 43 QR groups in Debug/ReleaseSafe/ReleaseFast, all 441 LAPACK-root bodies in ReleaseSafe, and 136 independent OpenBLAS configurations. Original/installed header C/C++ static/shared QR clients passed at installed and relocated prefixes; existing 184 numerical exports, C++ lifecycle, Hermitian, FFTW and operator consumers were separately rerun against the fresh Main SDK.

Main HPP MPI 1/2/4 passed 13/16/17 groups. Main C8 MPI4 cold SCF passed the frozen energy/eigenvalue reference in 11 iterations; no reference density comparison or actual QR application call was claimed. Build inventory and test structure checks passed with migration windows closed; 72 native bindings and 222 foreign rows remain. `zig-out/module-complex-qr-20261013/MAIN-acceptance-receipt.json` records this phase separately from candidate evidence. Complete library/profile coverage remains pending, and the previous benchmark still identifies its earlier SDK.

### Explicit Q installed SDK candidate (2026-10-13)

The optional `-Dlapacke-qr=true` profile now also generates LP64 SORGQR/DORGQR/CUNGQR/ZUNGQR high/work entries. Explicit Q expansion supports all four scalar families through the same Core, operation and scratch ownership. The compiler-time generator provides independent 8-entry Q generation, complete 24-entry QR and combined 80-entry LAPACK profiles. Original MKL header declarations match parameter order, pointer depth and tau constness under LP64 aliases. The default/false selection retains the earlier 56-entry LAPACK profile.

The installed candidate unified SDK has 287 exports: 80 LAPACK, 48 FFTW, 136 checked/Sparse, 21 operator and two diagnostics. Both optional and default shared SDKs contain one actual default Context. New Q generation and previous QR C/C++ clients passed original/installed headers, static/shared linking and installed/relocated prefixes. Existing 184 numerical exports, lifetime, Hermitian, FFTW and operator consumers passed separate fresh installed-SDK regression. Malformed 80-symbol lists with missing, duplicate or foreign symbols were rejected before linking outputs.

Candidate numerical evidence includes 51 QR groups in Debug/ReleaseSafe/ReleaseFast, 449 executed LAPACK-root bodies in ReleaseSafe and 44 independent OpenBLAS Q generation configurations (28 Core, 16 adapter). Canonical enumeration records 449 names in all three modes; that is separate from executing numerical bodies. The inventory retains 72 native bindings and 222 foreign rows pending. HPP MPI 1/2/4 passed 13/16/17 groups; C8 MPI4 cold SCF converged in 11 iterations and passed the compact energy/eigenvalue reference. No reference density comparison or actual new QR call dependency in those ABPLaS fixtures is claimed.

This checkpoint is candidate installed SDK/application evidence. Main publication and fresh Main acceptance remain pending. See `zig-out/module-q-generation-20261013/candidate-acceptance-receipt.json` for the phase boundary. Complete LAPACK/Sparse and full vendor interface coverage remain pending; the earlier benchmark retains its recorded SDK identity.

### Explicit Q Main acceptance (2026-10-14)

The ORGQR/UNGQR expansion is now accepted in Main. Fresh Main SDKs verified both optional selections: 287 exports / 80 LAPACK entries with QR enabled, and 263 exports / 56 LAPACK entries with QR disabled. Each shared SDK has one actual default Context. Main executed all 51 QR groups in Debug, ReleaseSafe and ReleaseFast, all 449 LAPACK-root bodies in ReleaseSafe, and 44 independent OpenBLAS explicit-Q configurations. New Q generation and previous QR original/installed-header C/C++ static/shared clients passed at installed and relocated prefixes. Existing numerical, lifetime, Hermitian, FFTW and operator consumers were independently rerun against the fresh Main SDK.

Main HPP MPI 1/2/4 passed 13/16/17 groups with each rank's SDK path independently recorded. Main C8 MPI4 cold SCF passed the frozen energy/eigenvalue reference after 11 iterations. No reference density comparison or actual new QR call in these fixtures is claimed. Main build/test inventory checks passed with migration windows closed; 72 native bindings and 222 foreign rows remain. `zig-out/module-q-generation-20261013/MAIN-acceptance-receipt.json` records the phase separately from candidate evidence. Broader LAPACK/Sparse and complete external profile coverage remain pending; the persistent development goal stays active.

### Optional Sparse triangular solves and inverse operator Main acceptance (2026-10-14)

`-Dsparse-solves=true` generates the finite LP64 `sparse-csr-csc-mv-mm-trsv-trsm-ordinary-lp64-v1` frontend. It adds the eight S/D/C/Z TRSV/TRSM functions to the existing 25-function CSR/CSC create/export/MV/MM profile. With QR, FFTW and the operator frontend also enabled, the unified installed SDK exports 320 functions (80 LAPACK, 48 FFTW, 136 checked/Sparse, 33 standard Sparse, 21 operator and two diagnostics) and contains one actual default Context. Disabling only Sparse solves retains the exact earlier 287-export unified surface and 25-function standalone Sparse profile. Both installation entry points share the same selected header/manifest installation Steps.

The numerical implementation uses the existing Core, independent offset/ordinal views, operation admission, captured Context scratch and handle family. All four scalar families support checked triangular N/T/H solves, unit/nonunit diagonal, stored-order duplicate diagonal aggregation and mixed i32/u32/i64/u64 offsets/ordinals. The standard frontend uses an explicitly reviewed own IEEE contract. Its triangular and diagonal descriptors are finite scope; TRSM currently accepts the reviewed base-zero/row-major and base-one/column-major combinations. Other descriptor/configuration domains return not_supported. Original MKL declarations match argument order, types and constness; full vendor error priority, nonfinite trajectories, rounding and exception flags are not equated. The borrowed CSR triangular inverse backend uses the existing unified operator and caller workspace without another Context, allocator, thread pool or handle manager.

Fresh Main executed all 203 Sparse-root test bodies in Debug, ReleaseSafe and ReleaseFast, independently verified the same 203 canonical names in each mode, and passed 48 checked plus 48 finite IEEE OpenBLAS TRTRS configurations. Thirty-two original/installed-header C/C++ static/shared solve clients passed at original and relocated prefixes. Eight lifecycle clients each performed 512 create/solve/destroy cycles with in-place panels/vectors, sparse-storage alias rejection, empty RHS, own IEEE behavior and nested LAPACKE diagnostic callbacks. Sixteen additional consumers combined all installed LAPACK, Sparse, FFTW, checked and operator headers in both LAPACK/Sparse include orders.

Existing numerical, lifetime, QR/Q, Hermitian, FFTW and operator clients were rerun against the fresh Main SDK. HPP MPI1/2/4 passed 13/16/17 groups and Hermitian MPI1/2/4 passed 1/2/3, with each rank's actual Main SDK path recorded. Native/split C8 MPI4 cold SCF passed the frozen compact energy/eigenvalue reference after eleven iterations; no reference density comparison or actual new solve/QR call in those fixtures is claimed. Main build/test inventory checks passed with all migration windows closed; 72 native bindings and 222 foreign rows remain. Precise older FFT/QR source fixtures were reconciled while retaining owner, resource-split and reverse-import rejection checks.

`zig-out/module-complex-sparse-triangular-20261014/MAIN-acceptance-receipt.json` records this phase separately from candidate evidence. Complete LAPACK/Sparse, remaining descriptors/configurations and full external profiles remain pending. The earlier benchmark remains tied to its recorded ReleaseFast SDK; the new solve/frontend expansion was not benchmarked, and no performance optimization was performed. The continuous development goal remains active.


### Optional ILP64 Sparse solve profile Main acceptance (2026-10-14)

The finite `mkl2026.ilp64` Sparse profile now includes ordinary CSR/CSC TRSV/TRSM for all four scalar families. `zig build install-sparse-ilp64 -Dsparse-solves=true -Dcpu=baseline -Ddispatch=specialized -Doptimize=safe --prefix <sdk>` installs 33 API symbols; disabling `sparse-solves` retains the existing 25-symbol package. The same option selects the existing LP64 33-symbol package and its unified LP64 exports. `pkg-config --cflags --libs zynum_sparse_ilp64` supplies `MKL_ILP64`; the installed header rejects mismatched integer width. LP64 and ILP64 use the same public symbol names, so select one integer interface when linking; handles retain their original index width.

This is an external representation and packaging extension over the original Sparse binding, Core, Context, scratch leases, and borrowed handles. No numerical source changed. The generator still requires an exact finite plan/source acceptance and keeps full vendor readiness false. The LP64 public declarations and export list remain byte-identical; the LP64 unified package retains 320 exports. Each tested library has one actual default Context definition.

Fresh Main acceptance covers 16 original/installed-header C/C++ static/shared ILP64 clients, including relocated SDKs, and eight LP64 clients. Each ILP64 client checks 48 prior diagonal/status configurations and 48 nontrivial triangular configurations, including complex T/H differences; the full client matrix executes 2,304 handle create/destroy cycles, 1,536 in-place solves, and 2,304 legal empty-panel calls with positive integer arguments above INT32_MAX. These empty extents verify argument width without allocating giant matrices. Both selected/default integer packages, exact exports, header width guards, pkg-config flags, and original header signatures are verified.

Fresh Main also passes 203 actual Sparse test bodies and 203 canonical enumerated names in each of Debug/Safe/Fast; seven finite generation mutation cases and two exact resource-owner fixture cases pass. Build and test inventories are current-only, with the original 72 native bindings and 222 foreign matrix rows still pending. HPP MPI1/2/4 passes 13/16/17 cases and Hermitian MPI1/2/4 passes 1/2/3, with actual new-library loads checked at every rank. C8 MPI4 cold SCF converges in 11 iterations at -47.57100950477337 Ha; compact energy/eigenvalue reference comparison passes. Reference density was not compared, and these ABPLaS fixtures do not establish calls to the new ILP64 solve APIs.

Evidence is retained in `zig-out/module-sparse-solve-ilp64-20261014/MAIN-acceptance-receipt.json`. Large nonempty ILP64 workloads, broader descriptors, vendor nonfinite/error-priority/rounding equivalence, and full LAPACK/Sparse coverage remain pending. The existing benchmark report in `zig-out/module-comparison-20261013/report.md` remains tied to its recorded older ReleaseFast SDK; this phase does not relabel those timings as measurements of the new SDK.


### Borrowed complex SVD and rank least-squares Core Main acceptance (2026-10-14)

Main now provides allocation-free C32/C64 rectangular thin SVD, complex minimum-norm reconstruction and rank-revealing least squares with optional tall full-rank residual coordinates. They borrow existing native matrix descriptors and caller scratch under the original Core, Context, admission, threading and resource lifecycle. Public Zig SVD supports all four scalar families and adds `svdRectangularWorkspaceLength`, `svdRectangularScratchLength` and `svdDecomposeRectangular`. Complex factorization uses U*S*V^H; complex workspaces use existing widened real/imaginary pairs. Real workspace types and the original real Core implementation remain unchanged. All prospective public outputs commit only after successful validation/computation; padding and unused suffixes are preserved.

The complex SVD can prove representable exact input relations with native coefficients and error-free widened product expansions before constructing a nullspace. This is a certificate path, not a relative rank cutoff. Tests preserve genuine independent directions separated by the native smallest subnormal, including phase and rational multi-column cases. Unproved relations follow the original widened one-sided Jacobi iteration. No arbitrary-input exact-rank completeness or relative singular-value accuracy is claimed. Minimum-norm/rank reconstruction keeps internal singular values widened, uses conjugate projections and retains the Core's strict `s > rcond*s[0]` rule.

Fresh Main executes 493 LAPACK-root bodies and verifies the same 493 compiler-enumerated names in each of Debug, ReleaseSafe and ReleaseFast: all earlier 449 names plus 44 new owned groups. New tests cover mixed byte-span aliases, overflow, finite preflight, empty/native-width dimensions, residuals, failed-output preservation, single admission, and complete-before-close behavior without child admissions or resource pins. Three exact public-surface contracts and two exact resource-owner fixtures pass. The configured Core/resource graph and build projection are unchanged; current-only inventory checks pass with 72 native bindings and 222 foreign rows still pending.

Independent installed OpenBLAS comparisons cover 192 configurations against the exact published numerical-source hashes. A test-only GELSD policy projection passes all 192. The unprojected strict Core passes 172 and retains 20 direct-comparison failures at `rcond=1`: its rank-zero rule differs from GELSD's epsilon fallback. Those red results and the original numerical failure evidence are preserved, not waived or relabeled. The test projection normalizes the external cutoff policy without changing Core rules. This is numerical test-bridge evidence; new C/Z GESVD or GELSD LAPACKE high/work/query frontends, installed ABI and vendor behavior remain pending.

The fresh selected SDK retains the existing 320-export shared unified surface, 80 finite LP64 LAPACK entries, 48 ILP64 LAPACK entries and 33 Sparse entries per integer interface. LAPACKE's separately exported error handler and static implementation globals are recorded separately from operation counts. Ten installed static/shared artifacts each contain one actual default Context. Twenty-four existing original/installed-header C/C++ static/shared QR, Q generation and LP64 Sparse clients pass, plus sixteen ILP64 clients including relocated installations. The nineteen finite generation plans are refreshed only for source provenance; all existing finite header and export source bytes remain unchanged. BLAS, FFT, Sparse and existing interop numerical sources are unchanged.

HPP MPI1/2/4 passes 13/16/17 groups; Hermitian MPI1/2/4 passes 1/2/3, with the actual fresh SDK path recorded at every rank. C8 MPI4 cold SCF converges in eleven iterations at -47.57100950477337 Ha and passes its compact energy/eigenvalue reference. Reference density was not compared, and these fixtures do not establish application calls to the new complex algorithms. An initial disk-space interruption is retained; rebuilding-cache cleanup and retries complete the affected gates. `zig-out/module-complex-svd-20261014/MAIN-acceptance-receipt.json` records the accepted boundary. The previous benchmark in `zig-out/module-comparison-20261013/report.md` keeps its recorded older ReleaseFast SDK identity. Broader coverage and full external compatibility remain pending; the continuous development goal stays active.


### Optional LP64 complex GELSD Main acceptance (2026-10-14)

`-Dlapacke-complex-gelsd=true` selects the generated C32/C64 `LAPACKE_cgelsd`, `LAPACKE_zgelsd` and their `_work` frontends. With `-Dlapacke-qr=true`, the selected finite Apple arm64 LP64 LAPACKE profile has 84 operation entries. The fresh unified SDK with Sparse solves, FFTW and the operator frontend enabled exports 324 functions; each of its ten installed static/shared artifacts has one actual default Context. ILP64 LAPACKE retains 48 operation entries and each selected Sparse integer profile retains 33. LAPACKE error handlers and static implementation globals are counted separately. To install this accepted selection, use `zig build install-abplas-lp64 install-lapacke-lp64 -Dlapacke-complex-gelsd=true -Dlapacke-qr=true -Dsparse-solves=true -Dfft-interface=fftw3 -Doperator-interface=zynum_v1 -Dcpu=baseline -Ddispatch=specialized -Doptimize=safe --prefix <sdk>`. Its installed LP64 header is `include/zynum/lapacke/lp64/lapacke.h`; `pkg-config --cflags --libs zynum_lapacke_lp64` supplies the standalone package flags.

The optional compile-time generator owns external declarations and forwarders. The numerical implementation, matrix descriptors, memory, operation admission, thread, scratch and handle ownership remain Zynum-defined. The new adapter borrows the existing complex rank least-squares Core and the original Context, including minimum-workspace fallback leases. BLAS, FFT, Sparse and numerical Core leaves are unchanged. Original MKL LP64 declarations have been checked for exact argument order, complex representation, real singular values/cutoff/RWORK, pointer depth and constness. High/work row and column layouts, query without operand access, empty extents, zero matrices, rank-deficient tall/wide solves, mixed-type alias rejection, output preservation, allocation failure and complete-before-close behavior have separate acceptance evidence.

Workspace queries return this backend's recommendations through WORK[0], RWORK[0] and IWORK[0]. Selected Reference-LAPACK minimum WORK is accepted through the original Context fallback; the ILAENV6 boundary uses the reference single-precision conversion rather than exact integer arithmetic. The generated GELSD frontend normalizes nonpositive or at-least-one cutoff values using the selected real epsilon policy, while the native Core retains its strict cutoff rule. Positive INFO1 is Zynum's bounded numerical failure token, not a claim to reproduce a vendor unfinished bidiagonal-edge count. Full vendor nonfinite payloads, error priority, floating-point flags and rounding trajectories are not equated.

Fresh Main executes all 527 LAPACK-root bodies in Debug, ReleaseSafe and ReleaseFast. Their actual runtime names match the independently compiler-enumerated candidate protocol in each mode: all previous 493 names plus 34 owned adapter groups. The original capture is preserved separately from subsequent source-only Python fixture expectation repairs. Three exact public-surface contracts and three focused ownership/classification fixtures pass. Build and test inventories are current-only, with all NEXT slots closed, 72 native bindings and 222 foreign rows pending. Thirty-two original/installed-header C/C++ static/shared standalone/unified complex GELSD clients pass 1,120 processes against the fresh Main SDK; 24 existing QR/Q/LP64 Sparse clients and 16 ILP64 Sparse clients including relocated installations also pass.

HPP MPI1/2/4 passes 13/16/17 groups and Hermitian MPI1/2/4 passes 1/2/3, with each rank's actual new SDK load recorded. C8 MPI4 cold SCF passes the compact energy/eigenvalue reference in eleven iterations at -47.57100950477337 Ha. Reference density is not compared, and those existing application fixtures do not establish new complex GELSD calls. `zig-out/module-complex-gelsd-20261014/MAIN-acceptance-receipt.json` records this accepted phase. The separate four-entry and combined60-entry generation sets retain source/generation evidence; their runtime installation was not newly tested in this phase. Broader LAPACK/Sparse and complete external profiles remain pending. The benchmark in `zig-out/module-comparison-20261013/report.md` retains its recorded earlier ReleaseFast SDK identity; no performance optimization or retiming of this new interface was performed.

### Native operator numerical boundary acceptance (2026-10-14)

Core Lanczos now remeasures the scaled initial vector when its first norm is subnormal, using the same local or borrowed global metric and the original admitted operation and two scratch leases. This repairs incorrect Ritz values caused by the coarse spacing of a rounded subnormal norm; ordinary starts keep their callback count. ABPLaS now agrees finite real and imaginary Chebyshev coefficient bits before native admission, computes its complex norm through a common maximum scale, and retains the actual local failure rank when a distributed inner product fails. Pure global overflow is rejected by all participants. Its existing transport workspaces, borrowed communicators and completion/drain gate retain ownership.

All 530 canonical numerical bodies pass in Debug, ReleaseSafe and ReleaseFast in the immutable candidate tree. All 349 numerical/test inputs remain byte-identical after Main promotion; the subsequent two Python fixture repairs have separate evidence. These bodies were not rerun under the Main directory. The original 527 names remain present, with three new normalization, failure/close and empty-shard groups. Independent compiler enumeration matches 530 names in each mode. Four related build fixtures and one positive inventory fixture pass; direct source facts and strict CURRENT-only checks also pass. The broad Python direct-command tooling method timed out and was subsequently interrupted; its full execution remains pending and is not counted as passed. All prior assertions were retained.

A fresh Main selected SDK passes 232 installed C++ ABI cases across static and shared linking: four scalar families, subnormal signs/complex phases, default and supplied metrics, second-metric zero/NaN/Inf/negative/callback failure, complete output-byte preservation and retry. Its headers and shared exports remain unchanged: 324 unified functions, 84 LP64 LAPACKE, 48 ILP64 LAPACKE and 33 Sparse functions per selected integer interface. Ten installed artifacts each contain one actual default Context. The 22 generation plans retain the original 66 header/export files; changes are source provenance only. Complete external profiles remain pending.

The actual integrated ABPLaS CPP and test objects were newly compiled and linked with guarded accepted unchanged application objects and dependencies. Native operator MPI 1/2/4 passes 20/20/15 groups, HPP passes 13/16/17 and Hermitian passes 1/2/3; every rank's fresh Main SDK load is recorded. The newly linked physical executable passes a fresh C8 cold MPI4 regression in 11 iterations at -47.571009504773365 Ha against the compact energy/eigenvalue reference. Reference density was not compared. This is an object-level application rebuild, not a fresh complete application build. Arbitrary distributed dot cancellation accuracy is unchanged. `zig-out/module-operator-boundaries-20261014/MAIN-acceptance-receipt.json` records the accepted boundary and its remaining scope. The earlier benchmark retains its recorded SDK and timings; the continuous development objective remains unfinished.

### Optional four-scalar full GESVD Main acceptance (2026-10-14)

`-Dlapacke-svd=true` enables LP64 `LAPACKE_sgesvd`, `dgesvd`, `cgesvd`, `zgesvd` and their `_work` entries. The optional generator has one standalone eight-entry set and four combinations with the existing interfaces, containing 64/68/88/92 entries. Existing choices and their header/export bytes are preserved. With QR, complex GELSD, Sparse solves, FFTW and the operator interface selected, the fresh Main SDK contains 92 LP64 LAPACKE entries and 332 unified exports. ILP64 LAPACKE retains 48 entries; Sparse retains 33 entries per selected integer width. Each of the ten installed static/shared libraries has one actual default Context.

```sh
zig build install-abplas-lp64 install-lapacke-lp64 \
  -Dlapacke-svd=true -Dlapacke-qr=true -Dlapacke-complex-gelsd=true \
  -Dsparse-solves=true -Dfft-interface=fftw3 -Doperator-interface=zynum_v1 \
  -Dcpu=baseline -Ddispatch=specialized -Doptimize=safe --prefix <sdk>
```

All four scalar families support legal A/S/O/N job pairs, tall/wide matrices, row/column layouts, full and thin orthogonal/unitary vectors, and V-transpose/conjugate-transpose output. The Core supplies rectangular SVD and allocation-free basis completion. Generated representation adapters borrow the original Context, one admitted operation, captured error services and original Context scratch; they add no allocator, thread pool or handle manager. Query uses only WORK[0] and ignores complex RWORK and nonreferenced numerical buffers. The standard minimum-work contract and the implementation's managed scratch are separate. Failed numerical completion preserves A/S/U/VT; successful high-level SUPERB has a finite zero prefix with its suffix preserved.

The original MKL 2026 LP64 declarations, parameter order, real SUPERB/RWORK types, complex representation and C/C++ signatures are verified. Legal stride validation follows the documented contract and Reference LAPACK v3.12.1; newer OpenBLAS wrapper behavior for illegal unused strides differs and is recorded separately. Positive INFO=1 is the implementation's bounded Jacobi numerical failure token. It does not claim the vendor's unfinished bidiagonal count or SUPERB failure payload. MKL runtime, complete vendor semantics, ILP64 GESVD and complete LAPACK coverage remain unverified.

The immutable numerical candidate executed all 545 actual bodies in Debug/Safe/Fast, retaining the original 530 names. Its 348 native Src/Test input paths are byte-identical after Main promotion; these bodies were not rerun from the Main directory. Independent compiler enumeration confirms 545 names per mode, with 72 native observation bindings, 222 foreign rows pending, and all NEXT windows closed. Focused coverage includes 3,840 analytical SVD configurations per mode, 432 allocation-fault configurations, captured defaults/configuration replacement during close, raw alignment/address extent/capacity checks and mixed-byte SUPERB aliases. Twelve related build/generation methods and nine checked Sparse profile methods pass. The broader Python direct-command method remains incomplete and is not counted as a full tooling-suite pass.

Fresh actual Main installation passed 8,064 external calls across installed/original-MKL headers, standalone/unified products, and static/shared linking: 1,920 numerical configurations with query/minimum/one-short/high calls, plus 384 boundary calls. The four new combined symbol profiles linked successfully; 24 forged/missing/extra selections were rejected before output creation. Native operator MPI1/2/4 passes 20/20/15 groups, HPP passes 13/16/17 and Hermitian passes 1/2/3, with all 21 rank loads verified against the new Main library. Two application executables were relinked using guarded candidate CPP objects and original accepted dependencies; this Main check performed no CPP recompilation or complete application rebuild. Fresh C8 cold MPI4 converged in 11 iterations at -47.571009504773365 Ha and passed compact energy/eigenvalue references. Reference density comparison and ABPLaS adoption of new GESVD calls are still unverified.

`zig-out/module-full-svd-20261014/MAIN-acceptance-receipt.json` records this phase. The earlier 38-item LAPACK/Sparse/FFT benchmark retains its original ReleaseFast SDK, ABPLaS primary plus general-scale workload, and requested 1/4-thread configurations. This expansion was not retimed or optimized. Cross-library process singleton behavior and the broader continuous objective remain incomplete.

### Full Hermitian Lanczos coefficient chain (2026-10-14)

The optional `-Doperator-interface=zynum_v1` frontend adds `zynum_operator_lanczos_chain_f32`, `_f64`, `_complex_f32` and `_complex_f64`. Native implementations remain in `Core.internal.numerics.LanczosChain`; the generator emits only external declarations and forwarding functions. All four scalar families use the existing Operator, borrowed global reductions, the original Operation and two captured ContextScratch leases. Existing Lanczos Ritz behavior is unchanged.

The chain performs at most `max_order` actions. It stops at an exact zero residual or `residual < breakdown_tolerance`, with a strict comparison and no implicit global-dimension cap. It computes no Ritz values and performs no reorthogonalization. Successful calls publish `alpha[0..order]`, `beta[0..order-1]` and the scalar summary; failures preserve all declared outputs. Initial data, coefficient spans, mixed byte aliases, finite options and workspace extents are checked before callbacks. Subnormal normalization remeasures the scaled vector through the same metric; subnormal coefficients retain their scalar rounding limitations.

The immutable numerical candidate passed all 559 actual bodies in Debug, ReleaseSafe and ReleaseFast, including 11 chain groups and three raw-boundary groups. Four-scalar analytic phased tridiagonal coefficients and independent finite-eta spectral measures, strict thresholds, callback/reduction faults, empty shards, all mixed aliases, allocation failures and close/retry lifetimes are covered. Current installed C/C++ static/shared clients passed 64 chain calls, a relocated package passed 16 calls, and old LAPACK/Sparse/FFT external calls passed. Every one of the ten installed artifacts has one default Context; the combined shared package has 336 public symbols, including 25 operator entries.

ABPLaS source compilation and chain integration are deferred from this publication at the user's request. The referenced remote synchronization and conflict cleanup have finished, while new-engine workbench/SDK adaptation remains active. The earlier application candidate predates that synchronization and must be rebased against the updated consumer and its actual dependencies before application acceptance. No new ABPLaS MPI, Haydock production, cold C8 or density acceptance is claimed for this publication. Deep-chain/small-eta accuracy, reorthogonalization, process-wide singleton behavior across separately loaded libraries and complete vendor profiles remain pending.

The old ignored `zig-out` evidence tree disappeared during an interrupted turn. This phase uses freshly reconstructed numerical and installed-package evidence stored outside that tree; earlier document references record historical results and are not current acceptance receipts.

### Optional unified LP64 BLAS installation (2026-10-14)

The optional `-Dblas-interface=lp64` selection adds the existing 150 CBLAS and 161 classic Fortran BLAS declarations to the unified artifact. This closes an actual packaging gap: the former unified SDK did not export the CBLAS GEMM/GEMV functions used by the updated ABPLaS source. All adapters import the original `blas_abi_mod` and the same actual Core. The new root checks `OwnerIdentity`; it creates no allocator, execution state, thread pool, scratch cache or handle registry. Numerical implementations are unchanged.

With SVD, QR, complex GELSD, Sparse solves, FFTW and operator interfaces selected, the combined shared library has 646 public symbols. The one shared `xerbla_` name resolves to the existing strong LP64 BLAS implementation over the weak default LAPACK definition. The separate weak `LAPACKE_xerbla` remains overridable by the client. Each of ten installed static/shared artifacts contains one actual default Context. The four generated BLAS header/module/manifest files are byte-identical after fresh generation. The disabled `none` profile retains all previous installed headers, pkg-config bytes and its 336-symbol combined surface.

```sh
zig build install-abplas-lp64 \
  -Dblas-interface=lp64 -Dfft-interface=fftw3 -Doperator-interface=zynum_v1 \
  -Dlapacke-svd=true -Dlapacke-qr=true -Dlapacke-complex-gelsd=true \
  -Dsparse-solves=true -Dcpu=baseline -Ddispatch=specialized \
  -Doptimize=safe --prefix <sdk>
```

`pkg-config --cflags --libs zynum_abplas_lp64` now supplies the installed CBLAS and LP64 LAPACKE include paths and selects one numerical library. `test-unified-blas` is a reproducible installed-client gate requiring `-Dblas-interface=lp64 -Dfft-interface=fftw3`; it accepts the same remaining selection flags and prefix as installation. The gate checks C/C++ static/shared clients with one and four configured threads, without measuring performance.

The immutable candidate executed all 44 existing CBLAS and 101 existing Fortran ABI bodies in Debug, ReleaseSafe and ReleaseFast, for 435 actual bodies. Installed clients check real row/column GEMM and transposed GEMV with padding, complex conjugate-transpose Gram matrices feeding LAPACKE Cholesky, classic Fortran GEMM, all four scalar families, invalid argument output preservation, and a user LAPACKE error-handler override. Large BLAS calls and shutdown/reset are exercised while an FFT DCT-I plan remains alive; FFT execution and LAPACK calls continue afterward. C/C++ static/shared clients pass in the original and relocated installations. Existing installed LAPACK/Sparse/FFT and all four chain calls are revalidated. Eighteen forged declaration cases and a standalone-owner misuse are rejected before creating an output directory. Full vendor semantics and process-wide singleton behavior across separately loaded libraries remain unverified.

ABPLaS application compilation and integration are still deferred. Its updated consumer uses representation-independent `Hmlt::Apply` over exact-basis vectors and borrowed communicators, with both CSR and split-projector backends. The old CSR-only application candidate cannot establish this integration. A current two-rank standalone MPI test reproduces an existing Core boundary gap: all callback products are finite, but local Chebyshev recurrence overflow makes one rank return before its peer enters a third collective callback. Extra supervisor collectives are used only to drain that private test. Local-phase outcome agreement, distributed failure/lifetime validation, and actual application numerical regression must be completed before claiming MPI integration. This publication changes neither the algorithm's local failure behavior nor the application source.

### Optional global operator phase agreement candidate (2026-10-15)

`-Doperator-interface=zynum_v1 -Doperator-phase-agreement=true` adds sixteen generated entries: four scalar families of `zynum_operator_chebyshev_step_agreed_*`, `chebyshev_agreed_*`, `lanczos_agreed_*` and `lanczos_chain_agreed_*`. The external generator emits declarations and forwarding functions. Core defines the synchronization points and retains the existing numerical algorithms, Context, Operation, allocator, threads and scratch ownership. A fully selected LP64 BLAS/LAPACKE/Sparse/FFTW installation has 662 shared exports and 41 operator entries; every installed static/shared artifact retains one actual default Context.

The caller supplies synchronous global agreement over its borrowed communicator. First agree any local failure and its status/info; only after all participants succeed compare the complete stage/index/control point. Common controls include algorithm/scalar kind, exact real/imaginary coefficient bits, reduction presence, dimensions, budgets, tolerances, reductions and stopping decisions. Admission, scratch acquisition, callback completion, local arithmetic and output publication are agreed before a peer can proceed into another callback. High-level outputs are committed only after successful global agreement. A remote failure unwinds the original admitted operation and any successful scratch leases.

Every participant must enter the same agreed algorithm with a valid agreement callback and communicator. Tokens are borrowed only during a callback; admission failure may supply a null token. Apply, reduction and agreement callbacks must finish their own communication before returning. The outer protocol cannot recover an abandoned callback collective or an invalid bootstrap. Status 21 reports an agreement callback failure or invalid reply; status 22 reports mismatched common controls. Separately loaded numerical libraries retain separate default owners.

The option defaults to false. Both disabled LP64 BLAS and disabled BLAS installations preserve their previous 646/336-symbol surfaces, every installed C/Fortran header and pkg-config byte, and C/C++ static/shared legacy consumers. New phase declarations do not alter the old operator descriptor layouts or entry signatures. The old entries retain their local execution behavior; distributed consumers select the agreed entries explicitly. An enabled phase option with `operator-interface=none` is rejected during build configuration, before installing any files.

The immutable integrated numerical candidate passed 564 actual bodies in Debug, ReleaseSafe and ReleaseFast, retaining all 559 previous names and adding five phase/failure/lifetime groups. Its installed public C MPI client passed 296 actual parameter, callback and numerical scenarios across two ranks, four scalar families and all four algorithms. Analytic diagonal polynomial/Ritz/chain results, empty shards, finite control differences, callback and nonfinite failures, subnormal normalization, full output spans and commit rejection are checked. Another 48 cases test only the external agreement callback's point contract; they are counted separately. Each shared client rank loads the specified SDK image, and no extra collectives drain an unmatched numerical callback. These are standalone Zynum candidate results; fresh Main publication and application integration require their separate acceptance.

ABPLaS compilation remains deferred until the referenced source-update work is ready and this interface is integrated against the updated consumer. The previous CSR-only application candidate and historical ignored-tree receipts do not establish current integration. The requested local LAPACK/Sparse/FFT competitor benchmark remains a separate pending acceptance step, using ABPLaS workloads plus general scale scans and one-thread/host-parallel configurations.



### Affine operator BiCGStab candidate (2026-10-16)

The optional operator generator adds regular and agreed BiCGStab entries for real and complex single/double precision. Native `BiCGStab` solves `(action_scale*A + diagonal_shift*I)x=rhs`, with an optional stationary right inverse operator, common scalar reductions, borrowed admission and ContextScratch. A zero action scale ignores the A callback. The generator defines only the external representation and forwards calls to this implementation. Internal dimensions and counts remain native `usize`; checked wire dimensions and counts are signed 64-bit.

The 48-byte result distinguishes convergence, iteration limit, and rho/alpha/omega breakdown. Controlled stops publish a finite candidate and a verified true residual; callback, validation, arithmetic, reduction, allocation or agreement errors preserve output and summary. Initial/output alias is supported, while RHS/logical-output alias is rejected. Estimated convergence is checked against the true residual and may restart within the original iteration budget. Action counts include true-residual callbacks. Affine coefficients and all distributed controls are agreed before callbacks.

Private numerical evidence covers 156 actual numerical-root tests in each of Debug, ReleaseSafe and ReleaseFast, plus four-scalar wire tests in all three modes. It includes subnormal inputs, analytic nonsymmetric systems and breakdowns, false-convergence restart, empty local shards, failure rollback, allocation and close/retry behavior. Four generator option combinations compile as C11/C++20, and 24 malformed manifests are rejected before producing a library. Sixteen installed and relocated C/C++ static/dynamic consumers pass at thread limits one and four. The selected SDK has 81 operator entries and 794 combined public symbols; every one of its ten static/shared artifacts has one actual default Context. This is a per-artifact ownership observation, not a singleton claim across independently loaded libraries.

The current ABPLaS source compiles against the prior SDK and the new SDK with Zynum or ELPA dense solves. MPI4 passes 168 Core cases in each configuration and 849/853/853 Solver cases respectively. Four shifted-linear groups pass at MPI 1/2/4 and Core thread limits 1/4, with empty shards, child communicators, initial/output alias, controlled limits, divergent controls, failure preservation and retry. Four additional configurations exercise the real HmltCSR/VecRepacker mixed-index ghost path, an independent complex shifted solution oracle and failure after halo start. Each focused rank loads the selected SDK. The consumer's former MPI norm was independently shown to overflow a finite 1e200 RHS and misclassify a 1e-200 RHS as zero. A shared component scale and compensated unit-square sum fix the callback's global norm, using its original borrowed communicator; the identical solution/residual oracle passes after the fix. No new resource owner is introduced.

The original non-Gamma C8 target/guard validator passes for Zynum MPI1/MPI4 and ELPA MPI4. Target and guard residual maxima are 6.126826e-9 and 1.679776e-7 Ha. The full workspace is not all converged. The observed 24-band Zynum MPI4 values differ from the prior accepted Main values by at most 1.1657341758564144e-15 Ha. The additional MPI1/MPI4 1e-10 Ha criterion remains unmet at 1.7295886944879157e-10 Ha, with original thresholds unchanged. These bounded observations do not establish a complete Bands SDK or a physics certificate. The full Python suite was not repeated for these C++ changes; earlier 739-pass/11-skip evidence retains its original scope.

Complete FEAST/IFEAST contour quadrature, interval selection, FPM semantics and production adoption remain pending. Fresh canonical gates, source sealing and Main installation have separate receipts. Previous benchmark results retain their original SDK identity; this phase performs no performance optimization or new timing campaign.
