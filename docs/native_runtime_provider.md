# Native RuntimeProvider SDK (development)

The explicit-provider native FP64 NN path has bounded local CPU qualification.
Candidate 2 reference Core Debug, Safe and Fast each passed 1839 tests; its
host gate failed. After an independently reviewed Python fixture-only repair,
a new source generation passed host and CPU integration with installed C/C++
consumers, with full-source before/after identity independently checked. These
are combined evidence across generations, not one complete Core rerun.
The additional explicit-provider FP64 TN path is implemented in this development
source; its qualification is pending. Other packed, SIMD, GEMV, FFT, Sparse and
legacy resource paths need further migration
and qualification. No cluster, materials/SCF, GPU or performance qualification
is established. Earlier checked-bounds evidence retains its own source identity.

## Profiles and builds

The root builder offers `reference` and `integrated_native` execution profiles.
`reference` provides compatibility services and a reference default; the default
is not an integrated RuntimeProvider. `integrated_native` requires explicit
provider binding and does not create a local pool, allocator, Runtime or
implicit fallback. The profiles share the original Core, with one main
`libzynum` artifact; no second native Core archive is required.

Provide an **absolute header-only** Heteris ABI SDK directory containing
`heteris.h` and `heteris_runtime_provider.h`. No Rust, MPI or CUDA is needed to
consume those headers or to build Zynum against this SDK.
The caller supplies and links an actual Runtime implementation separately when
it needs actual-provider execution. Use the same normalized SDK path for the
CLI and inherited environment, especially for nested build probes:

```sh
SDK=/absolute/path/to/heteris/abi/include
OUT=/absolute/path/to/external-build
export ZYNUM_HETERIS_INCLUDE="$SDK"
zig build platform -Dexecution-profile=integrated_native -Doptimize=safe \
  -Dheteris-include="$SDK" --prefix "$OUT/install" --cache-dir "$OUT/cache"
```

`platform` installs the main artifact and headers to the prefix. Standalone
compatibility provider is separately opt-in with `-Dstandalone=true`; it is not
an implicit provider for native calls. To run the original reference baseline,
select it explicitly and use separate external caches:

```sh
SDK=/absolute/path/to/heteris/abi/include
OUT=/absolute/path/to/external-reference
export ZYNUM_HETERIS_INCLUDE="$SDK"
zig build test test-lapacke-posv test-checked-cholesky-lifecycle \
  -Dexecution-profile=reference -Dheteris-include="$SDK" -Dcpu=baseline \
  -Doptimize=safe -Dtest-optimize=safe --summary all \
  --prefix "$OUT/install" --cache-dir "$OUT/cache"
```

The full project qualification workflow must additionally isolate global and
child caches and record its inputs and results. CLI-only configuration works
for ordinary root builds; nested historical Python probes need the inherited
variable. Mismatched paths fail rather than selecting an adjacent SDK.

Actual-provider test targets require a **prebuilt** Runtime archive; without
it they are not registered, and the builder does not build or discover Rust:

```sh
SDK=/absolute/path/to/heteris/abi/include
RUNTIME=/absolute/path/to/libheteris_runtime.a
OUT=/absolute/path/to/external-provider-tests
export ZYNUM_HETERIS_INCLUDE="$SDK"
zig build test-native-provider-math test-native-runtime-c \
  -Dexecution-profile=integrated_native -Doptimize=safe \
  -Dheteris-include="$SDK" -Dheteris-runtime-archive="$RUNTIME" \
  --prefix "$OUT/install" --cache-dir "$OUT/cache"
```

A build or example command is not a claim that this worktree passed a gate.

## Context and calls

`zynum_native.h` declares the versioned C API. Query storage size and alignment,
then supply stable caller-owned storage to `zynum_native_context_init_v1`.
Keep storage, provider-owned state and the implementation image alive through
close and every returning API call. The provider and policy descriptors are
borrowed for initialization; their required contents are copied and the provider
is retained. Follow the header's ABI version, struct size and reserved-field
requirements. Report storage must not overlap Context, descriptors or mutable
mathematical data. No implicit Context or Runtime is created.

Operations are admitted through the Context. Requesting close prevents new
admissions; progress waits for admitted operations and their resource pins.
Integrated close releases its retained provider token and does **not** drain or
reset the shared Runtime. If release fails, the Context retains the token in
`ZYNUM_NATIVE_CLOSE_RELEASE_FAILED`; an explicit later progress call can retry.
Keep the provider and storage alive until close succeeds. Inspect each returned
status and report: resource stage, submission and output states, close state,
raw provider/execution/callback/cleanup statuses and callback counts. Output may
be unchanged, complete, or partial/invalid; do not assume success after failure.

The native mathematical entry points include checked FP64 column-major NN and
TN GEMM using explicit
`ZYNUM_NATIVE_ALGORITHM_PORTABLE_GENERIC`. Buffers are borrowed and their backing
extents and leading dimensions are checked. Every nonempty output write goes
through Heteris, including one-task and alpha-zero or zero-k work. A provider
rejection, allocation failure or execution failure is reported; the operation
does not fall back locally or replay potentially partial writes. The numerical
seam uses the original Core implementation.

For reference-only Zig use, `ReferenceContext.initInPlace` creates a wrapper
whose internal references require a stable address: do not move it after
initialization. Its caller-owned storage and allocator state must outlive close
and returning calls. The new reference default is a reference service, not an
integrated provider. Consult the actual declarations and current inventory
checker before relying on a profile or test result.

## FP64 TN contract

The zynum_native_gemm_f64_tn_v1 symbol has the same fixed-width primitive-buffer
signature as NN. A is physically k-by-m with leading dimension at least k when
nonempty; B is k-by-n and C is m-by-n. The minimum nonempty A backing is
k + (m - 1) * lda elements. Backing may start at an adjusted, bounded pointer
subview; the caller supplies the remaining backing extent. Physical extents,
checked arithmetic, narrowing, alignment, null rules, scalar checks and
A-C/B-C and metadata aliases are validated before slices, admission or packing.
Read-read overlap is allowed under the existing rules.

The Zig matrixMultiplyF64TN takes the same MatrixProductF64 type. Its left view
retains physical row_count=k and column_count=m and must explicitly carry
operation=.transposed; the right view must carry operation=.normal. Normal or
adjoint A and transformed B are unsupported. NN continues to reject transposes.
The C finite-alpha/beta policy and existing Zig boundary policy are unchanged.

One admitted Operation spans two synchronous phases. Provider scratch holds the
m*k packed A, one typed packing descriptor and its Invocation. A single provider
packing task calls the original Core packTransposedA helper. Once that bulk
returns quiescent to the synchronous caller, the same Operation enters the
original explicit NN planner with packed A(m,k), lda=m. The planner still
allocates its own Tasks and Invocation, so those allocations may fail after
successful packing. C is unchanged until its bulk begins. No provider worker
submits or waits for a nested NN call, and no public NN call re-admits the work.
The reused packing helper has existing vector loads and an x86 tiled shape
branch; this adds no new dispatch policy or performance qualification.

Empty and scaling-only calls retain structural, alias and admission checks,
skip packing and reuse the NN empty/scaling path. Every nonempty scaling write
still uses the provider. Requesting close between phases prevents new admission
but does not revoke this admitted Operation or drain the shared Runtime.

Each phase and allocation sequence uses a fresh private report. The public
report describes C and preserves the original terminal provider fields:

| Terminal result | C output state and diagnostic |
| --- | --- |
| Packed-A or packing descriptor/control allocation failure | Unchanged; original allocation diagnostic |
| Packing rejection or quiescent execution/callback failure | Unchanged; exact packing diagnostic; no NN submission |
| Postpack NN Task/Invocation allocation failure or NN rejection | Unchanged; fresh terminal diagnostic, without inherited packing counts |
| C-writing quiescent execution/callback failure | Partial or invalid; original terminal C diagnostic; no replay |
| Success | Complete; terminal C phase diagnostic only |

Callback counts are not accumulated across phases and reserved fields stay zero.
The flat v1 report cannot always identify packing versus C phase; it does not
expose both histories. Packed A remains live until its C consumer is quiescent.
Each task/control lease remains live through its own bulk, and all call-local
leases are released before the one Operation ends. Malformed/nonquiescent
completion or scratch-release failure retains the existing fail-stop and
lifetime protection. Context provider-release failure remains separately
retryable through explicit close progress.

The component's actual-provider tests reuse a pure test-only extraction of the
original Fortran TN oracle; the integrated product imports neither that oracle
nor a reference default owner. New TN cases, both-phase failure/lifetime checks,
controlled fail-stop children, original Core gates and installed C/C++ consumers
must pass before TN is described as locally qualified. NT/TT, complex, new SIMD
policies, asynchronous tickets, MPI/cluster, materials/SCF, GPU and performance
are outside this increment.

## Native periodic Laplacian

The additive `zynum_native_laplacian_periodic_3d_f64_v1` entrypoint borrows one
contiguous FP64 orthogonal periodic grid, with axis 2 fastest and index
`(i0*n1+i1)*n2+i2`. Its three scalar coefficients are finite, strictly positive
inverse squared spacings. It evaluates the original seven-point arithmetic
shared with the legacy signed-stride adapter; it does not apply the physical
kinetic factor -1/2 or a local potential. Singleton axes retain both periodic
center reads and length-two axes count the same neighbor twice.

The Context and report must be live accessible objects, mutually disjoint and
separate from both complete declared payload spans. The new entrypoint proves
computable control and payload ranges before any report write. Invalid or
uncheckable control separation returns INVALID_ARGUMENT with the report
untouched; these numeric checks do not prove pointer lifetime or provenance.
This rule is local to the new function; released NN/TN precedence is unchanged.

After control preflight, the order is: initialize report; checked individual
axis conversions and nonempty product/byte extent; X then Y null/alignment and
capacity checks; coefficients; full-declared-span X/Y separation; original
checked Core admission. A zero axis short-circuits multiplication, not the other
validation or Context admission. Null requires zero declared capacity and empty
geometry. Unrepresentable declared spans fail control preflight, whereas a
nonempty geometry product/byte overflow returns OVERFLOW with an initialized
unchanged report. All checks remain ordinary runtime checks in ReleaseFast.

Each nonempty call holds one Operation, allocates bounded Task descriptors and
Invocation scratch from its Context provider, and submits one checked bulk even
at task limit one. Empty calls allocate, submit and write nothing and return
UNCHANGED/NOT_SUBMITTED if admission succeeds. Closing still rejects empty
calls. A managed-worker nonempty invocation follows the actual provider's
WOULD_DEADLOCK response; empty calls do not invent a submit-time rejection.

The output is direct: prevalidation/allocation/rejected-submit failures preserve
Y; quiescent execution or callback failures may leave PARTIAL_OR_INVALID Y.
Malformed/nonquiescent completion and valid scratch-release failures retain
fail-stop protection. Success means completed IEEE arithmetic, including
possible NaN/Inf or finite-input overflow; it does not promise a finite or
scientifically valid result. Capacity tails and input remain untouched.
ABPLaS physical composition and finite staged public-output commit are separate
work. No new Runtime, resource owner, allocator, fallback or Heteris ABI is added.

The appended component test declarations and C/C++ cases are source coverage,
not execution evidence. Source inventory registration, three-mode actual-provider
runs, original Core/host, installed consumers and full CPU integration are required
before qualification or release.


## Borrowed native Chebyshev series and raw moments (source implementation)

`zynum_native_spectral.h` is included by `zynum_native.h`; either include order,
including repeated inclusion, declares the same v1 API. It uses the build's
existing single C translation rooted at `zynum_native.h`. Install and ship both
headers. This source adds five functions: passive guard query/init, workspace
count query, checked series and raw moments. New declarations and test cases
are implemented source; their source-inventory enrollment, compilation, three-mode
execution and installed qualification remain pending. Earlier qualifications
above retain their original source identities.

Every mathematical call borrows an explicit live native Context, FP64 primitive
buffers, synchronous callback, caller workspace, initialized passive guard and
fresh per-call report. It constructs no resources table, owner, Runtime,
allocator, pool or fallback. Heteris remains authoritative for shared Data,
resource ownership, allocation, execution and global policy; these primitive
panels are a bounded numerical interface. Ordinary standalone does not gain a
Rust, MPI or CUDA dependency. Serial initialization, finite checks, recurrence,
wide dots and commit run on the API caller. The outer Context operation admits
lifetime; it does not claim provider bulk charging or complete budget migration
for those loops. A callback can independently invoke the same Context's original
provider-backed mathematics.

### Guard storage and supported interoperation

Query the guard's actual bytes/alignment, allocate raw stable storage through the
explicit appropriate provider, then call guard init once, exclusively before
publication. Init constructs only the internal prefix; bytes beyond that prefix
are untouched. Keep the complete declared storage alive, at the same address,
through all returning calls. Never copy, move, reinitialize or free it while it
can be used. There is no provider, Context, payload allocation or destructor in
the guard. A distinct guard does not license overlapping workspace access.

The implementation owns a Zig `std.atomic.Value(u32)` inside its private Guard.
Only Zig initializes its ordinary zero state before publication and subsequently
performs inactive-to-active strong CAS (acquire, failed CAS monotonic) and release
store. No C `_Atomic`, C++ `std::atomic`, Rust atomic, or foreign ordinary integer
object is cast to this type. Callers cannot inspect or access the live internal
prefix; they may inspect separate excess bytes. Successful acquisition is paired
with exactly one release on every recoverable return. BUSY initializes a fresh
report, performs no admission/callback and never clears another caller's flag.
Control preflight still runs before CAS, so INVALID_CONTROL precedes BUSY.

The supported initial interoperation target is x86-64 Linux CPU, with global
Zig 0.17 and the project's C11/C++20 toolchains. The storage contract is the
formal provider's aligned host raw allocation, whose backing remains live until
explicit lease release. In C11, allocate storage with no declared Guard/atomic
type; Z initializes and accesses its own object rather than accessing a C atomic.
In C++20, the fixture obtains aligned raw provider storage and provides an
unsigned-byte backing buffer, then delegates opaque prefix construction/access
to the extern-C Zig implementation. Typed FP64 arrays are explicitly placement
constructed and remain live through calls. Neither language creates a foreign
Guard object or uses foreign atomic operations. Storage reuse/destruction occurs
only after calls have returned and all guards are inactive. Guard/query sizes
and report layouts describe the current supported ABI, not foreign object
lifetime proof by themselves.

This is an explicit supported compiler/target foreign-storage convention,
consistent with the existing caller-storage Context API. The C/C++ standards do
not define Zig object lifetime or atomics, and no universal language-standard
interoperability theorem is claimed. Review must retain this construction,
alignment, exclusive initialization, single-language atomic access and storage
lifetime rationale; actual C11/C++20 and ReleaseFast tests are additional evidence,
not substitutes for it. A new target/compiler/storage mechanism requires its
own review and qualification. Numeric address checks alone cannot prove raw
storage provenance, accessible objects, successful initialization or exclusion.

### Sizes, borrowing and exact validation precedence

Let N=rows, W=columns and Q=coefficient_count/orders, with Q>=1. Inputs and outputs
are column-major FP64 with row stride one; nonempty LD must be at least the row
count. Required backing is `(columns-1)*LD+rows`, or zero if either logical peer
is zero. Empty LD is unused and does not incur an invented LD-byte check; a
one-column view likewise does not materialize an unused inter-column offset.
Series uses 4*N*W workspace elements for previous/current/next/sum. Moments uses
3*N*W+Q*W for recurrence plus staged raw moments. Public output is N*W logical
values for series or Q*W for moments, with caller-chosen padding. There is no
borrowed-GEMM i32 dimension cap on spectral calls. Child operators may separately
reject dimensions that their own physical/numerical contract does not support.

The workspace query first validates its uint64 output control, then kind and
Q>=1, then all scalar/prefix/byte constraints below. It writes once on success
and leaves the output untouched on every failure. Guard query validates both
uint64 output controls and their disjointness before either write; guard init
validates its entire declared storage, capacity and queried alignment before
initializing its prefix. Neither query/init performs Context admission.

Actual-call order is fixed:

1. Before any caller write or payload pointer/slice construction, prove full
   address intervals/alignment for Context, report, entire guard and declared
   userdata. Userdata must be NULL/0 or nonnull/positive. Controls are mutually
   disjoint. Compute complete declared X, output, coefficients (series) and
   workspace byte intervals in that order, rejecting NULL/positive backing,
   unrepresentable count/bytes/end and any control/payload overlap. Nonnull/zero
   payload is allowed here. INVALID_CONTROL preserves report and all guard bytes.
2. Initialize ABI/size and VALIDATION/UNCHANGED report placeholders; acquire guard.
   BUSY returns that report without releasing the active caller's guard.
3. Require nonnull callback, Q>=1, finite lower, finite upper and lower<upper,
   in that order. Failure is INVALID_ARGUMENT, before payload initialization.
4. Independently check N/W/Q in host index range; 4W or 3W; N*8 in host/u64 and
   <=INT64_MAX even for W0; Q*8 in host/u64, also <=INT64_MAX for moments even W0.
   Check NW, its bytes, every recurrence prefix, QW/moment bytes when applicable,
   and the required workspace sum/count/bytes. Failure is OVERFLOW. Zero peers
   do not erase these old virtual-stride restrictions.
5. Compute physical X and output extents, in that order, completing both before
   any capacity/alignment check. Nonempty LD<rows is VIEW_INVALID; extent/byte
   overflow is OVERFLOW. Then check capacities X/output/coefficients/work, and
   only then alignments in that same order. Every nonnull pointer, even one with
   zero backing, requires double alignment; short capacity or misalignment is
   VIEW_INVALID. Unused tails and padding are never scientific inputs.
6. Check full declared payload aliases: work separate from X/output/coefficients,
   and series coefficients separate from output. Failure is ALIASING, including
   unused tail overlap. X/output (even partial overlap) and readonly X/coefficient
   sharing are allowed. Moments permits X/moments overlap.
7. Scan finite series coefficients first, then logical X j/i; moments scans X j/i.
   NONFINITE_INPUT precedes admission, workspace initialization and callback.
   Advertised unused padding/tails may contain NaN and are not scanned.
8. Set private ADMISSION state and begin exactly one original Context operation.
   Closing/counter overflow produces CONTEXT_FAILURE, attempted1/succeeded0 and
   the original native admission status. A successful operation stays live
   through all callbacks, arithmetic and final commit, then ends before return.
9. Invoke the one checked Core engine shared with the legacy mapped facade. Only
   after every order succeeds, copy staged logical values to public output.
   No recoverable failure remains during this copy. Release guard before return.

All full declared payload/control objects and access rights, Context/provider,
implementation/callback images, userdata, model, H/scaling and probes remain
stable/live through return, including commit after a close request. Userdata
preflight covers only its declared control object, not its recursively reachable
V/B/D/model/workspace graph. The caller/AB adapter must preflight that complete
graph against mutable/live borrowed buffers and controls. These checks cannot
prove provenance, accessibility, immutability or that exclusion is actually held.

### Arithmetic, callbacks, reports and publication

The original interval arithmetic remains `center=(f128(lower)+upper)/2` and
`half=(f128(upper)-lower)/2`. Callback computes the unscaled original H. Original
`recurrence_arithmetic.shifted` retains all affine/subtraction terms wide until
one final f64 narrowing. Series initializes with original `linear(c0,X,0,0)`,
then for each element performs checked product, shifted state and linear
accumulation in the original j/i and swap order. Moments keeps products and
entire row-order sum in f128 and narrows once. There is no endpoint halving,
damping, averaging, spin/probe/density weight or alternate affine rounding.
Bounds are the caller's spectral-enclosure promise, not a checked Hermiticity,
physical model or H-generation proof.

For each nonempty callback, compact disjoint X/Y recurrence panels have ld=N
and backing=N*W. Empty panels pass NULL pointers/backing0, preserving N/W/ld=N,
without forming zero-length pointer offsets. Callback completes synchronously,
retains no borrowed pointers and cannot unwind across C. Any nonzero int32,
including INT32_MIN and collisions with enum values, yields CALLBACK_FAILURE
with that exact raw status. After callback zero, produced nonfinite math is
NUMERICAL_FAILURE and retains the returned raw zero as valid callback evidence.
The old resources facade retains its original raw callback error propagation,
arbitrary admitted strides, metadata snapshots, validation order and checks.

Consume reports only after the function returns; public reports are not a live
callback progress channel. Returned flags identify validity, not live handles.
The private engine records phases before work, and the facade projects the
terminal result. Report ABI/size/reserved fields are fixed; flags are 0/1:

| Returned state | Valid diagnostics |
| --- | --- |
| INVALID_CONTROL | Incoming report and guard untouched; no new report validity |
| BUSY/other validation failure | VALIDATION/UNCHANGED, no admission/order/callback; zero placeholders |
| CONTEXT_FAILURE | ADMISSION/UNCHANGED, attempted1/succeeded0, native admission_status valid |
| Initial produced failure | INITIALIZE or initial MOMENT, successful admission, order0, no callback |
| Callback nonzero | APPLY, current order and last returned raw callback/order/count valid |
| Produced failure after callback zero | RECURRENCE or MOMENT, last callback zero/count preserved |
| Success | COMPLETE, output COMPLETE iff logical output nonempty; otherwise UNCHANGED |

Every positive order updates current order before calling. Last returned callback
fields persist through subsequent arithmetic and commit and count whole operator
returns, not native child calls, tasks, provider entries or accepted work. Order
is valid only under order_valid; callback status/order only under callback_returned.
Unused status/order fields remain zero placeholders. Initial moments copy is
INITIALIZE, order0 dot is MOMENT, later shifted/accumulation work is RECURRENCE,
later dots MOMENT, final copying COMMIT, and successful completion COMPLETE.

Q1 admits and initializes but never calls. Series W0 admits, marks order0 and
runs Q-1 empty callbacks ascending; it scans all Q coefficients first. Moments W0
admits and completes without a valid order, callback or payload/workspace write.
N0/W>0 series runs Q-1 zero-row callbacks and writes no public elements; moments
runs Q-1 callbacks, stages Q*W zero moments and commits them only after success.
All empty calls retain every validation/admission constraint above. Child model
validation remains independent of generic Z emptiness.

Every recoverable failure preserves all public output bytes, including padding
and tails; workspace prefixes can change, excess workspace remains untouched.
Failure-preserving publication assumes caller exclusion, not concurrent atomic
stores. Caller may permit X/output overlap because the complete result is staged.

Same-guard reentry/concurrency returns BUSY with separate report storage. Distinct
guards/disjoint workspace may share a Context. Each child call admits separately;
the outer operation grants no token for child calls after close. Close before
outer begin fails admission. Close during a call keeps progress-close BUSY until
outer return and rejects later child admissions; spectral preserves that child's
raw callback failure and the successful outer admission. After the final
successful callback, local arithmetic/commit can finish after close, using still
live Context/provider/image and payload leases. The caller must not release them
when one child operation ends. A Runtime-worker nonempty inner bulk can return
actual WOULD_DEADLOCK; this becomes callback failure with its raw child code,
not outer admission failure. Q1/no-bulk cases remain permitted. No provider
fail-stop boundary is weakened by this recoverable interface.

Source coverage adds independent diagonal/dense degree-polynomial references,
adjacent/extreme interval and wide-dot cancellation, callback/raw/failure reports,
real-provider C/C++ controls/guard/tails, actual same-guard concurrency, reentry,
same-Context nesting, empty schedules, child-close behavior and real worker bulk
rejection versus Q1 permission. None of those added cases has run in this source
phase. Original legacy spectral tests and old native assertions remain intact;
source enrollment, independent review, focused three-mode execution, Core/host
and root integration/installed ON/OFF/relocated/header-only checks are still
required. No material/SCF, cluster/MPI/Slurm, GPU, performance or full C2-C4
qualification is implied.
