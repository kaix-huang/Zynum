# Level 1 Optimization Notes

This document records durable rules for BLAS Level 1 semantics, kernel ownership,
dispatch, and performance acceptance. It intentionally excludes dated run logs,
host identities, scheduler records, and raw benchmark paths. Store that material
outside the repository.

## Ownership

The implementation is divided by responsibility:

- `src/blas/core/vector/` owns BLAS semantics, stride normalization, fast-path
  eligibility, task composition, and portable fallback.
- `src/blas/kernels/shared/vector/` owns reusable contiguous and fixed-width SIMD
  bodies, kernel descriptors, coverage, and tuning policy.
- `src/blas/kernels/aarch64/vector/` and
  `src/blas/kernels/x86_64/vector/` own target-specific entrypoints and assembly.
- `src/blas/abi/` owns CBLAS and Fortran argument conversion only.

ABI wrappers must not choose instruction families or duplicate numerical loops.
Architecture modules enforce hard feasibility; tuning modules own measured
preferences; core modules own whole-operation fallback.

## Semantic Contract

Every optimized path must preserve the portable behavior for:

- positive, zero where the operation permits it, and negative increments;
- empty vectors and single-element tails;
- real and complex scalar domains, including conjugated dot products;
- BLAS index return conventions for `iamax`;
- NaN, infinity, signed-zero, and accumulation behavior allowed by the API;
- permitted aliasing, overlap, and in-place updates;
- checked-build dimension and pointer validation; and
- balanced architecture state on every return path.

Contiguous kernels are not substitutes for general-stride operations. Dispatch
must prove the required stride, alignment, non-overlap, and length predicates
before entering a leaf. A rejected predicate falls back before any output is
modified.

Reduction reassociation may change low bits. Acceptance therefore uses the
operation's documented tolerance and also checks adversarial cancellation,
non-finite inputs, odd tails, and boundary lengths. Faster unchecked rows are
not evidence.

## Reusable Kernel Patterns

### Copy and swap

Treat real and complex copy as byte movement when the public semantics and
overlap rule permit it. Select a byte kernel from the element size, not from the
source scalar type. Swap requires distinct, non-overlapping ranges unless a leaf
explicitly implements overlap-safe behavior.

### Real-lane reuse

Complex storage may reuse a real contiguous leaf only when the operation is
exactly equivalent over interleaved components. Real-alpha complex scale and
AXPY-like updates are common examples. General complex coefficients, conjugated
operations, complex absolute-value reductions, and index reductions require
their own semantics.

The operand category must remain explicit through architecture dispatch and
parallel task descriptors. A real vector and the real-component stream of a
complex vector can prefer different SIMD widths even when they share an
instruction body.

### Reductions

Use multiple independent accumulators to shorten dependency chains, then reduce
them in a fixed and testable order. Keep scalar cleanup outside the main vector
loop. Do not add a wider vector body merely because the build target exposes a
wider ISA: width is an operation-and-type tuning decision.

For `nrm2`, preserve the scaled sum-of-squares strategy or another overflow-safe
equivalent. A direct sum of squares is not an acceptable optimization.

### Tails

Predicated targets should express tails with native predicates when doing so
does not add state-transition cost. Fixed-width targets may use a smaller vector
or scalar cleanup. Tests must cover every tail length reachable from the chosen
unroll factor.

## Dispatch And Registry

`src/blas/kernels/shared/vector/catalog.zig` is the compile-time contract catalog.
Descriptors provide stable identities and state hard requirements such as:

- operation and scalar domain;
- complete-operation versus sub-operation semantics;
- stride, alignment, alias, and tail behavior;
- required target capability and architecture state;
- lifecycle (`production`, `experimental`, `rejected`, or unavailable); and
- a total whole-operation fallback.

`src/blas/kernels/shared/vector/coverage.zig` enumerates implemented, missing,
unsupported, and experimental cells. Build coverage, native correctness, and
native performance are independent and monotonic evidence dimensions. A
cross-build proves only that a target can be compiled.

`src/blas/kernels/shared/vector/tuning.zig` owns named production profiles and
measured preferences. Length thresholds, byte ranges, and preferred SIMD widths
belong there. Instruction bodies must not acquire anonymous target-specific
gates. Experimental candidates may be exposed through an explicit benchmark
build, but ordinary builds must keep their production profile unchanged.

Kernel IDs describe semantics, implementation, and capability rather than file
paths or processor product names. A shared fallback does not become a distinct
ISA kernel merely because it was compiled with a wider target.

## Architecture Boundaries

### x86_64

Baseline, AVX-family, and AVX-512-family artifacts are separately compiled
capability tiers. The widest compiled tier is not automatically the fastest.
Use same-ISA candidate/control comparisons when changing only a leaf or tuning
rule; otherwise code layout and target flags confound the result.

Large positive stride-two candidates may cross a private fixed-layout object
boundary. The isolated object must not create a second worker pool. It submits
work through the process-wide task runtime, keeps internal symbols hidden, and
falls back as a whole operation when its gate fails.

### AArch64

ASIMD, non-streaming SVE, and streaming SME families are separate capabilities.
SME availability does not prove that non-streaming SVE instructions are valid.
Enable and validate each tier independently on native hardware.

Streaming descriptors declare whether they own SM only or SM plus ZA, any
streaming-vector-length constraint, and optional arithmetic features. Every
entry must balance state on all exits. A function whose coefficient restrictions
cover only part of a public operation remains a sub-operation or experimental
body; it must not be registered as the complete operation.

Emulation and cross-compilation are useful for syntax, tail, and functional
coverage, but never establish native throughput or state-transition cost.

## Parallelism

Level 1 parallelism is useful only when saved memory or arithmetic time exceeds
task submission, wake-up, synchronization, and final-reduction cost.

- Use the shared `std.Io` task runtime described in
  [`zig_0_16_std_io_threading.md`](zig_0_16_std_io_threading.md).
- Derive usable concurrency from the runtime's assigned CPU capacity; do not
  assume the physical-machine total.
- Split contiguous ranges at element boundaries and keep write ownership
  disjoint.
- Give reductions private partials and merge after all tasks complete.
- If helper submission is partial or unavailable, complete the remaining ranges
  synchronously; never return success with unfinished work.
- Keep single-task execution direct. Do not submit a task merely to wait for it.

Thread-count caps are diagnostics, not production gates. Default acceptance uses
the documented runtime policy with `ZYNUM_MAXIMUM_THREADS` unset.

## Acceptance Matrix

A candidate is eligible for production only after all applicable evidence exists:

1. Build the portable baseline and each changed capability tier.
2. Run API, ABI, registry, and forced-path correctness tests.
3. Exercise zero, boundary, odd-tail, negative-stride, alias, and non-finite cases.
4. Compare production and candidate in isolated fresh processes with identical
   target flags, runtime controls, and inputs.
5. Cover small, medium, parallel-threshold, and memory-bandwidth-bound sizes.
6. Inspect at least one mechanism signal: disassembly, sampling, tracing, task
   timing, or selected-kernel instrumentation.
7. Repeat apparent regressions and gains until they exceed run-to-run variance.

Retain a dispatch rule only when selected cases improve and off-gate controls
remain neutral within the declared regression threshold. Record the rule,
capability, shape or length boundary, correctness status, artifact identity,
sample policy, and rollback condition in curated evidence. Raw CSVs, binary
hash inventories, and scheduler output stay private.

## Rollback Rules

Disable or narrow a path when any of these occurs:

- the forced path fails semantics, state restoration, or fallback tests;
- a default build selects it outside its proven capability or operand category;
- an off-gate control regresses beyond the declared threshold;
- the observed gain is smaller than measurement noise;
- the implementation requires an independent worker lifecycle or unbounded
  workspace; or
- the mechanism cannot be explained by code, disassembly, or runtime evidence.

Rejected experiments should leave behind only the durable reason: for example,
submission overhead dominated, the wider vector reduced frequency, a hot
function grew enough to perturb layout, or a partial semantic body was mistaken
for a complete operation. Do not preserve a public diary of individual runs.
