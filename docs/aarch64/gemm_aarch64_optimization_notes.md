# AArch64 GEMM Optimization Notes

This page records portable AArch64 rules for GEMM backends, capability dispatch,
architecture state, and performance acceptance. Machine-specific run logs and
raw artifacts belong outside the repository.

## Backends And Capabilities

Zynum may build several independent AArch64 kernel families:

- portable scalar and compiler-vectorized fallbacks;
- ASIMD fixed-width kernels;
- non-streaming SVE or SVE2 kernels where the target and host support them;
- explicitly enabled Apple AMX kernels reached through a private, documented
  ABI boundary; and
- SME or SME2 streaming kernels with explicit SM/ZA ownership.

These are distinct capabilities. SME availability does not prove that ordinary
SVE instructions can execute. A cross-build proves compilation only; emulation
can add functional coverage but cannot establish native throughput or
state-transition cost.

## Dispatch Order

Dispatch first proves hard feasibility, then applies a named tuning profile:

1. scalar type, layout, transpose/conjugate, alpha/beta, and dimensions;
2. compiled target capability and native runtime capability;
3. state requirements, including streaming vector length and ZA use;
4. alignment, tail, packing, and bounded-workspace requirements;
5. lifecycle and complete fallback availability; then
6. measured shape preference.

Architecture files contain hard constraints. Shape thresholds, preferred tiles,
and backend ranking belong in shared tuning descriptors. A failed predicate must
fall back before caller output changes.

## ASIMD

ASIMD is the baseline architecture-specific tier for contemporary AArch64
systems. Shared fixed-width skeletons should own common packed-panel traversal,
K loops, tail handling, and epilogues. Add a separate assembly body only when a
target instruction or register-allocation requirement materially changes the
loop.

Validate ASIMD-only artifacts without optional streaming or matrix-extension
features. Their disassembly and forced-path tests provide independent tier
evidence and prevent a wider build from silently supplying the tested body.

## SVE And SVE2

SVE code must be vector-length agnostic unless a descriptor states and checks an
exact vector length. Predicated tails must preserve GEMM epilogue semantics, and
planner geometry must not assume a development host's lane count.

Non-streaming SVE and streaming SME have separate runtime lifecycles. Keep SVE
routes experimental until a native SVE system supplies correctness and
fresh-process performance evidence for the exact compiled tier. Do not infer
support from SME feature bits or from successful emulation.

## SME And SME2

Every streaming entrypoint declares:

- whether it enters SM only or SM plus ZA;
- the required streaming vector length;
- any optional arithmetic feature;
- which component saves and restores state; and
- a balanced exit for normal, tail, rejection, and error paths.

Keep `smstart`/`smstop`, ZA initialization, ZA reduction, and spills inside the
measured route. A microbenchmark that starts in streaming mode cannot support an
end-to-end dispatch claim unless callers have the same state contract.

State transition overhead can dominate small or narrow shapes. Tuning profiles
therefore require boundary controls on both sides of every SME gate. Returning
from a streaming kernel must leave ordinary scalar/vector code and subsequent
BLAS calls correct.

## Apple AMX

AMX is an implementation-specific backend, not a public API promise. It is
compiled out by default and can be enabled only for an AArch64 macOS target with
`-Dapple-amx=true`. Keep its instruction interface behind a small private ABI
with explicit register, stack, alignment, and clobber contracts. Public and core
modules select only a stable kernel descriptor; they do not import low-level
opcode details.

macOS provides no reliable public runtime capability signal for this private
ISA. CPU family, ASIMD, SME, and successful compilation must not enable it.
Before opting in, the deployment owner must validate that the exact target and
runtime—including any virtualization layer—can execute these instructions. A
disabled or rejected route returns before allocation, output mutation, or AMX
state entry and continues through the ordinary fallback path.

An AMX route must state:

- supported scalar types and accumulator format;
- M/N/K tile and remainder behavior;
- packed panel layout, alignment, and padding;
- alpha/beta restrictions and epilogue ownership;
- state save/restore responsibilities; and
- total fallback on a rejected shape or unavailable workspace.

Do not assume the matrix engine wins for small K, narrow N, or vector-edge
shapes. Direct ASIMD, SME, or GEMV-like routes may avoid packing and state costs.

## Packing And Tails

Packing should be shared with the cross-platform planner wherever possible.
Architecture modules may define the exact panel layout required by a microkernel,
but allocation, lifetime, reuse, and fallback remain planner responsibilities.

- Acquire workspace before writing output.
- Zero or otherwise define padded lanes before a kernel reads them.
- Reuse packed panels across enough output tiles to amortize copies.
- Test every M/N/K remainder supported by the tile geometry.
- Keep vector-edge paths separate when general packing cannot amortize.

## Threading On Heterogeneous Systems

macOS does not expose a supported general API for pinning each task to a chosen
performance class. Topology queries are capacity hints, not affinity contracts.
Use the shared process-wide task runtime and measure actual task durations or
placement when investigating heterogeneous scheduling.

Do not encode processor numbers, assumed core classes, or a single host's helper
ordering into dispatch. A split that depends on heterogeneous capacity must
remain correct when all work lands on one class, when helpers are fewer than
requested, or when the process receives a restricted CPU allocation.

See
[`../common/cpu_affinity_and_heterogeneous_scheduling.md`](../common/cpu_affinity_and_heterogeneous_scheduling.md)
for the portable measurement policy.

## Target Matrix

Track these evidence dimensions independently for every backend and scalar type:

| Dimension | Meaning |
| --- | --- |
| Build | The exact capability tier compiles and links. |
| Forced correctness | The backend executes boundary, tail, epilogue, and state tests. |
| Native correctness | Tests run on hardware that exposes the capability. |
| Native performance | Isolated measurements cover target and control shapes. |
| Production selection | A named tuning profile selects the path by default. |

Evidence is monotonic but not interchangeable. A production gate requires all
preceding applicable dimensions; a build or emulator run cannot skip native
validation.

## Benchmark And Gate Requirements

Follow [`../common/benchmarking.md`](../common/benchmarking.md). For an AArch64
GEMM promotion, retain:

- exact source and binary identities;
- compiler version, target triple, CPU features, and optimization mode;
- runtime-observed thread capacity with `ZYNUM_MAXIMUM_THREADS` unset for the
  default gate;
- isolated, interleaved candidate/control samples;
- correctness status and maximum error for every timed row;
- representative square, rectangular, narrow, high-K, transpose, complex, and
  remainder shapes;
- selected-kernel evidence; and
- a mechanism signal such as disassembly, sampling, tracing, or phase timing.

Low-thread caps are useful diagnostics. They are not the production gate unless
the documented public route is explicitly single-threaded.

## Retention And Rejection

Retain a backend or expand a gate only when the complete call is correct, target
shapes improve beyond noise, off-gate controls remain within the declared
regression threshold, and architecture state is balanced.

Reject or narrow a route when:

- target capability is inferred rather than executed natively;
- state transition, packing, or merge cost dominates;
- an epilogue or transpose restriction is not represented in the descriptor;
- workspace failure can occur after output mutation;
- the result depends on one host identity or one benchmark sample; or
- an apparent gain cannot be tied to the intended kernel through selected-path
  evidence or inspection.

Public notes preserve these rules and current boundaries, not dated tuning
chronology.
