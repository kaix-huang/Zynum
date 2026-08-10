# Zig 0.16 `std.Io` And Threading

Zynum BLAS uses Zig 0.16 `std.Io` as the single lifecycle for optional parallel
CPU work. BLAS calls can be shorter than general asynchronous scheduling
overhead, so task composition and low-latency publication remain measured
internal policy.

## Current Policy

- Correctness and portable fallbacks do not depend on global asynchronous state.
- Coarse Level 1, Level 2, and Level 3 work uses the shared
  `std.Io.Threaded` runtime with `std.Io.Group.concurrent`.
- Narrow low-latency publication may exist only behind a measured internal gate
  and still uses `std.Io.Threaded` for helper ownership.
- Worker strategy is not selected by environment variables.
- `ZYNUM_MAXIMUM_THREADS` is the only runtime thread control: unset uses
  available execution capacity; a positive value caps it.

## Why Scheduling Cost Matters

For short numerical kernels, these costs can dominate:

- closure and task construction;
- helper wake-up and waiting;
- per-call allocation;
- cache-line contention in shared claim counters;
- private-result merge; and
- lazy worker initialization.

Persistent helpers can also affect later measurements in the same process.
Use fresh-process isolation for reportable comparisons.

## Shared Runtime Contract

`src/blas/core/execution/thread_pool.zig` owns the process-wide helper lifecycle.
Parallel callers submit planned tasks to it instead of creating operation- or
architecture-specific pools.

Every task path must:

- derive usable concurrency from CPU capacity available to the process;
- bound task count, stack use, and private workspace;
- assign disjoint output or use explicit private reductions;
- keep caller participation unless a measured plan says otherwise;
- complete unsubmitted work synchronously after partial submission;
- propagate failures without reporting partial success; and
- provide an explicit shutdown before dynamic-library unloading.

Worker stack size must satisfy every supported OS/runtime minimum and the
maximum bounded kernel frame. Validate stack changes on representative targets;
do not derive them from one environment's minimum accepted value.

## Coarse Submission

Use `std.Io.Group.concurrent` when each task contains enough arithmetic or memory
traffic to hide publication and waiting. The planner owns task shapes and
submits them to the shared runtime. Operation leaves do not manage workers.

Fixed task ownership is usually preferable to helper races for short work. A
shared claim counter adds contention and can disturb partition balance. Use a
dynamic claim design only when traces show meaningful load imbalance and
complete-call measurements justify it.

## Low-Latency Publication

A low-latency path may reuse process-lifetime helpers with generations and
bounded wake/wait behavior. It remains valid only when:

- the task set is fixed and bounded;
- each helper receives explicit work ownership;
- the caller can finish missing work;
- shutdown waits for all helpers;
- focused and broad fresh-process measurements show a repeatable gain; and
- off-gate shapes remain on the ordinary submission path.

Helper identity, processor number, and publication offset are not portable
dispatch inputs. A retained asymmetric design expresses work classes or capacity
ratios and remains correct under arbitrary placement.

## Affinity

Linux helpers may use CPUs only from the inherited affinity mask when a measured
policy explicitly enables it. Do not pin outside the assigned cpuset and do not
leave the caller competing unexpectedly with a helper.

On platforms without an exact supported affinity contract, topology and QoS are
hints only. See
[`cpu_affinity_and_heterogeneous_scheduling.md`](cpu_affinity_and_heterogeneous_scheduling.md).

## Dynamic-Library Lifetime

Persistent helpers may execute code from the loaded Zynum library. A process
using `dlopen`/`dlclose` must call `zynum_blas_shutdown` or
`zynum_blas_shutdown_` before closing the handle. Process termination is the
reliable cleanup boundary for benchmark isolation.

## Rejected Designs

- Do not add raw `std.Thread.spawn` pools around BLAS dispatch.
- Do not add worker-strategy environment variables.
- Do not instantiate a second pool in GEMM, an architecture backend, or an
  isolated object.
- Do not promote from a single best sample or a thread-capped diagnostic.
- Do not return success after only part of a fixed task set executes.
- Do not unload a library while its helpers may still run.

## Source Locations

- Thread-count policy: `src/blas/runtime.zig`.
- Shared runtime: `src/blas/core/execution/thread_pool.zig`.
- GEMM planning: `src/blas/core/matrix_matrix/planner.zig`.
- Level 2 task shaping: operation-family modules under
  `src/blas/core/matrix_vector/`.
- Fresh-process GEMM isolation: `bench/tools/run_gemm_sweep_isolated.py`.

## Validation

Run ordinary tests with the default runtime, then explicit one-thread and
low-thread diagnostics. For reportable timing:

```sh
unset ZYNUM_MAXIMUM_THREADS
python3 bench/tools/run_gemm_sweep_isolated.py \
  --gemm-sweep zig-out/bin/gemm-sweep \
  --zynum-blas zig-out/lib/libzynum_blas.so \
  --csv zig-out/perf-report/gemm_threading.csv \
  --reps 30 \
  --process-repeats 4 \
  --check
```

Record runtime-observed concurrency, task count, task/merge timing, selected
path, sample distribution, and the exact candidate/control identities. Keep a
new path experimental until default-thread broad evidence is stable.
