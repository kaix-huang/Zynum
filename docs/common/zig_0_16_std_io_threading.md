# Zig 0.16 `std.Io` And Threading

Zig 0.16 introduced a new `std.Io` model. Zynum BLAS uses it carefully because
BLAS tasks are often too small for generic asynchronous scheduling overhead.

## Current Policy

- Correctness code should not depend on global async state.
- GEMM and selected Level 1/2 parallelism use explicit dispatch policy.
- Coarse CPU-bound helper tasks should prefer `std.Io.Threaded` with
  `std.Io.Group.concurrent`.
- Low-latency helper paths may exist only as narrow internal dispatch rules with
  focused benchmark evidence. They should still use `std.Io.Threaded` for helper
  ownership instead of ad hoc thread spawning.
- Worker strategy is internal implementation policy and must not be selected by
  Zynum environment variables.

## Why This Matters

BLAS kernels can be extremely short. A scheduling strategy that looks clean at
the API level can lose to raw thread or direct execution because:

- Closure setup costs are visible.
- Worker wake-up costs are visible.
- Per-call allocation costs are visible.
- Persistent workers can affect later benchmark libraries in the same process.

For this reason, benchmark worker strategies with fresh-process isolation before
making them default.

## Retained Patterns

- Use `std.Io.Group.concurrent` for normal coarse task submission. It keeps the
  code simple and is appropriate when each task has enough arithmetic or memory
  traffic to hide scheduler overhead.
- Use process-lifetime `std.Io.Threaded` helpers only behind a dispatch gate that
  is narrower than the evidence. The Level 2 DGER/DSYMV low-latency paths in
  `src/blas/core/pool.zig` are examples: helpers are created by
  `std.Io.Threaded`, then reused with per-helper generations, futex wake/wait,
  and bounded spin waits.
- Prefer fixed task assignment over helper races for very small work. The DGER
  128/256 probes showed that waking extra helpers and letting them race on a
  shared claim counter was slower than publishing only the helpers needed.
- Keep caller participation unless measured otherwise. Fully offloading short
  BLAS calls made the caller pay publication and wait overhead without doing
  useful work.
- Treat helper identity or offset tuning as machine-local evidence. If retained,
  it must be behind shape predicates and documented in the relevant optimization
  notes.

## Rejected Patterns

- Do not introduce raw `std.Thread.spawn` pools for BLAS dispatch while
  `std.Io.Threaded` can provide the helper lifecycle.
- Do not add runtime environment variables for scheduler modes. Use
  `ZYNUM_MAXIMUM_THREADS` only as a thread cap.
- Do not promote a low-latency worker path from a single focused best result.
  Require repeated fresh-process samples, comparator reruns, and nearby-shape
  checks.

## Current Source Locations

- Thread-count policy: `src/blas/runtime.zig`.
- GEMM dispatch: `src/blas/gemm/dispatch.zig`.
- Process-lifetime GEMM workers: `src/blas/gemm/pool.zig`.
- Shared Level 1/2 runners and low-latency helpers:
  `src/blas/core/pool.zig`.
- Level 2 DGER task shaping: `src/blas/core/level2/rank_update.zig`.
- Level 2 DSYMV task shaping: `src/blas/core/level2/symmetric.zig`.
- Fresh-process comparator isolation: `bench/tools/run_gemm_sweep_isolated.py`.

## Development Rules

- Do not add environment variables for worker strategy; only
  `ZYNUM_MAXIMUM_THREADS` may cap Zynum threads.
- Do not unload a dynamic library while its worker threads may still be running.
- Prefer small focused probes before full sweeps, then confirm with full sweeps.
- Document target features, thread cap, and relevant dispatch predicates in
  benchmark notes.
- Treat process boundaries as the reliable cleanup boundary for reportable data.

## Recommended Validation

```sh
ZYNUM_MAXIMUM_THREADS=1 zig build bench-gemm-sweep --release=fast -- --reps 30
ZYNUM_MAXIMUM_THREADS=6 zig build bench-gemm-sweep --release=fast -- --reps 30
```

For comparator numbers:

```sh
python3 bench/tools/run_gemm_sweep_isolated.py \
  --gemm-sweep zig-out/bin/gemm-sweep \
  --zynum-blas zig-out/lib/libzynum_blas.dylib \
  --csv zig-out/gemm_sweep_io_check.csv \
  --reps 30
```

Keep dispatch gates narrow until isolated data is stable.
