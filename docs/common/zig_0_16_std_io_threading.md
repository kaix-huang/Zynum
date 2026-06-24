# Zig 0.16 `std.Io` And Threading

Zig 0.16 introduced a new `std.Io` model. Zynum BLAS uses it carefully because
GEMM tasks are often too small for generic asynchronous scheduling overhead.

## Current Policy

- Correctness code should not depend on global async state.
- GEMM parallelism uses explicit dispatch policy.
- Parallel GEMM work should prefer `std.Io.Threaded` with
  `std.Io.Group.concurrent` for CPU-bound helper tasks.
- Worker strategy is internal implementation policy and must not be selected by
  Zynum environment variables.

## Why This Matters

GEMM kernels can be extremely short. A scheduling strategy that looks clean at
the API level can lose to raw thread or direct execution because:

- Closure setup costs are visible.
- Worker wake-up costs are visible.
- Per-call allocation costs are visible.
- Persistent workers can affect later benchmark libraries in the same process.

For this reason, benchmark worker strategies with fresh-process isolation before
making them default.

## Current Source Locations

- Thread-count policy: `src/blas/runtime.zig`.
- GEMM dispatch: `src/blas/gemm/dispatch.zig`.
- Process-lifetime GEMM workers: `src/blas/gemm/pool.zig`.
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
