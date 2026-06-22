# Zig 0.16 `std.Io` And Threading

Zig 0.16 introduced a new `std.Io` model. Zynum BLAS uses it carefully because
GEMM tasks are often too small for generic asynchronous scheduling overhead.

## Current Policy

- Correctness code should not depend on global async state.
- GEMM default parallelism uses explicit dispatch policy.
- Experimental `std.Io` worker strategies are controlled by
  `ZYNUM_BLAS_GEMM_IO`.
- Persistent worker experiments must stay opt-in until they are proven not to
  pollute long benchmark sweeps or comparator libraries.

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
- Worker-pool experiments: `src/blas/gemm/pool.zig`.
- Benchmark isolation warning: `bench/gemm_sweep.zig`.

## Development Rules

- Keep stateful worker experiments behind environment variables.
- Do not unload a dynamic library while its worker threads may still be running.
- Prefer small focused probes before full sweeps, then confirm with full sweeps.
- Document the exact `ZYNUM_BLAS_GEMM_IO` mode used in benchmark notes.
- Treat process boundaries as the reliable cleanup boundary for reportable data.

## Recommended Validation

```sh
ZYNUM_BLAS_GEMM_IO=0 zig build bench-gemm-sweep --release=fast -- --reps 30
ZYNUM_BLAS_GEMM_IO=pool zig build bench-gemm-sweep --release=fast -- --reps 30
```

For comparator numbers:

```sh
python3 bench/tools/run_gemm_sweep_isolated.py \
  --gemm-sweep zig-out/bin/gemm-sweep \
  --zynum-blas zig-out/lib/libzynum_blas.dylib \
  --csv zig-out/gemm_sweep_io_check.csv \
  --reps 30
```

Keep the default path conservative until the isolated data is stable.
