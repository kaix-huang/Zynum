# AArch64 GEMM Notes

This document covers AArch64-specific GEMM work in Zynum BLAS.

## Backends

Current source locations:

- `src/blas/kernels/aarch64/features.zig`: compile-time feature probes.
- `src/blas/kernels/aarch64/asimd.zig`: ASIMD/FMA packed-B backend.
- `src/blas/kernels/aarch64/sve2.zig`: SVE2 dispatch placeholder.
- `src/blas/kernels/aarch64/sme.zig`: SME dispatch and guarded fast paths.
- `src/blas/kernels/aarch64/sme_f32_gemm.S`: f32 SME assembly kernels.
- `src/blas/kernels/aarch64/sme_f64_gemm.S`: f64 SME assembly kernels.
- `src/blas/kernels/aarch64/amx_gemm.zig`: Apple AMX experiments.

## Dispatch Order

Dispatch must be capability-based:

1. SME or SME F64 support when the target and shape gate permit it.
2. SVE2 once true scalable-vector kernels exist.
3. ASIMD/FMA packed-B kernels.
4. Generic fallback.

CPU model names may appear in benchmark records, but code should prefer target
features such as ASIMD, FMA, SVE2, SME, SME2, and SME F64 support.

## Target Matrix

AArch64 performance records should identify the exact hardware and feature set;
cross-target builds are coverage, not throughput evidence.

| Target | What to record | Reportable evidence requirement |
| --- | --- | --- |
| `aarch64-macos` Apple M-series | `-Dcpu`, macOS version, P/E-core context if known, `ZYNUM_BLAS_AMX`, thread counts, Accelerate/OpenBLAS settings. | Native correctness, focused shape probes, and full fresh-process sweep before any comparator claim. |
| `aarch64-linux-gnu` ASIMD/FMA | CPU model as a label, target features, kernel backend, thread counts, OpenBLAS/BLIS settings if used. | Native Linux sweep on the measured host. Cross-compiled macOS data must not be reused as Linux performance evidence. |
| AArch64 SVE2/SME/SME F64 | Vector/SME feature set, streaming-mode gate, shape predicate, OS support, and cleanup path. | Native hardware run for the feature being claimed. Emulators or compile checks are not performance evidence. |

## Gate Record Requirements

Every retained AArch64 gate should record:

- Feature predicate: ASIMD/FMA, SVE2, SME, SME2, SME F64, and any Apple AMX
  opt-in requirement.
- Shape predicate: dtype, transpose flags, `m`, `n`, `k`, alpha/beta path, and
  thread-count assumptions.
- Backend effect: ASIMD, SVE2 placeholder/future path, SME assembly path, AMX
  experiment, or generic fallback.
- Evidence chain: correctness command, focused probe, full sweep, isolation
  level, runtime environment, raw CSV, and summary.
- Boundary evidence: nearby shapes that failed, regressed, or were deliberately
  excluded from the default gate.

If a gate depends on `ZYNUM_BLAS_AMX=1`, a non-default `ZYNUM_BLAS_GEMM_IO`, or a
worker-pool mode, keep it documented as experimental until fresh-process sweeps
show stable behavior. Do not convert an opt-in gate to default from one focused
win.

## SME Rules

- Keep streaming-mode and ZA state handling local to SME code.
- Pair every state-enter operation with cleanup.
- Keep large SME loops in assembly files.
- Use inline assembly only for small state-control snippets.
- Keep shape gates in `src/blas/gemm/dispatch.zig` or in a narrow SME helper,
  not in ABI wrappers.

SME can improve high-throughput real GEMM, but small shapes are sensitive to
fixed streaming-mode and packing costs.

## Apple AMX Rules

Apple AMX paths are experimental and controlled by `ZYNUM_BLAS_AMX`.

Rules:

- Default auto gates must remain narrow.
- Explicit `ZYNUM_BLAS_AMX=1` may enable broader experiments.
- `ZYNUM_BLAS_AMX=0` must disable AMX.
- Every AMX state setup must have paired cleanup.
- AMX benchmark results should include both focused probes and full sweeps.

Do not generalize a single AMX win into a broad default gate without full-sweep
evidence.

## Threading On Apple Silicon

Apple P/E core topology makes thread policy shape-dependent. Small matrices can
lose from thread startup. Tall/narrow and short/wide matrices can lose from
repeated packing across splits. Large square matrices can benefit from more
parallel work.

Use `ZYNUM_BLAS_NUM_THREADS` for explicit thread-count experiments. Keep the
default heuristic conservative.

## Benchmark Commands

Recommended environment for reportable local sweeps:

```sh
export ZYNUM_BLAS_GEMM_POOL=0
export ZYNUM_BLAS_GEMM_IO=0
export ZYNUM_BLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENBLAS_DYNAMIC=0
```

Native Apple M-series target:

```sh
zig build test -Dtarget=aarch64-macos -Dcpu=apple_m4 --release=fast
zig build bench-gemm-sweep -Dtarget=aarch64-macos -Dcpu=apple_m4 --release=fast -- --reps 30
```

Focused SME/AMX probe:

```sh
ZYNUM_BLAS_AMX=0 zig build bench-gemm-sweep \
  -Dtarget=aarch64-macos -Dcpu=apple_m4 --release=fast -- \
  --kind sgemm \
  --shape sq128:128:128:128 \
  --shape m128_n128_k4096:128:128:4096 \
  --reps 100
```

Comparator data should use the isolated runner and record process repeats:

```sh
python3 bench/tools/run_gemm_sweep_isolated.py \
  --gemm-sweep zig-out/bin/gemm-sweep \
  --zynum-blas zig-out/lib/libzynum_blas.dylib \
  --csv zig-out/aarch64_gemm_isolated.csv \
  --reps 30 \
  --process-repeats 2
```

Use level 3 isolation for any SME or AMX outlier that might be affected by cache,
streaming-mode setup, AMX state setup, or worker history.

## Current Priorities

1. Maintain a clean evidence chain for every AArch64 gate: correctness, focused
   probe, full sweep, isolation level, environment, CSV, and summary.
2. Improve small and medium SGEMM/DGEMM latency only with gates that do not
   regress full sweeps.
3. Keep SME and AMX gates narrow until nearby shapes and full sweeps justify a
   default rule.
4. Build a dedicated complex GEMM strategy for remaining CGEMM gaps, with
   separate evidence for workspace and dispatch overhead.
5. Keep worker-pool and non-default IO experiments opt-in until fresh-process
   comparator data is stable.

## Rejection Criteria

Reject or keep opt-in if an experiment:

- Wins only one focused shape.
- Regresses small square shapes.
- Pollutes later comparator libraries in the same process.
- Requires ABI or public API layers to know SME/AMX details.
- Depends on a CPU marketing name instead of an ISA feature.
