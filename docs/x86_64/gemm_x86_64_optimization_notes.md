# x86_64 GEMM Notes

This document covers x86_64-specific GEMM work in Zynum BLAS.

## Backends

Current source locations:

- `src/blas/kernels/arch/x86_64/features.zig`: compile-time feature probes.
- `src/blas/kernels/arch/x86_64/matrix_matrix/simd.zig`: thin packed-B SGEMM/DGEMM wrapper that
  selects lane and panel parameters from x86_64 target features.
- `src/blas/kernels/arch/x86_64/vector/binary.zig`,
  `src/blas/kernels/arch/x86_64/vector/unary.zig`, and
  `src/blas/kernels/arch/x86_64/matrix_vector.zig`: thin Level 1/2 wrappers that
  select lane widths and call shared fixed-SIMD skeletons.
- `src/blas/kernels/shared/matrix_matrix/packed_simd.zig`: shared fixed-width packed-B SIMD
  skeleton used by the x86_64 wrapper.
- `src/blas/kernels/shared/vector/fixed_simd.zig` and
  `src/blas/kernels/shared/matrix_vector/fixed_simd.zig`: shared Level 1/2 SIMD
  skeletons used by x86_64 and AArch64 wrappers.
- `src/blas/kernels/shared/matrix_matrix/epilogue.zig`: shared real-GEMM alpha/beta write-back
  helpers.

The wrapper currently selects vector width from target features:

- SSE2: 128-bit lanes.
- AVX or AVX2: 256-bit lanes.
- AVX512F: 512-bit lanes.
- FMA: use multiply-add lowering where available.

The x86_64 file should stay a configuration and feature-boundary layer. Packed
panel preparation, the K loop, scalar/vector row tails, and generic fallback for
unsupported shapes live in the shared `packed_simd.zig` skeleton unless native
x86 evidence justifies a different instruction sequence.

## Dispatch Rules

Dispatch must be feature-based:

1. AVX512F/FMA when the target supports it.
2. AVX2/FMA or AVX/FMA when available.
3. SSE2 baseline.
4. Generic fallback when no x86_64 vector backend is selected.

Do not branch on Intel or AMD product names in kernel selection. Use CPU names
only in benchmark labels and tuning notes.

## Target Matrix

x86_64 performance records should distinguish compile coverage from native
throughput evidence.

| Target tier | What to record | Reportable evidence requirement |
| --- | --- | --- |
| Baseline/SSE2 | Target triple, `-Dcpu`, OS, CPU label, thread counts, comparator settings. | Native correctness plus full sweep on the measured host. |
| AVX or AVX2/FMA | Feature tier, vector width, tile choice, tail path, Intel/AMD CPU label, MKL/OpenBLAS settings. | Native machine with the advertised features. Do not infer AVX2 throughput from cross-compiled checks. |
| AVX512F/FMA | AVX512 feature set, mask-tail status, downclock-sensitive thread count, MKL/OpenBLAS settings. | Native AVX512 host, focused probes, full fresh-process sweep, and outlier reruns when frequency behavior is suspected. |
| Cross-target checks | Zig target and CPU tier. | Build/correctness coverage only; not performance evidence. |

## Gate Record Requirements

Every retained x86_64 gate should record:

- Feature predicate: SSE2, AVX/FMA, AVX2/FMA, AVX512F/FMA, and any required tail
  support.
- Shape predicate: dtype, transpose flags, `m`, `n`, `k`, alpha/beta path, and
  thread-count assumptions.
- Backend effect: vector width, tile choice, packing policy, tail path, and split
  policy.
- Evidence chain: correctness command, focused benchmark, full-sweep CSV,
  isolation level, comparator libraries, runtime environment, and summary.
- Boundary evidence: nearby shapes excluded from the gate, Intel/AMD differences,
  and any frequency or throttling notes.

Keep new AVX2 or AVX512 behavior opt-in until the record includes native data for
the feature tier being enabled by default.

## Current Limitations

- The current development machine may cross-compile x86_64 but cannot produce
  real AVX2/AVX512 performance numbers.
- Packed-SIMD GEMM has shared partial-column handling enabled by the x86_64
  wrapper. Native measurement is still required before making any x86
  throughput claim for a particular feature tier.
- AVX512 mask-tail paths are not complete.
- AVX2 and AVX512 probably need separate tile choices.
- MKL and OpenBLAS threading defaults differ, so comparator benchmarks must pin
  thread counts.

## Level 1/2 Coverage

The x86_64 Level 1 and Level 2 files should stay wrapper-only where possible.
Current shared skeleton coverage includes:

- Level 1 real copy, swap, scal, axpy, axpby, dot, asum, nrm2, iamax, and rot.
- Level 1 complex scal, axpy, axpby, and dot.
- Level 2 real GEMV-N, GEMV-T, beta-handling GEMV wrappers, and GER.
- Level 2 complex GEMV-N and GEMV-T under conservative work gates.
- Real GEMV-N packed-row hooks that pack `alpha*x` once and reuse the shared
  GEMV-N skeleton inside row-split tasks.

Add new x86_64 microkernel coverage by extending the shared skeletons with
comptime lane, unroll, tail, or packing parameters first. Only add hand-written
x86 assembly after native measurements show that Zig vector lowering cannot
cover the required instruction shape.

## Benchmark Commands

Compile checks:

```sh
zig build -Dtarget=x86_64-linux-gnu -Dcpu=baseline --summary failures
zig build -Dtarget=x86_64-linux-gnu -Dcpu=x86_64_v3 --release=fast --summary failures
zig build -Dtarget=x86_64-linux-gnu -Dcpu=x86_64_v4 --release=fast --summary failures
```

Run `zig build test ...` only on a host that can execute the selected x86_64
target binaries, or under an explicitly documented runner/emulator. Cross-target
builds from Apple Silicon are compile coverage only.

Linux benchmark with MKL and AOCL-BLIS:

```sh
zig build bench-gemm-sweep \
  -Dtarget=x86_64-linux-gnu \
  -Dcpu=x86_64_v4 \
  --release=fast \
  -Dbench-mkl=/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so \
  -Dbench-aocl-blis=/path/to/libblis-mt.so -- \
  --reps 30
```

Single-thread focused environment:

```sh
export ZYNUM_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENBLAS_DYNAMIC=0
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_NUM_THREADS=1
export BLIS_NUM_THREADS=1
```

Then sweep thread counts explicitly. Record whether the result came from the
in-process sweep or the isolated runner; use fresh-process data for comparator
claims.

For reportable x86 data, prefer the isolated runner so each library gets a fresh
process:

```sh
python3 bench/tools/run_gemm_sweep_isolated.py \
  --gemm-sweep zig-out/bin/gemm-sweep \
  --zynum-blas zig-out/lib/libzynum_blas.so \
  --openblas /path/to/libopenblas.so \
  --mkl /path/to/libmkl_rt.so \
  --aocl-blis /path/to/libblis-mt.so \
  --reps 30 \
  --process-repeats 3 \
  --isolate-kind \
  --csv zig-out/perf-report/x86_gemm_isolated.csv \
  --skip-missing

python3 bench/tools/check_gemm_sweep.py \
  zig-out/perf-report/x86_gemm_isolated.csv \
  --comparator OpenBLAS \
  --comparator MKL \
  --comparator AOCL-BLIS
```

Do not include LIBXSMM in this BLAS sweep unless a shim or dedicated runner
exports the same GEMM semantics and the benchmark record describes that adapter.

## Optimization Priorities

1. Establish real Intel and AMD baselines with correctness, focused probes, full
   sweeps, fresh-process comparator data, and recorded runtime environment.
2. Add stack packing for small and medium shapes only behind documented shape
   predicates.
3. Implement packed partial-column tail handling through shared
   `packed_simd.zig` prologue/epilogue hooks where possible, and record the
   tail-shape boundary.
4. Add AVX512 mask load/store tail paths with separate native AVX512 evidence.
5. Split AVX2/FMA and AVX512/FMA tile choices when native data justifies the
   separate gates.
6. Benchmark alpha/beta write-back paths separately, but keep shared formulas in
   `src/blas/kernels/shared/matrix_matrix/epilogue.zig`.
7. Compare against MKL, OpenBLAS, and AOCL-BLIS with pinned thread counts and
   isolated processes. Treat LIBXSMM as a separate comparator integration unless
   a documented BLAS-compatible shim is used.

## Acceptance Criteria

A retained x86_64 optimization needs:

- Passing correctness tests on the target.
- At least one real x86_64 machine measurement for the feature tier being
  enabled.
- Pinned Zynum and comparator thread counts, plus dynamic-threading variables.
- Focused shape data, full sweep data, raw CSV paths, command lines, runtime
  environment, and isolation level.
- Fresh-process data for any MKL/OpenBLAS/AOCL-BLIS comparison.
- No broad regression across square, tall/narrow, short/wide, and high-K shapes.
- A shape/feature gate that is narrower than the measured evidence.

Do not present cross-compiled AVX2/AVX512 checks as performance results.
