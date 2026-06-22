# x86_64 GEMM Notes

This document covers x86_64-specific GEMM work in Zynum BLAS.

## Backends

Current source locations:

- `src/blas/kernels/x86_64/features.zig`: compile-time feature probes.
- `src/blas/kernels/x86_64/simd.zig`: packed-B SGEMM/DGEMM vector backend.

The backend currently selects vector width from target features:

- SSE2: 128-bit lanes.
- AVX or AVX2: 256-bit lanes.
- AVX512F: 512-bit lanes.
- FMA: use multiply-add lowering where available.

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
- Tail columns still need better packed handling.
- AVX512 mask-tail paths are not complete.
- AVX2 and AVX512 probably need separate tile choices.
- MKL and OpenBLAS threading defaults differ, so comparator benchmarks must pin
  thread counts.

## Benchmark Commands

Compile checks:

```sh
zig build test -Dtarget=x86_64-linux-gnu -Dcpu=baseline
zig build test -Dtarget=x86_64-linux-gnu -Dcpu=x86_64_v3 --release=fast
zig build test -Dtarget=x86_64-linux-gnu -Dcpu=x86_64_v4 --release=fast
```

Linux benchmark with MKL:

```sh
zig build bench-gemm-sweep \
  -Dtarget=x86_64-linux-gnu \
  -Dcpu=x86_64_v4 \
  --release=fast \
  -Dbench-mkl=/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so -- \
  --reps 30
```

Recommended environment:

```sh
export ZYNUM_BLAS_GEMM_POOL=0
export ZYNUM_BLAS_GEMM_IO=0
export ZYNUM_BLAS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENBLAS_DYNAMIC=0
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_NUM_THREADS=1
```

Then sweep thread counts explicitly. Record whether the result came from the
in-process sweep or the isolated runner; use fresh-process data for comparator
claims.

## Optimization Priorities

1. Establish real Intel and AMD baselines with correctness, focused probes, full
   sweeps, fresh-process comparator data, and recorded runtime environment.
2. Add stack packing for small and medium shapes only behind documented shape
   predicates.
3. Implement packed partial-column tail handling and record the tail-shape
   boundary.
4. Add AVX512 mask load/store tail paths with separate native AVX512 evidence.
5. Split AVX2/FMA and AVX512/FMA tile choices when native data justifies the
   separate gates.
6. Benchmark alpha/beta write-back paths separately.
7. Compare against MKL and OpenBLAS with pinned thread counts and isolated
   processes.

## Acceptance Criteria

A retained x86_64 optimization needs:

- Passing correctness tests on the target.
- At least one real x86_64 machine measurement for the feature tier being
  enabled.
- Pinned Zynum and comparator thread counts, plus dynamic-threading variables.
- Focused shape data, full sweep data, raw CSV paths, command lines, runtime
  environment, and isolation level.
- Fresh-process data for any MKL/OpenBLAS comparison.
- No broad regression across square, tall/narrow, short/wide, and high-K shapes.
- A shape/feature gate that is narrower than the measured evidence.

Do not present cross-compiled AVX2/AVX512 checks as performance results.
