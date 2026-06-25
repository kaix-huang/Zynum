# AArch64 GEMM Notes

This document covers AArch64-specific GEMM work in Zynum BLAS.

## Backends

Current source locations:

- `src/blas/kernels/matrix_matrix/catalog.zig`: shared AArch64 descriptor parameters.
- `src/blas/kernels/matrix_matrix/tuning.zig`: shape/scalar candidate matching and
  execution-plan parameter selection.
- `src/blas/kernels/matrix_matrix/executor.zig`: selected `KernelId` execution bridge.
- `src/blas/kernels/matrix_matrix/packed_simd.zig`: shared packed-B SIMD skeleton used by
  ASIMD after AArch64 supplies lane, row-group, panel, tail, and stack-pack
  parameters.
- `src/blas/kernels/matrix_matrix/epilogue.zig`: shared scalar/vector alpha/beta
  write-back helpers used by portable and packed SIMD kernels.
- `src/blas/kernels/aarch64/features.zig`: compile-time feature probes.
- `src/blas/kernels/aarch64/asimd.zig`: thin ASIMD/FMA wrapper that configures
  the shared packed-B SIMD skeleton.
- `src/blas/kernels/aarch64/sve2.zig`: SVE2 dispatch placeholder.
- `src/blas/kernels/aarch64/sme.zig`: SME execution wrapper, feasibility checks,
  shared f32/f64 pack-workspace prologue, tail-row fallback, and hardware state
  boundaries for plan-selected fast paths.
- `src/blas/kernels/aarch64/sme_f32_gemm.S`: f32 SME assembly kernels.
- `src/blas/kernels/aarch64/sme_f64_gemm.S`: f64 SME assembly kernels.
- `src/blas/kernels/aarch64/amx_gemm.zig`: Apple AMX experiments.

## Dispatch Order

Dispatch must be capability-based:

1. SME or SME F64 support when descriptor matching and `ExecutionPlan` permit it.
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
| `aarch64-macos` Apple M-series | `-Dcpu`, macOS version, P/E-core context if known, target features, `ZYNUM_MAXIMUM_THREADS`, detected CPU count, Accelerate/OpenBLAS settings. | Native correctness, focused shape probes, and full fresh-process sweep before any comparator claim. |
| `aarch64-linux-gnu` ASIMD/FMA | CPU model as a label, target features, kernel backend, thread counts, OpenBLAS/BLIS settings if used. | Native Linux sweep on the measured host. Cross-compiled macOS data must not be reused as Linux performance evidence. |
| AArch64 SVE2/SME/SME F64 | Vector/SME feature set, streaming-mode gate, shape predicate, OS support, and cleanup path. | Native hardware run for the feature being claimed. Emulators or compile checks are not performance evidence. |

## Gate Record Requirements

Every retained AArch64 gate should record:

- Feature predicate: ASIMD/FMA, SVE2, SME, SME2, SME F64, and any Apple AMX
  internal auto-gate requirement.
- Shape predicate: dtype, transpose flags, `m`, `n`, `k`, alpha/beta path, and
  thread-count assumptions.
- Backend effect: selected `KernelId`, `ExecutionPlan` fields, ASIMD/SVE2/SME
  assembly path, AMX variant, pack path, workspace budget, or generic fallback.
- Evidence chain: correctness command, focused probe, full sweep, isolation
  level, runtime environment, raw CSV, and summary.
- Boundary evidence: nearby shapes that failed, regressed, or were deliberately
  excluded from the default gate.

If a gate depends on a new AMX/SME/threading predicate, keep it documented as
experimental until fresh-process sweeps show stable behavior. Do not convert an
exploratory gate to default from one focused win.

## SME Rules

- Keep streaming-mode and ZA state handling local to SME code.
- Pair every state-enter operation with cleanup.
- SME assembly that is called through the normal AArch64 C ABI must preserve the
  ABI-visible callee-saved FP state it actually uses. The AArch64 ABI preserves
  only the low 64 bits of `v8`-`v15`; in streaming mode those architectural
  registers correspond to the low 64 bits of `z8`-`z15`. Prefer using
  caller-saved `v`/`z` registers first. If an SME/SVE helper uses any of
  `v8`-`v15` or `z8`-`z15`, save and restore only the touched low-64-bit
  `d8`-`d15` lanes rather than blindly spilling the whole range.
- Do not save `d0` merely because `SMSTOP` changes the normal FP register view.
  `d0` is caller-saved and is the scalar FP return register. If an SME helper
  returns a scalar FP value, move the result or input value through a GPR before
  `SMSTOP` and write `d0` once after leaving streaming mode. Helpers that return
  `void` and do not need `d0` after `SMSTOP` should not spend instructions
  preserving it.
- Treat the SME ABI prologue/epilogue as part of the kernel contract. If a
  prologue macro is shared by several kernels, it may intentionally over-save
  for simplicity; performance-sensitive hand-written kernels should audit their
  actual `v8`-`v15`/`z8`-`z15` use and narrow the save set when that is safe.
- Keep large SME loops in assembly files.
- Use inline assembly only for small state-control snippets.
- Keep performance shape gates in `src/blas/gemm/dispatch.zig` or
  `src/blas/kernels/matrix_matrix/tuning.zig`, not in `src/blas/kernels/aarch64/sme.zig`
  or ABI wrappers.
- Keep SME Zig-side pack helpers parameterized by `T`, panel width, and panel
  count where possible. Instruction files may check tile feasibility, but pack
  mode, panel variant, batch count, and workspace budget should come from
  `ExecutionPlan`.
- Keep f32/f64 direct-path prologue code shared where it is not instruction
  specific: tile feasibility, stack-vs-cache workspace acquisition, and scalar
  tail rows should be parameterized by `T` and tile width. Only the assembly
  call sequence and SME state ownership should remain dtype-specific.

SME can improve high-throughput real GEMM, but small shapes are sensitive to
fixed streaming-mode and packing costs.

## Apple AMX Rules

Apple AMX paths are experimental native kernels selected only by internal
`ExecutionPlan` rules.

Rules:

- Default auto gates must remain narrow.
- Do not add environment variables to enable, disable, or broaden AMX.
- Every AMX state setup must have paired cleanup.
- AMX wrapper functions should only check ABI, nonzero sizes, alignment, and
  tile feasibility. Shape preference between N8/N16/N32 variants belongs in
  `src/blas/kernels/matrix_matrix/tuning.zig`.
- Stack/cache workspace budgets for AMX are execution-plan parameters. The AMX
  file may retain hard fixed-stack safety caps, but it should not contain
  tiered performance thresholds.
- Keep AMX inline assembly clobbers as narrow as the instruction semantics
  allow. `AMXLDX`, `AMXLDY`, `AMXSTZ`, state setup, and state cleanup interact
  with normal memory or opaque hardware state and should keep a `memory`
  clobber so the compiler cannot reorder pack/C memory operations across them.
  Pure AMX compute ops such as FMA/MATFP consume AMX register state and an
  encoded operand but do not read or write normal memory; they should stay
  `volatile` but do not need a `memory` clobber. This avoids unnecessary
  compiler barriers in the inner loop while preserving ordering around real
  loads, stores, and state transitions.
- When changing AMX clobbers, inspect the emitted code. A valid sequence keeps
  B/A AMX loads before dependent FMA/MATFP operations and keeps `AMXSTZ` in the
  epilogue after accumulation. If removing a clobber from a load/store/state
  wrapper changes memory ordering or correctness, restore the clobber rather
  than compensating with broader dispatch gates.
- AMX benchmark results should include both focused probes and full sweeps.

Do not generalize a single AMX win into a broad default gate without full-sweep
evidence.

## Contaminated Historical Evidence

Several earlier SME/SVE/SME2 and AMX experiments were evaluated while the SME
ABI boundary was still under investigation. Data from runs that might have
crossed an incorrect `SMSTART`/`SMSTOP` prologue/epilogue, returned a scalar FP
value through a stale `d0`, or failed to preserve used `d8`-`d15` lanes is not
reliable performance evidence. Treat those results as contaminated correctness
debugging data, not as proof that the instruction sequence, gate, or threading
strategy was slow.

Rules for old rollbacks:

- Do not cite a pessimistic rollback of an SME/SVE/SME2 or AMX candidate unless
  the candidate was rerun after the ABI and clobber fixes with correctness
  checks enabled.
- If old data showed comparator slowdown after a Zynum SME/AMX run, rerun with
  fresh-process isolation before attributing it to thermal state, worker state,
  AMX state, or the candidate kernel itself.
- If a benchmark run fails correctness, produces scalar-return mismatches, or
  changes when an unrelated SME helper runs first, discard all timing numbers
  from that process.
- Re-enable previously rejected SME/SVE/SME2 candidates only behind the same
  narrow feature and shape predicates used for new experiments, then rebuild the
  evidence chain from correctness, focused probes, and isolated sweeps.

## Threading On Apple Silicon

Apple P/E core topology makes thread policy shape-dependent. Small matrices can
lose from thread startup. Tall/narrow and short/wide matrices can lose from
repeated packing across splits. Large square matrices can benefit from more
parallel work.

Use `ZYNUM_MAXIMUM_THREADS` for explicit thread-cap experiments. When unset, the
cap is the runtime CPU count; GEMM dispatch may choose fewer threads internally.

Current default planner notes for Apple Silicon:

- Direct SME square-ish real GEMM is allowed only for `alpha=1,beta=0` and
  descriptor minimum block/work thresholds. Small direct square-ish work starts
  parallelism at `96^3` work units; f64 may use up to six threads below
  `256^3`, while other small square-ish work remains capped more tightly.
- Narrow-N real GEMM (`n <= 4 * n_panel`, `m >= 512`, and enough K work) is
  capped at four requested workers by default. This avoids the repeated-B-pack
  cost seen when row splitting is too broad.
- Vector-edge real GEMM (`m == 1` or `n == 1`) is intentionally sent to the
  generic vector-edge kernels before packed ASIMD/SME, because packing has poor
  reuse for those shapes.
- The task `ExecutionPlan` carries pack layout (`natural`, `dynamic`, or
  `transpose4`), f32 SME panel variant, SME panel batch count, AMX variant, and
  stack/cache workspace budgets. AArch64 instruction files must treat these as
  inputs and only reject them for feasibility or correctness. ASIMD should pass
  lane/tile/row-group configuration to `packed_simd.zig` instead of owning a
  separate packed-B loop body.

Focused probes on Apple M5 improved vector-edge and some narrow-N OpenBLAS
ratios, but Accelerate remains much faster for many SGEMM narrow-N and
medium-square points. Do not describe the current default as generally matching
Accelerate without a fresh-process sweep that proves it.

## Benchmark Commands

Recommended environment for reportable local sweeps:

```sh
export ZYNUM_MAXIMUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENBLAS_DYNAMIC=0
```

Native Apple M-series target:

```sh
zig build test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme2p1 --release=fast
zig build bench-gemm-sweep -Dtarget=aarch64-macos -Dcpu=apple_m4+sme2p1 --release=fast -- --reps 30
```

Focused SME/AMX probe:

```sh
ZYNUM_MAXIMUM_THREADS=1 zig build bench-gemm-sweep \
  -Dtarget=aarch64-macos -Dcpu=apple_m4+sme2p1 --release=fast -- \
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

For this local Apple M5 tuning pass, the reportable final sweep should leave
`ZYNUM_MAXIMUM_THREADS` unset and run the isolated helper against Accelerate and
OpenBLAS, then plot the CSV with `bench/tools/plot_gemm_sweep.py`.

## 2026-06-23 Apple M5 Tuning Result

Validation commands:

```sh
zig fmt --check build.zig build.zig.zon src test bench examples tools
zig build --global-cache-dir .zig-cache/global test --summary failures
zig build --global-cache-dir .zig-cache/global test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme2p1 --release=fast --summary failures
zig build --global-cache-dir .zig-cache/global -Dtarget=aarch64-macos -Dcpu=apple_m4+sme2p1 --release=fast --summary failures
```

Final sweep command:

```sh
env -u ZYNUM_MAXIMUM_THREADS \
  OPENBLAS_DYNAMIC=0 \
  OPENBLAS_NUM_THREADS=10 \
  VECLIB_MAXIMUM_THREADS=10 \
  OMP_NUM_THREADS=10 \
  python3 bench/tools/run_gemm_sweep_isolated.py \
    --gemm-sweep zig-out/bin/gemm-sweep \
    --zynum-blas zig-out/lib/libzynum_blas.dylib \
    --accelerate /System/Library/Frameworks/Accelerate.framework/Accelerate \
    --openblas /opt/homebrew/opt/openblas/lib/libopenblas.dylib \
    --csv zig-out/gemm_sweep_m5_final.csv \
    --reps 30 \
    --process-repeats 3 \
    --isolate-kind \
    --skip-missing
```

Plot local artifact:

```sh
python3 bench/tools/plot_gemm_sweep.py \
  zig-out/gemm_sweep_m5_final.csv \
  zig-out/gemm_sweep_m5_final.svg
```

Curated README performance charts are refreshed separately with the current
all-level process in `docs/common/benchmarking.md#readme-performance-charts`;
do not keep historical one-off sweep SVGs under `docs/assets/benchmarks/`.

Summary for 168 complete points (`sgemm`, `dgemm`, `cgemm`, `zgemm` across 42
default shapes):

- vs Accelerate: `36/168` points at or above comparator speed, `42/168` within
  10%.
- vs OpenBLAS: `50/168` points at or above comparator speed, `72/168` within
  10%.
- vs max(Accelerate, OpenBLAS): `29/168` points at or above the best comparator,
  `32/168` within 10%.

Worst remaining gaps versus the best comparator are SGEMM narrow-N and
high-K panel points: `m2048_n64_k512`, `m1024_n64_k1024`,
`m512_n64_k2048`, `sq128`, and the dgemm/cgemm/zgemm `n=64` high-K cases.
This pass improves the architecture boundary and keeps parameterized dispatch
in place, but it does not support a claim that Zynum is generally no slower
than Accelerate or OpenBLAS on Apple M5.

After the tuning pass, the AArch64 kernel layer was refactored around shared
kernel mechanics:

- ASIMD is a thin feature/config wrapper around
  `src/blas/kernels/matrix_matrix/packed_simd.zig`.
- Real-GEMM alpha/beta write-back is centralized in
  `src/blas/kernels/matrix_matrix/epilogue.zig`, which keeps the `beta=0` no-read path
  explicit.
- SME direct paths share dtype-parameterized tile feasibility, stack-vs-cache
  workspace prologue, and scalar tail-row handling. SME assembly calls and
  streaming/ZA state cleanup remain local to the AArch64 SME files.
- AMX/SME workspace budgets are supplied by `ExecutionPlan`; instruction files
  retain only hard safety caps and hardware-state requirements.

The validation commands above were rerun after this refactor.

## 2026-06-24 Apple M5 Focused AMX Gate Update

Validation commands:

```sh
zig fmt --check build.zig build.zig.zon src test bench examples tools
zig build --global-cache-dir .zig-cache/global test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme2p1 --release=fast --summary failures
python3 -m py_compile bench/tools/check_gemm_sweep.py
```

Focused isolated command:

```sh
env OPENBLAS_DYNAMIC=0 OPENBLAS_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10 OMP_NUM_THREADS=10 \
  python3 bench/tools/run_gemm_sweep_isolated.py \
    --gemm-sweep zig-out/bin/gemm-sweep \
    --zynum-blas zig-out/lib/libzynum_blas.dylib \
    --accelerate /System/Library/Frameworks/Accelerate.framework/Accelerate \
    --openblas /opt/homebrew/opt/openblas/lib/libopenblas.dylib \
    --csv zig-out/gemm_dispatch_focus_after2.csv \
    --kind sgemm --kind dgemm \
    --shape m1024_n1024_k64:1024:1024:64 \
    --shape m1024_n1024_k128:1024:1024:128 \
    --shape m1024_n1024_k256:1024:1024:256 \
    --shape m1536_n256_k256:1536:256:256 \
    --shape m256_n1536_k256:256:1536:256 \
    --shape m1024_n64_k1024:1024:64:1024 \
    --shape m2048_n64_k512:2048:64:512 \
    --shape m512_n64_k2048:512:64:2048 \
    --reps 30 --process-repeats 2 --isolate-shape --skip-missing
```

Strict checker result:

```sh
python3 bench/tools/check_gemm_sweep.py zig-out/gemm_dispatch_focus_after2.csv --worst 30
# checked=16 passed=12 failed=4 missing=0 ratio=1
```

Retained default gates:

- `sgemm`, no-transpose real, `alpha=1,beta=0`, AMX `f32_n32` when
  `m >= 256`, `n >= 256`, `k <= 256`, and the existing AMX alignment checks
  pass. Focused isolated wins include `m1024_n1024_k64/128/256` and
  `m1536_n256_k256`; `m256_n1536_k256` remained slightly below Accelerate
  (`0.953x`) but was well above OpenBLAS.
- `dgemm`, no-transpose real, `alpha=1,beta=0`, AMX `f64_n32` when
  `m >= 256`, `n >= 256`, `k <= 256`, and alignment checks pass. Focused
  isolated data passed every tested low-K point versus both Accelerate and
  OpenBLAS.
- `dgemm`, no-transpose real, `alpha=1,beta=0`, AMX `f64_n32` for
  tall/narrow high-K panels with `m >= 512`, `32 <= n <= 64`, and
  `512 <= k <= 1024`, plus existing alignment checks. Focused isolated data
  passed `m1024_n64_k1024`, `m2048_n64_k512`, and `m512_n64_k2048`.

Rejected or still-open experiments:

- Disabling the `sgemm sq128` pre-planner AMX shortcut was slower than the
  existing shortcut, so the shortcut remains.
- Forcing `sgemm m1024_n64_k1024` from AMX `f32_n32` to `f32_n16` was slower.
- SGEMM narrow-N/high-K remains below Accelerate for
  `m1024_n64_k1024`, `m2048_n64_k512`, and `m512_n64_k2048`; the latter two are
  close in focused isolation, but the first remains a major gap.
- The focused gates improve selected default points but do not support a full
  default-sweep claim that Zynum is no slower than Accelerate/OpenBLAS.

## 2026-06-25 Apple M5 Follow-up

Validation commands:

```sh
zig fmt --check build.zig build.zig.zon src test bench examples tools
zig build --global-cache-dir .zig-cache/global test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme2p1 --release=fast --summary failures
```

Retained cleanup rules:

- Generic real GEMM now routes `m == 1`, `k >= 16` to the existing
  `rowVectorColumns` fallback before the tiled generic kernels. Focused isolated
  probes improved the default row-vector shapes, but they remain well below
  Accelerate/OpenBLAS.
- Real no-transpose GEMM with `n == 1` and `k > 0` may reuse the core Level 2
  GEMV path. The `k > 0` predicate is required so GEMM `k == 0` keeps the
  original `beta*C` scaling semantics.
- SGEMM narrow-N/high-K direct-kernel planning caps the retained local f32 gate
  at two threads when `n <= 2 * desc.tile.n_panel`, `m >= 512`, and `k >= 512`.
  A one-thread cap was tested and rejected because it regressed nearby default
  shapes.

Focused CSVs:

```text
zig-out/perf-report/gemm_m1_sgemm_after.csv
zig-out/perf-report/gemm_m1_dgemm_after.csv
zig-out/perf-report/gemm_n1_sgemm_after.csv
zig-out/perf-report/gemm_n1_dgemm_after.csv
zig-out/perf-report/gemm_sgemm_narrow_cap2_after.csv
zig-out/perf-report/gemm_sgemm_narrow_cap1_after.csv
```

Final quick full-sweep CSV:

```text
zig-out/perf-report/level3_after_final_fast.csv
```

Environment: `ZYNUM_MAXIMUM_THREADS=unset`, detected max threads 10,
`OPENBLAS_DYNAMIC=0`, `OPENBLAS_NUM_THREADS=10`,
`VECLIB_MAXIMUM_THREADS=10`, `OMP_NUM_THREADS=10`.

The quick full sweep still had 101/168 points below the fastest comparator.
The retained narrow-N cap improved the focused `sgemm m2048_n64_k512` and
`sgemm m512_n64_k2048` points to near parity, but it did not close
`sgemm m1024_n64_k1024`, the vector-edge GEMM cases, small square latency, or
the complex GEMM gaps. This pass does not support a broad no-slower-than claim.

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
