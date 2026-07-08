# Level 1 Optimization Notes

This document records BLAS Level 1 performance lessons for Zynum. It is meant
to guide future tuning work across real, complex, ABI, and benchmark paths.
Architecture-specific instruction details still belong in the matching
architecture notes when they become permanent dispatch rules.

## Ownership

Current Level 1 performance code is split across:

- `src/blas/core/vector.zig`: stable Level 1 facade used by API, ABI, and other
  core levels.
- `src/blas/core/vector/operations.zig`: BLAS semantics, strides, complex
  behavior, contiguous fast-path dispatch, portable Zig vector loops, tail
  handling, and coarse parallel splitting.
- `src/blas/core/execution/thread_pool.zig`: shared `std.Io.Threaded` task runner for large
  contiguous Level 1 work.
- `src/blas/kernels/dispatch/vector_unary.zig`: architecture facade for one-vector
  operations such as `scal` and `asum`.
- `src/blas/kernels/dispatch/vector_binary.zig`: architecture facade for two-vector
  operations such as `copy`, `axpy`, and `dot`.
- `src/blas/kernels/arch/aarch64/vector/unary.zig`: AArch64 SVE/SME candidates for
  one-vector work.
- `src/blas/kernels/arch/aarch64/vector/binary.zig`: AArch64 SVE/SME candidates for
  copy, axpy, dot, and complex dot.
- `bench/level1_probe.zig` and `bench/dcopy_probe.zig`: focused fresh-process
  probes.
- `bench/tools/run_level1_report.py` and `bench/tools/plot_level1_report.py`:
  reportable Level 1 coverage runner and operation-grouped SVG plots.

Keep ABI wrappers thin. A BLAS symbol should translate arguments into core
semantics, not select kernels or embed shape policy.

## Semantic Rules

Level 1 kernels must preserve all BLAS argument semantics before optimizing:

- `n <= 0` returns without touching memory.
- `incx == 0` and `incy == 0` are no-op guards where existing core semantics
  define them as invalid work.
- Negative strides start at `(1 - n) * inc`, then advance by `inc`.
- Non-unit strides stay on scalar/reference-safe loops unless a candidate
  explicitly handles the stride.
- Unit-stride complex data is interleaved real/imag storage. It may be viewed as
  `2*n` real values only when the operation is algebraically equivalent on each
  lane, such as real-alpha scaling, swap, copy, `asum`, `nrm2`, and real
  Givens rotations.
- Complex-alpha operations require true complex arithmetic:
  - `(a + bi) * (x + yi) = (ax - by) + i(ay + bx)`
  - `axpy`: `y += alpha*x`
  - `axpby`: `y = alpha*x + beta*y`
  - `dotu`: no conjugation
  - `dotc`: conjugate the first vector

Do not use comparator libraries as a Zynum compute path. They are benchmark and
compatibility comparators only.

## Effective Patterns

### Treat Copy As Bytes

All BLAS copy variants can be reduced to byte copy when both strides are one.
This removes dtype-specific duplication and lets one byte kernel serve `scopy`,
`dcopy`, `ccopy`, and `zcopy`.

Use this path only for unit-stride copy. Strided copy still needs element
semantics so source and destination indexing remain correct.

### Reuse Real Kernels For Complex Real-Alpha Work

Complex operations with real scalar coefficients should reuse real kernels by
viewing the buffer as `2*n` real values:

- `csscal` and `zdscal`
- complex `rscal`
- complex `swap`
- complex `asum`
- complex `nrm2`
- complex `rot` when `s` is real
- complex `axpy` and `axpby` when every complex scalar has zero imaginary part

This is usually both faster and simpler than maintaining a duplicate complex
loop. It also makes f32/f64 architecture kernels available to complex callers.

### Keep True Complex Kernels Interleaved

For complex-alpha `scal`, `axpy`, and `axpby`, keep the data interleaved and use
shuffle/sign vectors:

- Load `[xr, xi, xr, xi, ...]`.
- Pair-swap to `[xi, xr, xi, xr, ...]`.
- Multiply the swapped lanes by a sign vector `[-imag, +imag, ...]`.
- FMA with the real scalar part.

The same pattern works for `cscal`, `zscal`, `caxpy`, `zaxpy`,
`caxpby`, and `zaxpby`. Use the scalar tail for the final one complex value
when the real lane count is not a multiple of the vector length.

### Use Multiple Accumulators For Reductions

Dot and asum loops need independent accumulators to avoid one long
loop-carried dependency. The core real loops use four vector accumulators.
Complex dot should follow the same shape with separate real and imaginary
accumulators.

For `ComplexF64` dot on Apple M-series, adding coarse parallel splitting was
more valuable than attempting to make a single SVE `ld2d` loop win against
Accelerate. Keep this distinction in mind: one kernel can be instruction-bound,
while the whole operation can still win through partitioning.

### Tail Handling

For portable Zig vector loops, use descending SIMD tails before scalar cleanup:

```text
main vector width -> width/2 -> width/4 -> width/8 -> scalar
```

This helped the core Level 1 paths avoid large scalar tails while keeping code
simple. Use the same pattern for real and complex kernels where the data can be
expressed as a real lane stream.

### Parallelism

Level 1 is mostly memory-bandwidth work. Parallelism helps only when the vector
is large enough to amortize:

- `std.Io.Threaded` worker handoff,
- extra SME/SVE state entry/exit,
- additional memory-stream pressure,
- final reduction merging.

The Apple M5 work showed f32 SME single-thread kernels can already saturate the
useful bandwidth for around one million elements. Broad f32 parallel thresholds
made `saxpy`, `sdot`, and `asum` slower until gates were raised or made more
specific. Complex F64 dot, however, benefited from task splitting at large sizes
because the per-element arithmetic is heavier and the single-thread path was
below Accelerate.

Future threshold changes must be tested both single-thread and default-thread,
with fresh-process comparator data before changing defaults.

## x86_64 Lessons

### Coarse Parallel Splitting

On large unit-stride Level 1 work, x86_64 benefits from splitting more broadly
than the Apple M-series defaults once the core `std.Io.Threaded` pool is known
to run correctly. The retained H3C Xeon Gold 6326 tuning uses 32-way splits for
large real and complex unit-stride arithmetic, copy, reductions, and `nrm2`.

Keep this split behind `x86_64` predicates. The same thresholds are not a
portable rule for AArch64, and they should be revalidated on AMD or smaller
Intel systems before claiming a general x86 rule.

### Thread-Pool Publication

For the one-million-element hot-cache probes, the persistent
`core_pool.runLowLatency` path was materially faster than per-call
`std.Io.Group.concurrent` submission once the worker stack size was fixed.
However, helper publication is still part of the measured cost, so keep the
number of tasks tied to evidence rather than assuming more tasks are always
better.

On Linux/x86_64, pinning only Zynum's persistent helper threads inside the
current `sched_getaffinity` CPU set improved H3C Level 1 full-report coverage.
This policy deliberately does not pin the caller thread and does not choose CPUs
outside the scheduler-provided cpuset. Treat it as a Linux helper lifecycle
detail, not a user-facing mode.

Rejected scheduling experiments are just as important:

- A shared-generation claimed-worker pool reduced the number of futex wake
  addresses but broke fixed per-worker partitioning and regressed copy, axpy,
  dot, and complex axpy. Do not retry helper racing for Level 1 without a new
  design and focused evidence.
- Increasing the generic x86 hot-loop unroll from four vector accumulators to
  eight increased register pressure and did not close the OpenBLAS gap. Keep the
  portable core loops conservative; put materially different x86 instruction
  choices behind the architecture kernel facades.

### Missing Architecture Kernels

The current x86_64 Level 1 fast path still mostly uses the shared Zig `@Vector`
loops. Remaining gaps against OpenBLAS on H3C are concentrated in real
`axpy/dot/asum`, complex-as-real `asum`, and true-complex `axpy/dot`.

The preferred next step is to add x86_64 Level 1 kernels through
`src/blas/kernels/dispatch/vector_unary.zig` and `src/blas/kernels/dispatch/vector_binary.zig`,
with implementation files under `src/blas/kernels/arch/x86_64/`. Start with kernels
that are materially different from the core fallback, such as AVX2-width versus
AVX512-width variants, feature-gated reduction trees, or true-complex kernels
with x86-specific shuffle/FMA structure. Do not add an x86 facade that simply
duplicates the portable loop; focused probes already showed that this does not
move the result.

## AArch64 Lessons

### SME State Boundaries

SME kernels must use the same ABI discipline as GEMM and GEMV:

- Start/stop streaming mode and ZA in paired paths.
- Preserve only ABI-visible callee-saved FP lanes that are actually used.
- If a scalar result must survive `SMSTOP`, move the result bits through a GPR
  and restore them after leaving streaming mode.
- Treat `SMSTART`/`SMSTOP` correctness as a precondition for timing. Any
  benchmark gathered before this is correctness-debugging data, not dispatch
  evidence.

### f32 `asum`

The first SME f32 `asum` path was fast but unstable around the complex-as-real
`scasum` case. A non-streaming SVE f32 `sasum` loop avoided ZA entry/exit and
ZA reduction overhead and made `scasum` competitive in the focused runs.

This is a useful pattern: reductions that do not need ZA accumulation may be
better served by plain SVE even when SME2 is available.

### f32 `scal`, `axpy`, and `dot`

SME2 streaming f32 kernels are valuable for large unit-stride real work, but
parallel splitting can erase the win. Keep gates narrow:

- use SME2 only when the streaming vector length and feature set are known,
- keep minimum length thresholds explicit,
- avoid splitting f32 work too early.

### Complex F64 Dot

A dedicated SVE `ld2d` complex F64 dot kernel was not enough to beat
Accelerate by itself. The retained win came from using the unit kernel inside
a coarse parallel reduction. Keep the unit kernel, but do not assume
structure-load SVE is always better than the generic interleaved vector path.

## ABI And Test Coverage

Complex scalar ABI correctness needs direct tests. The BLAS entry points pass
complex scalars by pointer for Fortran and CBLAS:

- `cscal_`, `zscal_`
- `caxpy_`, `zaxpy_`
- `caxpby_`, `zaxpby_`
- `cblas_cscal`, `cblas_zscal`
- `cblas_caxpy`, `cblas_zaxpy`
- `cblas_caxpby`, `cblas_zaxpby`

The tests should cover:

- unit stride,
- positive non-unit stride,
- negative stride,
- nonzero imaginary scalar values,
- untouched padding/sentinel slots.

If a testable compatibility facade lacks a symbol re-export, add the re-export
instead of bypassing the ABI layer in the test.

## Benchmarking

Use correctness first:

```sh
zig build test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
```

Build the focused probes:

```sh
zig build-exe bench/level1_probe.zig -OReleaseFast \
  -target aarch64-macos -mcpu apple_m4+sme+sme2+sme2p1 \
  --global-cache-dir .zig-cache/global \
  -femit-bin=zig-out/perf-report/bin/level1_probe

zig build-exe bench/dcopy_probe.zig -OReleaseFast \
  -target aarch64-macos -mcpu apple_m4+sme+sme2+sme2p1 \
  --global-cache-dir .zig-cache/global \
  -femit-bin=zig-out/perf-report/bin/dcopy_probe
```

Run the reportable fresh-process coverage sweep:

```sh
python3 bench/tools/run_level1_report.py \
  --level1-probe zig-out/perf-report/bin/level1_probe \
  --copy-probe zig-out/perf-report/bin/dcopy_probe \
  --zynum zig-out/lib/libzynum_blas.dylib \
  --accelerate /System/Library/Frameworks/Accelerate.framework/Accelerate \
  --openblas /opt/homebrew/lib/libopenblas_armv8p-r0.3.33.dylib \
  --n 1048576 \
  --seconds 1 \
  --copy-seconds 1 \
  --process-repeats 3 \
  --csv zig-out/level1_final_report.csv
```

Plot grouped bar charts, using operation names on the x-axis:

```sh
python3 bench/tools/plot_level1_report.py \
  zig-out/level1_final_report.csv \
  --bars-svg zig-out/level1_final_bars.svg \
  --ratio-svg zig-out/level1_final_ratio.svg
```

Use grouped bars for Level 1 coverage because the operations are categorical,
not a continuous shape sweep. Use line charts only for GEMM or other cases where
the x-axis is an ordered continuous or quasi-continuous shape progression.

Copy should be charted as bandwidth (`GB/s`). Arithmetic Level 1 operations
should be charted as `Gops` using a consistent operation-count convention.
Avoid mixing copy bandwidth and arithmetic Gops on one axis.

## Current Apple M5 Snapshot

Snapshot from `zig-out/level1_final_report.csv`, generated on 2026-06-24 with
`ZYNUM_MAXIMUM_THREADS` unset and detected thread count 10. Each operation and
library ran in a fresh process with `n=1048576` and one timed second.

Measured below the fastest comparator in that snapshot:

| Operation | Group | Zynum | Best comparator | Ratio |
| --- | --- | ---: | ---: | ---: |
| `scopy` | copy GB/s | 213.032 | 259.149 | 0.822 |
| `sscal` | f32 Gops | 65.127 | 66.938 | 0.973 |
| `scasum` | complex f32 Gops | 64.583 | 66.742 | 0.968 |
| `zdscal` | complex f64 Gops | 21.292 | 23.660 | 0.900 |
| `zscal` | complex f64 Gops | 25.142 | 25.173 | 0.999 |
| `zaxpy` | complex f64 Gops | 18.776 | 19.666 | 0.955 |

Measured at or above the fastest comparator in that snapshot included `dcopy`,
`ccopy`, `zcopy`, `saxpy`, `sdot`, `sasum`, `snrm2`, all real f64 arithmetic
cases, `csscal`, `cscal`, `caxpy`, `caxpby`, `cdotu`, `cdotc`, `scnrm2`,
`zaxpby`, `zdotu`, `zdotc`, `dzasum`, and `dznrm2`.

Treat one-second coverage data as a map of current risk, not a dispatch gate.
For promotion or rollback, rerun the affected operation with longer seconds and
process repeats.

## 2026-06-25 Apple M5 Follow-up

Validation commands:

```sh
zig fmt --check build.zig build.zig.zon src test bench examples tools
zig build --global-cache-dir .zig-cache/global test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
```

Focused copy probes showed that fully disabling the SME byte-copy path regressed
4 MiB and 8 MiB copy cases, while the 16 MiB `zcopy` point benefited from the
system `memcpy` fallback. The retained gate keeps SME byte copy for
`8 KiB <= n_bytes < 16 MiB` and lets larger copies fall through. A narrow
parallel-copy experiment is retained only for `4 MiB <= n_bytes < 8 MiB`, which
targets the default `scopy` report shape without moving the 8 MiB `dcopy` and
`ccopy` points onto the same helper path.

Fresh-process coverage CSV:

```text
zig-out/perf-report/level1_after_final.csv
```

Environment: `ZYNUM_MAXIMUM_THREADS=unset`, detected max threads 10,
`OPENBLAS_DYNAMIC=0`, `OPENBLAS_NUM_THREADS=10`,
`VECLIB_MAXIMUM_THREADS=10`, `OMP_NUM_THREADS=10`.

The final one-second coverage run still had 10/30 operations below the fastest
available comparator. The largest remaining gaps were `scopy`, `dcopy`,
`daxpy`, `caxpy`, and `zaxpy`. Treat the copy gate as a focused cleanup, not as
a claim that Level 1 is fully no-slower-than Accelerate/OpenBLAS.

## 2026-06-29 Cross-Level Semantic Fast-Path Cleanup

Retained changes:

- `scal` and complex `rscal` return immediately for alpha equal to one. This is
  a semantic no-op for every stride and avoids entering vector, architecture, or
  parallel paths unnecessarily.
- `axpby` routes common scalar cases through existing lower-cost kernels:
  alpha zero becomes `scal(beta)`, beta one becomes `axpy(alpha)`, and
  alpha one with beta zero becomes `copy`.

These are operation-family rules, not shape-tuning rules. They apply before
kernel selection and reduce overhead for Level 1 callers and for Level 2/3 paths
that reuse Level 1 helpers.

Validation:

```sh
zig fmt --check build.zig build.zig.zon src test bench examples tools
zig build --global-cache-dir .zig-cache/global -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
zig build --global-cache-dir .zig-cache/global test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
```

## 2026-06-29 AArch64 f64 AXPY Microkernel Follow-up

This pass followed the shared optimization order: add kernel coverage first,
determine feasible ranges, then retain only single-thread routes that improve a
range rather than one shape point.

Kernel coverage:

- Added an AArch64 SME2 streaming `daxpy` microkernel for unit-stride f64.
  It mirrors the retained f32 SME `saxpy` family and updates 256 doubles per
  main block when the streaming vector length is 64 bytes.
- Added an AArch64 SVE `daxpy` candidate, but left it disabled after focused
  runs showed it was slower than both the existing core path and OpenBLAS in
  the tested medium and large ranges.

Feasible range rule:

- The retained SME2 f64 route is gated to
  `64K <= n < 8 MiB / sizeof(f64)` with SVL equal to 64 bytes.
- This gate is expressed as a cache-size rule, not as a matrix-size special
  case. It keeps medium vectors on the SME stream path and lets larger vectors
  fall back to the existing core path once the one-vector footprint reaches
  8 MiB.
- The `size=1024` Level 1/2 sweep case has `n=1048576`, exactly 8 MiB of f64
  data per vector, so it intentionally falls through to the core path.

Single-thread evidence from fresh `vector-matrix-sweep` processes with
`ZYNUM_MAXIMUM_THREADS=1`, `OPENBLAS_DYNAMIC=0`, `OPENBLAS_NUM_THREADS=1`,
`VECLIB_MAXIMUM_THREADS=1`, and `OMP_NUM_THREADS=1`:

| Sweep size | Effective n | Zynum Gops | Accelerate Gops | OpenBLAS Gops | Route |
| ---: | ---: | ---: | ---: | ---: | --- |
| 256 | 65536 | 48.402 | 47.663 | 13.054 | SME2 |
| 512 | 262144 | 38.599 | 39.695 | 11.771 | SME2 |
| 640 | 409600 | 41.393 | 42.742 | 12.062 | SME2 |
| 768 | 589824 | 43.223 | 46.795 | 12.114 | SME2 |
| 896 | 802816 | 28.736 | 29.895 | 13.188 | SME2 |
| 960 | 921600 | 16.923 | 21.474 | 13.069 | SME2 |
| 1024 | 1048576 | 13.049 | 12.007 | 13.111 | core fallback |

Rejected SVE candidate evidence:

- `size=512`: Zynum SVE candidate 10.655 Gops, Accelerate 46.778 Gops,
  OpenBLAS 12.134 Gops.
- `size=1024`: Zynum SVE candidate 12.087 Gops, Accelerate 11.573 Gops,
  OpenBLAS 13.436 Gops.
- `size=2048`: Zynum SVE candidate 10.386 Gops, Accelerate 9.608 Gops,
  OpenBLAS 10.700 Gops.

Conclusion:

- The SME2 route fixes the medium-vector OpenBLAS gap and makes the 256 point
  slightly faster than Accelerate, but it does not yet beat Accelerate across
  all medium sizes.
- The large-vector path remains on the existing core implementation. At
  `size=1024`, the retained route is effectively tied with OpenBLAS and faster
  than Accelerate in the repeated focused run.
- Do not add multi-thread splitting for this medium f64 AXPY route until the
  single-thread kernel itself is no longer the limiting factor. The current
  retained range is still memory-stream dominated, and prior Level 1 work
  showed that broad splitting can erase SME wins.

## 2026-06-30 Microkernel Organization Cleanup

This pass reorganized Level 1 microkernel coverage without making a new
comparator-performance claim.

Retained organization:

- `src/blas/kernels/shared/vector/fixed_simd.zig` is the shared fixed-width SIMD
  skeleton for real copy, swap, scal, axpy, axpby, dot, asum, nrm2, iamax, rot,
  and complex scal, axpy, axpby, and dot.
- AArch64 and x86_64 Level 1 facades configure those skeletons with comptime
  lane and unroll settings before falling back to the core implementation.
- AArch64 SVE and SME inline-assembly bodies now reuse builders in
  `src/blas/kernels/arch/aarch64/asm/builders.zig` for lane suffixes, unroll count,
  complex conjugation, reduction shape, ZA tile load/store shape, and streaming
  prologue/epilogue text. Longer SME streaming `scal`, `asum`, AXPY, and dot
  bodies should be generated from those shared builders instead of duplicated
  per lane.
- AArch64 SVE real AXPY and SME streaming byte-copy assembly are also generated
  from `builders.zig`. The copy path only starts streaming mode; it does
  not enable ZA because the kernel uses streaming vector load/store predicates
  but no ZA state.
- Assembly builders must generate legal addressing for the target ISA. For SVE
  single-vector loads and stores, large `MUL VL` offsets are split through
  temporary base registers instead of emitting invalid offsets.

Validation for the current Apple M5 local target:

```sh
zig fmt --check build.zig build.zig.zon src test bench examples tools
zig build --global-cache-dir .zig-global-cache test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
zig build --global-cache-dir .zig-global-cache test --summary failures
zig build --global-cache-dir .zig-global-cache -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
zig build --global-cache-dir .zig-global-cache -Dtarget=x86_64-linux-gnu -Dcpu=x86_64_v4 --release=fast --summary failures
```

## 2026-07-05 AArch64 Large-Vector Follow-up

Retained changes:

- AArch64 `nrm2` now uses the existing Level 1 partial-norm task split for
  large unit-stride real streams. The gate starts at `512 Ki` real elements and
  uses `min_items_per_task = 128 Ki` with a 10-task cap. This applies to real
  `snrm2`/`dnrm2` and to complex-as-real `scnrm2`/`dznrm2`; the per-task kernel
  still uses the robust scale/ssq algorithm and the final merge uses the
  existing stable partial-norm combiner.
- AArch64 large unit-stride true-complex `caxpy` and `zaxpy` now use the
  existing coarse Level 1 task split from `n >= 512 Ki`. This does not affect
  the small Level 2 column calls because those stay below the task-count gate.

Focused evidence with `ZYNUM_MAXIMUM_THREADS` unset, detected max 10, and
comparator thread env pinned to 10:

| Case | Before Zynum | After Zynum | Best comparator in after run |
| --- | ---: | ---: | ---: |
| `snrm2 n=1048576` | 11.58-11.64 Gops | 32.14-32.95 Gops | 17.70 Gops |
| `dnrm2 n=1048576` | 5.77-5.78 Gops | 17.43-18.46 Gops | 9.30 Gops |
| `scnrm2 n=1048576` | 11.55-11.57 Gops | 33.23-33.28 Gops | 17.89 Gops |
| `dznrm2 n=1048576` | 5.72-5.73 Gops | 21.14-21.75 Gops | 2.39 Gops |
| `caxpy n=1048576` | 47.44-47.65 Gops in full smoke | 84.80-90.05 Gops | 48.17 Gops |
| `zaxpy n=1048576` | 18.48-22.11 Gops in full smoke | 34.98-35.89 Gops | 20.76 Gops |

Evidence CSVs:

- `zig-out/perf-report/level1_copy_nrm2_focus_baseline.csv`
- `zig-out/perf-report/level1_nrm2_parallel_probe.csv`
- `zig-out/perf-report/level1_complex_axpy_parallel_probe.csv`
- `zig-out/perf-report/level1_after_nrm2_parallel.csv`
- `zig-out/perf-report/level1_after_nrm2_complex_axpy.csv`

Rejected experiment:

- Disabling the AArch64 MOPS/`@memcpy` copy candidate did not improve the
  4 MiB `scopy` point and regressed the 16 MiB `zcopy` point. The default copy
  order remains MOPS, then the retained SME streaming-copy gate, then ASIMD or
  libc fallback.

## 2026-07-05 AArch64 Copy and Complex-Scale Follow-up

Retained changes:

- AArch64 unit-stride byte copy now uses the low-latency worker pool for the
  medium copy windows that benefit from fresh helper wakeup. The retained
  windows are `4 MiB <= bytes < 8 MiB` with 512 KiB minimum chunks, and the
  exact 8 MiB boundary with 2 MiB minimum chunks. Subtasks below 8 MiB prefer
  fixed ASIMD copy before MOPS; larger copies still use the existing MOPS path.
- AArch64 parallel real `scal` now uses the low-latency worker pool once the
  existing task gate has selected a parallel path. The 1 Mi scalar `dscal`
  point remains single-kernel, while complex-as-real `zdscal` at `2*n` f64
  benefits from lower helper overhead.
- AArch64 full complex `cscal` and `zscal` now use the existing complex-scale
  task split for `n >= 512 Ki`, capped at 10 tasks. This leaves small Level 2
  column work below the gate and gives the large Level 1 `cscal`/`zscal`
  shapes enough bandwidth headroom over Accelerate.
- AArch64 SVE f32 `asum` uses an unroll of 8 instead of 16. Focused probes
  showed a small improvement for the `scasum` real-view boundary without
  regressing `sasum`.

Focused evidence with `ZYNUM_MAXIMUM_THREADS` unset and comparator thread env
pinned to 10:

| Case | Retained Zynum | Best comparator in same/follow-up probe |
| --- | ---: | ---: |
| `scopy n=1048576` | 354-377 GB/s | 193-290 GB/s |
| `dcopy n=1048576` | 297-334 GB/s | 180-189 GB/s |
| `ccopy n=1048576` | 285-313 GB/s | 127-150 GB/s |
| `zcopy n=1048576` | 107 GB/s copy-only best-of-3 | 105.6 GB/s Accelerate best-of-3 |
| `cscal n=1048576` | 184-190 Gops | 47-50 Gops |
| `zscal n=1048576` | 46-55 Gops | 20-28 Gops |
| `zdscal n=1048576` | 26.7-27.9 Gops | 18-22 Gops |

Report CSVs:

- `zig-out/perf-report/level1_after_copy_8m_special_r3.csv`
- `zig-out/perf-report/level1_after_zscal_parallel_r3.csv`
- `zig-out/perf-report/level1_after_complex_scal_parallel_r3.csv`
- `zig-out/perf-report/level1_after_scal_lowlatency_r3.csv`

Rejected experiments:

- Using MOPS for the 4-8 MiB copy subtasks regressed `dcopy`/`ccopy` versus the
  fixed-ASIMD subtask path.
- Expanding the parallel copy window broadly to include 8 MiB with many small
  chunks regressed `dcopy`/`ccopy`; the retained 8 MiB rule uses four 2 MiB
  chunks only.
- Parallelizing the 16 MiB `zcopy` point with 4 MiB or 2 MiB subtasks regressed
  sharply. The retained 16 MiB path is the original large-copy MOPS/libc-class
  path.
- Routing 8-16 MiB copies through SME streaming copy or libc `memcpy` did not
  beat the retained MOPS/fixed-ASIMD mix.
- Parallelizing the exact 2 Mi f32 `asum` or f32 real `scal` boundary regressed
  `scasum`/`csscal`; those stay on the single-kernel SVE/SME paths.
- Routing `csscal` through the new parallel `ComplexF32` full-complex scale
  path was slower than the existing complex-as-real SME scale path.

The one-second full Level 1 smoke runs remained sensitive to copy/cache/thermal
state, especially `scopy` and near-tie `asum`/`scal` rows. Treat the retained
changes above as focused large-vector rules; use the 2026-07-06 rebaseline below
for the current Apple M5 large-vector status.

## 2026-07-06 Apple M5 Level 1 Rebaseline

Current target build:

```sh
zig build --global-cache-dir .zig-cache/global test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
zig build --global-cache-dir .zig-cache/global -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
zig build-exe bench/level1_probe.zig -O ReleaseFast -target aarch64-macos -mcpu apple_m4+sme+sme2+sme2p1 --global-cache-dir .zig-cache/global -femit-bin=zig-out/perf-report/bin/level1_probe
zig build-exe bench/dcopy_probe.zig -O ReleaseFast -target aarch64-macos -mcpu apple_m4+sme+sme2+sme2p1 --global-cache-dir .zig-cache/global -femit-bin=zig-out/perf-report/bin/dcopy_probe
```

Fresh-process Level 1 report:

```sh
env -u ZYNUM_MAXIMUM_THREADS OPENBLAS_DYNAMIC=0 OPENBLAS_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10 OMP_NUM_THREADS=10 \
  python3 bench/tools/run_level1_report.py \
  --level1-probe zig-out/perf-report/bin/level1_probe \
  --copy-probe zig-out/perf-report/bin/dcopy_probe \
  --zynum zig-out/lib/libzynum_blas.dylib \
  --accelerate /System/Library/Frameworks/Accelerate.framework/Accelerate \
  --openblas /opt/homebrew/opt/openblas/lib/libopenblas.dylib \
  --n 1048576 --seconds 1 --copy-seconds 1 --process-repeats 3 \
  --csv zig-out/perf-report/level1_m5_current_baseline_20260706.csv \
  --skip-missing
```

The full report kept the best fresh-process repeat per library/op. All rows
were at or above the fastest comparator except three near-ties against
Accelerate:

| Operation | Metric | Zynum | Accelerate | Ratio |
| --- | --- | ---: | ---: | ---: |
| `zcopy` | GB/s | 115.424 | 115.709 | 0.9975 |
| `dasum` | Gops | 33.363 | 33.370 | 0.9998 |
| `scasum` | Gops | 66.558 | 66.695 | 0.9979 |

Focused follow-up with longer processes did not show a stable Level 1 gap:

| Probe | Zynum best/median | Accelerate best/median | Notes |
| --- | ---: | ---: | --- |
| `zcopy`, 5x2s | 115.618 / 114.352 GB/s | 115.154 / 111.526 GB/s | Zynum exceeded Accelerate best and median. |
| `dasum`, 10x3s | 33.408 / 33.367 Gops | 33.429 / 33.357 Gops | Median slightly favored Zynum; best differed by 0.06%. |
| `scasum`, 10x3s | 66.790 / 66.686 Gops | 66.769 / 66.634 Gops | Zynum exceeded Accelerate best and median. |

Additional diagnostic files:

- `zig-out/perf-report/level1_m5_focus_zcopy_dasum_scasum_20260706.csv`
- `zig-out/perf-report/level1_m5_focus_dasum_10x3s_20260706.csv`
- `zig-out/perf-report/level1_m5_focus_scasum_10x3s_20260706.csv`
- `/tmp/zynum_scasum_sample.txt`

Sampling the Zynum `scasum` focused run placed 4235 of 4257 samples in
`kernels.arch.aarch64.vector.unary.smeSasumF32StreamingBits`, with only small
wrapper samples. The near-tie is therefore the f32 absolute-sum kernel body and
memory/cache state, not ABI dispatch, thread scheduling, or SM state transition
overhead. The previously rejected exact 2 Mi f32 `asum` parallel split remains
rejected; the current evidence does not justify adding a new Level 1 dispatch
rule.

For the local Apple M5 large-vector Level 1 shape set covered by this runner,
there is no stable remaining slower-than-Accelerate/OpenBLAS gap. Keep `zcopy`,
`dasum`, and `scasum` on the watch list for future README refreshes, but move
the main local effort to Level 2 before revisiting Level 1.

## Next Priorities

1. Move the main effort to Level 2 GEMV/GER/HEMV. Current Level 1 focused probes
   no longer show a stable large-vector gap, while Level 2 still has broad
   128/256/512 GEMV and complex GER/HEMV gaps.
2. Keep `zcopy`, `dasum`, and `scasum` on the watch list during README-quality
   refreshes because one-second best-repeat reports can still show sub-0.3%
   comparator outliers. Do not add a dispatch rule for these without a mechanism
   beyond normal cache/thermal variance.
3. For any future f32 `asum`/`scal` boundary work, avoid broad task splitting.
   The exact 2 Mi f32 `asum` and `scal` experiments both regressed.

## Retention Checklist

Before retaining a new Level 1 optimization:

- Add or update correctness tests for all affected scalar types and ABI entry
  points.
- Include unit, positive non-unit, and negative stride cases if the optimized
  path changes dispatch around stride handling.
- Run `zig build test` on the target feature set.
- Run focused probes against Zynum, Accelerate, and OpenBLAS in fresh processes.
- For complex scalar changes, include both Fortran and CBLAS ABI coverage.
- Record raw CSV and metadata paths.
- Prefer operation-grouped bar charts for categorical Level 1 summaries.
- Document any operation that remains below the fastest comparator, including
  the exact ratio and next investigation step.
