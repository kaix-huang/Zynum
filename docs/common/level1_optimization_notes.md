# Level 1 Optimization Notes

This document records BLAS Level 1 performance lessons for Zynum. It is meant
to guide future tuning work across real, complex, ABI, and benchmark paths.
Architecture-specific instruction details still belong in the matching
architecture notes when they become permanent dispatch rules.

## Ownership

Current Level 1 performance code is split across:

- `src/blas/core/level1.zig`: BLAS semantics, strides, complex behavior,
  contiguous fast-path dispatch, portable Zig vector loops, tail handling, and
  coarse parallel splitting.
- `src/blas/core/pool.zig`: shared `std.Io.Threaded` task runner for large
  contiguous Level 1 work.
- `src/blas/kernels/vector_unary.zig`: architecture facade for one-vector
  operations such as `scal` and `asum`.
- `src/blas/kernels/vector_binary.zig`: architecture facade for two-vector
  operations such as `copy`, `axpy`, and `dot`.
- `src/blas/kernels/aarch64/vector_unary.zig`: AArch64 SVE/SME candidates for
  one-vector work.
- `src/blas/kernels/aarch64/vector_binary.zig`: AArch64 SVE/SME candidates for
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
zig build test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme2p1 --release=fast --summary failures
```

Build the focused probes:

```sh
zig build-exe bench/level1_probe.zig -OReleaseFast \
  -target aarch64-macos -mcpu apple_m4+sme2p1 \
  --global-cache-dir .zig-cache/global \
  -femit-bin=zig-out/perf-report/bin/level1_probe

zig build-exe bench/dcopy_probe.zig -OReleaseFast \
  -target aarch64-macos -mcpu apple_m4+sme2p1 \
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
  --process-repeats 1 \
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
zig build --global-cache-dir .zig-cache/global test -Dtarget=aarch64-macos -Dcpu=apple_m4+sme2p1 --release=fast --summary failures
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

## Next Priorities

1. `scopy`: byte copy is generally good, but f32-sized copy lagged Accelerate in
   the coverage run. Investigate small element-count/byte-count interaction,
   warmup behavior, and whether the SME copy block size or fallback threshold
   should differ for 4-byte element entry points.
2. `zdscal`: real-alpha complex f64 scaling reuses real `dscal` over `2*n`
   values, but the coverage run lagged Accelerate. Check whether `2*n` crosses a
   threshold that chooses a worse path, and compare real `dscal --n 2097152`.
3. `zaxpy`: complex f64 alpha path is close to Accelerate and noisy. Run longer
   isolated repeats before changing code. If it remains low, compare the
   interleaved shuffle/sign loop against a dedicated SVE `ld2d` update kernel.
4. `sscal` and `scasum`: both are within a few percent of Accelerate. Do not
   broaden parallelism blindly; prior experiments showed f32 splitting can
   slow these paths. Prefer focused SVE/SME single-thread micro-kernel work.
5. `zscal`: ratio is effectively tied. Keep it on the watch list, but avoid a
   new specialized path unless longer repeated runs show a stable gap.

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
