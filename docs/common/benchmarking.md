# Benchmarking

BLAS benchmarks are sensitive to target features, thermal state, thread pools,
cache history, and comparator library defaults. Treat benchmark results as
evidence for a specific change under recorded conditions, not as a portable
guarantee. Do not quote speedups or regressions unless the command, runtime
environment, target, raw data, and isolation level are reproducible.

## Rules

1. Run correctness tests for the target before performance tests.
2. Inspect the per-row correctness field before interpreting timing. For Level
   1/2 reports this is `check_status`; for GEMM sweeps it is `check`.
   `sampled-ok` and `checked-ok` are valid performance evidence. Rows marked
   `correctness_failed`, `error`, `missing`, unchecked, or unknown are
   diagnostics only.
3. Use the same shapes, repetitions, initialization, warmup, and timing method
   for every library in a comparison.
4. For default Zynum runs, leave `ZYNUM_MAXIMUM_THREADS` unset and record the
   detected max; set it only for explicit cap or single-thread runs. Pin
   comparator thread counts and dynamic-threading policy.
5. Record target tuple, CPU, target features, OS, Zig version, source revision,
   command, runtime environment, raw CSV path, summary, and isolation level.
6. Treat in-process sweeps as smoke checks unless no comparator libraries are
   loaded.
7. Use fresh-process isolation for reportable comparator data; prefer level 2
   and use level 3 for outliers or dispatch-gate evidence.
8. Re-test outliers in isolation before using them to justify a gate.
9. Keep optimization gates narrower than the measured evidence. Do not retain a
   default gate from one lucky focused run.
10. Discard timing data from any run with possible numerical pollution. If a
   kernel later proves to have missing register preservation, stale scalar
   return state, incorrect hardware-state cleanup, or invalid inline-assembly
   memory ordering, rerun every affected benchmark before using it for either a
   promotion or a rollback.

## Evidence Chain

A benchmark-backed performance change should leave an inspectable chain:

1. Correctness tests for the exact target or the closest available target.
2. Focused single-shape runs for the candidate shape class.
3. A full sweep covering square, rectangular, tall/narrow, short/wide, and high-K
   shapes.
4. Fresh-process comparator data when claiming relative performance against
   Accelerate, OpenBLAS, MKL, or another BLAS.
5. A retained dispatch rule or shape gate that cites the evidence and is narrower
   than the measured region.

If any link is missing, label the result as exploratory. Cross-compilation proves
build coverage, not throughput.

## Runtime Environment

Record every runtime variable that can affect Zynum's thread cap or comparator
behavior. Record variables that are intentionally unset as `unset`.

| Variable | Applies to | Record because |
| --- | --- | --- |
| `ZYNUM_MAXIMUM_THREADS` | Zynum | Maximum number of threads Zynum may use; values above the runtime CPU count are capped to that count; unset means runtime CPU count. |
| `OPENBLAS_NUM_THREADS` | OpenBLAS | Comparator thread count. |
| `OPENBLAS_DYNAMIC` | OpenBLAS | Dynamic thread-count policy. |
| `VECLIB_MAXIMUM_THREADS` | Accelerate/vecLib | Comparator thread limit on macOS. |
| `MKL_NUM_THREADS` | Intel MKL | Comparator thread count. |
| `MKL_DYNAMIC` | Intel MKL | Dynamic thread-count policy. |
| `OMP_NUM_THREADS` | OpenMP-based comparators | Thread count when a comparator uses OpenMP. |
| `BLIS_NUM_THREADS` | BLIS/AOCL-BLIS | Comparator thread count when a BLIS-family comparator honors it. |

Recommended single-process smoke baseline:

```sh
export OPENBLAS_DYNAMIC=0
export MKL_DYNAMIC=FALSE
```

For default reportable Apple Silicon comparisons, leave
`ZYNUM_MAXIMUM_THREADS` unset and record the detected Zynum max thread count in
the metadata. Pin comparator library thread settings explicitly so their policy
is not left to lazy runtime defaults:

```sh
unset ZYNUM_MAXIMUM_THREADS
export OPENBLAS_NUM_THREADS=10
export VECLIB_MAXIMUM_THREADS=10
export MKL_NUM_THREADS=10
export OMP_NUM_THREADS=10
```

Set `ZYNUM_MAXIMUM_THREADS` only for explicit cap experiments or single-thread
verification runs, and label those runs separately from default-thread gates.

## Quick Benchmark

```sh
zig build bench --release=fast -- --size 1024 --reps 10
```

Pass comparator libraries when defaults are not available:

```sh
zig build bench --release=fast \
  -Dbench-openblas=/path/to/libopenblas.dylib \
  -Dbench-accelerate=/System/Library/Frameworks/Accelerate.framework/Accelerate \
  -Dbench-mkl=/path/to/libmkl_rt.so \
  -Dbench-aocl-blis=/path/to/libblis-mt.so \
  -- --size 1024 --reps 10
```

## GEMM Sweep

The build step below is a single-process smoke check. It is useful for validating
the tool and plotting path, but not for published comparator claims:

```sh
zig build bench-gemm-sweep --release=fast -- --reps 30
python3 bench/tools/plot_gemm_sweep.py zig-out/gemm_sweep.csv zig-out/gemm_sweep.svg
```

GEMM sweep CSV rows keep `best_ns` and best-based `gflops` for continuity, but
also report `median_ns`, `p95_ns`, and `max_ns`. Use the distribution fields when
deciding whether a gate is robust; do not promote a dispatch rule from a lone
best-time win. Correctness-checked rows use `check=checked-ok`; older archived
CSV files may use `sampled-ok`.

When the isolated runner uses `--process-repeats`, the merged CSV adds
`process_repeats` and combines the per-process timing summaries instead of
discarding every process except the fastest one.

Use the CSV checker to turn a comparator sweep into a pass/fail gate. The
strict form requires Zynum to be at least as fast as the fastest requested
comparator for every selected `(kind, shape)` group:

```sh
python3 bench/tools/check_gemm_sweep.py zig-out/gemm_sweep.csv
```

Pass explicit comparator labels when checking a non-default set such as
MKL and AOCL-BLIS:

```sh
python3 bench/tools/check_gemm_sweep.py zig-out/gemm_sweep.csv \
  --comparator MKL \
  --comparator AOCL-BLIS
```

Use `--ratio 0.98` only when the benchmark note explicitly accepts a 2%
measurement tolerance. The default `--ratio 1.0` is the no-slower-than gate.

Use custom shapes for focused testing:

```sh
zig build bench-gemm-sweep --release=fast -- \
  --kind sgemm \
  --shape sq128:128:128:128 \
  --shape m128_n128_k4096:128:128:4096 \
  --reps 100
```

## Rank-K And Rank-2K Comparator Sweep

The Level 3 rank-k tooling is opt-in and does not change the default benchmark
or build steps. It covers `ssyrk`, `dsyrk`, `csyrk`, `zsyrk`, `cherk`, `zherk`,
`ssyr2k`, `dsyr2k`, `csyr2k`, `zsyr2k`, `cher2k`, and `zher2k` through
dynamically loaded Fortran BLAS symbols. Build the probe, then run the
controller with explicit comparator paths as needed:

```sh
zig build build-rank-k-probe --release=fast

python3 bench/tools/run_rank_k_report.py \
  --probe zig-out/bin/rank-k-probe \
  --zynum zig-out/lib/libzynum_blas.so \
  --mkl /path/to/libmkl_rt.so \
  --openblas /path/to/libopenblas.so \
  --aocl-blis /path/to/libblis-mt.so \
  --extra-blas Upstream-BLIS=/path/to/libblis.so \
  --shape n128_k32:128:32 \
  --shape n128_k512:128:512 \
  --process-repeats 3 \
  --csv zig-out/perf-report/rank_k_broad.csv \
  --skip-missing
```

With no routine, transpose, or triangle filters, the runner expands both `U`
and `L` and every legal transpose: `N/T` for SYRK/SYR2K and `N/C` for
HERK/HER2K. The four default shapes, 12 routines, two triangles, and two legal
transpose modes produce 192 logical groups per library for one alpha/beta pair.
Repeat `--alpha RE[,IM]` and `--beta RE[,IM]` to sweep scalars. Complex alpha
and beta are valid for CSYRK/ZSYRK/CSYR2K/ZSYR2K. HERK uses real alpha and beta;
HER2K uses complex alpha and real beta, matching the Fortran ABI. Each CSV row
records the complete routine, shape, storage, transpose, scalar, `lda`, optional
rank-2k `ldb`, `ldc`, timing, and correctness parameters.

Every library/case/process repeat runs in a fresh process. The probe checks the
result against an independent scalar reference before timing. Rank-2k cases
allocate and initialize A and B independently. The SYR2K reference evaluates
both symmetric products; the HER2K reference evaluates the alpha and
conjugate-alpha products separately. Small cases check the full matrix and
larger cases check deterministic samples. Both modes require the unstored
triangle to remain bitwise unchanged and require Hermitian diagonal values to
be real. Any incorrect or failed repeat makes the aggregate row ineligible even
if another repeat was fast. The process samples use each probe's median timing,
and the checker compares their median by default:

```sh
python3 bench/tools/check_rank_k_report.py \
  zig-out/perf-report/rank_k_broad.csv \
  --comparator MKL \
  --comparator OpenBLAS
```

`zig build bench-rank-k-report --release=fast -- <runner arguments>` builds the
probe and Zynum shared library and runs the same opt-in controller. Use
`--stat best` or `--stat min` only for an explicitly documented diagnostic;
the default `--stat median` is the rank-k promotion gate.

## SYMM/HEMM Comparator Sweep

The opt-in SYMM/HEMM tooling covers `ssymm`, `dsymm`, `csymm`, `zsymm`,
`chemm`, and `zhemm` through dynamically loaded Fortran BLAS symbols. Its
default broad set is 72 logical groups: `square128`, `tall512x128`, and
`wide128x512`, all six routines, both `L/R` sides, and both `U/L` triangles.
Real routines use real default scalars; complex routines use complex alpha and
beta. Build and run it with explicit x86 comparators as needed:

```sh
zig build build-symm-probe --release=fast

python3 bench/tools/run_symm_report.py \
  --probe zig-out/bin/symm-probe \
  --zynum zig-out/lib/libzynum_blas.so \
  --mkl /path/to/libmkl_rt.so \
  --openblas /path/to/libopenblas.so \
  --aocl-blis /path/to/libblis-mt.so \
  --extra-blas Upstream-BLIS=/path/to/libblis.so \
  --process-repeats 3 \
  --csv zig-out/perf-report/symm_broad.csv \
  --skip-missing
```

Every process first checks all `m*n` output elements against an independent
scalar reference. The probe puts unrelated values in A's unstored triangle and
nonzero imaginary values on Hermitian diagonals; the reference reconstructs A
only from the stored triangle and treats Hermitian diagonals as real. This
checks the structured-input contract as well as the complete C result. The
short worker leaves the dynamically loaded BLAS mapped until process exit so a
threaded BLAS is not explicitly unloaded before its runtime destructors.

The controller uses one fresh process per library, complete case, and process
repeat. Any bad repeat makes the aggregate ineligible. The checker gates on the
median of per-process median throughput by default:

```sh
python3 bench/tools/check_symm_report.py \
  zig-out/perf-report/symm_broad.csv \
  --comparator MKL \
  --comparator OpenBLAS
```

`zig build bench-symm-report --release=fast -- <runner arguments>` builds the
probe and shared library before running the controller. Repeat `--shape
NAME:M:N`, `--alpha RE[,IM]`, or `--beta RE[,IM]` for focused coverage; real
routines ignore explicitly requested complex scalar values and require at
least one real value.

## TRMM/TRSM Comparator Sweep

The opt-in triangular matrix tooling covers `strmm`, `dtrmm`, `ctrmm`,
`ztrmm`, `strsm`, `dtrsm`, `ctrsm`, and `ztrsm` through dynamically loaded
Fortran BLAS symbols. The default broad set has 480 logical groups over
`square128`, `tall512x128`, and `wide128x512`; both `L/R` sides, `U/L`
triangles, and `N/U` diagonal modes; and `N/T` operations plus `C` for complex
routines. Build the probe and run explicit x86 comparators with:

```sh
zig build build-triangular-matrix-probe --release=fast

python3 bench/tools/run_triangular_matrix_report.py \
  --probe zig-out/bin/triangular-matrix-probe \
  --zynum zig-out/lib/libzynum_blas.so \
  --mkl /path/to/libmkl_rt.so \
  --openblas /path/to/libopenblas.so \
  --aocl-blis /path/to/libblis-mt.so \
  --atlas /path/to/libatlas.so \
  --extra-blas Upstream-BLIS=/path/to/libblis.so \
  --process-repeats 3 \
  --csv zig-out/perf-report/triangular_matrix_broad.csv \
  --skip-missing
```

TRMM and TRSM overwrite B, so the probe restores the same initialized input
before every timed call. Before timing, it checks every output element against
an independent scalar reference: TRMM evaluates the triangular product, while
TRSM performs its own left- or right-side forward/back substitution. Unstored
A values and stored unit-diagonal values are deliberately unrelated poison and
must not affect the reference or result. The CSV records the complete family,
routine, shape, side, triangle, transpose, diagonal, scalar, leading
dimensions, timing distribution, and correctness fields.

Every library, complete case, and process repeat runs in a fresh process. Any
bad repeat makes the aggregate ineligible, and the default checker gate uses
the median of per-process median throughput:

```sh
python3 bench/tools/check_triangular_matrix_report.py \
  zig-out/perf-report/triangular_matrix_broad.csv \
  --comparator MKL \
  --comparator OpenBLAS
```

`zig build bench-triangular-matrix-report --release=fast -- <runner
arguments>` builds the probe and shared library before invoking the same
controller. Repeat `--routine`, `--shape NAME:M:N`, `--side`, `--uplo`,
`--trans`, `--diag`, or `--alpha RE[,IM]` for focused coverage.

## Level 1/2 Sweep

Use `bench-vector-matrix-sweep` for focused Level 1 and Level 2 tuning gates. The
benchmark exercises contiguous f64 BLAS ABI paths including copy, scal, axpy,
dot, asum, nrm2, GEMV, SYMV, and GER:

```sh
zig build bench-vector-matrix-sweep -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast -- --size 1024 --reps 120
```

Run the SME-targeted AArch64 sweep only on hosts where the requested SME feature
set is executable. On other Apple Silicon hosts, use the same feature string as
compile coverage and run benchmark binaries built for the host's
runtime-supported target.

The `--size` value is the matrix dimension for Level 2 cases. Level 1 cases use
`size * size` elements so the same sweep stresses comparable memory footprints;
for example `--size 1024` measures Level 1 vectors with 1,048,576 f64 elements,
not 1024 elements.

Use `--case <name>` to isolate a single kernel while tuning noisy paths such as
`dgemv_t` or `dger`:

```sh
zig-out/bin/vector-matrix-sweep --zynum-blas zig-out/lib/libzynum_blas.dylib --size 1024 --reps 240 --case dgemv_t
```

For reportable comparator data, run one library per process by passing the
library under test as `--zynum-blas`; the tool label is reused, so record the
path in the benchmark notes:

```sh
zig-out/bin/vector-matrix-sweep --zynum-blas zig-out/lib/libzynum_blas.dylib --size 1024 --reps 120
zig-out/bin/vector-matrix-sweep --zynum-blas /System/Library/Frameworks/Accelerate.framework/Accelerate --size 1024 --reps 120
zig-out/bin/vector-matrix-sweep --zynum-blas /opt/homebrew/opt/openblas/lib/libopenblas.dylib --size 1024 --reps 120
```

The command-line benchmark tools can label MKL and AOCL-BLIS separately when the
libraries export the standard Fortran BLAS symbols. LIBXSMM is not a drop-in
Fortran BLAS comparator for these tools; use a documented shim or a separate
LIBXSMM-specific runner before including it in a no-slower-than gate.

Use `run_level2_report.py` for fresh-process rectangular Level 2 coverage. Keep
`--n` for square sizes and repeat `--shape NAME:M:N` for general matrix shapes:

```sh
python3 bench/tools/run_level2_report.py \
  --zynum zig-out/lib/libzynum_blas.so \
  --shape tall4096x256:4096:256 \
  --shape wide256x4096:256:4096 \
  --process-repeats 3 \
  --csv zig-out/perf-report/level2_rectangular.csv
```

Rectangular shapes cover GEMV N/T, complex GEMV C, and GER/GERU/GERC. SYMV and
HEMV are square-only and are omitted unless `m == n`. New CSVs record `shape`,
`m`, and `n`; the checker remains compatible with legacy square CSVs. With
`--process-repeats N`, every library/shape pair runs in `N` independent worker
processes. The historical primary metric remains the best repeat, while
`metric_min`, `metric_median`, `metric_max`, and `metric_samples` retain the
distribution. Use `check_level2_report.py --stat median` for noisy promotion
decisions; the default `--stat best` preserves old-report behavior. Any failed,
missing, or incorrect repeat makes the aggregate row ineligible for a gate.

Dense TRMV/TRSV coverage is opt-in so existing report commands retain the
historical case set. `--op triangular` expands s/d/c/z TRMV and TRSV across
upper/lower storage, unit/non-unit diagonal, and `incx=1`. Real routines use
N/T, while complex routines use N/T/C. This produces 80 logical groups per
shape; `--op trmv` or `--op trsv` selects one four-type family. Use square
shapes explicitly for the broad 128/512/2048 sweep:

```sh
python3 bench/tools/run_level2_report.py \
  --zynum zig-out/lib/libzynum_blas.so \
  --shape sq128:128:128 \
  --shape sq512:512:512 \
  --shape sq2048:2048:2048 \
  --op triangular \
  --process-repeats 3 \
  --csv zig-out/perf-report/level2_triangular.csv
```

Each triangular row records `uplo`, `trans`, `diag`, and `incx` and runs its own
scalar reference check before timing. The complex C reference conjugates every
selected matrix value; unstored values and stored unit-diagonal values do not
participate. TRSV uses a safely conditioned real or complex triangular matrix
and a right-hand side generated from a known solution. The checker includes all
four triangular parameters in its group key and accepts matching filters such
as `--uplo U --trans C --diag N --incx 1`.

Level 1/2 architecture gates must be retained only when the focused sweep shows
repeatable improvement over the shared Zig vector fallback. Keep rejected ASIMD,
SVE, SME, or AMX candidates disabled behind internal predicates rather than
leaving a slower path active by default.

Level 1/2 low-latency threading changes need extra scrutiny. Some retained paths
use process-lifetime `std.Io.Threaded` helpers and per-helper futex publication,
so a single isolated process can still contain warm helper state after the first
measured repetition. For short paths such as DGER 128, keep the final evidence as
multiple fresh processes per library, not only one process with many repetitions.
Record both the best retained sample and nearby slow outliers when they affect
the dispatch decision.

When adding a full-call Level 2 kernel that bypasses core beta scaling or task
splitting, such as an SME2 GEMV-N or GEMV-T kernel, record both the focused
comparator result and the reason the full-call gate belongs before the shared
parallel path. Sampling, LLDB disassembly, and comparator traces are acceptable
supporting evidence, but the retained gate must still be narrower than the
benchmarked shapes and ISA assumptions.

## Fresh-Process Sweep

Use the isolated runner for reportable comparator data:

```sh
python3 bench/tools/run_gemm_sweep_isolated.py \
  --gemm-sweep zig-out/bin/gemm-sweep \
  --zynum-blas zig-out/lib/libzynum_blas.dylib \
  --csv zig-out/gemm_sweep_isolated.csv \
  --reps 30 \
  --process-repeats 3 \
  --check
```

Isolation levels:

- Level 0: one process loads all libraries. Use only for smoke checks and tool
  validation.
- Level 1: one process loads all libraries for quick local comparisons. Use this
  for smoke checks, not for published comparator claims.
- Level 2: one fresh process per library. Use this as the default reportable
  comparator level.
- Level 3: one fresh process per library, kind, and shape. Use this for outliers,
  shape-gate promotion, and cases where cache or worker history may explain a
  result.

Zynum GEMM and selected Level 1/2 kernels may keep process-lifetime
`std.Io.Threaded` worker state or cached temporary workspaces after the first
parallel call. Comparator libraries may also initialize their own dispatch or
worker state lazily. Do not rely on single-process mixed-library sweeps for
published comparator numbers; use fresh-process isolation instead.

When a comparator appears faster only in occasional low-tail samples, repeat the
same library/path in several fresh processes before changing a gate. The 2026
Level 2 DGER work showed that thread placement, helper warm state, and comparator
worker policy can move 128x128 timings by more than the intended optimization
margin.

Thread placement experiments must record the platform mechanism. Linux affinity
mask changes should include the inherited cpuset and the exact helper-pinning
policy. Apple Silicon experiments should record QoS, `hw.perflevel*` topology,
and whether Mach affinity tags were probed as supported; do not describe macOS
results as CPU-pinned unless a public API and trace actually prove that.

A reportable isolated run should include process repeats, the exact CSV path, and
the per-library command line. Focused reruns should use the same environment as
the full sweep unless the difference is explicitly part of the experiment.

## ROTG And ROTMG Scalar Latency

The Level 1 scalar generators use a dedicated latency tool instead of the
large-vector Gops report. It covers `srotg`, `drotg`, `crotg`, `zrotg`,
`srotmg`, and `drotmg` over zero inputs, ordinary magnitude ratios, extreme
exponents, and the principal ROTMG flag branches. Build and run it with:

```sh
zig build build-rotg-latency-probe --release=fast

python3 bench/tools/run_rotg_latency_report.py \
  --probe zig-out/bin/rotg-latency-probe \
  --zynum zig-out/lib/libzynum_blas.so \
  --mkl /path/to/libmkl_rt.so \
  --openblas /path/to/libopenblas.so \
  --aocl-blis /path/to/libblis-mt.so \
  --extra-blas Upstream-BLIS=/path/to/libblis.so \
  --process-repeats 3 \
  --csv zig-out/perf-report/rotg_latency_broad.csv \
  --skip-missing
```

Every library, routine, corpus case, and process repeat runs in a fresh process.
The probe checks every corpus member against an independent scalar reference
before timing. Each timing sample pairs a full batch with a baseline batch that
performs the same parameter reset, loop, and output-bit consumption but omits
the dynamic BLAS call; AB/BA order alternates, and the CSV reports the paired
difference as `ns/call`. Any incorrect or nonpositive pair makes that repeat
ineligible, and any bad repeat contaminates the aggregate row.

The checker first rejects bad `status` or `check_status` rows, then compares the
median of per-process median `ns/call` values. Lower is better:

```sh
python3 bench/tools/check_rotg_latency_report.py \
  zig-out/perf-report/rotg_latency_broad.csv \
  --comparator MKL \
  --comparator OpenBLAS
```

`zig build bench-rotg-latency-report --release=fast -- <runner arguments>`
builds the probe and shared library before invoking the same controller.

## README Performance Charts

The README performance charts are curated documentation assets. Refresh them
only from fresh-process reports, and keep the chart convention stable:

- Every chart must visibly state `Higher is better`.
- Library order must be `Zynum`, `Accelerate`, then `OpenBLAS`.
- Level 1 and Level 2 charts must include real and complex f32/f64 coverage, not
  only double-precision cases.
- Level 3 must include SGEMM, DGEMM, CGEMM, and ZGEMM across the default GEMM
  sweep shapes unless the README caption explicitly says otherwise.
- Remove old performance SVGs from `docs/assets/benchmarks/` before copying the
  refreshed current charts, so the README does not drift between stale images.

On the Apple Silicon development machine used for the current README snapshot,
build the benchmark artifacts with the measured target first:

```sh
zig build -Dtarget=aarch64-macos -Dcpu=apple_m4+sme+sme2+sme2p1 --release=fast --summary failures
zig build-exe bench/level1_probe.zig -O ReleaseFast -target aarch64-macos -mcpu apple_m4+sme+sme2+sme2p1 --global-cache-dir .zig-cache/global -femit-bin=zig-out/perf-report/bin/level1_probe
zig build-exe bench/dcopy_probe.zig -O ReleaseFast -target aarch64-macos -mcpu apple_m4+sme+sme2+sme2p1 --global-cache-dir .zig-cache/global -femit-bin=zig-out/perf-report/bin/dcopy_probe
```

Run with `ZYNUM_MAXIMUM_THREADS` unset unless the benchmark note explicitly
states a single-thread experiment. Pin comparator library thread settings:

```sh
env OPENBLAS_DYNAMIC=0 OPENBLAS_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10 OMP_NUM_THREADS=10 \
  python3 bench/tools/run_level1_report.py \
  --level1-probe zig-out/perf-report/bin/level1_probe \
  --copy-probe zig-out/perf-report/bin/dcopy_probe \
  --csv zig-out/perf-report/level1_all_types_three_libs.csv \
  --process-repeats 3 \
  --skip-missing
```

Level 1 report CSV copy rows use the concrete `scopy`/`dcopy`/`ccopy`/`zcopy`
operation name. For these rows, `n` is the function element count,
`copy_bytes` is the measured byte footprint, and `copy_kind` records the
selected `s/d/c/z` ABI family. The default copy sweep covers small-cache sizes
and the 4-16 MiB dispatch/cache boundaries; pass `--copy-byte-size <size>` more
than once to replace the default set, or `--copy-only` for a focused copy
report.

For a negative-stride vector sweep, pass `--skip-copy-byte-coverage`. This
disables only the independent byte-size COPY sweep; the signed-stride
`scopy`/`dcopy`/`ccopy`/`zcopy` ABI cases remain in the Level 1 operation set.
Without this flag, the report also contains an unrelated unit-stride COPY row
that must not be merged into the signed-stride evidence.

```sh
python3 bench/tools/plot_level1_report.py \
  zig-out/perf-report/level1_all_types_three_libs.csv \
  --bars-svg zig-out/perf-report/level1_all_types_three_libs_bars.svg \
  --ratio-svg zig-out/perf-report/level1_all_types_three_libs_ratio.svg

python3 bench/tools/check_level1_report.py \
  zig-out/perf-report/level1_all_types_three_libs.csv

env OPENBLAS_DYNAMIC=0 OPENBLAS_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10 OMP_NUM_THREADS=10 \
  python3 bench/tools/run_level2_report.py \
  --csv zig-out/perf-report/level2_all_types_three_libs.csv \
  --skip-missing

python3 bench/tools/plot_level2_report.py \
  zig-out/perf-report/level2_all_types_three_libs.csv \
  --bars-svg zig-out/perf-report/level2_all_types_three_libs_bars.svg

env OPENBLAS_DYNAMIC=0 OPENBLAS_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10 OMP_NUM_THREADS=10 \
  python3 bench/tools/run_gemm_sweep_isolated.py \
  --gemm-sweep zig-out/bin/gemm-sweep \
  --zynum-blas zig-out/lib/libzynum_blas.dylib \
  --reps 6 \
  --process-repeats 3 \
  --check \
  --isolate-kind \
  --csv zig-out/perf-report/level3_all_types_more_shapes_three_libs.csv \
  --skip-missing

python3 bench/tools/plot_gemm_sweep.py \
  zig-out/perf-report/level3_all_types_more_shapes_three_libs.csv \
  zig-out/perf-report/level3_all_types_more_shapes_three_libs_lines.svg
```

When `--process-repeats` is greater than one, the Level 1 CSV keeps the best
repeat in the primary `rate_gops` or `bandwidth_gbps` field for compatibility
and also records `metric_min`, `metric_median`, `metric_max`, and the ordered
`metric_samples`. Use the same statistic for Zynum and every comparator. The
default checker preserves the historical best-repeat gate; use `--stat median`
to reject a route whose apparent win depends on one lucky process, or
`--stat min` only for an explicitly documented low-tail stability gate.

Pass explicit comparators when checking an x86 report that used MKL and OpenBLAS:

```sh
python3 bench/tools/check_level1_report.py \
  zig-out/perf-report/level1_h3c.csv \
  --comparator MKL \
  --comparator OpenBLAS
```

Then replace the checked-in README image assets:

```sh
find docs/assets/benchmarks -maxdepth 1 -type f -name '*.svg' -delete
cp zig-out/perf-report/level1_all_types_three_libs_bars.svg docs/assets/benchmarks/current_level1_all_types_three_libs.svg
cp zig-out/perf-report/level2_all_types_three_libs_bars.svg docs/assets/benchmarks/current_level2_all_types_three_libs.svg
cp zig-out/perf-report/level3_all_types_more_shapes_three_libs_lines.svg docs/assets/benchmarks/current_level3_all_types_more_shapes.svg
```

Keep CSV and metadata files under `zig-out/perf-report/` unless a release note
explicitly links them as external artifacts. The repository should normally track
only the three curated SVG files under `docs/assets/benchmarks/`.

### Cross-Level H3C Report

After every independent SLURM family has finished, collect its CSV and metadata
files into one directory and render the synchronized x86 report with the
canonical comparator set:

```sh
python3 bench/tools/render_full_benchmark_report.py \
  --input-dir zig-out/perf-report/full-YYYYMMDD \
  --output-dir zig-out/perf-report/full-YYYYMMDD-report \
  --comparator MKL \
  --comparator OpenBLAS \
  --comparator AOCL-BLIS \
  --comparator ATLAS \
  --comparator Upstream-BLIS \
  --expected-process-repeats 3
```

The renderer emits seven SVGs plus `index.html`, `summary.csv`, and
`summary.json`. It covers vector Level 1, ROTG/ROTMG scalar latency, Level 2,
GEMM, rank-k/rank-2k, SYMM/HEMM, and TRMM/TRSM. Throughput ratios are Zynum
divided by the fastest eligible comparator; scalar-latency ratios invert that
formula so higher is still better. Every SVG shows the 1.0 gate and all
available library medians in native metric units as an upper all-library line
chart plus a lower per-case grouped bar chart. Level 1 aggregate rows may carry
`successful_repeats` without `process_repeats`; GEMM rows may carry
`process_repeats` with `reps`, and both schemas are validated separately.

The renderer is a final evidence filter, not a replacement for the family
checkers. It rejects unchecked or incorrect rows, nonpositive metrics, a
process-repeat mismatch, and rows whose recorded successful repeat count is
incomplete. A requested comparator that is missing or fails correctness is
reported explicitly. Retain every per-family checker output so an aggregate
failure can still be traced to the exact logical case and raw CSV.

## Default Sweep Shape Classes

The default sweep includes:

- Small and medium square matrices.
- Tall/narrow matrices.
- Short/wide matrices.
- High-K shapes.
- Medium rectangular shapes.

These shape classes expose different costs: thread startup, B packing, row or
column splitting, cache pressure, and comparator-library dispatch policy. Name
custom shapes consistently so a focused run can be matched back to a full sweep.

## Shape Gate And Dispatch Rule Records

Every default shape gate or dispatch rule justified by benchmark data should have
a record in the relevant optimization notes. The record should include:

- Exact predicate: dtype, transpose flags, `m`, `n`, `k`, alpha/beta constraints,
  thread-count assumptions, and target-feature requirements.
- Evidence boundary: shapes that were tested, shapes that were intentionally left
  outside the gate, and any nearby regressions.
- Artifact links or paths: correctness command, focused command, full-sweep CSV,
  isolated CSV if comparators are mentioned, and summary output.
- Runtime environment: all Zynum and comparator variables from the table above.
- Rollback rule: what observation should disable, narrow, or move the gate behind
  an internal dispatch predicate or explicit non-env API/build option.

Do not broaden a gate because a single machine, shape, or non-isolated run looked
better. CPU model names are acceptable in benchmark labels, but dispatch logic
should be expressed in target features and shapes.

## Target Matrix

Use the matrix below to decide what evidence is required before a performance
statement can be called reportable. A cross-target build or test is useful for
coverage, but it is not performance evidence for hardware that did not run the
benchmark.

| Target family | Minimum validation | Reportable performance evidence |
| --- | --- | --- |
| `aarch64-macos` Apple Silicon | Native correctness tests for the selected `-Dcpu`; record target features, `ZYNUM_MAXIMUM_THREADS`, detected CPU count, and comparator settings. | Native focused runs plus full fresh-process sweep. Use Accelerate/OpenBLAS comparator data only when thread counts and isolation are recorded. |
| `aarch64-linux-gnu` ASIMD/SVE2/SME | Correctness on matching hardware or a clearly labeled cross-target check. | Native hardware sweep for the advertised feature set. Do not infer SVE2/SME throughput from ASIMD-only hosts or emulators. |
| `x86_64-linux-gnu` baseline/AVX/AVX2/AVX512 | Correctness for the selected `-Dcpu`; compile checks may cover unsupported local features. | Native Intel and/or AMD measurements for the feature tier being claimed, with MKL/OpenBLAS thread counts pinned. |
| Other supported build targets | Correctness or compile coverage as available. | Mark as unmeasured until a native fresh-process sweep exists. |

## Outlier Handling

Re-test when:

- One point is far slower than nearby shapes.
- All libraries become slower in the same region.
- A long sweep disagrees with focused single-shape runs.
- A result changes when benchmark order changes.
- A stateful worker option was enabled.
- Correctness, scalar return values, or unrelated comparator timings change
  after an SME, AMX, or other stateful instruction path has run in the same
  process.

Do not merge a performance gate based only on one lucky focused benchmark.
Do not keep a pessimistic rollback based on a contaminated run; use it only as a
pointer to shapes that need fresh correctness and isolated benchmark evidence.

## Regression Criteria

A performance change is usually acceptable only when:

- Correctness tests pass.
- Correctness still passes after any stateful kernel path used by the benchmark
  has run, including SME streaming-mode and AMX-state paths.
- Target shapes improve repeatedly.
- Non-target shapes do not show broad regressions.
- Full sweep data does not introduce catastrophic slow points.
- The retained rule is narrower than the evidence.

When in doubt, keep the optimization behind an internal dispatch predicate or
explicit non-environment API/build option until more data exists.
