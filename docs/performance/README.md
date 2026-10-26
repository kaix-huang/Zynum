# Performance

The README compares Zynum with Apple Accelerate and OpenBLAS using measurements
from **2026-10-10 (Asia/Shanghai)** at clean source
[`4ef1065`](https://github.com/kaix-huang/Zynum/commit/4ef1065691d80439e9286f2f4d5ff758f0818e05).
The host is Apple M5 (10 logical CPUs, 32 GiB memory), macOS 27.0.1 (26A434),
with Zig 0.17.0 and OpenBLAS 0.3.34. Accelerate is the system framework supplied
with that macOS build.

Zynum is built with `-Dcpu=baseline -Ddispatch=dynamic -Doptimize=fast`;
baseline compilation still permits runtime selection of supported ISA kernels.
The build inventory, test-inventory structure, inventory-gated baseline `fast`
tests, host-tool smoke checks, and 132 dynamic-dispatch tests passed before
measurement. Native evidence here covers this AArch64 macOS host; pending
inventory rows for other targets remain pending.

## Scope and statistics

Each case has six cyclically interleaved fresh-process measurements per library.
Zynum uses its default thread policy. Comparator thread caps are 10, with
dynamic threading disabled where supported. Actual worker counts were not
instrumented. Library order cycles through Zynum, Accelerate, and OpenBLAS by
case and repeat, balancing execution positions. Compilation, tests and
profiling are excluded from the timing interval.

| Family | Cases | Measurement | Plotted statistic |
| --- | ---: | --- | --- |
| Level 1 | 46 | One-second windows; n=1,048,576 plus 8 KiB/8 MiB copies | Median process rate, Gops or GB/s |
| Level 2 | 60 | n=128/256/512; minimum positive individual Python/ctypes call time per process | Median process rate, GFLOP/s |
| GEMM | 168 | Four scalar types, 42 NN column-major shapes; 30 calibrated batches per process | GFLOP/s derived from median process-median per-call time |

Level 1 excludes the four nonstandard AXPBY extensions. Level 2 includes
foreign-call overhead and uses 100 repetitions at n=128/256 or 30 at n=512;
it is not a native batched-call benchmark. GEMM calibrates batches to at least
100 microseconds and records their call counts. It does not represent all
Level 3 routines.

The three chart datasets contain 822 aggregate rows and 4,932 process-case
samples. Level 1/2 use `sampled-ok` reference checks; GEMM uses `checked-ok`
from its sampled reference checker. These checks do not independently verify every output
element. Selected cases on one host do not establish an all-function or
cross-platform performance guarantee.

The metadata reader recognizes kernel-coverage schema 1 while the generator
emits schema 3. Its retained `kernel_coverage_invalid_document` diagnostic
means the metadata does not validate per-call kernel selection.

## Observed comparison

Each case compares Zynum's plotted median rate with the higher median rate of
Accelerate and OpenBLAS. Ratios above 1 favor Zynum. The geometric mean weights
each measured case equally; it is not a weighted application speedup. Counts
use a strict ratio of at least 1, with no noise tolerance or statistical
significance claim.

| Family | Cases at least as fast as the faster comparator | Geometric mean rate ratio |
| --- | ---: | ---: |
| Level 1 | 23 / 46 | 1.125x |
| Level 2 | 46 / 60 | 1.207x |
| GEMM | 97 / 168 | 0.942x |

These results do not establish an all-case performance lead. The CSVs retain
all six ordered samples and their dispersion, including slower cases. Small
differences may be measurement noise. Cache-resident copy bandwidth is not
DRAM bandwidth. This run also changes the OS and source relative to the older
bundle, so it does not isolate a compiler-version effect.

### Repeatability diagnostics

Large sample ranges prompted focused repeats after the complete suite, using
the same six-round interleaved policy. Level 1 repeats DDOT, DZASUM, and
8 MiB DCOPY. Level 2 repeats the 20 legacy cases at n=128, including the
variable SGEMV cases. GEMM repeats all four scalar types at M/N/K of
1024/64/1024, 256/1536/256, and 128/128/4096 with kind and shape isolation.
These diagnostics add 105 aggregate rows and 630 process-case samples; they
do not replace the chart data.

The table reports the largest within-row `(max - min) / median` sample-rate
range among the rechecked library/case rows, and the largest absolute relative
change in a row's plotted median rate between the full run and recheck.

| Diagnostic | Full-run maximum range | Recheck maximum range | Maximum median shift | Evidence |
| --- | ---: | ---: | ---: | --- |
| Level 1 | 102.6% | 68.0% | 40.1% | [CSV](../assets/benchmarks/current/level1-recheck.csv), [metadata](../assets/benchmarks/current/level1-recheck.csv.meta.json) |
| Level 2 | 68.6% | 65.1% | 8.2% | [CSV](../assets/benchmarks/current/level2-recheck.csv), [metadata](../assets/benchmarks/current/level2-recheck.csv.meta.json) |
| GEMM | 64.4% | 81.1% | 44.5% | [CSV](../assets/benchmarks/current/gemm-recheck.csv), [metadata](../assets/benchmarks/current/gemm-recheck.csv.meta.json) |

The cause of this variability was not isolated. The results describe the
recorded runs; a dispatch change would need further controlled evidence rather
than treating these medians as stable performance guarantees.

## Current data

All public benchmark assets use stable paths under
[`docs/assets/benchmarks/current/`](../assets/benchmarks/current/).

| Family | Data | Metadata | Figure |
| --- | --- | --- | --- |
| Level 1 | [CSV](../assets/benchmarks/current/level1.csv) | [JSON](../assets/benchmarks/current/level1.csv.meta.json) | [SVG](../assets/benchmarks/current/level1.svg) |
| Level 2 | [CSV](../assets/benchmarks/current/level2.csv) | [JSON](../assets/benchmarks/current/level2.csv.meta.json) | [SVG](../assets/benchmarks/current/level2.svg) |
| GEMM | [CSV](../assets/benchmarks/current/gemm.csv) | [JSON](../assets/benchmarks/current/gemm.csv.meta.json) | [SVG](../assets/benchmarks/current/gemm.svg) |

The [source manifest](../assets/benchmarks/current/source.json) identifies the
measured source contents, binaries, build settings, and host. This run used a
clean source commit; source hashes also identify the measured
source, test, build, header, and tooling contents. The manifest binds the
curated CSVs, metadata, and figures by SHA-256. Timing values are
observations and will vary when reproduced.

## Reproduce

Use the source commit and Zig 0.17.0 toolchain identified by the manifest.
Run the correctness commands recorded in `source.json` first. The commands
below reproduce the published measurement profile; each new run records its
own Git identity and binary hashes. Use a fresh output directory and run the
three families sequentially on an otherwise idle machine. Keep outputs local
until their samples, correctness checks, and metadata have been reviewed.
The sanitized environment leaves Zynum's thread and ISA caps unset.

```sh
env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" sh <<'ZYNUM_BENCHMARK'
set -eu
zig build -Dcpu=baseline -Ddispatch=dynamic -Doptimize=fast
unset ZYNUM_MAXIMUM_THREADS
export OPENBLAS_NUM_THREADS=10 OPENBLAS_DYNAMIC=0 VECLIB_MAXIMUM_THREADS=10
export MKL_NUM_THREADS=10 MKL_DYNAMIC=FALSE OMP_NUM_THREADS=10 BLIS_NUM_THREADS=10
mkdir -p zig-out/benchmarks
python3 bench/tools/run_level1_report.py \
  --level1-probe zig-out/bin/level1-probe --copy-probe zig-out/bin/dcopy-probe \
  --n 1048576 --copy-byte-size 8KiB --copy-byte-size 8MiB \
  --seconds 1 --copy-seconds 1 \
  --op sswap --op dswap --op cswap --op zswap --op isamax --op idamax \
  --op icamax --op izamax --op sscal --op saxpy --op sdot --op sasum \
  --op snrm2 --op srot --op srotm --op dscal --op daxpy --op ddot \
  --op dasum --op dnrm2 --op drot --op drotm --op sdsdot --op dsdot \
  --op csscal --op cscal --op caxpy --op cdotu --op cdotc --op scasum \
  --op scnrm2 --op csrot --op zdscal --op zscal --op zaxpy --op zdotu \
  --op zdotc --op dzasum --op dznrm2 --op zdrot --op scopy --op dcopy \
  --process-repeats 6 --process-schedule interleaved \
  --build-target aarch64-macos --build-cpu baseline --build-optimize fast \
  --csv zig-out/benchmarks/level1.csv --zynum zig-out/lib/libzynum_blas.dylib
python3 bench/tools/run_level2_report.py \
  --n 128 --n 256 --n 512 --op legacy --reps-small 100 --reps-large 30 \
  --process-repeats 6 --process-schedule interleaved \
  --build-target aarch64-macos --build-cpu baseline --build-optimize fast \
  --csv zig-out/benchmarks/level2.csv --zynum zig-out/lib/libzynum_blas.dylib
python3 bench/tools/run_gemm_sweep_isolated.py \
  --gemm-sweep zig-out/bin/gemm-sweep --zynum-blas zig-out/lib/libzynum_blas.dylib \
  --reps 30 --process-repeats 6 --process-schedule interleaved \
  --isolate-kind --isolate-shape --check \
  --build-target aarch64-macos --build-cpu baseline --build-optimize fast \
  --csv zig-out/benchmarks/gemm.csv
python3 bench/tools/plot_level1_report.py zig-out/benchmarks/level1.csv \
  --bars-svg zig-out/benchmarks/level1.svg \
  --ratio-svg zig-out/benchmarks/level1-ratio.svg --stat median
python3 bench/tools/plot_level2_report.py zig-out/benchmarks/level2.csv \
  --bars-svg zig-out/benchmarks/level2.svg --stat median
python3 bench/tools/plot_gemm_sweep.py zig-out/benchmarks/gemm.csv \
  zig-out/benchmarks/gemm.svg --stat median

# Focused repeatability diagnostic after the full suite.
python3 bench/tools/run_level1_report.py \
  --level1-probe zig-out/bin/level1-probe --copy-probe zig-out/bin/dcopy-probe \
  --n 1048576 --copy-byte-size 8KiB --copy-byte-size 8MiB --seconds 1 --copy-seconds 1 \
  --op ddot --op dzasum --op dcopy \
  --process-repeats 6 --process-schedule interleaved \
  --build-target aarch64-macos --build-cpu baseline --build-optimize fast \
  --csv zig-out/benchmarks/level1-recheck.csv \
  --zynum zig-out/lib/libzynum_blas.dylib
python3 bench/tools/run_level2_report.py \
  --n 128 --op legacy --reps-small 100 --reps-large 30 \
  --process-repeats 6 --process-schedule interleaved \
  --build-target aarch64-macos --build-cpu baseline --build-optimize fast \
  --csv zig-out/benchmarks/level2-recheck.csv \
  --zynum zig-out/lib/libzynum_blas.dylib
python3 bench/tools/run_gemm_sweep_isolated.py \
  --gemm-sweep zig-out/bin/gemm-sweep --zynum-blas zig-out/lib/libzynum_blas.dylib \
  --shape m1024_n64_k1024:1024:64:1024 \
  --shape m256_n1536_k256:256:1536:256 \
  --shape m128_n128_k4096:128:128:4096 \
  --reps 30 --process-repeats 6 --process-schedule interleaved \
  --isolate-kind --isolate-shape --check \
  --build-target aarch64-macos --build-cpu baseline --build-optimize fast \
  --csv zig-out/benchmarks/gemm-recheck.csv
ZYNUM_BENCHMARK
```

## Kernel design

- [Benchmark methodology](../common/benchmarking.md)
- [Level 1 kernels](../common/level1_optimization_notes.md)
- [Level 2 kernels](../common/level2_optimization_notes.md)
- [GEMM and structured Level 3](../common/gemm_optimization_notes.md)
- [AArch64](../aarch64/gemm_aarch64_optimization_notes.md) and
  [x86_64](../x86_64/gemm_x86_64_optimization_notes.md) kernels

Public notes describe current contracts and implementation. Raw profiling,
disassembly, experimental variants, and run-by-run records belong outside the
repository. The current measurement bundle contains the curated evidence
needed to interpret and reproduce the figures.

## Historical LAPACK, Sparse and FFT comparison (2026-10-15)

This separate campaign used an earlier finite scientific SDK on Apple M5,
macOS 27.0.1 and Zig 0.17.0 ReleaseFast with dynamic dispatch. It measured
72 shapes, including 15 ABPLaS source/trace-scale models and general size
scans, using 1,276 qualified fresh processes and 17,588 numerically checked
timed calls. Available comparators were OpenBLAS 0.3.34, Accelerate/vDSP,
FFTW 3.3.11 and benchmark-only Eigen 5.0.0. These are local eligible routes,
not a worldwide fastest-vendor ranking.

The measured Zynum shared image had SHA-256
`581b47cbaac59d00e3b7227d8fcbacb7007e0f3a91f8abeec2f9858046d5dc95`.
It predates the subsequent shared-SVD and interface revisions. These numbers
are retained historical results, not measurements of the `scientific` branch.

Cells show Zynum / fastest local comparator time in microseconds, followed
by the comparator. **Time ratios above 1 mean Zynum is slower.** This is the
inverse direction of the BLAS rate ratios earlier in this guide.

| ABPLaS scale model | Thread cap 1: times (us) | Time ratio | Thread cap 4: times (us) | Time ratio |
| --- | ---: | ---: | ---: | ---: |
| abp-lattice3-rhs1 | 0.083 / 0.166 (OpenBLAS) | 0.50x | 0.084 / 0.166 (OpenBLAS) | 0.51x |
| abp-lattice3-rhs3 | 0.125 / 0.250 (OpenBLAS) | 0.50x | 0.125 / 0.250 (OpenBLAS) | 0.50x |
| abp-anderson8 | 2775.750 / 8.375 (OpenBLAS) | 331.43x | 2803.333 / 10.167 (Accelerate) | 275.73x |
| abp-projected48-zpotrf | 2459.375 / 12.542 (OpenBLAS) | 196.09x | 2391.209 / 12.500 (OpenBLAS) | 191.30x |
| abp-projected48-zheev | 37531.250 / 505.458 (OpenBLAS) | 74.25x | 37521.083 / 499.291 (OpenBLAS) | 75.15x |
| abp-projected48-zhegv | 62693.000 / 592.792 (OpenBLAS) | 105.76x | 62857.625 / 604.042 (OpenBLAS) | 104.06x |
| abp-lanczos19-values | 18.833 / 8.875 (Accelerate) | 2.12x | 18.584 / 8.875 (Accelerate) | 2.09x |
| abp-kinetic1728-real_spmv | 80.917 / 24.708 (Accelerate) | 3.27x | 80.875 / 27.834 (Eigen) | 2.91x |
| abp-kinetic1728-complex_spmv | 100.145 / 71.980 (Eigen) | 1.39x | 100.292 / 37.999 (Eigen) | 2.64x |
| abp-kinetic1728-real_spmm | 1143.166 / 1195.875 (Accelerate) | 0.96x | 1143.709 / 1178.959 (Eigen) | 0.97x |
| abp-kinetic1728-complex_spmm | 2191.041 / 3453.959 (Eigen) | 0.63x | 2190.291 / 1722.188 (Eigen) | 1.27x |
| abp-cheby32768-c2c1d | 80596.917 / 210.917 (FFTW MEASURE) | 382.13x | 81001.688 / 131.958 (FFTW MEASURE) | 613.84x |
| abp-cheby32768-r2c1d | 78862.709 / 115.959 (FFTW MEASURE) | 680.09x | 78627.666 / 76.000 (FFTW MEASURE) | 1034.57x |
| abp-kerker12cubed-r2c3d | 75033.917 / 2.750 (fftw) | 27285.06x | 75451.959 / 25.875 (fftw) | 2916.02x |
| abp-kerker12cubed-c2r3d | 74672.333 / 2.708 (fftw) | 27574.72x | 74687.292 / 25.833 (fftw) | 2891.16x |

The statistic is the median of process medians, with three or four fresh
processes per library/shape/thread setting and a balanced cyclic order. Each
ABP model used 21 timed calls per process; general models used a predeclared
1/3/21-call policy shared by all libraries for a shape. Sparse/FFT had an
untimed warmup; LAPACK did not. Single-call general cases measure first-call
latency. Data generation, resets, queries, checks and FFT planning were outside
the execution interval; public-interface validation and internal scratch were
included. FFTW MEASURE planning can cost more than ESTIMATE; selecting it as
the fastest execution route does not compare total plan-plus-execution cost.

Caps 1/4 are requested policies, not observed worker counts. Zynum FFT was
serial; Apple caps were environment requests. vDSP comparisons covered only
the adapter's radix-2 C2C routes. Zynum Sparse revalidated mixed-index CSR on
each call, while Eigen compression/index conversion was setup work.

Model inputs were independently constructed, rather than extracted application
matrices. The 12-cubed order-6 CSR model had 58,752 entries and omitted
nonlocal projectors; SpMM with 48 columns was a batching model. The 48-order
Hermitian cases computed all vectors, the 19-order STEV case computed values,
and the 32,768-point Chebyshev models were C2C/R2C. These results do not measure
complete Hamiltonian, ELPA, MPI ghost exchange or SCF wall time. Tiny 3x3
observations approach timer granularity and do not establish a stable 2x lead.

An interrupted attempt without a terminal receipt was excluded. The original
1,188 completed processes were rechecked and 88 were resumed with the original
policy; no qualified sample was filtered by speed. Raw process records and the
original full comparison report remain external development artifacts. No
performance tuning or new measurement accompanies this consolidation.
