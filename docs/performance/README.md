# Performance

The README compares Zynum with Apple Accelerate and OpenBLAS on Apple M5
(10 logical CPUs), macOS 27.0 (26A428), Zig 0.16.0, and OpenBLAS 0.3.34.
Zynum is built with `-Dcpu=baseline -Ddispatch=dynamic -Doptimize=ReleaseFast`;
the baseline compilation target still permits runtime selection of supported
ISA kernels.

## Scope and statistics

Each case has six cyclically interleaved fresh-process measurements per library.
Zynum uses its default thread policy. Comparator thread caps are 10, with
dynamic threading disabled where supported. Compilation, tests and profiling
are excluded from the timing interval.

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

The bundle contains 822 aggregate rows and 4,932 process-case samples. Level
1/2 use `sampled-ok` reference checks; GEMM uses `checked-ok` from its sampled
reference checker. These checks do not independently verify every output
element. Selected cases on one host do not establish an all-function or
cross-platform performance guarantee.

The metadata reader recognizes kernel-coverage schema 1 while the generator
emits schema 3. Its retained `kernel_coverage_invalid_document` diagnostic
means the metadata does not validate per-call kernel selection.

## Current data

All public benchmark assets use stable paths under
[`docs/assets/benchmarks/current/`](../assets/benchmarks/current/).

| Family | Data | Metadata | Figure |
| --- | --- | --- | --- |
| Level 1 | [CSV](../assets/benchmarks/current/level1.csv) | [JSON](../assets/benchmarks/current/level1.csv.meta.json) | [SVG](../assets/benchmarks/current/level1.svg) |
| Level 2 | [CSV](../assets/benchmarks/current/level2.csv) | [JSON](../assets/benchmarks/current/level2.csv.meta.json) | [SVG](../assets/benchmarks/current/level2.svg) |
| GEMM | [CSV](../assets/benchmarks/current/gemm.csv) | [JSON](../assets/benchmarks/current/gemm.csv.meta.json) | [SVG](../assets/benchmarks/current/gemm.svg) |

The [source manifest](../assets/benchmarks/current/source.json) identifies the
measured source contents, binaries, build settings, and host. Source hashes
identify the working-tree code used for timing; the recorded base revision
alone does not identify uncommitted source changes. Timing values are
observations and will vary when reproduced.

## Reproduce

Use the source contents and host/toolchain identified by the manifest. The
commands collect their own Git identity and binary hashes. Run the three
families sequentially on an otherwise idle machine; keep outputs local until
the data and metadata have been reviewed for publication.

```sh
zig build -Dcpu=baseline -Ddispatch=dynamic -Doptimize=ReleaseFast
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
  --build-target aarch64-macos --build-cpu baseline --build-optimize ReleaseFast \
  --csv zig-out/benchmarks/level1.csv --zynum zig-out/lib/libzynum_blas.dylib
python3 bench/tools/run_level2_report.py \
  --n 128 --n 256 --n 512 --op legacy --reps-small 100 --reps-large 30 \
  --process-repeats 6 --process-schedule interleaved \
  --build-target aarch64-macos --build-cpu baseline --build-optimize ReleaseFast \
  --csv zig-out/benchmarks/level2.csv --zynum zig-out/lib/libzynum_blas.dylib
python3 bench/tools/run_gemm_sweep_isolated.py \
  --gemm-sweep zig-out/bin/gemm-sweep --zynum-blas zig-out/lib/libzynum_blas.dylib \
  --reps 30 --process-repeats 6 --process-schedule interleaved \
  --isolate-kind --isolate-shape --check \
  --build-target aarch64-macos --build-cpu baseline --build-optimize ReleaseFast \
  --csv zig-out/benchmarks/gemm.csv
python3 bench/tools/plot_level1_report.py zig-out/benchmarks/level1.csv \
  --bars-svg zig-out/benchmarks/level1.svg \
  --ratio-svg zig-out/benchmarks/level1-ratio.svg --stat median
python3 bench/tools/plot_level2_report.py zig-out/benchmarks/level2.csv \
  --bars-svg zig-out/benchmarks/level2.svg --stat median
python3 bench/tools/plot_gemm_sweep.py zig-out/benchmarks/gemm.csv \
  zig-out/benchmarks/gemm.svg --stat median
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
